"""User endpoints: create a user, set preferences, upload a resume, get matches.

The Telegram bot (Phase 7) calls the same services directly; these HTTP endpoints exist
for testing and demos, and are protected by the admin key like everything non-public.
"""

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agent.loop import handle_message
from app.agent.tools import AgentContext
from app.api.admin import require_admin
from app.db.models import TailoredResumeRecord, User
from app.db.queries import get_or_create_preferences, get_user
from app.db.session import get_engine, get_sessionmaker
from app.llm.client import LLMClient, LLMError, get_agent_llm_client, get_llm_client
from app.matching.embeddings import Embedder, get_embedder
from app.matching.service import NoResumeError, find_matches
from app.resume.parse import ResumeError
from app.resume.schemas import ResumeProfile
from app.resume.service import save_resume
from app.resume.tailor_service import TailorError, tailor_for_job, tailored_pdf

router = APIRouter(prefix="/users", tags=["users"], dependencies=[Depends(require_admin)])

Sessions = Annotated[async_sessionmaker[AsyncSession], Depends(get_sessionmaker)]
LLM = Annotated[LLMClient, Depends(get_llm_client)]
AgentLLM = Annotated[LLMClient, Depends(get_agent_llm_client)]
Emb = Annotated[Embedder, Depends(get_embedder)]


class UserIn(BaseModel):
    name: str | None = None
    telegram_chat_id: int | None = None


class PreferencesIn(BaseModel):
    """Partial update: only fields that are sent are changed."""

    roles: list[str] | None = None
    locations: list[str] | None = None
    remote_only: bool | None = None
    min_salary: int | None = Field(default=None, ge=0)
    salary_currency: str | None = Field(default=None, min_length=3, max_length=3)
    seniority: Literal["intern", "junior", "mid", "senior", "lead"] | None = None
    excluded_companies: list[str] | None = None
    excluded_industries: list[str] | None = None


class PreferencesOut(PreferencesIn):
    notes: list[str] = []


class MatchOut(BaseModel):
    match_id: int
    job_id: int
    title: str
    company: str
    location: str | None
    remote: bool
    url: str
    similarity: float
    fit_score: int | None
    reasons: list[str]
    gap: str | None


async def _require_user(session: AsyncSession, user_id: int) -> User:
    user = await get_user(session, user_id)
    if user is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "user not found")
    return user


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_user(body: UserIn, sessions: Sessions) -> dict:
    async with sessions() as session, session.begin():
        user = User(name=body.name, telegram_chat_id=body.telegram_chat_id)
        session.add(user)
        await session.flush()
        await get_or_create_preferences(session, user.id)
        return {"id": user.id, "name": user.name}


@router.put("/{user_id}/preferences")
async def update_preferences(
    user_id: int, body: PreferencesIn, sessions: Sessions
) -> PreferencesOut:
    async with sessions() as session, session.begin():
        await _require_user(session, user_id)
        prefs = await get_or_create_preferences(session, user_id)
        for field, value in body.model_dump(exclude_unset=True).items():
            if value is not None:
                setattr(prefs, field, value.upper() if field == "salary_currency" else value)
        return PreferencesOut.model_validate(prefs, from_attributes=True)


@router.post("/{user_id}/resume")
async def upload_resume(
    user_id: int,
    sessions: Sessions,
    llm: LLM,
    embedder: Emb,
    file: Annotated[UploadFile, File(description="Resume as PDF")],
) -> ResumeProfile:
    async with sessions() as session:
        await _require_user(session, user_id)
    try:
        _, profile = await save_resume(
            sessions, llm, embedder, user_id=user_id, file_name=file.filename,
            pdf_bytes=await file.read(),
        )  # fmt: skip
    except ResumeError as exc:
        raise HTTPException(422, str(exc)) from exc
    except LLMError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "LLM unavailable, retry") from exc
    return profile


@router.post("/{user_id}/matches")
async def get_matches(
    user_id: int, sessions: Sessions, llm: LLM, embedder: Emb, n: int = 3
) -> list[MatchOut]:
    async with sessions() as session:
        await _require_user(session, user_id)
    try:
        results = await find_matches(sessions, llm, embedder, user_id, n=min(n, 10))
    except NoResumeError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, "upload a resume first") from exc
    return [
        MatchOut(
            match_id=r.match_id,
            job_id=r.candidate.job_id,
            title=r.candidate.title,
            company=r.candidate.company,
            location=r.candidate.location,
            remote=r.candidate.remote,
            url=r.candidate.url,
            similarity=round(r.candidate.similarity, 3),
            fit_score=r.ranking.fit_score if r.ranking else None,
            reasons=r.ranking.reasons if r.ranking else [],
            gap=r.ranking.gap if r.ranking else None,
        )
        for r in results
    ]


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=4000)


@router.post("/{user_id}/chat")
async def chat(
    user_id: int, body: ChatIn, sessions: Sessions, llm: LLM, agent_llm: AgentLLM, embedder: Emb
) -> dict:
    """Talk to the agent (the Telegram bot sends free-text messages here internally)."""
    async with sessions() as session:
        await _require_user(session, user_id)
    ctx = AgentContext(sessions, get_engine(), agent_llm, embedder, user_id, task_llm=llm)
    reply = await handle_message(ctx, body.message)
    return {
        "reply": reply.text,
        "steps": reply.steps,
        "tailored_resume_ids": reply.tailored_resume_ids,
    }


@router.post("/{user_id}/tailor/{job_id}")
async def tailor(user_id: int, job_id: int, sessions: Sessions, llm: LLM) -> dict:
    """Tailor the active resume for a job; returns the content and what was fact-checked out."""
    async with sessions() as session:
        await _require_user(session, user_id)
    try:
        record = await tailor_for_job(sessions, llm, user_id, job_id)
    except TailorError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    except LLMError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "LLM unavailable, retry") from exc
    return {"id": record.id, "content": record.content, "dropped": record.dropped}


@router.get("/{user_id}/tailored/{tailored_id}.pdf")
async def tailored_resume_pdf(user_id: int, tailored_id: int, sessions: Sessions) -> Response:
    async with sessions() as session:
        record = await session.get(TailoredResumeRecord, tailored_id)
    if record is None or record.user_id != user_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "tailored resume not found")
    pdf = await tailored_pdf(sessions, record)
    return Response(
        pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="resume-{tailored_id}.pdf"'},
    )
