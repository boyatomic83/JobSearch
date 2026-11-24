"""The agent's tools. Each = name + description + Pydantic args model + async handler.

Arguments are validated with Pydantic before a handler runs, and every handler returns a
JSON-able dict. Errors are returned to the model as {"error": ...} (never raised), so the
model can read them and recover in its next step.
"""

import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.agent.preferences import PreferenceChanges, apply_changes, describe
from app.agent.text_to_sql import answer_question
from app.db.models import Job, Match
from app.db.queries import get_active_resume, get_or_create_preferences
from app.llm.base import ToolCall, ToolSpec
from app.llm.client import LLMClient
from app.matching.embeddings import Embedder
from app.matching.service import NoResumeError, find_matches
from app.resume.tailor_service import TailorError, tailor_for_job

logger = logging.getLogger(__name__)


@dataclass
class AgentContext:
    sessionmaker: async_sessionmaker[AsyncSession]
    engine: AsyncEngine
    llm: LLMClient  # the agent's own turns
    embedder: Embedder
    user_id: int
    task_llm: LLMClient | None = None  # used *inside* tools (rerank, SQL); defaults to llm

    @property
    def tools_llm(self) -> LLMClient:
        return self.task_llm or self.llm


Handler = Callable[[AgentContext, Any], Awaitable[dict[str, Any]]]


@dataclass
class Tool:
    name: str
    description: str
    args: type[BaseModel]
    handler: Handler

    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, _compact(self.args.model_json_schema()))


def _compact(schema: Any) -> Any:
    """Drop Pydantic's auto-generated "title" keys: ~20% fewer tokens on every agent call."""
    if isinstance(schema, dict):
        return {
            k: _compact(v) for k, v in schema.items() if not (k == "title" and isinstance(v, str))
        }
    if isinstance(schema, list):
        return [_compact(v) for v in schema]
    return schema


# ---- argument models --------------------------------------------------------------


class NoArgs(BaseModel):
    pass


class SearchFilters(BaseModel):
    """One-off overrides for this search only; lasting changes go to update_preferences."""

    remote_only: bool | None = None
    locations: list[str] | None = None
    min_salary: int | None = Field(default=None, ge=0)
    seniority: Literal["intern", "junior", "mid", "senior", "lead"] | None = None


class SearchArgs(BaseModel):
    query: str | None = Field(
        default=None, description="What to look for, e.g. 'ML infrastructure roles'"
    )
    filters: SearchFilters | None = None


class JobArgs(BaseModel):
    # Bounded to the DB's int range: models sometimes pass a number copied from a URL.
    job_id: int = Field(ge=1, le=2_147_483_647, description="Our job_id (not a URL number)")


class SQLArgs(BaseModel):
    question: str = Field(description="A question about the job data, in plain English")


# ---- handlers ---------------------------------------------------------------------


async def search_jobs(ctx: AgentContext, args: SearchArgs) -> dict[str, Any]:
    override = args.filters.model_dump(exclude_none=True) if args.filters else None
    try:
        results = await find_matches(
            ctx.sessionmaker, ctx.tools_llm, ctx.embedder, ctx.user_id,
            query=args.query, prefs_override=override,
        )  # fmt: skip
    except NoResumeError:
        return {"error": "No resume on file. Ask the user to upload one with /resume."}
    if not results:
        return {"jobs": [], "note": "No new jobs match the current preferences."}
    return {
        "jobs": [
            {
                "job_id": r.candidate.job_id,
                "title": r.candidate.title,
                "company": r.candidate.company,
                "location": r.candidate.location,
                "remote": r.candidate.remote,
                "url": r.candidate.url,
                "fit_score": r.ranking.fit_score if r.ranking else None,
                "reasons": r.ranking.reasons if r.ranking else [],
                "gap": r.ranking.gap if r.ranking else None,
            }
            for r in results
        ]
    }


async def update_preferences(ctx: AgentContext, args: PreferenceChanges) -> dict[str, Any]:
    async with ctx.sessionmaker() as session, session.begin():
        prefs = await get_or_create_preferences(session, ctx.user_id)
        changed = apply_changes(prefs, args)
        return {"changed": changed or ["nothing (already set)"], "preferences": describe(prefs)}


async def get_preferences(ctx: AgentContext, _: NoArgs) -> dict[str, Any]:
    async with ctx.sessionmaker() as session, session.begin():
        return describe(await get_or_create_preferences(session, ctx.user_id))


async def explain_match(ctx: AgentContext, args: JobArgs) -> dict[str, Any]:
    async with ctx.sessionmaker() as session:
        job = await session.get(Job, args.job_id)
        if job is None:
            return {"error": f"job {args.job_id} not found"}
        match = await session.scalar(
            select(Match).where(Match.user_id == ctx.user_id, Match.job_id == job.id)
        )
        resume = await get_active_resume(session, ctx.user_id)
    # The candidate's real profile travels with the job: without it the model fills in
    # "experience" from imagination (seen in testing: invented NLP projects, GCP, CI/CD).
    profile = resume.profile if resume else None
    return {
        "candidate_profile": profile,
        "job": {
            "job_id": job.id, "title": job.title, "company": job.company,
            "location": job.location, "remote": job.remote, "url": job.url,
            "salary": [job.salary_min, job.salary_max, job.salary_currency],
            "posted_at": job.posted_at.isoformat() if job.posted_at else None,
            "ghost_score": job.ghost_score, "ghost_reasons": job.ghost_reasons,
            "is_duplicate": job.is_duplicate_of is not None,
            "description": job.embedding_text.split("\n\n", 1)[-1][:1500],
        },
        "match": {"fit_score": match.fit_score, "reasons": match.reasons, "gap": match.gap}
        if match
        else None,
    }  # fmt: skip


async def tailor_resume(ctx: AgentContext, args: JobArgs) -> dict[str, Any]:
    try:
        record = await tailor_for_job(ctx.sessionmaker, ctx.tools_llm, ctx.user_id, args.job_id)
    except TailorError as exc:
        return {"error": str(exc)}
    content = record.content
    return {
        "tailored_resume_id": record.id,
        "pdf": "will be sent to the user as a file",
        "summary": content["summary"],
        "top_skills": content["skills"][:8],
        "claims_removed_by_fact_check": [d["claim"] for d in record.dropped],
    }


async def run_sql_readonly(ctx: AgentContext, args: SQLArgs) -> dict[str, Any]:
    return await answer_question(ctx.tools_llm, ctx.engine, args.question, user_id=ctx.user_id)


TOOLS: dict[str, Tool] = {
    t.name: t
    for t in [
        Tool(
            "search_jobs",
            "Find the best new matching jobs for the user (uses their resume and saved "
            "preferences; returns up to 3 with fit score and reasons).",
            SearchArgs,
            search_jobs,
        ),
        Tool(
            "update_preferences",
            "Save a lasting change to the user's job preferences (their long-term memory). "
            "Use add/remove lists; only send fields that change.",
            PreferenceChanges,
            update_preferences,
        ),
        Tool("get_preferences", "Read the user's saved preferences.", NoArgs, get_preferences),
        Tool(
            "explain_match",
            "Details for one job: description, salary, ghost-job signals and, if it was "
            "recommended, the stored fit score/reasons/gap.",
            JobArgs,
            explain_match,
        ),
        Tool(
            "tailor_resume",
            "Rewrite the user's resume for one job using only facts from their original "
            "resume (fact-checked); the PDF is sent to them automatically.",
            JobArgs,
            tailor_resume,
        ),
        Tool(
            "run_sql_readonly",
            "Answer counting/statistics questions about the job database, e.g. 'how many "
            "remote Python jobs in Bangalore were posted this week?'.",
            SQLArgs,
            run_sql_readonly,
        ),
    ]
}


async def run_tool(ctx: AgentContext, call: ToolCall) -> dict[str, Any]:
    tool = TOOLS.get(call.name)
    if tool is None:
        return {"error": f"unknown tool {call.name!r}; available: {sorted(TOOLS)}"}
    if "__invalid_json__" in call.arguments:
        return {"error": "arguments were not valid JSON"}
    try:
        args = tool.args.model_validate(call.arguments)
    except ValidationError as exc:
        return {"error": f"invalid arguments: {exc.errors(include_url=False)}"}
    try:
        return await tool.handler(ctx, args)
    except Exception as exc:  # a tool bug must not crash the conversation
        logger.exception("tool %s failed", call.name)
        return {"error": f"{call.name} failed: {type(exc).__name__}"}


def result_json(result: dict[str, Any]) -> str:
    return json.dumps(result, default=str, ensure_ascii=False)
