"""Find matches for a user: profile → query vector → SQL retrieval → LLM rerank → store.

LLM calls per run: exactly one (the rerank). Resume extraction happens once at upload.
"""

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models import Match
from app.db.queries import get_active_resume, get_or_create_preferences
from app.llm.client import LLMClient
from app.matching.embeddings import Embedder
from app.matching.rerank import RankedJob, rerank
from app.matching.retrieval import Candidate, retrieve_candidates
from app.resume.schemas import ResumeProfile


class NoResumeError(Exception):
    """The user must upload a resume before we can match."""


@dataclass
class MatchResult:
    match_id: int
    candidate: Candidate
    ranking: RankedJob | None  # None = LLM rerank unavailable, shown in retrieval order


async def find_matches(
    sessionmaker: async_sessionmaker[AsyncSession],
    llm: LLMClient,
    embedder: Embedder,
    user_id: int,
    *,
    n: int = 3,
    query: str | None = None,
) -> list[MatchResult]:
    async with sessionmaker() as session:
        resume = await get_active_resume(session, user_id)
        if resume is None or resume.profile is None:
            raise NoResumeError(user_id)
        prefs = await get_or_create_preferences(session, user_id)
        await session.commit()
    profile = ResumeProfile.model_validate(resume.profile)

    # Re-embed at match time (fast, local): target roles in preferences may have changed
    # since the upload, and an explicit query ("ML infra roles") should steer the search.
    query_text = profile.embedding_text(prefs.roles)
    if query:
        query_text = f"Looking for: {query}\n{query_text}"
    query_vec = await asyncio.to_thread(embedder.embed_query, query_text)

    async with sessionmaker() as session, session.begin():
        candidates = await retrieve_candidates(session, query_vec, prefs, user_id)

    ranked = await rerank(llm, profile, prefs, candidates, user_id=user_id, n=n)

    now = datetime.now(UTC)
    results: list[MatchResult] = []
    async with sessionmaker() as session, session.begin():
        for cand, ranking in ranked:
            match = Match(
                user_id=user_id,
                job_id=cand.job_id,
                retrieval_score=round(cand.similarity, 4),
                fit_score=ranking.fit_score if ranking else None,
                reasons=ranking.reasons if ranking else [],
                gap=ranking.gap if ranking else None,
                shown_at=now,
            )
            session.add(match)
            await session.flush()
            results.append(MatchResult(match.id, cand, ranking))
    return results
