"""Shared eval helpers: load the synthetic resumes and run retrieval without side effects."""

import json
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Preference
from app.matching.retrieval import Candidate, retrieve_candidates
from app.resume.schemas import ResumeProfile

EVALS = Path(__file__).parent
EVAL_USER_ID = -1  # no matches exist for this id, so nothing is excluded as "already shown"


def load_resumes() -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted((EVALS / "resumes").glob("*.json"))]


def preferences(resume: dict) -> Preference:
    """A transient (never saved) Preference with the resume's eval preferences."""
    p = resume["preferences"]
    return Preference(
        roles=p.get("roles", []),
        locations=p.get("locations", []),
        remote_only=p.get("remote_only", False),
        min_salary=p.get("min_salary"),
        salary_currency=p.get("salary_currency", "USD"),
        seniority=p.get("seniority"),
        excluded_companies=[],
        excluded_industries=[],
        notes=[],
    )


def profile(resume: dict) -> ResumeProfile:
    return ResumeProfile.model_validate(resume["profile"])


async def retrieve(session: AsyncSession, embedder, resume: dict, k: int) -> list[Candidate]:
    prefs = preferences(resume)
    vec = embedder.embed_query(profile(resume).embedding_text(prefs.roles))
    return await retrieve_candidates(session, vec, prefs, EVAL_USER_ID, k=k)


def job_key(source: str, source_id: str) -> str:
    """Stable across databases (local / Neon), unlike our auto-increment job ids."""
    return f"{source}:{source_id}"
