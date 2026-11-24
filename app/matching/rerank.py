"""LLM rerank: 20 retrieved candidates → the best 3, with reasons, in ONE call.

One call (not one per job) keeps us far inside free-tier request limits and lets the
model compare jobs against each other, which gives better-calibrated scores.
"""

import json
import logging

from pydantic import BaseModel, Field

from app.db.models import Preference
from app.llm.client import LLMClient, LLMError
from app.matching.retrieval import Candidate
from app.resume.schemas import ResumeProfile

logger = logging.getLogger(__name__)

RERANK_K = 20
JOB_SNIPPET_CHARS = 500  # per candidate; 20 x 500 chars keeps the prompt ~3k tokens


class RankedJob(BaseModel):
    job_id: int
    fit_score: int = Field(ge=0, le=100)
    reasons: list[str] = Field(min_length=1, max_length=3, description="2-3 concrete reasons")
    gap: str = Field(description="The single biggest gap or risk for this candidate")


class RerankResult(BaseModel):
    matches: list[RankedJob]


RERANK_SYSTEM = """You are a precise technical recruiter. Pick the jobs that best fit the
candidate and explain why.
Rules:
- Only use job_ids from the list.
- Ground every statement in the text shown: the candidate profile and the job snippet.
  Never use outside knowledge about a company or its tech stack.
- fit_score 0-100: 90+ near-perfect, 70-89 strong, 50-69 partial, <50 weak.
- reasons: 2-3 short, specific bullets that tie a resume fact to something the job text
  asks for (e.g. "5 yrs Python + FastAPI matches 'build Python APIs'"). No generic praise.
  Don't cite location, seniority or salary: preferences were already applied as filters.
- gap: the single most important requirement in the job text the resume doesn't show.
  If the snippet is too short to tell, say what is unclear."""


def _salary(c: Candidate) -> str:
    if c.salary_min or c.salary_max:
        return f"{c.salary_currency or ''} {c.salary_min or '?'}-{c.salary_max or '?'}".strip()
    return "salary not listed"


def _snippet(c: Candidate) -> str:
    # embedding_text starts with "title\ncompany\n\n"; we already show those in the header.
    body = c.embedding_text.split("\n\n", 1)[-1]
    return body[:JOB_SNIPPET_CHARS].replace("\n", " ")


def build_prompt(
    profile: ResumeProfile, prefs: Preference, candidates: list[Candidate], n: int
) -> str:
    pref_lines = {
        "target_roles": prefs.roles,
        "locations": prefs.locations,
        "remote_only": prefs.remote_only,
        "min_salary": f"{prefs.min_salary} {prefs.salary_currency}" if prefs.min_salary else None,
        "seniority": prefs.seniority,
        "notes": prefs.notes,
    }
    jobs = "\n\n".join(
        f"job_id: {c.job_id} | {c.title} | {c.company} | {c.location or 'location n/a'}"
        f"{' | remote' if c.remote else ''} | {_salary(c)}\n{_snippet(c)}"
        for c in candidates
    )
    return (
        f"Candidate profile:\n{profile.model_dump_json(exclude_none=True)}\n\n"
        f"Preferences:\n{json.dumps({k: v for k, v in pref_lines.items() if v})}\n\n"
        f"Jobs:\n{jobs}\n\n"
        f"Return the best {n} jobs, best first."
    )


async def rerank(
    llm: LLMClient,
    profile: ResumeProfile,
    prefs: Preference,
    candidates: list[Candidate],
    *,
    user_id: int | None,
    n: int = 3,
) -> list[tuple[Candidate, RankedJob | None]]:
    """Return up to n (candidate, ranking) pairs, best first.

    If the LLM fails or returns too few valid ids, we fill up from retrieval order with
    ranking=None, so the user still gets jobs (clearly marked as unranked).
    """
    pool = candidates[:RERANK_K]
    by_id = {c.job_id: c for c in pool}
    ranked: list[RankedJob] = []
    if pool:
        try:
            result = await llm.structured(
                purpose="rerank",
                system=RERANK_SYSTEM,
                prompt=build_prompt(profile, prefs, pool, n),
                schema=RerankResult,
                user_id=user_id,
            )
            seen: set[int] = set()
            for m in sorted(result.matches, key=lambda m: m.fit_score, reverse=True):
                if m.job_id in by_id and m.job_id not in seen:  # drop hallucinated ids
                    seen.add(m.job_id)
                    ranked.append(m)
        except LLMError:
            logger.exception("rerank failed; falling back to retrieval order")

    out: list[tuple[Candidate, RankedJob | None]] = [(by_id[m.job_id], m) for m in ranked[:n]]
    used = {c.job_id for c, _ in out}
    for c in pool:
        if len(out) >= n:
            break
        if c.job_id not in used:
            out.append((c, None))
    return out
