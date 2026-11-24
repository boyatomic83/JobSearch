"""Hybrid retrieval: hard filters in SQL + pgvector cosine ranking, in one query.

Why filters live in SQL rather than in the LLM prompt: they're cheap, exact and
explainable ("remote only" must never be violated), and they leave the LLM to judge only
what SQL can't — fit.
"""

import re
from dataclasses import dataclass

from sqlalchemy import Select, exists, func, not_, or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Company, Job, Match, Preference
from app.matching.ghost import GHOST_THRESHOLD

RETRIEVE_K = 50

# Titles to exclude for each seniority preference (Postgres regex, \m \M = word bounds).
_SENIORITY_EXCLUDE = {
    "junior": r"\m(senior|sr|staff|principal|lead|director|head|vp|manager)\M",
    "mid": r"\m(principal|director|head|vp|distinguished)\M",
    "senior": r"\m(junior|jr|new grad|graduate|entry)\M",
    "lead": r"\m(junior|jr|new grad|graduate|entry|associate)\M",
}
_INTERN = r"\m(intern|internship|co-?op|apprentice)\M"
# A remote job's location with no region restriction ("Remote", "Anywhere", "Worldwide").
# "Remote - USA" or "US Only" are restricted, and must match a preferred location instead.
_UNRESTRICTED_REMOTE = r"^\s*(fully\s+)?(remote|anywhere|worldwide|global)\s*(\(.*\))?\s*$"


@dataclass
class Candidate:
    job_id: int
    similarity: float
    title: str
    company: str
    location: str | None
    remote: bool
    salary_min: int | None
    salary_max: int | None
    salary_currency: str | None
    url: str
    embedding_text: str


def _like_escape(value: str) -> str:
    return re.sub(r"([\\%_])", r"\\\1", value.strip())


def build_query(query_vec: list[float], prefs: Preference, user_id: int, k: int) -> Select:
    distance = Job.embedding.cosine_distance(query_vec)
    already_shown = exists().where(Match.user_id == user_id, Match.job_id == Job.id)
    conditions = [
        Job.is_active,
        Job.embedding.is_not(None),
        Job.is_duplicate_of.is_(None),
        Job.ghost_score <= GHOST_THRESHOLD,
        not_(already_shown),
    ]

    # Location. A job's location text must mention a preferred location (city or country),
    # except fully unrestricted remote jobs, which anyone can take. Remote-only additionally
    # requires the job to be remote. Measured bug this prevents: a "Remote - USA" job being
    # recommended to someone in Bangalore.
    in_preferred = [Job.location.ilike(f"%{_like_escape(loc)}%") for loc in prefs.locations]
    unrestricted_remote = Job.remote & func.coalesce(Job.location, "Remote").regexp_match(
        _UNRESTRICTED_REMOTE, flags="i"
    )
    if prefs.remote_only:
        conditions.append(Job.remote)
    if prefs.locations:
        conditions.append(or_(unrestricted_remote, *in_preferred))

    # Salary: only drop jobs whose *known* pay in the *same currency* is below the minimum.
    # Half of postings have no salary; excluding them would hide most of the market.
    if prefs.min_salary:
        best_pay = func.coalesce(Job.salary_max, Job.salary_min)
        conditions.append(
            or_(
                best_pay.is_(None),
                func.coalesce(Job.salary_currency, "") != prefs.salary_currency,
                best_pay >= prefs.min_salary,
            )
        )

    if prefs.excluded_companies:
        lowered = [c.strip().lower() for c in prefs.excluded_companies]
        conditions.append(func.lower(Job.company).not_in(lowered))

    if prefs.excluded_industries:
        lowered = [i.strip().lower() for i in prefs.excluded_industries]
        conditions.append(
            or_(Company.industry.is_(None), func.lower(Company.industry).not_in(lowered))
        )

    if prefs.seniority == "intern":
        conditions.append(Job.title.regexp_match(_INTERN, flags="i"))
    else:
        # Interns' postings are noise for everyone else.
        conditions.append(not_(Job.title.regexp_match(_INTERN, flags="i")))
        if pattern := _SENIORITY_EXCLUDE.get(prefs.seniority or ""):
            conditions.append(not_(Job.title.regexp_match(pattern, flags="i")))

    return (
        select(
            Job.id,
            (1 - distance).label("similarity"),
            Job.title,
            Job.company,
            Job.location,
            Job.remote,
            Job.salary_min,
            Job.salary_max,
            Job.salary_currency,
            Job.url,
            Job.embedding_text,
        )
        .outerjoin(Company, Company.id == Job.company_id)
        .where(*conditions)
        .order_by(distance)
        .limit(k)
    )


async def retrieve_candidates(
    session: AsyncSession,
    query_vec: list[float],
    prefs: Preference,
    user_id: int,
    k: int = RETRIEVE_K,
) -> list[Candidate]:
    # With filters, a plain HNSW scan returns the ~40 nearest vectors and *then* filters,
    # which can silently leave far fewer than k rows. pgvector 0.8's iterative scan keeps
    # searching until k rows pass the filters. SET LOCAL lasts only for this transaction.
    await session.execute(text("SET LOCAL hnsw.iterative_scan = 'relaxed_order'"))
    await session.execute(text("SET LOCAL hnsw.ef_search = 200"))
    rows = (await session.execute(build_query(query_vec, prefs, user_id, k))).all()
    candidates = [
        Candidate(
            job_id=r.id,
            similarity=float(r.similarity),
            title=r.title,
            company=r.company,
            location=r.location,
            remote=r.remote,
            salary_min=r.salary_min,
            salary_max=r.salary_max,
            salary_currency=r.salary_currency,
            url=r.url,
            embedding_text=r.embedding_text,
        )
        for r in rows
    ]
    # relaxed_order may return rows slightly out of order; restore exact ranking.
    return sorted(candidates, key=lambda c: c.similarity, reverse=True)
