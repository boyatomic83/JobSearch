"""Public, unauthenticated endpoints: the landing page and its live numbers.

Only aggregate job-market numbers are exposed — never anything about users.
"""

import time
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.session import get_sessionmaker
from app.matching.ghost import GHOST_THRESHOLD

router = APIRouter(tags=["public"])

INDEX = Path(__file__).resolve().parent.parent / "web" / "index.html"
CACHE_SECONDS = 600  # page views must not hammer the free-tier database
_cache: dict = {"at": 0.0, "data": None}

_STATS_SQL = text("""
    SELECT count(*) FILTER (WHERE is_duplicate_of IS NULL AND ghost_score <= :ghost) AS open_jobs,
           count(DISTINCT company) AS companies,
           count(*) FILTER (WHERE remote AND is_duplicate_of IS NULL) AS remote_jobs,
           count(*) FILTER (WHERE is_duplicate_of IS NOT NULL) AS duplicates_hidden,
           count(*) FILTER (WHERE ghost_score > :ghost) AS ghost_jobs_hidden,
           count(*) FILTER (WHERE salary_min IS NOT NULL OR salary_max IS NOT NULL) AS with_salary,
           max(last_seen_at) AS last_refreshed
    FROM jobs WHERE is_active
""")


@router.get("/", include_in_schema=False)
async def landing() -> FileResponse:
    return FileResponse(INDEX, media_type="text/html")


@router.get("/public/stats")
async def public_stats(
    sessions: Annotated[async_sessionmaker[AsyncSession], Depends(get_sessionmaker)],
) -> dict:
    """Live job-market numbers for the landing page (cached for 10 minutes)."""
    if _cache["data"] is not None and time.monotonic() - _cache["at"] < CACHE_SECONDS:
        return _cache["data"]
    async with sessions() as session:
        row = (await session.execute(_STATS_SQL, {"ghost": GHOST_THRESHOLD})).mappings().one()
        sources = (
            await session.execute(
                text("SELECT source, count(*) FROM jobs WHERE is_active GROUP BY source")
            )
        ).all()
    data = {k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in row.items()}
    data["sources"] = {s: n for s, n in sources}
    _cache.update(at=time.monotonic(), data=data)
    return data
