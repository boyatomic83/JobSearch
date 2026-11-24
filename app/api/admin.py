"""Admin endpoints, protected by the ADMIN_API_KEY sent as the `X-Admin-Key` header."""

import asyncio
import secrets
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, status

from app.config import get_settings
from app.db.session import get_sessionmaker
from app.ingest.service import build_connectors, run_ingest
from app.matching.embeddings import get_embedder
from app.matching.process import run_processing

router = APIRouter(prefix="/admin", tags=["admin"])

# Prevents two overlapping runs (e.g. a double-click while the scheduler is also running).
_ingest_lock = asyncio.Lock()


def require_admin(x_admin_key: Annotated[str | None, Header()] = None) -> None:
    expected = get_settings().admin_api_key
    if not expected:  # fail closed: no key configured means admin is disabled
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "admin API disabled")
    if not x_admin_key or not secrets.compare_digest(x_admin_key, expected):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid admin key")


@router.post("/ingest", dependencies=[Depends(require_admin)])
async def ingest(
    source: Annotated[list[Literal["greenhouse", "lever", "remoteok"]] | None, Query()] = None,
    process: bool = True,
) -> dict:
    """Run ingestion now (+ embed / dedup / ghost-score unless process=false) and report.

    Takes ~1 minute for all feeds, plus embedding time for new jobs.
    """
    if _ingest_lock.locked():
        raise HTTPException(status.HTTP_409_CONFLICT, "an ingest run is already in progress")
    async with _ingest_lock:
        report = await run_ingest(
            get_sessionmaker(), build_connectors(list(source) if source else None)
        )
        result = report.to_dict()
        if process:
            processed = await run_processing(get_sessionmaker(), get_embedder())
            result["processing"] = processed.to_dict()
    return result


@router.post("/digest", dependencies=[Depends(require_admin)])
async def digest(request: Request) -> dict:
    """Send the daily digest now (demo / external cron when the instance was asleep)."""
    from app.scheduler import send_daily_digest

    tg = getattr(request.app.state, "telegram", None)
    if tg is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "bot not running")
    return await send_daily_digest(tg.bot, request.app.state.bot_deps)
