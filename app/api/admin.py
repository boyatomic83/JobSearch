"""Admin endpoints, protected by the ADMIN_API_KEY sent as the `X-Admin-Key` header."""

import asyncio
import secrets
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status

from app.config import get_settings
from app.db.session import get_sessionmaker
from app.ingest.service import build_connectors, run_ingest

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
) -> dict:
    """Run ingestion now and return the per-feed report (takes ~1 minute for all feeds)."""
    if _ingest_lock.locked():
        raise HTTPException(status.HTTP_409_CONFLICT, "an ingest run is already in progress")
    async with _ingest_lock:
        report = await run_ingest(
            get_sessionmaker(), build_connectors(list(source) if source else None)
        )
    return report.to_dict()
