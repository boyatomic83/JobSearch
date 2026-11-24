"""Admin endpoints, protected by the ADMIN_API_KEY sent as the `X-Admin-Key` header."""

import asyncio
import secrets
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, status
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

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


_STATS_SQL = {
    "jobs": """
        SELECT count(*) AS total,
               count(*) FILTER (WHERE is_active) AS active,
               count(*) FILTER (WHERE is_active AND is_duplicate_of IS NOT NULL) AS duplicates,
               count(*) FILTER (WHERE is_active AND ghost_score > :ghost) AS ghost_jobs,
               count(*) FILTER (WHERE is_active AND remote) AS remote,
               max(last_seen_at) AS last_ingest
        FROM jobs""",
    "by_source": """
        SELECT source, count(*) AS active FROM jobs WHERE is_active GROUP BY source
        ORDER BY active DESC""",
    "users": """
        SELECT count(*) AS total,
               count(*) FILTER (WHERE telegram_chat_id IS NOT NULL) AS telegram,
               count(*) FILTER (WHERE EXISTS (SELECT 1 FROM resumes r
                                WHERE r.user_id = users.id AND r.is_active)) AS with_resume
        FROM users""",
    "matches": """
        SELECT count(*) AS shown,
               count(*) FILTER (WHERE feedback = 1) AS thumbs_up,
               count(*) FILTER (WHERE feedback = -1) AS thumbs_down
        FROM matches""",
    "llm_by_purpose": """
        SELECT purpose, model, count(*) AS calls,
               count(*) FILTER (WHERE cached) AS cached,
               count(*) FILTER (WHERE status <> 'ok') AS failed,
               sum(prompt_tokens + completion_tokens) AS tokens,
               round(avg(latency_ms) FILTER (WHERE NOT cached)) AS avg_latency_ms,
               sum(cost_usd) AS est_cost_usd
        FROM llm_calls WHERE created_at > now() - interval '7 days'
        GROUP BY purpose, model ORDER BY tokens DESC NULLS LAST""",
    # Per user per day (IST): the number the spec asks for — what one user costs us daily.
    "per_user_per_day": """
        SELECT (created_at AT TIME ZONE 'Asia/Kolkata')::date AS day, user_id,
               count(*) AS calls, sum(prompt_tokens + completion_tokens) AS tokens,
               sum(cost_usd) AS est_cost_usd
        FROM llm_calls
        WHERE created_at > now() - interval '7 days' AND user_id IS NOT NULL
        GROUP BY 1, 2 ORDER BY 1 DESC, 5 DESC""",
}


@router.get("/stats", dependencies=[Depends(require_admin)])
async def stats(
    sessions: Annotated[async_sessionmaker[AsyncSession], Depends(get_sessionmaker)],
) -> dict:
    """Operational numbers: ingestion, data quality, users, feedback, LLM usage and cost.

    Costs are *estimates* at the provider's paid list prices (app/llm/pricing.py); on the
    free tier the actual bill is $0.
    """
    from sqlalchemy import text

    from app.agent.text_to_sql import _jsonable
    from app.matching.ghost import GHOST_THRESHOLD

    out: dict = {}
    async with sessions() as session:
        for key, sql in _STATS_SQL.items():
            rows = (await session.execute(text(sql), {"ghost": GHOST_THRESHOLD})).mappings().all()
            data = [{k: _jsonable(v) for k, v in row.items()} for row in rows]
            out[key] = data[0] if key in ("jobs", "users", "matches") else data
    per_day = out["per_user_per_day"]
    costs = [r["est_cost_usd"] or 0 for r in per_day]
    out["avg_est_cost_per_user_per_day_usd"] = round(sum(costs) / len(costs), 6) if costs else 0
    out["note"] = "Costs are estimates at paid list prices; the free tier bills $0."
    return out
