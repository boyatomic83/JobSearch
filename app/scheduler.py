"""Scheduled jobs (APScheduler, IST): nightly ingest+processing, 08:00 daily digest.

Note for Render's free tier: instances sleep after ~15 min without traffic, and a sleeping
process can't fire its scheduler. The same jobs are exposed as admin endpoints
(POST /admin/ingest, POST /admin/digest) so an external free cron can trigger them.
"""

import logging

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from telegram.error import Forbidden

from app.bot import service
from app.bot.service import Deps

logger = logging.getLogger(__name__)

TIMEZONE = "Asia/Kolkata"


async def nightly_pipeline() -> dict:
    from app.db.session import get_sessionmaker
    from app.ingest.service import build_connectors, run_ingest
    from app.matching.embeddings import get_embedder
    from app.matching.process import run_processing

    report = await run_ingest(get_sessionmaker(), build_connectors())
    processed = await run_processing(get_sessionmaker(), get_embedder())
    logger.info("nightly pipeline: %s / %s", report.totals, processed)
    return {"ingest": report.totals, "processing": processed.to_dict()}


async def send_daily_digest(bot, deps: Deps) -> dict:
    """Send each user their top new matches. One user's failure never stops the rest."""
    from app.bot.telegram_app import send

    sent = skipped = failed = 0
    for user_id, chat_id in await service.digest_recipients(deps):
        outs: list = []
        try:
            outs = await service.digest_for(deps, user_id)
            if not outs:
                skipped += 1
                continue
            await send(bot, chat_id, outs)
            sent += 1
        except Exception as exc:
            if isinstance(exc, Forbidden):  # the user blocked the bot
                logger.info("digest: user %s blocked the bot", user_id)
            else:
                logger.exception("digest failed for user %s", user_id)
            await service.forget_matches(deps, outs)
            failed += 1
    result = {"sent": sent, "nothing_new": skipped, "failed": failed}
    logger.info("daily digest: %s", result)
    return result


def build_scheduler(bot, deps: Deps) -> AsyncIOScheduler:
    scheduler = AsyncIOScheduler(timezone=TIMEZONE)
    opts = {"coalesce": True, "max_instances": 1, "misfire_grace_time": 3600}
    scheduler.add_job(nightly_pipeline, "cron", hour=2, minute=30, id="nightly_pipeline", **opts)
    scheduler.add_job(send_daily_digest, "cron", hour=8, minute=0, id="daily_digest",
                      args=[bot, deps], **opts)  # fmt: skip
    return scheduler
