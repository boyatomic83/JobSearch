"""FastAPI entrypoint."""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from telegram import Update

from app.api.admin import router as admin_router
from app.api.telegram import router as telegram_router
from app.api.users import router as users_router
from app.config import get_settings
from app.db.session import get_engine, get_session
from app.matching.embeddings import get_embedder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


async def _start_bot(app: FastAPI) -> None:
    """Webhook mode if TELEGRAM_WEBHOOK_URL is set (production), else long polling (local)."""
    from app.bot.service import Deps
    from app.bot.telegram_app import build_application, webhook_secret
    from app.scheduler import build_scheduler

    settings = get_settings()
    deps = Deps.from_app()
    tg = build_application(deps)
    await tg.initialize()
    await tg.start()
    if settings.telegram_webhook_url:
        url = settings.telegram_webhook_url.rstrip("/") + "/telegram/webhook"
        await tg.bot.set_webhook(
            url, secret_token=webhook_secret(settings.telegram_bot_token),
            allowed_updates=Update.ALL_TYPES,
        )  # fmt: skip
        logger.info("telegram: webhook mode")
    else:
        await tg.bot.delete_webhook()
        await tg.updater.start_polling(allowed_updates=Update.ALL_TYPES)
        logger.info("telegram: polling mode")
    app.state.telegram = tg
    app.state.bot_deps = deps
    if settings.scheduler_enabled:
        app.state.scheduler = build_scheduler(tg.bot, deps)
        app.state.scheduler.start()
        logger.info("scheduler started (nightly 02:30, digest 08:00 IST)")


async def _stop_bot(app: FastAPI) -> None:
    if scheduler := getattr(app.state, "scheduler", None):
        scheduler.shutdown(wait=False)
    if tg := getattr(app.state, "telegram", None):
        if tg.updater and tg.updater.running:
            await tg.updater.stop()
        await tg.stop()
        await tg.shutdown()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Load the embedding model once, at startup, so the first request isn't slow and the
    # model is never loaded twice (it's the biggest chunk of our RAM on Render's free tier).
    await asyncio.to_thread(get_embedder)
    if get_settings().telegram_bot_token:
        await _start_bot(app)
    else:
        logger.info("TELEGRAM_BOT_TOKEN not set: bot disabled")
    yield
    await _stop_bot(app)
    await get_engine().dispose()


app = FastAPI(title="JobScout", version="0.1.0", lifespan=lifespan)
app.include_router(admin_router)
app.include_router(users_router)
app.include_router(telegram_router)


@app.get("/health")
async def health(session: Annotated[AsyncSession, Depends(get_session)]) -> JSONResponse:
    """Liveness + DB check. Returns 503 if Postgres is unreachable so Render can tell."""
    try:
        await session.execute(text("SELECT 1"))
    except Exception:
        logger.exception("health check: database unreachable")
        return JSONResponse({"status": "degraded", "database": "down"}, status_code=503)
    return JSONResponse({"status": "ok", "database": "up"})
