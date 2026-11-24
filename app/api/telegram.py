"""Telegram webhook endpoint (production). Locally the bot uses polling instead."""

import secrets

from fastapi import APIRouter, Header, HTTPException, Request, status
from telegram import Update

from app.bot.telegram_app import webhook_secret
from app.config import get_settings

router = APIRouter(tags=["telegram"])


@router.post("/telegram/webhook")
async def telegram_webhook(
    request: Request,
    x_telegram_bot_api_secret_token: str | None = Header(default=None),
) -> dict:
    tg = getattr(request.app.state, "telegram", None)
    if tg is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "bot not running")
    expected = webhook_secret(get_settings().telegram_bot_token)
    if not x_telegram_bot_api_secret_token or not secrets.compare_digest(
        x_telegram_bot_api_secret_token, expected
    ):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "bad secret token")
    # Queue and return immediately: Telegram retries webhooks that take too long, and our
    # handlers may wait on the LLM for tens of seconds.
    await tg.update_queue.put(Update.de_json(await request.json(), tg.bot))
    return {"ok": True}
