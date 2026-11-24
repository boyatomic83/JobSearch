"""FastAPI entrypoint."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.admin import router as admin_router
from app.db.session import get_engine, get_session

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    yield
    await get_engine().dispose()


app = FastAPI(title="JobScout", version="0.1.0", lifespan=lifespan)
app.include_router(admin_router)


@app.get("/health")
async def health(session: Annotated[AsyncSession, Depends(get_session)]) -> JSONResponse:
    """Liveness + DB check. Returns 503 if Postgres is unreachable so Render can tell."""
    try:
        await session.execute(text("SELECT 1"))
    except Exception:
        logger.exception("health check: database unreachable")
        return JSONResponse({"status": "degraded", "database": "down"}, status_code=503)
    return JSONResponse({"status": "ok", "database": "up"})
