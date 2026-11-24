"""Shared fixtures.

DB tests run against a dedicated `jobscout_test` database on the local docker Postgres,
migrated with Alembic, so the real migration is what gets tested. We force DATABASE_URL
before any app import so `.env` (which may point at Neon) is never used by tests.
"""

import asyncio
import os

import pytest

TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL", "postgresql://jobscout:jobscout@localhost:5432/jobscout_test"
)
os.environ["DATABASE_URL"] = TEST_DATABASE_URL
# Tests must never use the real keys from .env (env vars take precedence over .env).
os.environ["ADMIN_API_KEY"] = "test-admin-key"
os.environ["GEMINI_API_KEY"] = ""
os.environ["TELEGRAM_BOT_TOKEN"] = ""
os.environ["TELEGRAM_WEBHOOK_URL"] = ""

from alembic import command  # noqa: E402
from alembic.config import Config  # noqa: E402
from sqlalchemy import text  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine  # noqa: E402

from app.config import to_async_url  # noqa: E402


async def _ensure_database() -> None:
    """Create the test database if it doesn't exist (connects to the default `postgres` db)."""
    base, _, db_name = TEST_DATABASE_URL.rpartition("/")
    admin = create_async_engine(to_async_url(f"{base}/postgres"), isolation_level="AUTOCOMMIT")
    try:
        async with admin.connect() as conn:
            exists = await conn.scalar(
                text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": db_name}
            )
            if not exists:
                await conn.execute(text(f'CREATE DATABASE "{db_name}"'))
    finally:
        await admin.dispose()


@pytest.fixture(scope="session")
def migrated_db() -> str:
    """Create + migrate the test DB once per session; skip DB tests if Postgres is down."""
    try:
        asyncio.run(_ensure_database())
    except OSError as exc:  # connection refused etc.
        pytest.skip(f"test Postgres not reachable ({exc}); run `docker compose up -d db`")
    cfg = Config("alembic.ini")
    cfg.cmd_opts = type("Opts", (), {"x": [f"db_url={TEST_DATABASE_URL}"]})()
    command.downgrade(cfg, "base")  # clean slate each session
    command.upgrade(cfg, "head")
    return TEST_DATABASE_URL


@pytest.fixture
async def engine(migrated_db: str) -> AsyncEngine:
    eng = create_async_engine(to_async_url(migrated_db))
    yield eng
    await eng.dispose()


@pytest.fixture
async def sessionmaker(engine: AsyncEngine):
    """Session factory on the test DB; wipes all rows after each test."""
    from sqlalchemy.ext.asyncio import async_sessionmaker

    yield async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as conn:
        tables = await conn.scalars(
            text(
                "SELECT tablename FROM pg_tables "
                "WHERE schemaname = 'public' AND tablename <> 'alembic_version'"
            )
        )
        await conn.execute(text(f"TRUNCATE {', '.join(tables)} RESTART IDENTITY CASCADE"))
