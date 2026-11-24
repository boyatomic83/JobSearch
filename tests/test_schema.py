"""Checks that the Alembic migration produces the schema the spec requires."""

import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from app.db.models import Base

EXPECTED_TABLES = {
    "users",
    "preferences",
    "resumes",
    "companies",
    "jobs",
    "matches",
    "messages",
    "llm_calls",
    "llm_cache",
}


async def test_all_tables_exist(engine):
    async with engine.connect() as conn:
        rows = await conn.execute(
            text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
        )
        tables = {r[0] for r in rows}
    assert tables >= EXPECTED_TABLES


async def test_hnsw_index_on_job_embedding(engine):
    async with engine.connect() as conn:
        indexdef = await conn.scalar(
            text("SELECT indexdef FROM pg_indexes WHERE indexname = 'ix_jobs_embedding_hnsw'")
        )
    assert indexdef is not None
    assert "hnsw" in indexdef and "vector_cosine_ops" in indexdef


async def test_jobs_source_source_id_is_unique(engine):
    insert = text(
        "INSERT INTO jobs (source, source_id, company, title, url) "
        "VALUES ('greenhouse', 'dup-1', 'Acme', 'Engineer', 'https://example.com')"
    )
    async with engine.connect() as conn:
        trans = await conn.begin()
        await conn.execute(insert)
        with pytest.raises(IntegrityError):
            await conn.execute(insert)
        await trans.rollback()


async def test_vector_roundtrip_and_cosine_search(engine):
    """Insert two jobs with 384-d vectors and confirm `<=>` ranks the closer one first."""
    near = "[" + ",".join(["1"] + ["0"] * 383) + "]"
    far = "[" + ",".join(["0", "1"] + ["0"] * 382) + "]"
    async with engine.connect() as conn:
        trans = await conn.begin()
        for sid, vec in (("near", near), ("far", far)):
            await conn.execute(
                text(
                    "INSERT INTO jobs (source, source_id, company, title, url, embedding) "
                    "VALUES ('test', :sid, 'Acme', 'Eng', 'https://x', CAST(:v AS vector))"
                ),
                {"sid": sid, "v": vec},
            )
        top = await conn.scalar(
            text(
                "SELECT source_id FROM jobs WHERE source = 'test' "
                "ORDER BY embedding <=> CAST(:q AS vector) LIMIT 1"
            ),
            {"q": near},
        )
        await trans.rollback()
    assert top == "near"


async def test_models_match_migration(engine):
    """Guards against drift: ORM models and the hand-written migration must agree."""

    def diff(sync_conn):
        ctx = MigrationContext.configure(sync_conn)
        return compare_metadata(ctx, Base.metadata)

    async with engine.connect() as conn:
        changes = await conn.run_sync(diff)
    assert changes == []
