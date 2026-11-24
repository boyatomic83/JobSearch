"""Initial schema: users, preferences, resumes, companies, jobs (+vector), matches,
messages, llm_calls, llm_cache.

Revision ID: 0001
Revises:
Create Date: 2026-11-24
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import ARRAY, JSONB

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Frozen here on purpose: migrations must not change if the app constant changes later.
EMBEDDING_DIM = 384


def _ts(name: str, nullable: bool = False) -> sa.Column:
    return sa.Column(
        name,
        sa.DateTime(timezone=True),
        server_default=None if nullable else sa.func.now(),
        nullable=nullable,
    )


def _text_array(name: str) -> sa.Column:
    return sa.Column(name, ARRAY(sa.Text), server_default="{}", nullable=False)


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "users",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("telegram_chat_id", sa.BigInteger, unique=True),
        sa.Column("name", sa.String(200)),
        _ts("created_at"),
    )

    op.create_table(
        "preferences",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column(
            "user_id",
            sa.Integer,
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            unique=True,
            nullable=False,
        ),
        _text_array("roles"),
        _text_array("locations"),
        sa.Column("remote_only", sa.Boolean, server_default="false", nullable=False),
        sa.Column("min_salary", sa.Integer),
        sa.Column("seniority", sa.String(50)),
        _text_array("excluded_companies"),
        _text_array("excluded_industries"),
        _text_array("notes"),
        _ts("updated_at"),
    )

    op.create_table(
        "resumes",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column(
            "user_id", sa.Integer, sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("file_name", sa.String(300)),
        sa.Column("raw_text", sa.Text, nullable=False),
        sa.Column("profile", JSONB),
        sa.Column("embedding", Vector(EMBEDDING_DIM)),
        sa.Column("is_active", sa.Boolean, server_default="true", nullable=False),
        _ts("created_at"),
    )
    op.create_index("ix_resumes_user_id", "resumes", ["user_id"])

    op.create_table(
        "companies",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("slug", sa.String(200), nullable=False),
        sa.Column("ats", sa.String(30), nullable=False),
        sa.Column("industry", sa.String(100)),
        _ts("created_at"),
        sa.UniqueConstraint("ats", "slug", name="uq_companies_ats_slug"),
    )

    op.create_table(
        "jobs",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("source", sa.String(30), nullable=False),
        sa.Column("source_id", sa.String(200), nullable=False),
        sa.Column("company_id", sa.Integer, sa.ForeignKey("companies.id", ondelete="SET NULL")),
        sa.Column("company", sa.String(200), nullable=False),
        sa.Column("title", sa.String(300), nullable=False),
        sa.Column("description", sa.Text, server_default="", nullable=False),
        sa.Column("location", sa.String(300)),
        sa.Column("remote", sa.Boolean, server_default="false", nullable=False),
        sa.Column("salary_min", sa.Integer),
        sa.Column("salary_max", sa.Integer),
        _ts("posted_at", nullable=True),
        sa.Column("url", sa.Text, nullable=False),
        _ts("first_seen_at"),
        _ts("last_seen_at"),
        sa.Column("is_active", sa.Boolean, server_default="true", nullable=False),
        sa.Column("times_reposted", sa.Integer, server_default="0", nullable=False),
        sa.Column("is_duplicate_of", sa.Integer, sa.ForeignKey("jobs.id", ondelete="SET NULL")),
        sa.Column("ghost_score", sa.Float, server_default="0", nullable=False),
        sa.Column("ghost_reasons", JSONB, server_default="[]", nullable=False),
        sa.Column("content_hash", sa.String(64)),
        sa.Column("embedding", Vector(EMBEDDING_DIM)),
        sa.UniqueConstraint("source", "source_id", name="uq_jobs_source_source_id"),
    )
    op.create_index("ix_jobs_company", "jobs", ["company"])
    op.create_index("ix_jobs_posted_at", "jobs", ["posted_at"])
    op.create_index("ix_jobs_is_duplicate_of", "jobs", ["is_duplicate_of"])
    op.create_index(
        "ix_jobs_embedding_hnsw",
        "jobs",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )

    op.create_table(
        "matches",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column(
            "user_id", sa.Integer, sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column(
            "job_id", sa.Integer, sa.ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("retrieval_score", sa.Float),
        sa.Column("fit_score", sa.SmallInteger),
        sa.Column("reasons", JSONB, server_default="[]", nullable=False),
        sa.Column("gap", sa.Text),
        sa.Column("feedback", sa.SmallInteger),
        sa.Column("feedback_reason", sa.Text),
        _ts("shown_at", nullable=True),
        _ts("created_at"),
        sa.UniqueConstraint("user_id", "job_id", name="uq_matches_user_job"),
    )
    op.create_index("ix_matches_user_id", "matches", ["user_id"])

    op.create_table(
        "messages",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column(
            "user_id", sa.Integer, sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("role", sa.String(20), nullable=False),
        sa.Column("content", sa.Text, server_default="", nullable=False),
        sa.Column("tool_name", sa.String(100)),
        sa.Column("tool_payload", JSONB),
        _ts("created_at"),
    )
    op.create_index("ix_messages_user_created", "messages", ["user_id", "created_at"])

    op.create_table(
        "llm_calls",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("user_id", sa.Integer, sa.ForeignKey("users.id", ondelete="SET NULL")),
        sa.Column("provider", sa.String(30), nullable=False),
        sa.Column("model", sa.String(100), nullable=False),
        sa.Column("purpose", sa.String(50), nullable=False),
        sa.Column("prompt_tokens", sa.Integer, server_default="0", nullable=False),
        sa.Column("completion_tokens", sa.Integer, server_default="0", nullable=False),
        sa.Column("cost_usd", sa.Numeric(12, 6), server_default="0", nullable=False),
        sa.Column("latency_ms", sa.Integer, server_default="0", nullable=False),
        sa.Column("cached", sa.Boolean, server_default="false", nullable=False),
        sa.Column("status", sa.String(20), server_default="ok", nullable=False),
        sa.Column("error", sa.Text),
        _ts("created_at"),
    )
    op.create_index("ix_llm_calls_user_created", "llm_calls", ["user_id", "created_at"])

    op.create_table(
        "llm_cache",
        sa.Column("key", sa.String(64), primary_key=True),
        sa.Column("model", sa.String(100), nullable=False),
        sa.Column("response", JSONB, nullable=False),
        _ts("created_at"),
    )


def downgrade() -> None:
    for table in (
        "llm_cache",
        "llm_calls",
        "messages",
        "matches",
        "jobs",
        "companies",
        "resumes",
        "preferences",
        "users",
    ):
        op.drop_table(table)
