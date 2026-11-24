"""SQLAlchemy ORM models — the whole schema lives here.

Design notes:
- The job embedding is a column on `jobs` (not a separate table): one row per job, one
  vector, simpler joins for filtered vector search.
- `content_hash` lets us re-embed only jobs whose text actually changed.
- JSONB is used for semi-structured data (LLM-extracted profile, ghost reasons, match
  reasons); anything we filter on in SQL gets a real column.
"""

from datetime import datetime
from decimal import Decimal

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from app.config import EMBEDDING_DIM


class Base(DeclarativeBase):
    pass


def _now() -> Mapped[datetime]:
    return mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    telegram_chat_id: Mapped[int | None] = mapped_column(BigInteger, unique=True)
    name: Mapped[str | None] = mapped_column(String(200))
    created_at: Mapped[datetime] = _now()

    preferences: Mapped["Preference | None"] = relationship(back_populates="user", uselist=False)


class Preference(Base):
    """Structured job preferences — the agent's long-term memory about the user."""

    __tablename__ = "preferences"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False
    )
    roles: Mapped[list[str]] = mapped_column(ARRAY(Text), server_default="{}", nullable=False)
    locations: Mapped[list[str]] = mapped_column(ARRAY(Text), server_default="{}", nullable=False)
    remote_only: Mapped[bool] = mapped_column(Boolean, server_default="false", nullable=False)
    min_salary: Mapped[int | None] = mapped_column(Integer)
    seniority: Mapped[str | None] = mapped_column(String(50))
    excluded_companies: Mapped[list[str]] = mapped_column(
        ARRAY(Text), server_default="{}", nullable=False
    )
    excluded_industries: Mapped[list[str]] = mapped_column(
        ARRAY(Text), server_default="{}", nullable=False
    )
    # Free-form notes the agent learned that don't fit a column yet ("hates on-call").
    notes: Mapped[list[str]] = mapped_column(ARRAY(Text), server_default="{}", nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    user: Mapped[User] = relationship(back_populates="preferences")


class Resume(Base):
    __tablename__ = "resumes"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )
    file_name: Mapped[str | None] = mapped_column(String(300))
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    profile: Mapped[dict | None] = mapped_column(JSONB)  # LLM-extracted, Pydantic-validated
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM))
    is_active: Mapped[bool] = mapped_column(Boolean, server_default="true", nullable=False)
    created_at: Mapped[datetime] = _now()


class Company(Base):
    __tablename__ = "companies"
    __table_args__ = (UniqueConstraint("ats", "slug", name="uq_companies_ats_slug"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    slug: Mapped[str] = mapped_column(String(200), nullable=False)
    ats: Mapped[str] = mapped_column(String(30), nullable=False)  # greenhouse | lever | remoteok
    industry: Mapped[str | None] = mapped_column(String(100))
    created_at: Mapped[datetime] = _now()


class Job(Base):
    __tablename__ = "jobs"
    __table_args__ = (
        UniqueConstraint("source", "source_id", name="uq_jobs_source_source_id"),
        # HNSW with cosine distance: approximate nearest neighbour search that stays fast
        # as the table grows, with no training step (unlike IVFFlat).
        Index(
            "ix_jobs_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index("ix_jobs_company", "company"),
        Index("ix_jobs_posted_at", "posted_at"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    source: Mapped[str] = mapped_column(String(30), nullable=False)
    source_id: Mapped[str] = mapped_column(String(200), nullable=False)
    company_id: Mapped[int | None] = mapped_column(ForeignKey("companies.id", ondelete="SET NULL"))
    company: Mapped[str] = mapped_column(String(200), nullable=False)
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    description: Mapped[str] = mapped_column(Text, server_default="", nullable=False)
    location: Mapped[str | None] = mapped_column(String(300))
    remote: Mapped[bool] = mapped_column(Boolean, server_default="false", nullable=False)
    salary_min: Mapped[int | None] = mapped_column(Integer)
    salary_max: Mapped[int | None] = mapped_column(Integer)
    posted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    url: Mapped[str] = mapped_column(Text, nullable=False)

    first_seen_at: Mapped[datetime] = _now()
    last_seen_at: Mapped[datetime] = _now()
    # True while the job appears in the latest ingest; lets us detect disappear → reappear.
    is_active: Mapped[bool] = mapped_column(Boolean, server_default="true", nullable=False)
    times_reposted: Mapped[int] = mapped_column(Integer, server_default="0", nullable=False)

    is_duplicate_of: Mapped[int | None] = mapped_column(
        ForeignKey("jobs.id", ondelete="SET NULL"), index=True
    )
    ghost_score: Mapped[float] = mapped_column(Float, server_default="0", nullable=False)
    ghost_reasons: Mapped[list[str]] = mapped_column(JSONB, server_default="[]", nullable=False)

    content_hash: Mapped[str | None] = mapped_column(String(64))
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM))


class Match(Base):
    """A job we recommended to a user, with the reranker's reasoning and user feedback."""

    __tablename__ = "matches"
    __table_args__ = (UniqueConstraint("user_id", "job_id", name="uq_matches_user_job"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False)
    retrieval_score: Mapped[float | None] = mapped_column(Float)  # cosine similarity
    fit_score: Mapped[int | None] = mapped_column(SmallInteger)  # 0-100 from the reranker
    reasons: Mapped[list[str]] = mapped_column(JSONB, server_default="[]", nullable=False)
    gap: Mapped[str | None] = mapped_column(Text)
    feedback: Mapped[int | None] = mapped_column(SmallInteger)  # +1 👍, -1 👎
    feedback_reason: Mapped[str | None] = mapped_column(Text)
    shown_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = _now()


class Message(Base):
    """Chat history, used as the agent's short-term memory (last ~20 messages)."""

    __tablename__ = "messages"
    __table_args__ = (Index("ix_messages_user_created", "user_id", "created_at"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    role: Mapped[str] = mapped_column(String(20), nullable=False)  # user | assistant | tool
    content: Mapped[str] = mapped_column(Text, server_default="", nullable=False)
    tool_name: Mapped[str | None] = mapped_column(String(100))
    tool_payload: Mapped[dict | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = _now()


class LLMCall(Base):
    """One row per LLM request — the basis for cost/latency observability."""

    __tablename__ = "llm_calls"
    __table_args__ = (Index("ix_llm_calls_user_created", "user_id", "created_at"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    provider: Mapped[str] = mapped_column(String(30), nullable=False)
    model: Mapped[str] = mapped_column(String(100), nullable=False)
    purpose: Mapped[str] = mapped_column(String(50), nullable=False)  # extract | rerank | agent…
    prompt_tokens: Mapped[int] = mapped_column(Integer, server_default="0", nullable=False)
    completion_tokens: Mapped[int] = mapped_column(Integer, server_default="0", nullable=False)
    cost_usd: Mapped[Decimal] = mapped_column(Numeric(12, 6), server_default="0", nullable=False)
    latency_ms: Mapped[int] = mapped_column(Integer, server_default="0", nullable=False)
    cached: Mapped[bool] = mapped_column(Boolean, server_default="false", nullable=False)
    status: Mapped[str] = mapped_column(String(20), server_default="ok", nullable=False)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = _now()


class LLMCache(Base):
    """Response cache keyed by a hash of (model, prompt, schema) to save free-tier quota."""

    __tablename__ = "llm_cache"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    model: Mapped[str] = mapped_column(String(100), nullable=False)
    response: Mapped[dict] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = _now()
