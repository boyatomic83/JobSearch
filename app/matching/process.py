"""Post-ingest processing: embed new/changed jobs → mark duplicates → ghost scores.

Runs after every ingest. Each step is idempotent, so re-running is always safe.
"""

import asyncio
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

from sqlalchemy import func, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models import Job
from app.matching.embeddings import Embedder
from app.matching.ghost import GHOST_THRESHOLD, GhostFacts, ghost_score

logger = logging.getLogger(__name__)

EMBED_PAGE = 256  # jobs embedded + committed per round trip; bounds memory use
DUPLICATE_SIMILARITY = 0.95


@dataclass
class ProcessReport:
    embedded: int = 0
    duplicates: int = 0
    ghost_jobs: int = 0
    scored: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


async def embed_pending_jobs(
    sessionmaker: async_sessionmaker[AsyncSession], embedder: Embedder
) -> int:
    """Embed active jobs that have no embedding (new, or text changed since last time).

    Ingest NULLs `embedding` whenever `content_hash` changes, so "pending" is just
    `embedding IS NULL` — a cheap query that never reads description text.
    """
    total, last_id = 0, 0
    while True:
        async with sessionmaker() as session:
            rows = (
                await session.execute(
                    select(Job.id, Job.embedding_text)
                    .where(Job.is_active, Job.embedding.is_(None), Job.id > last_id)
                    .order_by(Job.id)
                    .limit(EMBED_PAGE)
                )
            ).all()
            if not rows:
                return total
            texts = [r.embedding_text for r in rows]
            # Encoding is CPU-bound; run it off the event loop so the API stays responsive.
            vectors = await asyncio.to_thread(embedder.embed_documents, texts)
            await session.execute(
                update(Job),
                [{"id": r.id, "embedding": v} for r, v in zip(rows, vectors, strict=True)],
            )
            await session.commit()
        total += len(rows)
        last_id = rows[-1].id
        logger.info("embedded %d jobs so far", total)


# For each active job, find the *oldest* other active job with the same company, location
# and normalised title whose embedding is > threshold similar (`<=>` is cosine distance,
# so similarity = 1 - distance).
# - Same location: companies post one role per city as separate listings; collapsing those
#   would hide the Bangalore opening behind the San Francisco one.
# - Same title: measured on real data, embeddings alone merged distinct roles that share a
#   description ("Sr." vs "Staff Software Engineer", 60/60 sampled title-mismatched pairs
#   were > 0.95). The embedding check then confirms the description is really the same.
_DEDUP_SQL = text("""
WITH matches AS (
    SELECT DISTINCT ON (j.id) j.id, o.id AS original_id
    FROM jobs j
    JOIN jobs o
      ON o.company = j.company
     AND lower(coalesce(o.location, '')) = lower(coalesce(j.location, ''))
     AND trim(regexp_replace(lower(o.title), '[^a-z0-9]+', ' ', 'g'))
       = trim(regexp_replace(lower(j.title), '[^a-z0-9]+', ' ', 'g'))
     AND (coalesce(o.posted_at, o.first_seen_at), o.id)
       < (coalesce(j.posted_at, j.first_seen_at), j.id)
     AND o.is_active AND o.embedding IS NOT NULL
     AND 1 - (o.embedding <=> j.embedding) > :threshold
    WHERE j.is_active AND j.embedding IS NOT NULL
    ORDER BY j.id, coalesce(o.posted_at, o.first_seen_at), o.id
)
UPDATE jobs SET is_duplicate_of = matches.original_id
FROM matches WHERE jobs.id = matches.id
""")


async def mark_duplicates(
    sessionmaker: async_sessionmaker[AsyncSession], threshold: float = DUPLICATE_SIMILARITY
) -> int:
    """Recompute `is_duplicate_of` from scratch (originals can close, so links can change)."""
    async with sessionmaker() as session, session.begin():
        await session.execute(
            update(Job).where(Job.is_duplicate_of.is_not(None)).values(is_duplicate_of=None)
        )
        result = await session.execute(_DEDUP_SQL, {"threshold": threshold})
        return result.rowcount


async def update_ghost_scores(
    sessionmaker: async_sessionmaker[AsyncSession], now: datetime | None = None
) -> tuple[int, int]:
    """Score every active job. Returns (jobs scored, jobs above the ghost threshold)."""
    now = now or datetime.now(UTC)
    has_structure = Job.description.regexp_match(
        r"responsibilit|requirement|qualification|what you.ll do|you will|about the role",
        flags="i",
    )
    # Compute the text-derived facts in SQL so we never ship full descriptions to Python.
    query = select(
        Job.id,
        Job.title,
        Job.posted_at,
        Job.first_seen_at,
        Job.times_reposted,
        func.length(Job.description).label("description_length"),
        has_structure.label("has_structure"),
        (Job.salary_min.is_not(None) | Job.salary_max.is_not(None)).label("has_salary"),
        (func.coalesce(Job.location, "") != "").label("has_location"),
    ).where(Job.is_active)

    async with sessionmaker() as session, session.begin():
        rows = (await session.execute(query)).all()
        updates = []
        ghosts = 0
        for r in rows:
            facts = GhostFacts(
                title=r.title,
                posted_at=r.posted_at,
                first_seen_at=r.first_seen_at,
                times_reposted=r.times_reposted,
                description_length=r.description_length,
                has_structure=bool(r.has_structure),
                has_salary=bool(r.has_salary),
                has_location=bool(r.has_location),
            )
            score, reasons = ghost_score(facts, now)
            ghosts += score > GHOST_THRESHOLD
            updates.append({"id": r.id, "ghost_score": score, "ghost_reasons": reasons})
        if updates:
            await session.execute(update(Job), updates)
    return len(updates), ghosts


async def run_processing(
    sessionmaker: async_sessionmaker[AsyncSession], embedder: Embedder
) -> ProcessReport:
    report = ProcessReport()
    report.embedded = await embed_pending_jobs(sessionmaker, embedder)
    report.duplicates = await mark_duplicates(sessionmaker)
    report.scored, report.ghost_jobs = await update_ghost_scores(sessionmaker)
    logger.info("processing done: %s", report)
    return report
