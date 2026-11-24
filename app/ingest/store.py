"""Writing ingested postings to Postgres.

Lifecycle of a job row:
- first time seen      → INSERT (first_seen_at = last_seen_at = now, is_active = true)
- seen again           → UPDATE fields + last_seen_at
- missing from a run   → is_active = false (only for complete feeds, see Connector)
- back after missing   → is_active = true, times_reposted += 1 (a ghost-job signal)
"""

from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import case, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Company, Job
from app.ingest.base import CompanyConfig, JobPosting

BATCH_SIZE = 500  # keeps each INSERT well under Postgres' bind-parameter limit


@dataclass
class UpsertResult:
    new: int = 0
    updated: int = 0
    reposted: int = 0


async def upsert_company(session: AsyncSession, ats: str, cfg: CompanyConfig) -> int:
    stmt = (
        insert(Company)
        .values(ats=ats, slug=cfg.slug, name=cfg.name, industry=cfg.industry)
        .on_conflict_do_update(
            constraint="uq_companies_ats_slug", set_={"name": cfg.name, "industry": cfg.industry}
        )
        .returning(Company.id)
    )
    return (await session.execute(stmt)).scalar_one()


async def upsert_jobs(
    session: AsyncSession,
    postings: list[JobPosting],
    *,
    seen_at: datetime,
    company_id: int | None = None,
) -> UpsertResult:
    # A feed can list the same id twice; ON CONFLICT can't touch one row twice per statement.
    unique = list({p.source_id: p for p in postings}.values())
    result = UpsertResult()

    for start in range(0, len(unique), BATCH_SIZE):
        batch = unique[start : start + BATCH_SIZE]
        source = batch[0].source

        # Look up existing rows first so we can report new / updated / reposted counts.
        existing = dict(
            (
                await session.execute(
                    select(Job.source_id, Job.is_active).where(
                        Job.source == source, Job.source_id.in_([p.source_id for p in batch])
                    )
                )
            ).all()
        )
        result.new += sum(p.source_id not in existing for p in batch)
        result.updated += sum(p.source_id in existing for p in batch)
        result.reposted += sum(existing.get(p.source_id) is False for p in batch)

        rows = [
            p.model_dump()
            | {"company_id": company_id, "first_seen_at": seen_at, "last_seen_at": seen_at}
            for p in batch
        ]
        stmt = insert(Job).values(rows)
        updatable = [
            "company_id", "company", "title", "description", "location", "remote",
            "salary_min", "salary_max", "salary_currency", "posted_at", "url", "last_seen_at",
        ]  # fmt: skip
        stmt = stmt.on_conflict_do_update(
            constraint="uq_jobs_source_source_id",
            set_={col: stmt.excluded[col] for col in updatable}
            | {
                "is_active": True,
                # Evaluated against the *old* row: bump only if it had disappeared.
                "times_reposted": Job.times_reposted + case((Job.is_active.is_(False), 1), else_=0),
            },
        )
        await session.execute(stmt)
    return result


async def deactivate_missing(
    session: AsyncSession, *, source: str, company_id: int | None, seen_at: datetime
) -> int:
    """Mark jobs from this feed that weren't seen in the current run as inactive."""
    stmt = update(Job).where(
        Job.source == source, Job.is_active.is_(True), Job.last_seen_at < seen_at
    )
    if company_id is not None:
        stmt = stmt.where(Job.company_id == company_id)
    res = await session.execute(stmt.values(is_active=False))
    return res.rowcount
