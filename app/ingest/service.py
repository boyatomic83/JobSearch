"""Orchestrates one ingestion run across all feeds.

Each feed is fetched and saved independently (own DB transaction), so one broken company
board or API outage is logged and reported but never aborts the whole run.
"""

import asyncio
import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import yaml
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.ingest.base import CompanyConfig, Connector
from app.ingest.greenhouse import GreenhouseConnector
from app.ingest.http import PoliteClient
from app.ingest.lever import LeverConnector
from app.ingest.remoteok import RemoteOKConnector
from app.ingest.store import deactivate_missing, upsert_company, upsert_jobs

logger = logging.getLogger(__name__)

COMPANIES_FILE = Path(__file__).with_name("companies.yaml")
CONNECTORS: dict[str, type[Connector]] = {
    "greenhouse": GreenhouseConnector,
    "lever": LeverConnector,
    "remoteok": RemoteOKConnector,
}
MAX_CONCURRENT_FEEDS = 4


@dataclass
class FeedReport:
    feed: str
    fetched: int = 0
    new: int = 0
    updated: int = 0
    reposted: int = 0
    deactivated: int = 0
    error: str | None = None


@dataclass
class IngestReport:
    started_at: datetime
    feeds: list[FeedReport] = field(default_factory=list)

    @property
    def totals(self) -> dict[str, int]:
        keys = ("fetched", "new", "updated", "reposted", "deactivated")
        totals = {k: sum(getattr(f, k) for f in self.feeds) for k in keys}
        totals["failed_feeds"] = sum(f.error is not None for f in self.feeds)
        return totals

    def to_dict(self) -> dict:
        return {
            "started_at": self.started_at.isoformat(),
            "totals": self.totals,
            "feeds": [asdict(f) for f in self.feeds],
        }


def build_connectors(
    sources: list[str] | None = None, companies_file: Path = COMPANIES_FILE
) -> list[Connector]:
    """One connector per company board from companies.yaml, plus aggregator feeds."""
    wanted = set(sources or CONNECTORS)
    config = yaml.safe_load(companies_file.read_text()) or {}
    connectors: list[Connector] = []
    for source, cls in CONNECTORS.items():
        if source not in wanted:
            continue
        if cls.complete_feed:  # per-company boards
            connectors += [cls(CompanyConfig(**c)) for c in config.get(source, [])]
        else:
            connectors.append(cls())
    return connectors


async def _run_feed(
    connector: Connector,
    http: PoliteClient,
    sessionmaker: async_sessionmaker[AsyncSession],
    seen_at: datetime,
) -> FeedReport:
    report = FeedReport(feed=connector.label)
    try:
        postings = await connector.fetch(http)
        report.fetched = len(postings)
        async with sessionmaker() as session, session.begin():
            company_id = None
            if connector.company:
                company_id = await upsert_company(session, connector.source, connector.company)
            if postings:
                res = await upsert_jobs(session, postings, seen_at=seen_at, company_id=company_id)
                report.new, report.updated, report.reposted = res.new, res.updated, res.reposted
            if connector.complete_feed:
                report.deactivated = await deactivate_missing(
                    session, source=connector.source, company_id=company_id, seen_at=seen_at
                )
        logger.info("%s: %s", connector.label, report)
    except Exception as exc:  # one bad feed must not stop the run
        logger.exception("ingest failed for %s", connector.label)
        report.error = f"{type(exc).__name__}: {exc}"[:500]
    return report


async def run_ingest(
    sessionmaker: async_sessionmaker[AsyncSession],
    connectors: list[Connector],
    http: PoliteClient | None = None,
) -> IngestReport:
    seen_at = datetime.now(UTC)
    report = IngestReport(started_at=seen_at)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_FEEDS)

    async def guarded(c: Connector, client: PoliteClient) -> FeedReport:
        async with semaphore:
            return await _run_feed(c, client, sessionmaker, seen_at)

    async with http or PoliteClient() as client:
        report.feeds = list(await asyncio.gather(*(guarded(c, client) for c in connectors)))
    return report
