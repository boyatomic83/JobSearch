"""Job lifecycle against a real Postgres: insert → update → disappear → reappear (repost).

Postings here are synthetic test data.
"""

from datetime import UTC, datetime, timedelta

from sqlalchemy import select

from app.db.models import Company, Job
from app.ingest.base import CompanyConfig, Connector, JobPosting
from app.ingest.service import build_connectors, run_ingest
from app.ingest.store import deactivate_missing, upsert_company, upsert_jobs

T0 = datetime(2026, 11, 1, tzinfo=UTC)


def _posting(sid: str, title: str = "Backend Engineer", source: str = "greenhouse") -> JobPosting:
    return JobPosting(
        source=source, source_id=sid, company="Acme", title=title, url=f"https://x/{sid}"
    )


async def _jobs(session) -> dict[str, Job]:
    return {j.source_id: j for j in (await session.scalars(select(Job))).all()}


async def test_job_lifecycle_insert_update_disappear_repost(sessionmaker):
    async with sessionmaker() as s, s.begin():
        cid = await upsert_company(s, "greenhouse", CompanyConfig(slug="acme", name="Acme"))
        res = await upsert_jobs(s, [_posting("1"), _posting("2")], seen_at=T0, company_id=cid)
        assert (res.new, res.updated, res.reposted) == (2, 0, 0)

    # Run 2: job 1 updated, job 2 missing → deactivated.
    t1 = T0 + timedelta(days=1)
    async with sessionmaker() as s, s.begin():
        res = await upsert_jobs(s, [_posting("1", "Senior Backend")], seen_at=t1, company_id=cid)
        assert (res.new, res.updated) == (0, 1)
        assert await deactivate_missing(s, source="greenhouse", company_id=cid, seen_at=t1) == 1

    async with sessionmaker() as s:
        jobs = await _jobs(s)
    assert jobs["1"].title == "Senior Backend"
    assert jobs["1"].first_seen_at == T0 and jobs["1"].last_seen_at == t1
    assert jobs["2"].is_active is False

    # Run 3: job 2 is back → reposted.
    t2 = T0 + timedelta(days=2)
    async with sessionmaker() as s, s.begin():
        res = await upsert_jobs(s, [_posting("1"), _posting("2")], seen_at=t2, company_id=cid)
        assert res.reposted == 1

    async with sessionmaker() as s:
        jobs = await _jobs(s)
    assert jobs["2"].is_active is True and jobs["2"].times_reposted == 1
    assert jobs["1"].times_reposted == 0
    assert jobs["2"].first_seen_at == T0  # first_seen_at never moves


async def test_duplicate_ids_in_one_feed_are_collapsed(sessionmaker):
    async with sessionmaker() as s, s.begin():
        res = await upsert_jobs(s, [_posting("1"), _posting("1", "Dup")], seen_at=T0)
    assert res.new == 1


async def test_deactivation_is_scoped_to_one_company(sessionmaker):
    async with sessionmaker() as s, s.begin():
        a = await upsert_company(s, "greenhouse", CompanyConfig(slug="a", name="A"))
        b = await upsert_company(s, "greenhouse", CompanyConfig(slug="b", name="B"))
        await upsert_jobs(s, [_posting("a1")], seen_at=T0, company_id=a)
        await upsert_jobs(s, [_posting("b1")], seen_at=T0, company_id=b)
        # Company A refreshed later; B's job must not be touched.
        later = T0 + timedelta(hours=1)
        assert await deactivate_missing(s, source="greenhouse", company_id=a, seen_at=later) == 1
    async with sessionmaker() as s:
        jobs = await _jobs(s)
    assert jobs["b1"].is_active is True


class _FakeConnector(Connector):
    source = "greenhouse"

    def __init__(self, company, postings=None, error=None):
        super().__init__(company)
        self.postings, self.error = postings or [], error

    async def fetch(self, http):
        if self.error:
            raise self.error
        return self.postings


async def test_run_ingest_reports_and_survives_a_failing_feed(sessionmaker):
    ok = _FakeConnector(CompanyConfig(slug="acme", name="Acme"), [_posting("1"), _posting("2")])
    broken = _FakeConnector(CompanyConfig(slug="down", name="Down"), error=RuntimeError("500"))
    report = await run_ingest(sessionmaker, [ok, broken])

    by_feed = {f.feed: f for f in report.feeds}
    assert by_feed["greenhouse:acme"].new == 2 and by_feed["greenhouse:acme"].error is None
    assert "RuntimeError" in by_feed["greenhouse:down"].error
    assert report.totals["failed_feeds"] == 1
    async with sessionmaker() as s:
        assert len(await _jobs(s)) == 2
        company = await s.scalar(select(Company).where(Company.slug == "acme"))
        assert company is not None
        assert {j.company_id for j in (await _jobs(s)).values()} == {company.id}


def test_build_connectors_from_yaml(tmp_path):
    cfg = tmp_path / "companies.yaml"
    cfg.write_text("greenhouse:\n  - {slug: a, name: A}\nlever:\n  - {slug: b, name: B}\n")
    labels = [c.label for c in build_connectors(companies_file=cfg)]
    assert labels == ["greenhouse:a", "lever:b", "remoteok"]
    assert [c.label for c in build_connectors(["lever"], companies_file=cfg)] == ["lever:b"]


def test_shipped_companies_yaml_has_about_30_companies():
    feeds = build_connectors(["greenhouse", "lever"])
    assert 25 <= len(feeds) <= 40
