"""Connector parsing against recorded API responses (tests/fixtures, captured 2026-11-24).

No network: httpx.MockTransport serves the fixture files.
"""

import json
from pathlib import Path

import httpx

from app.ingest.base import CompanyConfig
from app.ingest.greenhouse import GreenhouseConnector
from app.ingest.http import PoliteClient
from app.ingest.lever import LeverConnector
from app.ingest.remoteok import RemoteOKConnector

FIXTURES = Path(__file__).parent / "fixtures"


async def _noop_sleep(_: float) -> None:
    pass


def _client_serving(payload, seen_urls: list[str] | None = None) -> PoliteClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if seen_urls is not None:
            seen_urls.append(str(request.url))
        return httpx.Response(200, json=payload)

    return PoliteClient(transport=httpx.MockTransport(handler), sleep=_noop_sleep, min_interval=0)


def _load(name: str):
    return json.loads((FIXTURES / name).read_text())


async def test_greenhouse_parses_recorded_board():
    urls: list[str] = []
    conn = GreenhouseConnector(CompanyConfig(slug="vercel", name="Vercel", industry="devtools"))
    async with _client_serving(_load("greenhouse_vercel.json"), urls) as http:
        jobs = await conn.fetch(http)

    assert "boards/vercel/jobs" in urls[0] and "content=true" in urls[0]
    assert len(jobs) == 3
    first = jobs[0]
    assert first.source == "greenhouse" and first.source_id == "6231864004"
    assert first.company == "Vercel"
    assert first.location == "Remote - Germany" and first.remote is True
    assert first.posted_at is not None and first.posted_at.tzinfo is not None
    assert first.url.startswith("https://job-boards.greenhouse.io/vercel/jobs/")
    # HTML was double-unescaped and stripped.
    assert "&lt;" not in first.description and "<div" not in first.description
    assert "About Vercel" in first.description
    assert jobs[1].remote is False  # "Hybrid - London"


async def test_greenhouse_uses_pay_ranges_when_present():
    payload = _load("greenhouse_vercel.json")
    payload["jobs"][0]["pay_input_ranges"] = [  # synthetic test value
        {"min_cents": 15000000, "max_cents": 20000000, "currency_type": "USD"}
    ]
    conn = GreenhouseConnector(CompanyConfig(slug="vercel", name="Vercel"))
    async with _client_serving(payload) as http:
        job = (await conn.fetch(http))[0]
    assert (job.salary_min, job.salary_max, job.salary_currency) == (150_000, 200_000, "USD")


async def test_lever_parses_recorded_postings():
    conn = LeverConnector(CompanyConfig(slug="zeta", name="Zeta", industry="fintech"))
    async with _client_serving(_load("lever_zeta.json")) as http:
        jobs = await conn.fetch(http)

    assert len(jobs) == 3
    legal, devsecops, director = jobs
    assert legal.title == "Associate Director - Legal"
    assert legal.location == "Mumbai / Bangalore" and legal.remote is False
    assert "Responsibilities" in legal.description  # titled list sections are included
    assert legal.url.startswith("https://jobs.lever.co/zeta/")
    assert legal.posted_at is not None
    assert director.remote is True  # workplaceType == "remote"
    assert devsecops.salary_min is None


async def test_lever_yearly_salary_range():
    payload = _load("lever_zeta.json")[:1]
    payload[0]["salaryRange"] = {  # synthetic test value
        "min": 2500000, "max": 4000000, "currency": "INR", "interval": "per-year-salary"
    }  # fmt: skip
    conn = LeverConnector(CompanyConfig(slug="zeta", name="Zeta"))
    async with _client_serving(payload) as http:
        job = (await conn.fetch(http))[0]
    assert (job.salary_min, job.salary_max, job.salary_currency) == (2_500_000, 4_000_000, "INR")


async def test_remoteok_skips_legal_notice_and_parses_jobs():
    async with _client_serving(_load("remoteok.json")) as http:
        jobs = await RemoteOKConnector().fetch(http)

    assert len(jobs) == 3  # 4 items in the fixture, the first is the legal notice
    thrive = jobs[0]
    assert thrive.company == "Thrive" and thrive.title == "Strategic Technical Advisor"
    assert thrive.remote is True and thrive.location == "Remote"  # empty location → "Remote"
    assert thrive.salary_min == 128091 and thrive.salary_currency == "USD"
    assert "Tags:" in thrive.description
    assert jobs[1].salary_min is None and jobs[1].salary_currency is None  # 0 means unknown
    assert thrive.url.startswith("https://remoteOK.com/")
