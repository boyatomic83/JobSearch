"""Greenhouse public job board API: one request returns every open job with its HTML body."""

import html
from datetime import datetime

from app.ingest.base import Connector, JobPosting
from app.ingest.clean import extract_usd_salary_range, html_to_text, looks_remote
from app.ingest.http import PoliteClient

API = "https://boards-api.greenhouse.io/v1/boards/{slug}/jobs"


class GreenhouseConnector(Connector):
    source = "greenhouse"

    async def fetch(self, http: PoliteClient) -> list[JobPosting]:
        assert self.company is not None
        data = await http.get_json(
            API.format(slug=self.company.slug),
            params={"content": "true", "pay_transparency": "true"},
        )
        return [self._parse(j) for j in data.get("jobs", [])]

    def _parse(self, j: dict) -> JobPosting:
        assert self.company is not None
        # `content` is HTML that has itself been HTML-escaped (&lt;p&gt;), so unescape first.
        description = html_to_text(html.unescape(j.get("content") or ""))
        location = (j.get("location") or {}).get("name")
        offices = " ".join(o.get("name", "") for o in j.get("offices") or [])

        salary_min = salary_max = currency = None
        ranges = j.get("pay_input_ranges") or []
        if ranges:
            r = ranges[0]
            salary_min = int(r["min_cents"]) // 100 if r.get("min_cents") else None
            salary_max = int(r["max_cents"]) // 100 if r.get("max_cents") else None
            currency = r.get("currency_type")
        elif found := extract_usd_salary_range(description):
            (salary_min, salary_max), currency = found, "USD"

        posted = j.get("first_published") or j.get("updated_at")
        return JobPosting(
            source=self.source,
            source_id=str(j["id"]),
            company=j.get("company_name") or self.company.name,
            title=j["title"],
            description=description,
            location=location,
            remote=looks_remote(location, offices),
            salary_min=salary_min,
            salary_max=salary_max,
            salary_currency=currency,
            posted_at=datetime.fromisoformat(posted) if posted else None,
            url=j["absolute_url"],
        )
