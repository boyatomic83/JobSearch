"""Lever public postings API: returns a JSON list of every published posting."""

from datetime import UTC, datetime

from app.ingest.base import Connector, JobPosting
from app.ingest.clean import html_to_text, looks_remote
from app.ingest.http import PoliteClient

API = "https://api.lever.co/v0/postings/{slug}"


class LeverConnector(Connector):
    source = "lever"

    async def fetch(self, http: PoliteClient) -> list[JobPosting]:
        assert self.company is not None
        data = await http.get_json(API.format(slug=self.company.slug), params={"mode": "json"})
        return [self._parse(j) for j in data]

    def _parse(self, j: dict) -> JobPosting:
        assert self.company is not None
        cats = j.get("categories") or {}
        location = cats.get("location") or ", ".join(cats.get("allLocations") or [])

        # Lever splits the body into intro, titled lists (Responsibilities…), and a footer.
        parts = [j.get("descriptionPlain") or ""]
        for lst in j.get("lists") or []:
            parts.append(f"{lst.get('text', '')}\n{html_to_text(lst.get('content'))}")
        parts.append(j.get("additionalPlain") or "")
        description = "\n\n".join(p.strip() for p in parts if p and p.strip())

        salary_min = salary_max = currency = None
        sr = j.get("salaryRange") or {}
        # Only keep yearly ranges so salary filters compare like with like.
        if sr and "year" in (sr.get("interval") or "per-year-salary"):
            salary_min, salary_max, currency = sr.get("min"), sr.get("max"), sr.get("currency")

        created = j.get("createdAt")
        return JobPosting(
            source=self.source,
            source_id=j["id"],
            company=self.company.name,
            title=j["text"],
            description=description,
            location=location,
            remote=j.get("workplaceType") == "remote" or looks_remote(location),
            salary_min=salary_min,
            salary_max=salary_max,
            salary_currency=currency,
            posted_at=datetime.fromtimestamp(created / 1000, tz=UTC) if created else None,
            url=j["hostedUrl"],
        )
