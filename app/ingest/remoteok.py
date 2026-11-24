"""RemoteOK public API (remote jobs only).

Terms of the API: link back to the RemoteOK job URL and credit Remote OK as the source —
we store their URL as the job link and credit them in the README.
"""

from datetime import datetime

from app.ingest.base import Connector, JobPosting
from app.ingest.clean import html_to_text
from app.ingest.http import PoliteClient

API = "https://remoteok.com/api"


class RemoteOKConnector(Connector):
    source = "remoteok"
    # The API returns only the ~100 most recent jobs: dropping off the feed means "older",
    # not "closed", so we must not mark missing jobs inactive.
    complete_feed = False

    async def fetch(self, http: PoliteClient) -> list[JobPosting]:
        data = await http.get_json(API)
        # The first element is a legal notice, not a job.
        return [self._parse(j) for j in data if isinstance(j, dict) and j.get("id")]

    def _parse(self, j: dict) -> JobPosting:
        description = html_to_text(j.get("description"))
        if tags := j.get("tags"):
            description += "\n\nTags: " + ", ".join(tags)
        salary_min = int(j.get("salary_min") or 0) or None  # RemoteOK uses 0 for "unknown"
        salary_max = int(j.get("salary_max") or 0) or None
        return JobPosting(
            source=self.source,
            source_id=str(j["id"]),
            company=j.get("company") or "Unknown",
            title=j.get("position") or "Untitled",
            description=description,
            location=j.get("location") or "Remote",
            remote=True,
            salary_min=salary_min,
            salary_max=salary_max,
            salary_currency="USD" if salary_min or salary_max else None,
            posted_at=datetime.fromisoformat(j["date"]) if j.get("date") else None,
            url=j.get("url") or f"https://remoteok.com/remote-jobs/{j['id']}",
        )
