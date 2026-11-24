"""The shared connector interface and the normalized job shape every source maps into."""

from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import ClassVar

from pydantic import BaseModel, field_validator, model_validator

from app.ingest.clean import plausible_annual_salary
from app.ingest.http import PoliteClient


class CompanyConfig(BaseModel):
    """One entry of companies.yaml."""

    slug: str
    name: str
    industry: str | None = None


class JobPosting(BaseModel):
    """A job normalized from any source. Field limits mirror the `jobs` table columns."""

    source: str
    source_id: str
    company: str
    title: str
    description: str = ""
    location: str | None = None
    remote: bool = False
    salary_min: int | None = None
    salary_max: int | None = None
    salary_currency: str | None = None
    posted_at: datetime | None = None
    url: str

    @field_validator("company", "title", "source_id")
    @classmethod
    def _short(cls, v: str) -> str:
        return v.strip()[:200]

    @field_validator("location")
    @classmethod
    def _location(cls, v: str | None) -> str | None:
        v = (v or "").strip()
        return v[:300] or None

    @field_validator("salary_currency")
    @classmethod
    def _currency(cls, v: str | None) -> str | None:
        return v.strip().upper()[:3] if v else None

    @field_validator("posted_at")
    @classmethod
    def _aware(cls, v: datetime | None) -> datetime | None:
        if v is not None and v.tzinfo is None:
            return v.replace(tzinfo=UTC)
        return v

    @model_validator(mode="after")
    def _drop_implausible_salary(self) -> "JobPosting":
        if not plausible_annual_salary(self.salary_min, self.salary_max, self.salary_currency):
            self.salary_min = self.salary_max = self.salary_currency = None
        return self


class Connector(ABC):
    """One feed of jobs (a company's board, or a whole aggregator).

    `company` is set for per-company boards. `complete_feed` says whether the feed lists
    *every* open job: only then does "missing from this run" mean "closed".
    """

    source: ClassVar[str]
    complete_feed: ClassVar[bool] = True

    def __init__(self, company: CompanyConfig | None = None) -> None:
        self.company = company

    @property
    def label(self) -> str:
        return f"{self.source}:{self.company.slug}" if self.company else self.source

    @abstractmethod
    async def fetch(self, http: PoliteClient) -> list[JobPosting]:
        """Download and normalize all postings for this feed."""
