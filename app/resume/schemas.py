"""The structured profile we extract from a resume (validated LLM output)."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

Seniority = Literal["intern", "junior", "mid", "senior", "lead"]


class Education(BaseModel):
    degree: str | None = None
    field: str | None = None
    institution: str | None = None
    year: int | None = Field(default=None, ge=1950, le=2100)


class ResumeProfile(BaseModel):
    name: str | None = None
    location: str | None = Field(default=None, description="City/country the person lives in")
    current_title: str | None = None
    years_experience: float | None = Field(
        default=None, ge=0, le=60, description="Total professional years, internships excluded"
    )
    seniority: Seniority | None = None
    roles: list[str] = Field(default_factory=list, description="Job titles held, newest first")
    skills: list[str] = Field(
        default_factory=list, description="Concrete technical skills, tools, languages"
    )
    education: list[Education] = Field(default_factory=list)
    summary: str = Field(default="", description="2 sentences: who they are, what they do best")

    @field_validator("roles", "skills")
    @classmethod
    def _clean_list(cls, items: list[str]) -> list[str]:
        """Strip, drop empties and case-insensitive duplicates, keep order."""
        seen: set[str] = set()
        out = []
        for item in (i.strip() for i in items):
            if item and item.lower() not in seen:
                seen.add(item.lower())
                out.append(item)
        return out

    def embedding_text(self, target_roles: list[str] | None = None) -> str:
        """What we embed to search jobs: the parts of a profile that a posting would mirror.

        Target roles from preferences come first: "what I want next" matters more than
        "what I did before" when someone is changing direction.
        """
        parts = []
        if target_roles:
            parts.append("Looking for: " + ", ".join(target_roles))
        if self.current_title:
            parts.append(self.current_title)
        if self.roles:
            parts.append("Experience as: " + ", ".join(self.roles[:5]))
        if self.years_experience is not None:
            parts.append(f"{self.years_experience:g} years of experience")
        if self.skills:
            parts.append("Skills: " + ", ".join(self.skills[:40]))
        if self.summary:
            parts.append(self.summary)
        return "\n".join(parts)
