"""Structured preference updates — how the agent's long-term memory gets written.

The LLM never writes SQL for this: it fills `PreferenceChanges` (validated by Pydantic)
and `apply_changes` maps it onto the row. add/remove lists make "no more fintech"
an append, not a full overwrite the model could get wrong.
"""

from typing import Literal

from pydantic import BaseModel, Field

from app.db.models import Preference


class PreferenceChanges(BaseModel):
    roles_add: list[str] = Field(default_factory=list, description="Target job titles to add")
    roles_remove: list[str] = Field(default_factory=list)
    locations_add: list[str] = Field(
        default_factory=list, description="Cities or countries, e.g. 'Bangalore', 'India'"
    )
    locations_remove: list[str] = Field(default_factory=list)
    remote_only: bool | None = Field(default=None, description="true for 'remote only'")
    min_salary: int | None = Field(default=None, ge=0, description="Yearly minimum")
    salary_currency: str | None = Field(default=None, description="ISO code, e.g. INR, USD")
    seniority: Literal["intern", "junior", "mid", "senior", "lead"] | None = None
    exclude_companies_add: list[str] = Field(default_factory=list)
    exclude_companies_remove: list[str] = Field(default_factory=list)
    exclude_industries_add: list[str] = Field(
        default_factory=list, description="e.g. 'fintech', 'crypto' (see known industries)"
    )
    exclude_industries_remove: list[str] = Field(default_factory=list)
    note: str | None = Field(
        default=None, description="Any other lasting preference, e.g. 'no on-call'"
    )


def _merge(current: list[str], add: list[str], remove: list[str]) -> list[str]:
    removed = {r.strip().lower() for r in remove}
    out = [c for c in current if c.lower() not in removed]
    for item in (a.strip() for a in add):
        if item and item.lower() not in {o.lower() for o in out}:
            out.append(item)
    return out


def apply_changes(prefs: Preference, ch: PreferenceChanges) -> list[str]:
    """Mutate `prefs`; return a human-readable list of what changed (for the agent/logs)."""
    changed: list[str] = []

    def set_list(attr: str, add: list[str], remove: list[str]) -> None:
        new = _merge(list(getattr(prefs, attr) or []), add, remove)
        if new != list(getattr(prefs, attr) or []):
            setattr(prefs, attr, new)
            changed.append(f"{attr} = {new}")

    set_list("roles", ch.roles_add, ch.roles_remove)
    set_list("locations", ch.locations_add, ch.locations_remove)
    set_list("excluded_companies", ch.exclude_companies_add, ch.exclude_companies_remove)
    set_list("excluded_industries", ch.exclude_industries_add, ch.exclude_industries_remove)
    for attr in ("remote_only", "min_salary", "seniority"):
        value = getattr(ch, attr)
        if value is not None and value != getattr(prefs, attr):
            setattr(prefs, attr, value)
            changed.append(f"{attr} = {value}")
    if ch.salary_currency and ch.salary_currency.upper() != prefs.salary_currency:
        prefs.salary_currency = ch.salary_currency.upper()[:3]
        changed.append(f"salary_currency = {prefs.salary_currency}")
    if ch.note:
        set_list("notes", [ch.note], [])
    return changed


def describe(prefs: Preference) -> dict:
    return {
        "roles": prefs.roles,
        "locations": prefs.locations,
        "remote_only": prefs.remote_only,
        "min_salary": prefs.min_salary,
        "salary_currency": prefs.salary_currency,
        "seniority": prefs.seniority,
        "excluded_companies": prefs.excluded_companies,
        "excluded_industries": prefs.excluded_industries,
        "notes": prefs.notes,
    }
