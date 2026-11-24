"""Ghost-job scoring: how likely a posting is stale, evergreen or not a real opening.

Rule-based on purpose: every point of the score comes with a human-readable reason, so a
user (or interviewer) can see exactly why a job was hidden. Scores are summed and capped
at 1.0; retrieval hides jobs above GHOST_THRESHOLD.

| Signal                                     | Points |
|--------------------------------------------|--------|
| Evergreen title ("talent pool", "general application", ...) | 0.50 |
| Posted > 90 days ago / > 45 days ago       | 0.35 / 0.15 |
| Reposted (disappeared and came back)       | 0.15 per repost, max 0.30 |
| Description < 400 chars / < 1200 chars     | 0.30 / 0.10 |
| No responsibilities/requirements section   | 0.10 |
| No salary                                  | 0.05 |
| No location                                | 0.10 |

Weights are chosen so no single *soft* signal (age, missing salary) can hide a job alone,
but stacked signals (old + reposted + thin description) do.
"""

import re
from dataclasses import dataclass
from datetime import datetime

GHOST_THRESHOLD = 0.7

_EVERGREEN_TITLE = re.compile(
    r"talent (pool|community|network)|general application|future (opportunit|opening|role)"
    r"|expression of interest|open application|don.t see (a|the) role|evergreen",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class GhostFacts:
    """Only what the score needs — selectable in SQL without loading full descriptions."""

    title: str
    posted_at: datetime | None
    first_seen_at: datetime
    times_reposted: int
    description_length: int
    has_structure: bool  # mentions responsibilities / requirements / qualifications
    has_salary: bool
    has_location: bool


def ghost_score(f: GhostFacts, now: datetime) -> tuple[float, list[str]]:
    points: list[tuple[float, str]] = []

    if _EVERGREEN_TITLE.search(f.title):
        points.append((0.50, "evergreen / talent-pool posting"))

    age_days = (now - (f.posted_at or f.first_seen_at)).days
    if age_days > 90:
        points.append((0.35, f"posted {age_days} days ago"))
    elif age_days > 45:
        points.append((0.15, f"posted {age_days} days ago"))

    if f.times_reposted:
        points.append((min(0.15 * f.times_reposted, 0.30), f"reposted {f.times_reposted}x"))

    if f.description_length < 400:
        points.append((0.30, f"very short description ({f.description_length} chars)"))
    elif f.description_length < 1200:
        points.append((0.10, f"short description ({f.description_length} chars)"))

    if not f.has_structure:
        points.append((0.10, "no responsibilities/requirements section"))
    if not f.has_salary:
        points.append((0.05, "no salary listed"))
    if not f.has_location:
        points.append((0.10, "no location listed"))

    score = round(min(sum(p for p, _ in points), 1.0), 2)
    return score, [f"{reason} (+{p:.2f})" for p, reason in points]
