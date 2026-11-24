"""What text we embed for a job, and its hash.

Kept in one place because ingestion (which computes `content_hash`) and embedding must
agree exactly: if the hash changes, the job gets re-embedded.
"""

import hashlib
import re
from collections import Counter

# bge-small reads at most 512 tokens; ~2000 chars of English is roughly 450 tokens. Cutting
# here keeps CPU time predictable instead of tokenizing 10k-char postings only to truncate.
MAX_DESCRIPTION_CHARS = 2000


def job_embedding_text(title: str, company: str, description: str) -> str:
    """Title and company first: they carry the most signal and always survive truncation.

    `description` should already have company boilerplate removed (see strip_boilerplate).

    Location is deliberately left out — it's filtered exactly in SQL, and including it
    would make the same role in two cities look less similar than it is.
    """
    return f"{title}\n{company}\n\n{description[:MAX_DESCRIPTION_CHARS]}".strip()


# A line is boilerplate if it appears in at least this share of one company's postings.
# Only judged for companies with enough postings for "shared" to mean something.
BOILERPLATE_MIN_SHARE = 0.2
BOILERPLATE_MIN_POSTINGS = 5
BOILERPLATE_MIN_REPEATS = 3  # else 20% of 5 postings = "appears once" = everything
_MIN_LINE_CHARS = 30  # short lines ("Requirements", "- Python") are never boilerplate


def find_boilerplate(descriptions: list[str]) -> set[str]:
    """Lines repeated across many postings of one company (intro, benefits, EEO text).

    Measured on our data: ~40% of all description text is such boilerplate. Left in, it
    fills the 2000-char embedding window, so two different roles at one company look
    near-identical and a resume ends up compared mostly against "About Stripe".
    """
    if len(descriptions) < BOILERPLATE_MIN_POSTINGS:
        return set()
    counts = Counter(
        line for d in descriptions for line in _lines(d) if len(line) >= _MIN_LINE_CHARS
    )
    min_count = max(BOILERPLATE_MIN_REPEATS, BOILERPLATE_MIN_SHARE * len(descriptions))
    return {line for line, n in counts.items() if n >= min_count}


def strip_boilerplate(description: str, boilerplate: set[str]) -> str:
    if not boilerplate:
        return description
    lines = description.splitlines()
    kept = "\n".join(line for line in lines if line.strip() not in boilerplate)
    return re.sub(r"\n\s*\n+", "\n\n", kept).strip()  # collapse the gaps left behind


def _lines(description: str) -> set[str]:
    return {line.strip() for line in description.splitlines()}


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
