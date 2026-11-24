"""Text helpers for turning raw postings into clean, comparable fields."""

import re

from bs4 import BeautifulSoup

_BLOCK_TAGS = ["p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "ul", "ol"]


def html_to_text(html: str | None) -> str:
    """Strip HTML to readable plain text, keeping paragraph/list breaks.

    Line breaks matter: later we embed this text and show snippets to the LLM, and
    "Requirements\\n- Python" reads far better than one run-on line. We add newlines only
    after block tags, so inline tags like <b> or <a> don't split sentences.
    """
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    for br in soup.find_all("br"):
        br.replace_with("\n")
    for li in soup.find_all("li"):
        li.insert(0, "- ")
    for tag in soup.find_all(_BLOCK_TAGS):
        tag.append("\n")

    out: list[str] = []
    for raw in soup.get_text().splitlines():
        line = re.sub(r"[ \t ]+", " ", raw).strip()
        if line in ("", "-"):
            if out and out[-1]:
                out.append("")  # keep one blank line between blocks
            continue
        out.append(line)
    return "\n".join(out).strip()


_AMOUNT = r"(\d{2,3}(?:,\d{3})+|\d{2,3}(?:\.\d+)?\s?[kK])"
_SALARY_RANGE = re.compile(rf"\$\s?{_AMOUNT}\s*(?:-|–|—|to)\s*\$?\s?{_AMOUNT}")


def _parse_amount(raw: str) -> int:
    raw = raw.replace(",", "").replace(" ", "")
    if raw[-1] in "kK":
        return int(float(raw[:-1]) * 1000)
    return int(raw)


def extract_usd_salary_range(text: str) -> tuple[int, int] | None:
    """Find a yearly USD range like "$150,000 - $200,000" or "$120k–$160k" in free text.

    Many US postings publish pay only in the description (pay-transparency laws). We only
    accept plausible yearly figures (10k–2M) so "$5B in funding" or "$50-$70/hr" don't match.
    """
    for m in _SALARY_RANGE.finditer(text):
        low, high = _parse_amount(m.group(1)), _parse_amount(m.group(2))
        if 10_000 <= low <= high <= 2_000_000:
            return low, high
    return None


_REMOTE = re.compile(r"\b(remote|anywhere|work from home|wfh)\b", re.IGNORECASE)


def looks_remote(*texts: str | None) -> bool:
    return any(t and _REMOTE.search(t) for t in texts)


# Currencies where any real yearly tech salary is at least ~15k units. Below that it's
# almost certainly hourly/monthly pay (e.g. intern stipends) rather than a yearly figure.
_HIGH_VALUE_CURRENCIES = {"USD", "EUR", "GBP", "CAD", "AUD", "SGD", "CHF"}


def plausible_annual_salary(low: int | None, high: int | None, currency: str | None) -> bool:
    """Reject salary ranges that are not believable *yearly* figures.

    Sources mix in hourly rates, monthly stipends and typos (we saw "$152,405–$179,300,152").
    A wrong salary silently breaks the min-salary filter, so we prefer "unknown" over wrong.
    """
    if low is None and high is None:
        return True
    low, high = low or high, high or low
    assert low is not None and high is not None
    if low > high or high > 10 * low or low < 1_000:
        return False
    return not (currency in _HIGH_VALUE_CURRENCIES and low < 15_000)
