"""Telegram text helpers: LLM markdown → Telegram HTML, message splitting, input parsing.

HTML parse mode instead of MarkdownV2: MarkdownV2 requires escaping 18 characters and a
single stray "." or "-" from the model makes Telegram reject the whole message.
"""

import html
import re

TELEGRAM_LIMIT = 4096


def md_to_html(text: str) -> str:
    """Convert the small markdown subset LLMs use into Telegram-safe HTML."""
    text = html.escape(text, quote=False)
    text = re.sub(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", r'<a href="\2">\1</a>', text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"^#{1,6}\s*(.+)$", r"<b>\1</b>", text, flags=re.MULTILINE)
    return text


def split_message(text: str, limit: int = TELEGRAM_LIMIT) -> list[str]:
    """Split on paragraph/line boundaries so no message exceeds Telegram's limit."""
    if len(text) <= limit:
        return [text]
    parts, current = [], ""
    for line in text.split("\n"):
        while len(line) > limit:  # a single giant line: hard cut
            parts.append(line[:limit])
            line = line[limit:]
        if len(current) + len(line) + 1 > limit:
            parts.append(current)
            current = line
        else:
            current = f"{current}\n{line}" if current else line
    if current:
        parts.append(current)
    return parts


_SALARY = re.compile(
    r"(?P<cur>\$|₹|usd|inr|eur|€|gbp|£)?\s*(?P<num>\d+(?:[.,]\d+)?)\s*"
    r"(?P<unit>lpa|lakhs?|l\b|k\b|m\b|cr\b|crores?)?\s*(?P<cur2>usd|inr|eur|gbp)?",
    re.IGNORECASE,
)
_CURRENCY = {"$": "USD", "usd": "USD", "₹": "INR", "inr": "INR", "€": "EUR", "eur": "EUR",
             "£": "GBP", "gbp": "GBP"}  # fmt: skip


def parse_salary(text: str) -> tuple[int, str] | None:
    """'40 LPA' → (4000000, 'INR'); '$150k' → (150000, 'USD'); 'skip' → None.

    LPA = lakhs per annum (1 lakh = 100,000 INR), the usual way salaries are quoted in
    India. A bare number defaults to INR if it looks like rupees (>= 100,000); a bare
    small number ("40") is ambiguous and returns None.
    """
    m = _SALARY.search(text.strip())
    if not m or not m.group("num"):
        return None
    amount = float(m.group("num").replace(",", ""))
    unit = (m.group("unit") or "").lower()
    cur_token = (m.group("cur") or m.group("cur2") or "").lower()
    currency = _CURRENCY.get(cur_token)
    if unit in ("lpa", "lakh", "lakhs", "l"):
        amount, currency = amount * 100_000, currency or "INR"
    elif unit in ("cr", "crore", "crores"):
        amount, currency = amount * 10_000_000, currency or "INR"
    elif unit == "k":
        amount *= 1_000
    elif unit == "m":
        amount *= 1_000_000
    elif amount < 1_000:
        return None  # "40" is ambiguous (40 LPA? $40k?) — ask again rather than guess
    if currency is None:
        currency = "INR" if amount >= 100_000 else "USD"
    return int(amount), currency


# Common alternate spellings, so "Bangalore" also matches postings that say "Bengaluru".
_ALIASES = {
    "bangalore": "Bengaluru", "bengaluru": "Bangalore", "bombay": "Mumbai",
    "mumbai": "Bombay", "gurgaon": "Gurugram", "gurugram": "Gurgaon",
    "madras": "Chennai", "chennai": "Madras", "calcutta": "Kolkata", "kolkata": "Calcutta",
    "usa": "United States", "us": "United States", "uk": "United Kingdom",
}  # fmt: skip


def parse_locations(text: str) -> list[str]:
    """'Bangalore, India' → ['Bangalore', 'Bengaluru', 'India'] (deduped, order kept)."""
    out: list[str] = []
    for part in re.split(r"[,/;]| or | and ", text):
        part = part.strip()
        if not part:
            continue
        for item in (part, _ALIASES.get(part.lower())):
            if item and item.lower() not in {o.lower() for o in out}:
                out.append(item)
    return out
