"""PDF → text, plus LLM text → structured profile."""

import io
import re

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.llm.client import LLMClient
from app.resume.schemas import ResumeProfile

MAX_PDF_BYTES = 5 * 1024 * 1024
MAX_PAGES = 10
MAX_RESUME_CHARS = 12_000  # ~3k tokens: plenty for a resume, bounded for the free tier


class ResumeError(ValueError):
    """The upload can't be used as a resume (with a message safe to show the user)."""


def extract_text(pdf_bytes: bytes) -> str:
    if len(pdf_bytes) > MAX_PDF_BYTES:
        raise ResumeError("PDF is larger than 5 MB")
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = reader.pages[:MAX_PAGES]
        text = "\n".join(page.extract_text() or "" for page in pages)
    except (PdfReadError, ValueError, KeyError) as exc:
        raise ResumeError("Could not read this PDF") from exc
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
    if len(text) < 200:
        # Usually a scanned image PDF: there is no text layer to extract.
        raise ResumeError("Found almost no text in this PDF — is it a scanned image?")
    return text[:MAX_RESUME_CHARS]


EXTRACT_SYSTEM = """You extract a structured profile from a resume.
Rules:
- Use ONLY information written in the resume. Never guess or invent.
- Unknown fields are null (or [] for lists).
- current_title and roles: job titles only, e.g. "Senior Backend Engineer" — never the
  company name or dates.
- years_experience: total professional years; count internships as 0.
- seniority: intern | junior (<2y) | mid (2-5y) | senior (5-10y) | lead (10y+ or
  staff/principal/manager titles).
- skills: concrete technologies, languages, frameworks, tools — not soft skills."""


async def extract_profile(llm: LLMClient, resume_text: str, user_id: int | None) -> ResumeProfile:
    return await llm.structured(
        purpose="resume_extract",
        system=EXTRACT_SYSTEM,
        prompt=f"Resume:\n<<<\n{resume_text}\n>>>",
        schema=ResumeProfile,
        user_id=user_id,
    )
