"""Render a tailored resume to PDF with reportlab.

reportlab rather than weasyprint: it's pure Python, while weasyprint needs Pango/Cairo
system libraries that would bloat the slim Docker image for Render's free tier.
"""

import io
import unicodedata
from dataclasses import dataclass
from xml.sax.saxutils import escape as xml_escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer

from app.resume.tailor import TailoredResume

# LLMs love typographic characters that the built-in PDF fonts (Windows-1252) don't have:
# seen in testing, U+2011 rendered as "high■performance". Map them to safe equivalents.
_REPLACEMENTS = str.maketrans({
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2212": "-",  # hyphens, minus
    "\u00a0": " ", "\u202f": " ", "\u2009": " ", "\u200b": "",  # odd spaces
    "\u20b9": "INR ",  # rupee sign
})  # fmt: skip


def _pdf_safe(text: str) -> str:
    text = text.translate(_REPLACEMENTS)
    out = []
    for ch in text:
        try:
            ch.encode("cp1252")
            out.append(ch)
        except UnicodeEncodeError:
            base = unicodedata.normalize("NFKD", ch).encode("cp1252", "ignore").decode("cp1252")
            out.append(base or "?")
    return "".join(out)


def escape(text: str) -> str:
    """Font-safe and XML-escaped (Paragraph parses a mini-markup)."""
    return xml_escape(_pdf_safe(text))


@dataclass
class ResumeHeader:
    name: str
    contact: str = ""  # e.g. "Bangalore, India | priya@example.com"
    education: list[str] | None = None


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "name": ParagraphStyle("name", parent=base["Title"], fontSize=18, spaceAfter=2),
        "contact": ParagraphStyle(
            "contact", parent=base["Normal"], alignment=1, textColor=colors.grey, fontSize=9
        ),
        "h": ParagraphStyle(
            "h",
            parent=base["Heading2"],
            fontSize=11,
            spaceBefore=8,
            spaceAfter=3,
            textColor=colors.HexColor("#1f3a5f"),
        ),  # fmt: skip
        "role": ParagraphStyle("role", parent=base["Normal"], fontSize=10, spaceBefore=4),
        "body": ParagraphStyle("body", parent=base["Normal"], fontSize=9.5, leading=13),
    }


def render_pdf(header: ResumeHeader, resume: TailoredResume) -> bytes:
    st = _styles()
    story: list = [Paragraph(escape(header.name), st["name"])]
    if header.contact:
        story.append(Paragraph(escape(header.contact), st["contact"]))
    story.append(Spacer(1, 4 * mm))

    if resume.summary:
        story += [Paragraph("Summary", st["h"]), Paragraph(escape(resume.summary), st["body"])]
    if resume.skills:
        story += [
            Paragraph("Skills", st["h"]),
            Paragraph(escape(", ".join(resume.skills)), st["body"]),
        ]
    if resume.experience:
        story.append(Paragraph("Experience", st["h"]))
        for role in resume.experience:
            dates = f" &nbsp;·&nbsp; {escape(role.dates)}" if role.dates else ""
            story.append(
                Paragraph(f"<b>{escape(role.title)}</b>, {escape(role.company)}{dates}", st["role"])
            )
            if role.bullets:
                story.append(
                    ListFlowable(
                        [ListItem(Paragraph(escape(b), st["body"])) for b in role.bullets],
                        bulletType="bullet", leftIndent=12, bulletFontSize=7,
                    )
                )  # fmt: skip
    if header.education:
        story.append(Paragraph("Education", st["h"]))
        story += [Paragraph(escape(e), st["body"]) for e in header.education]

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=15 * mm, bottomMargin=15 * mm, title=f"{header.name} - Resume",
    )  # fmt: skip
    doc.build(story)
    return buf.getvalue()
