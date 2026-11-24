"""Tailor the user's active resume for a job, store the result, render PDFs on demand."""

import re
from dataclasses import asdict

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models import Job, Resume, TailoredResumeRecord
from app.db.queries import get_active_resume
from app.llm.client import LLMClient
from app.resume.render import ResumeHeader, render_pdf
from app.resume.schemas import ResumeProfile
from app.resume.tailor import TailoredResume, tailor_resume

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"\+?\d[\d\s-]{8,}\d")


class TailorError(Exception):
    """Shown to the user: missing resume or job."""


async def tailor_for_job(
    sessionmaker: async_sessionmaker[AsyncSession], llm: LLMClient, user_id: int, job_id: int
) -> TailoredResumeRecord:
    async with sessionmaker() as session:
        resume = await get_active_resume(session, user_id)
        job = await session.get(Job, job_id)
    if resume is None:
        raise TailorError("Upload a resume first (/resume).")
    if job is None:
        raise TailorError(f"Job {job_id} not found.")

    result = await tailor_resume(
        llm,
        original_text=resume.raw_text,
        job_title=job.title,
        job_company=job.company,
        job_text=job.embedding_text.split("\n\n", 1)[-1],
        user_id=user_id,
    )
    async with sessionmaker() as session, session.begin():
        record = TailoredResumeRecord(
            user_id=user_id,
            job_id=job_id,
            resume_id=resume.id,
            content=result.resume.model_dump(mode="json"),
            dropped=[asdict(d) for d in result.dropped],
        )
        session.add(record)
    return record


async def tailored_pdf(
    sessionmaker: async_sessionmaker[AsyncSession], record: TailoredResumeRecord
) -> bytes:
    """Header (name, contact, education) comes straight from the original resume."""
    async with sessionmaker() as session:
        resume = await session.get(Resume, record.resume_id)
    profile = ResumeProfile.model_validate(resume.profile or {})
    contact = [p for p in (profile.location,) if p]
    if email := _EMAIL.search(resume.raw_text):
        contact.append(email.group())
    if phone := _PHONE.search(resume.raw_text):
        contact.append(phone.group())
    education = [
        ", ".join(str(x) for x in (e.degree, e.field, e.institution, e.year) if x)
        for e in profile.education
    ]
    header = ResumeHeader(
        name=profile.name or "Resume", contact=" | ".join(contact), education=education
    )
    return render_pdf(header, TailoredResume.model_validate(record.content))
