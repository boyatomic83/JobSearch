import json

import pytest
from sqlalchemy import select

from app.db.models import Resume, User
from app.resume.parse import ResumeError, extract_text
from app.resume.schemas import ResumeProfile
from app.resume.service import save_resume
from tests.fakes import SAMPLE_PROFILE, SAMPLE_RESUME, FakeEmbedder, make_llm, make_pdf


def test_extract_text_from_pdf():
    text = extract_text(make_pdf(SAMPLE_RESUME))
    assert "Senior Backend Engineer" in text and "FastAPI" in text


def test_rejects_textless_and_broken_pdfs():
    with pytest.raises(ResumeError, match="scanned"):
        extract_text(make_pdf("Hi"))
    with pytest.raises(ResumeError, match="Could not read"):
        extract_text(b"definitely not a pdf")
    with pytest.raises(ResumeError, match="5 MB"):
        extract_text(b"x" * (6 * 1024 * 1024))


def test_profile_cleans_lists_and_builds_embedding_text():
    profile = ResumeProfile.model_validate(
        SAMPLE_PROFILE | {"skills": ["Python", " python ", "", "SQL"]}
    )
    assert profile.skills == ["Python", "SQL"]
    text = profile.embedding_text(["ML Engineer"])
    assert text.startswith("Looking for: ML Engineer\nSenior Backend Engineer")
    assert "7 years of experience" in text and "Skills: Python, SQL" in text


def test_profile_rejects_impossible_values():
    with pytest.raises(ValueError):
        ResumeProfile.model_validate(SAMPLE_PROFILE | {"years_experience": 99})
    with pytest.raises(ValueError):
        ResumeProfile.model_validate(SAMPLE_PROFILE | {"seniority": "wizard"})


async def test_save_resume_replaces_previous_active_resume(sessionmaker):
    async with sessionmaker() as s, s.begin():
        user = User(name="Test")
        s.add(user)
    llm, provider = make_llm(sessionmaker, [json.dumps(SAMPLE_PROFILE)] * 2)
    pdf = make_pdf(SAMPLE_RESUME)

    _, profile = await save_resume(
        sessionmaker, llm, FakeEmbedder(), user_id=user.id, file_name="a.pdf", pdf_bytes=pdf
    )
    assert profile.current_title == "Senior Backend Engineer"
    assert "Priya" in provider.calls[0]["messages"][0]["content"]  # resume text in prompt

    await save_resume(  # second upload (served from the LLM cache)
        sessionmaker, llm, FakeEmbedder(), user_id=user.id, file_name="b.pdf", pdf_bytes=pdf
    )
    async with sessionmaker() as s:
        resumes = (await s.scalars(select(Resume).order_by(Resume.id))).all()
    assert [(r.file_name, r.is_active) for r in resumes] == [("a.pdf", False), ("b.pdf", True)]
    assert resumes[1].embedding is not None and len(resumes[1].embedding) == 384
    assert len(provider.calls) == 1  # identical resume → cached extraction
