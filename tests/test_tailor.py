"""Resume tailoring: invented claims must never survive. Fake LLM; synthetic resume/job."""

import io
import json

import pytest
from pypdf import PdfReader
from sqlalchemy import select

from app.agent.loop import handle_message
from app.agent.tools import AgentContext
from app.db.models import Job, Resume, TailoredResumeRecord, User
from app.resume.render import ResumeHeader, render_pdf
from app.resume.tailor import TailoredResume, rule_violations, tailor_resume
from app.resume.tailor_service import tailor_for_job, tailored_pdf
from tests.fakes import SAMPLE_PROFILE, SAMPLE_RESUME, FakeEmbedder, make_llm, tool_call

# What a (misbehaving) tailoring model might return: mostly faithful, plus inventions.
DRAFT = {
    "summary": "Senior backend engineer building Python FastAPI services on PostgreSQL. "
    "Led a team of 50 engineers.",  # 2nd sentence: invented (and has a new number)
    "skills": ["Python", "FastAPI", "Kubernetes", "Rust"],  # Rust: invented
    "experience": [
        {
            "title": "Senior Backend Engineer",
            "company": "Acme Payments",
            "dates": "2021-present",
            "bullets": [
                "Built Python FastAPI services handling 2k requests/second on PostgreSQL",
                "Led migration of batch jobs to Kubernetes; cut infra cost by 45%",  # was 30%
                "Mentored engineers on API design",  # invented, no number → verifier's job
            ],
        },
        {
            "title": "Staff Engineer",
            "company": "Google",
            "dates": "2016-2018",
            "bullets": ["Built search"],
        },  # invented employer  # fmt: skip
    ],
}


def verdicts(unsupported: set[str], claims_text: str) -> str:
    """A verifier reply: everything supported except `unsupported` ids."""
    ids = [line.split(":", 1)[0] for line in claims_text.splitlines() if line.strip()]
    return json.dumps(
        {
            "verdicts": [
                {"id": i, "supported": i not in unsupported, "reason": "checked"} for i in ids
            ]
        }  # fmt: skip
    )


def fake_verifier(unsupported: set[str]):
    def reply(messages):
        prompt = messages[-1]["content"]
        return verdicts(unsupported, prompt.split("CLAIMS:\n", 1)[1])

    return reply


def test_rule_checks_catch_numbers_skills_and_employers():
    bad = rule_violations(TailoredResume.model_validate(DRAFT), SAMPLE_RESUME)
    assert "50" in bad["summary.1"]
    assert "Rust" in bad["skill.3"]
    assert "45" in bad["role.0.bullet.1"]
    assert "Google" in bad["role.1"]
    assert "role.0.bullet.0" not in bad  # 2k / PostgreSQL are in the original
    assert "role.0" not in bad  # dates 2021 are in the original


async def test_tailoring_drops_everything_not_in_the_original(sessionmaker):
    # The LLM verifier only catches the number-free invention; rules catch the rest.
    llm, provider = make_llm(sessionmaker, [json.dumps(DRAFT), fake_verifier({"role.0.bullet.2"})])
    result = await tailor_resume(
        llm, original_text=SAMPLE_RESUME, job_title="Backend Engineer", job_company="GitLab",
        job_text="Build scalable Python APIs.", user_id=None,
    )  # fmt: skip
    r = result.resume
    assert r.summary == "Senior backend engineer building Python FastAPI services on PostgreSQL."
    assert r.skills == ["Python", "FastAPI", "Kubernetes"]
    assert [role.company for role in r.experience] == ["Acme Payments"]
    assert r.experience[0].bullets == [
        "Built Python FastAPI services handling 2k requests/second on PostgreSQL"
    ]
    reasons = {d.reason.split(":")[0] for d in result.dropped}
    assert reasons == {"verifier", "rule"}
    assert len(result.dropped) == 5
    # The verifier was shown the original resume and every claim.
    verify_prompt = provider.calls[1]["messages"][0]["content"]
    assert "Priya Sharma" in verify_prompt and "role.0.bullet.2:" in verify_prompt


async def test_claims_the_verifier_skips_are_dropped(sessionmaker):
    draft = {"summary": "Backend engineer.", "skills": ["Python"], "experience": []}
    llm, _ = make_llm(sessionmaker, [json.dumps(draft), json.dumps({"verdicts": []})])
    result = await tailor_resume(
        llm, original_text=SAMPLE_RESUME, job_title="x", job_company="y", job_text="z",
        user_id=None,
    )  # fmt: skip
    assert result.resume.summary == "" and result.resume.skills == []
    assert {d.reason for d in result.dropped} == {"not verified"}


def test_render_pdf_has_no_missing_glyphs():
    from app.resume.render import _pdf_safe

    # Seen in a real run: U+2011 rendered as a black box.
    assert _pdf_safe("high\u2011performance, cloud\u2011native") == "high-performance, cloud-native"
    assert _pdf_safe("40 \u20b9LPA, caf\u00e9, na\u00efve \u2014 \u201cok\u201d") == (
        "40 INR LPA, caf\u00e9, na\u00efve \u2014 \u201cok\u201d"
    )  # cp1252 has \u00e9, \u2014 and curly quotes, so they stay
    assert _pdf_safe("\u0101") == "a"  # decomposable → base letter
    assert _pdf_safe("\u0939") == "?"  # Devanagari: no glyph in the built-in font


def test_render_pdf_contains_the_content():
    pdf = render_pdf(
        ResumeHeader("Priya Sharma", "Bangalore | priya@example.com", ["B.Tech, IIT Delhi"]),
        TailoredResume(
            summary="Backend engineer & API <builder>.",  # XML chars must be escaped
            skills=["Python", "FastAPI"],
            experience=[
                {
                    "title": "Senior Backend Engineer",
                    "company": "Acme Payments",
                    "dates": "2021-present",
                    "bullets": ["Built FastAPI services"],
                }
            ],
        ),  # fmt: skip
    )
    assert pdf.startswith(b"%PDF")
    text = "\n".join(p.extract_text() for p in PdfReader(io.BytesIO(pdf)).pages)
    for expected in ("Priya Sharma", "API <builder>", "Python, FastAPI", "Acme Payments",
                     "Built FastAPI services", "IIT Delhi"):  # fmt: skip
        assert expected in text


async def _seed(sessionmaker):
    async with sessionmaker() as s, s.begin():
        user = User(name="t")
        s.add(user)
        await s.flush()
        s.add(Resume(user_id=user.id, raw_text=SAMPLE_RESUME, profile=SAMPLE_PROFILE))
        text = "Backend Engineer\nGitLab\n\nPython APIs"
        job = Job(source="t", source_id="1", company="GitLab", title="Backend Engineer",
                  url="https://x", embedding_text=text)  # fmt: skip
        s.add(job)
    return user.id, job.id


async def test_tailor_for_job_stores_and_renders(sessionmaker):
    user_id, job_id = await _seed(sessionmaker)
    llm, _ = make_llm(sessionmaker, [json.dumps(DRAFT), fake_verifier(set())])
    record = await tailor_for_job(sessionmaker, llm, user_id, job_id)
    async with sessionmaker() as s:
        stored = await s.get(TailoredResumeRecord, record.id)
    assert "Rust" not in stored.content["skills"]
    assert any("Google" in d["reason"] for d in stored.dropped)

    text = PdfReader(io.BytesIO(await tailored_pdf(sessionmaker, stored))).pages[0].extract_text()
    assert "Priya Sharma" in text and "priya@example.com" in text and "IIT Delhi" in text
    assert "Google" not in text and "Rust" not in text and "45%" not in text


async def test_agent_tailor_tool_reports_pdf_and_removed_claims(sessionmaker, engine):
    user_id, job_id = await _seed(sessionmaker)
    llm, _ = make_llm(
        sessionmaker,
        [tool_call("tailor_resume", job_id=job_id), json.dumps(DRAFT), fake_verifier(set()),
         "Here's your tailored resume (PDF attached)."],
    )  # fmt: skip
    ctx = AgentContext(sessionmaker, engine, llm, FakeEmbedder(), user_id)
    reply = await handle_message(ctx, "tailor my resume for that GitLab job")
    result = reply.steps[0]["result"]
    assert result["claims_removed_by_fact_check"]
    assert reply.tailored_resume_ids == [result["tailored_resume_id"]]


async def test_pdf_endpoint_checks_ownership(sessionmaker):
    from httpx import ASGITransport, AsyncClient

    from app.db.session import get_sessionmaker
    from app.main import app

    user_id, job_id = await _seed(sessionmaker)
    llm, _ = make_llm(sessionmaker, [json.dumps(DRAFT), fake_verifier(set())])
    record = await tailor_for_job(sessionmaker, llm, user_id, job_id)
    app.dependency_overrides[get_sessionmaker] = lambda: sessionmaker
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t",
                               headers={"X-Admin-Key": "test-admin-key"}) as c:  # fmt: skip
            ok = await c.get(f"/users/{user_id}/tailored/{record.id}.pdf")
            other = await c.get(f"/users/{user_id + 1}/tailored/{record.id}.pdf")
    finally:
        app.dependency_overrides.clear()
    assert ok.status_code == 200 and ok.content.startswith(b"%PDF")
    assert ok.headers["content-type"] == "application/pdf"
    assert other.status_code == 404


@pytest.mark.parametrize("missing", ["resume", "job"])
async def test_tailor_errors(sessionmaker, missing):
    from app.resume.tailor_service import TailorError

    async with sessionmaker() as s, s.begin():
        user = User(name="t")
        s.add(user)
    llm, _ = make_llm(sessionmaker)
    if missing == "job":
        async with sessionmaker() as s, s.begin():
            s.add(Resume(user_id=user.id, raw_text=SAMPLE_RESUME, profile=SAMPLE_PROFILE))
    with pytest.raises(TailorError):
        await tailor_for_job(sessionmaker, llm, user.id, 999)
    async with sessionmaker() as s:
        assert (await s.scalars(select(TailoredResumeRecord))).all() == []
