"""Bot logic without Telegram: service functions return `Out`s; a fake bot records sends.

All users, resumes and jobs are synthetic.
"""

import json

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from app.bot import service
from app.bot.format import md_to_html, parse_locations, parse_salary, split_message
from app.bot.service import Deps, Out
from app.bot.telegram_app import send, webhook_secret
from app.db.models import Company, Job, Match, Preference, Resume
from tests.fakes import SAMPLE_PROFILE, SAMPLE_RESUME, FakeEmbedder, make_llm, make_pdf, tool_call

EMB = FakeEmbedder()


# ---- pure helpers ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("40 LPA", (4_000_000, "INR")),
        ("₹ 25 lakhs", (2_500_000, "INR")),
        ("$150k", (150_000, "USD")),
        ("120000 EUR", (120_000, "EUR")),
        ("1.2 cr", (12_000_000, "INR")),
        ("2500000", (2_500_000, "INR")),
        ("40", None),
        ("skip", None),
    ],
)
def test_parse_salary(text, expected):
    assert parse_salary(text) == expected


def test_parse_locations_adds_aliases():
    assert parse_locations("Bangalore, India") == ["Bangalore", "Bengaluru", "India"]
    assert parse_locations("Gurgaon or Mumbai") == ["Gurgaon", "Gurugram", "Mumbai", "Bombay"]


def test_md_to_html_escapes_and_converts():
    html = md_to_html("**Fit** <script> & [Job](https://x.y/a_b) snake_case_name")
    assert (
        html == '<b>Fit</b> &lt;script&gt; &amp; <a href="https://x.y/a_b">Job</a> snake_case_name'
    )


def test_split_message_respects_limit():
    parts = split_message("line\n" * 2000, limit=1000)
    assert all(len(p) <= 1000 for p in parts) and "".join(parts).count("line") == 2000


# ---- fixtures --------------------------------------------------------------------------


def deps_with(sessionmaker, engine, replies=None):
    llm, provider = make_llm(sessionmaker, replies)
    return Deps(sessionmaker, engine, llm, llm, EMB), provider


async def seed_jobs(sessionmaker, n=3, **kw):
    async with sessionmaker() as s, s.begin():
        jobs = [
            Job(
                source="t",
                source_id=f"{i}",
                company=kw.get("company", "Acme"),
                company_id=kw.get("company_id"),
                title=f"Backend <Engineer> {i}",
                url=f"https://x/{i}",
                embedding_text=f"Backend Engineer {i}\nAcme\n\npython",
                embedding=EMB.embed_query(f"backend python {i}"),
                location="Bangalore",
            )
            for i in range(n)
        ]
        s.add_all(jobs)
    return [j.id for j in jobs]


async def user_with_resume(deps):
    user_id = await service.get_or_create_user(deps, chat_id=111, name="Test")
    async with deps.sessionmaker() as s, s.begin():
        s.add(Resume(user_id=user_id, raw_text=SAMPLE_RESUME, profile=SAMPLE_PROFILE))
    return user_id


def rerank(ids):
    return json.dumps(
        {
            "matches": [
                {"job_id": j, "fit_score": 80, "reasons": ["Python <3"], "gap": "no Go"}
                for j in ids
            ]
        }
    )


# ---- onboarding & prefs --------------------------------------------------------------------


async def test_user_creation_is_idempotent(sessionmaker, engine):
    deps, _ = deps_with(sessionmaker, engine)
    a = await service.get_or_create_user(deps, 42, "A")
    b = await service.get_or_create_user(deps, 42, "A")
    assert a == b


async def test_onboarding_saves_preferences(sessionmaker, engine):
    deps, _ = deps_with(sessionmaker, engine)
    uid = await service.get_or_create_user(deps, 1, "A")
    for step, text in [
        ("role", "Backend Engineer, ML Engineer"),
        ("location", "Bangalore, India"),
        ("remote", "remote_only"),
        ("salary", "40 LPA"),
    ]:
        assert await service.save_onboarding(deps, uid, step, text) is None
    assert "didn't get that" in await service.save_onboarding(deps, uid, "salary", "40")
    async with sessionmaker() as s:
        p = await s.scalar(select(Preference).where(Preference.user_id == uid))
    assert p.roles == ["Backend Engineer", "ML Engineer"]
    assert p.locations == ["Bangalore", "Bengaluru", "India"]
    assert (p.remote_only, p.min_salary, p.salary_currency) == (True, 4_000_000, "INR")
    text = await service.prefs_text(deps, uid)
    assert "4,000,000 INR" in text and "Remote only: yes" in text


# ---- resume & jobs -----------------------------------------------------------------------


async def test_resume_upload_and_bad_pdf(sessionmaker, engine):
    deps, _ = deps_with(sessionmaker, engine, [json.dumps(SAMPLE_PROFILE)])
    uid = await service.get_or_create_user(deps, 1, "A")
    [ok] = await service.handle_resume(deps, uid, "cv.pdf", make_pdf(SAMPLE_RESUME))
    assert "Senior Backend Engineer" in ok.text and "/jobs" in ok.text
    [bad] = await service.handle_resume(deps, uid, "cv.pdf", b"nope")
    assert bad.text.startswith("⚠️")


async def test_jobs_without_resume_asks_for_one(sessionmaker, engine):
    deps, _ = deps_with(sessionmaker, engine)
    uid = await service.get_or_create_user(deps, 1, "A")
    [out] = await service.jobs(deps, uid)
    assert "resume" in out.text


async def test_jobs_messages_are_escaped_and_have_buttons(sessionmaker, engine):
    job_ids = await seed_jobs(sessionmaker)
    deps, _ = deps_with(sessionmaker, engine, [rerank(job_ids)])
    uid = await user_with_resume(deps)
    outs = await service.jobs(deps, uid)
    assert len(outs) == 3
    first = outs[0]
    assert "&lt;Engineer&gt;" in first.text and "Python &lt;3" in first.text  # escaped
    assert "Fit: <b>80/100</b>" in first.text
    labels = [label for label, _ in first.buttons[0]]
    assert labels == ["👍", "👎", "📝 Tailor resume"]


# ---- feedback ----------------------------------------------------------------------------


async def _match(deps, uid, **job_kw):
    [job_id] = await seed_jobs(deps.sessionmaker, n=1, **job_kw)
    async with deps.sessionmaker() as s, s.begin():
        m = Match(user_id=uid, job_id=job_id)
        s.add(m)
    return m.id


async def _prefs(deps, uid):
    async with deps.sessionmaker() as s:
        return await s.scalar(select(Preference).where(Preference.user_id == uid))


async def test_thumbs_down_company_and_industry_update_preferences(sessionmaker, engine):
    deps, _ = deps_with(sessionmaker, engine)
    uid = await service.get_or_create_user(deps, 1, "A")
    async with sessionmaker() as s, s.begin():
        s.add(Company(id=7, name="PayCo", slug="payco", ats="t", industry="fintech"))
    mid = await _match(deps, uid, company="PayCo", company_id=7)

    [ask] = await service.feedback(deps, uid, mid, -1)
    assert [row[0][1] for row in ask.buttons][0] == f"why:{mid}:company"
    await service.feedback_reason(deps, uid, mid, "company")
    await service.feedback_reason(deps, uid, mid, "industry")
    p = await _prefs(deps, uid)
    assert p.excluded_companies == ["PayCo"] and p.excluded_industries == ["fintech"]
    async with sessionmaker() as s:
        m = await s.get(Match, mid)
    assert (m.feedback, m.feedback_reason) == (-1, "industry")


async def test_thumbs_up_and_ownership(sessionmaker, engine):
    deps, _ = deps_with(sessionmaker, engine)
    uid = await service.get_or_create_user(deps, 1, "A")
    other = await service.get_or_create_user(deps, 2, "B")
    mid = await _match(deps, uid)
    [out] = await service.feedback(deps, uid, mid, 1)
    assert out.text.startswith("👍")
    [denied] = await service.feedback(deps, other, mid, -1)
    assert "no longer exists" in denied.text


async def test_free_text_reason_goes_to_the_agent(sessionmaker, engine):
    deps, provider = deps_with(
        sessionmaker,
        engine,
        [tool_call("update_preferences", note="no on-call"), "Noted: no on-call roles."],
    )
    uid = await service.get_or_create_user(deps, 1, "A")
    mid = await _match(deps, uid)
    [out] = await service.feedback_text(deps, uid, mid, "it requires on-call")
    assert out.text == "Noted: no on-call roles."
    assert "requires on-call" in provider.calls[0]["messages"][-1]["content"]
    assert (await _prefs(deps, uid)).notes == ["no on-call"]


# ---- tailoring, chat, digest ----------------------------------------------------------


async def test_tailor_sends_text_and_pdf(sessionmaker, engine):
    draft = {"summary": "Backend engineer.", "skills": ["Python"], "experience": []}
    verdicts = {
        "verdicts": [{"id": "summary.0", "supported": True}, {"id": "skill.0", "supported": True}]
    }
    deps, _ = deps_with(sessionmaker, engine, [json.dumps(draft), json.dumps(verdicts)])
    uid = await user_with_resume(deps)
    [job_id] = await seed_jobs(sessionmaker, n=1)
    text, doc = await service.tailor(deps, uid, job_id)
    assert "tailored" in text.text
    assert doc.document[0] == "resume-Acme.pdf" and doc.document[1].startswith(b"%PDF")


async def test_chat_converts_markdown(sessionmaker, engine):
    deps, _ = deps_with(sessionmaker, engine, ["**Hi** there"])
    uid = await service.get_or_create_user(deps, 1, "A")
    [out] = await service.chat(deps, uid, "hello")
    assert out.text == "<b>Hi</b> there"


class FakeBot:
    def __init__(self, blocked: set[int] = frozenset()):
        self.sent: list[tuple] = []
        self.blocked = blocked

    async def send_message(self, chat_id, text, **kw):
        if chat_id in self.blocked:
            from telegram.error import Forbidden

            raise Forbidden("bot was blocked by the user")
        self.sent.append(("text", chat_id, text, kw.get("reply_markup")))

    async def send_document(self, chat_id, document, **kw):
        self.sent.append(("doc", chat_id, document.filename, None))


async def test_send_splits_long_text_and_attaches_buttons_to_last_chunk():
    bot = FakeBot()
    await send(
        bot, 5, [Out("x\n" * 3000, buttons=[[("👍", "fb:1:1")]]), Out(document=("r.pdf", b"%PDF"))]
    )
    texts = [s for s in bot.sent if s[0] == "text"]
    assert len(texts) == 2 and texts[0][3] is None and texts[1][3] is not None
    assert bot.sent[-1][:3] == ("doc", 5, "r.pdf")


async def test_daily_digest_sends_matches_and_survives_blocked_users(sessionmaker, engine):
    from app.scheduler import send_daily_digest

    job_ids = await seed_jobs(sessionmaker, n=3)
    deps, _ = deps_with(sessionmaker, engine, [rerank(job_ids)])
    await user_with_resume(deps)  # chat 111
    blocked_uid = await service.get_or_create_user(deps, 222, "Blocked")
    async with sessionmaker() as s, s.begin():
        s.add(Resume(user_id=blocked_uid, raw_text="x", profile=SAMPLE_PROFILE))
    await service.get_or_create_user(deps, 333, "No resume")  # not a recipient

    bot = FakeBot(blocked={222})
    result = await send_daily_digest(bot, deps)
    assert result == {"sent": 1, "nothing_new": 0, "failed": 1}
    chats = {s[1] for s in bot.sent}
    assert chats == {111}
    assert "Good morning" in bot.sent[0][2] and len(bot.sent) == 4  # intro + 3 matches

    # Undelivered matches are not recorded as shown.
    async with sessionmaker() as s:
        assert (await s.scalars(select(Match).where(Match.user_id == blocked_uid))).all() == []

    # Next day: user 111 saw everything → no "nothing new" spam; the blocked user still fails.
    result = await send_daily_digest(FakeBot(blocked={222}), deps)
    assert result == {"sent": 0, "nothing_new": 1, "failed": 1}


# ---- webhook -------------------------------------------------------------------------------


async def test_webhook_requires_secret_and_queues_update(monkeypatch):
    import asyncio

    from telegram import Bot

    from app.config import get_settings
    from app.main import app

    monkeypatch.setattr(get_settings(), "telegram_bot_token", "123:test-token")

    class FakeApp:
        bot = Bot("123:test-token")
        update_queue: asyncio.Queue = asyncio.Queue()

    update = {
        "update_id": 1,
        "message": {"message_id": 1, "date": 0, "text": "hi", "chat": {"id": 5, "type": "private"}},
    }
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        assert (await c.post("/telegram/webhook", json=update)).status_code == 503
        app.state.telegram = FakeApp()
        try:
            bad = await c.post(
                "/telegram/webhook",
                json=update,
                headers={"X-Telegram-Bot-Api-Secret-Token": "wrong"},
            )
            ok = await c.post(
                "/telegram/webhook",
                json=update,
                headers={"X-Telegram-Bot-Api-Secret-Token": webhook_secret("123:test-token")},
            )
        finally:
            del app.state.telegram
    assert bad.status_code == 403 and ok.status_code == 200
    queued = FakeApp.update_queue.get_nowait()
    assert queued.message.text == "hi"
