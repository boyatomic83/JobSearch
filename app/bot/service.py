"""Bot logic, independent of Telegram: every function returns `Out` messages to send.

Keeping this layer free of python-telegram-bot objects means it is unit-tested directly
(tests/test_bot.py) and the Telegram adapter (telegram_app.py) stays a thin shell.
"""

import html
import logging
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.agent.loop import handle_message
from app.agent.preferences import describe
from app.agent.tools import AgentContext
from app.bot.format import md_to_html, parse_locations, parse_salary
from app.db.models import Company, Job, Match, TailoredResumeRecord, User
from app.db.queries import get_active_resume, get_or_create_preferences
from app.llm.client import LLMClient, LLMError
from app.matching.embeddings import Embedder
from app.matching.service import MatchResult, NoResumeError, find_matches
from app.resume.parse import ResumeError
from app.resume.service import save_resume
from app.resume.tailor_service import TailorError, tailor_for_job, tailored_pdf

logger = logging.getLogger(__name__)

Button = tuple[str, str]  # (label, callback_data)


@dataclass
class Out:
    text: str = ""
    buttons: list[list[Button]] = field(default_factory=list)
    document: tuple[str, bytes] | None = None  # (filename, content)
    match_id: int | None = None  # set on job messages, so undelivered matches can be undone


@dataclass
class Deps:
    sessionmaker: async_sessionmaker[AsyncSession]
    engine: AsyncEngine
    llm: LLMClient  # tasks: extraction, rerank, SQL, tailoring
    agent_llm: LLMClient  # the chat agent's own turns
    embedder: Embedder

    @classmethod
    def from_app(cls) -> "Deps":
        from app.db.session import get_engine, get_sessionmaker
        from app.llm.client import get_agent_llm_client, get_llm_client
        from app.matching.embeddings import get_embedder

        return cls(
            get_sessionmaker(),
            get_engine(),
            get_llm_client(),
            get_agent_llm_client(),
            get_embedder(),
        )


HELP = (
    "<b>JobScout</b> finds real jobs that fit you and explains why.\n\n"
    "/start – set up (role, location, remote, salary)\n"
    "/resume – upload your resume (PDF)\n"
    "/jobs – get your best new matches now\n"
    "/prefs – see your saved preferences\n"
    "/help – this message\n\n"
    "Or just chat: <i>“no more fintech”</i>, <i>“remote only”</i>, "
    "<i>“how many Python jobs in Bangalore this week?”</i>, "
    "<i>“tailor my resume for the first one”</i>.\n\n"
    "Every morning at 08:00 IST I send your top 3 new matches."
)


# ---- users & onboarding ------------------------------------------------------------


async def get_or_create_user(deps: Deps, chat_id: int, name: str | None) -> int:
    async with deps.sessionmaker() as session, session.begin():
        user = await session.scalar(select(User).where(User.telegram_chat_id == chat_id))
        if user is None:
            user = User(telegram_chat_id=chat_id, name=name)
            session.add(user)
            await session.flush()
        await get_or_create_preferences(session, user.id)
        return user.id


async def save_onboarding(deps: Deps, user_id: int, step: str, text: str) -> str | None:
    """Store one onboarding answer. Returns an error message if it couldn't be parsed."""
    async with deps.sessionmaker() as session, session.begin():
        prefs = await get_or_create_preferences(session, user_id)
        if step == "role":
            prefs.roles = [r.strip() for r in text.split(",") if r.strip()][:5]
        elif step == "location":
            prefs.locations = parse_locations(text)[:8]
        elif step == "remote":
            prefs.remote_only = text == "remote_only"
        elif step == "salary":
            if text.strip().lower() in ("skip", "no", "none", "-"):
                prefs.min_salary = None
                return None
            parsed = parse_salary(text)
            if parsed is None:
                return (
                    "I didn't get that. Try e.g. <code>40 LPA</code>, <code>$150k</code> or skip."
                )
            prefs.min_salary, prefs.salary_currency = parsed
    return None


async def prefs_text(deps: Deps, user_id: int) -> str:
    async with deps.sessionmaker() as session, session.begin():
        p = describe(await get_or_create_preferences(session, user_id))
    salary = f"{p['min_salary']:,} {p['salary_currency']}" if p["min_salary"] else "any"
    lines = [
        "<b>Your preferences</b>",
        f"Roles: {html.escape(', '.join(p['roles']) or 'any')}",
        f"Locations: {html.escape(', '.join(p['locations']) or 'any')}",
        f"Remote only: {'yes' if p['remote_only'] else 'no'}",
        f"Min salary: {salary}",
        f"Seniority: {p['seniority'] or 'any'}",
    ]
    if p["excluded_companies"]:
        lines.append(f"Excluded companies: {html.escape(', '.join(p['excluded_companies']))}")
    if p["excluded_industries"]:
        lines.append(f"Excluded industries: {html.escape(', '.join(p['excluded_industries']))}")
    if p["notes"]:
        lines.append(f"Notes: {html.escape('; '.join(p['notes']))}")
    lines.append("\nChange anything by just telling me, e.g. “remote only”.")
    return "\n".join(lines)


# ---- resume ---------------------------------------------------------------------------


async def handle_resume(deps: Deps, user_id: int, file_name: str, pdf: bytes) -> list[Out]:
    try:
        _, profile = await save_resume(
            deps.sessionmaker,
            deps.llm,
            deps.embedder,
            user_id=user_id,
            file_name=file_name,
            pdf_bytes=pdf,
        )
    except ResumeError as exc:
        return [Out(f"⚠️ {html.escape(str(exc))}")]
    except LLMError:
        return [Out("⚠️ I couldn't analyse your resume right now. Please try again in a minute.")]
    skills = ", ".join(profile.skills[:10]) or "—"
    years = f"{profile.years_experience:g} yrs" if profile.years_experience is not None else "?"
    return [
        Out(
            f"✅ Got it! <b>{html.escape(profile.current_title or 'Your profile')}</b> · {years}\n"
            f"Skills: {html.escape(skills)}\n\nSend /jobs to see your matches."
        )
    ]


# ---- matches ---------------------------------------------------------------------------


def match_out(r: MatchResult) -> Out:
    c, k = r.candidate, r.ranking
    where = html.escape(c.location or "Location n/a") + (" · remote" if c.remote else "")
    lines = [f"<b>{html.escape(c.title)}</b> — {html.escape(c.company)}", where]
    if k:
        lines.append(f"Fit: <b>{k.fit_score}/100</b>")
        lines += [f"✓ {html.escape(reason)}" for reason in k.reasons]
        lines.append(f"⚠️ {html.escape(k.gap)}")
    else:
        lines.append("<i>(ranked by similarity — AI ranking unavailable right now)</i>")
    lines.append(f'<a href="{html.escape(c.url)}">View job</a>')
    return Out(
        match_id=r.match_id,
        text="\n".join(lines),
        buttons=[
            [
                ("👍", f"fb:{r.match_id}:1"),
                ("👎", f"fb:{r.match_id}:-1"),
                ("📝 Tailor resume", f"tailor:{c.job_id}"),
            ]
        ],
    )


async def jobs(deps: Deps, user_id: int, *, intro: str | None = None) -> list[Out]:
    try:
        results = await find_matches(deps.sessionmaker, deps.llm, deps.embedder, user_id)
    except NoResumeError:
        return [Out("First, send me your resume as a PDF (/resume).")]
    if not results:
        return [
            Out(
                "No new matches right now — I'll keep looking. Try widening your "
                "preferences, e.g. “add Remote” or “any seniority”."
            )
        ]
    head = [Out(intro)] if intro else []
    return head + [match_out(r) for r in results]


# ---- feedback ---------------------------------------------------------------------------

REASONS: list[Button] = [
    ("Not this company", "company"),
    ("Not this industry", "industry"),
    ("Wrong seniority", "seniority"),
    ("Wrong location", "location"),
    ("Other (type it)", "other"),
]


async def feedback(deps: Deps, user_id: int, match_id: int, value: int) -> list[Out]:
    async with deps.sessionmaker() as session, session.begin():
        match = await session.get(Match, match_id)
        if match is None or match.user_id != user_id:
            return [Out("That match no longer exists.")]
        match.feedback = value
    if value > 0:
        return [Out("👍 Noted — I'll look for more like this.")]
    return [
        Out(
            "Thanks! What was off?",
            buttons=[[(label, f"why:{match_id}:{code}")] for label, code in REASONS],
        )
    ]


async def feedback_reason(deps: Deps, user_id: int, match_id: int, code: str) -> list[Out]:
    """Turn a 👎 reason into a lasting preference where it clearly implies one."""
    async with deps.sessionmaker() as session, session.begin():
        match = await session.get(Match, match_id)
        if match is None or match.user_id != user_id:
            return [Out("That match no longer exists.")]
        job = await session.get(Job, match.job_id)
        prefs = await get_or_create_preferences(session, user_id)
        match.feedback_reason = code
        if code == "company":
            if job.company not in prefs.excluded_companies:
                prefs.excluded_companies = [*prefs.excluded_companies, job.company]
            return [Out(f"Got it — no more jobs from <b>{html.escape(job.company)}</b>.")]
        if code == "industry":
            industry = (
                await session.scalar(select(Company.industry).where(Company.id == job.company_id))
                if job.company_id
                else None
            )
            if not industry:
                return [Out("I don't know this company's industry, but I've noted your feedback.")]
            if industry not in prefs.excluded_industries:
                prefs.excluded_industries = [*prefs.excluded_industries, industry]
            return [Out(f"Got it — no more <b>{html.escape(industry)}</b> jobs.")]
        if code in ("seniority", "location"):
            note = f"disliked {code} of '{job.title}' ({job.location or 'n/a'})"
            prefs.notes = [*prefs.notes, note][-20:]
            hint = "“senior roles only”" if code == "seniority" else "“only Bangalore or remote”"
            return [Out(f"Noted. To change it for good, tell me e.g. {hint}.")]
    return [Out("Tell me in a few words what didn't fit, and I'll remember it.")]


async def feedback_text(deps: Deps, user_id: int, match_id: int, text: str) -> list[Out]:
    """Free-text 👎 reason → stored, then handed to the agent to update preferences."""
    async with deps.sessionmaker() as session, session.begin():
        match = await session.get(Match, match_id)
        job = await session.get(Job, match.job_id) if match else None
        if match is None or match.user_id != user_id or job is None:
            return [Out("That match no longer exists.")]
        match.feedback_reason = text[:500]
    prompt = (
        f"I gave 👎 to '{job.title}' at {job.company} because: {text}. If this reflects "
        "a lasting preference, update my preferences. Reply in one short sentence."
    )
    return await chat(deps, user_id, prompt)


# ---- tailoring & chat ---------------------------------------------------------------


async def _pdf_outs(deps: Deps, ids: list[int]) -> list[Out]:
    outs = []
    for tid in ids:
        async with deps.sessionmaker() as session:
            record = await session.get(TailoredResumeRecord, tid)
            job = await session.get(Job, record.job_id)
        pdf = await tailored_pdf(deps.sessionmaker, record)
        safe = "".join(ch if ch.isalnum() else "-" for ch in job.company)[:30]
        outs.append(Out(document=(f"resume-{safe}.pdf", pdf)))
    return outs


async def tailor(deps: Deps, user_id: int, job_id: int) -> list[Out]:
    try:
        record = await tailor_for_job(deps.sessionmaker, deps.llm, user_id, job_id)
    except TailorError as exc:
        return [Out(html.escape(str(exc)))]
    except LLMError:
        return [Out("⚠️ Tailoring failed (AI service busy). Please try again in a minute.")]
    text = "📝 Here's your resume tailored for this job — every line checked against your original."
    if record.dropped:
        removed = "\n".join(f"• {html.escape(d['claim'])}" for d in record.dropped[:5])
        text += f"\n\nRemoved because they weren't in your original resume:\n{removed}"
    return [Out(text), *await _pdf_outs(deps, [record.id])]


async def chat(deps: Deps, user_id: int, text: str) -> list[Out]:
    ctx = AgentContext(
        deps.sessionmaker, deps.engine, deps.agent_llm, deps.embedder, user_id, task_llm=deps.llm
    )
    reply = await handle_message(ctx, text)
    return [Out(md_to_html(reply.text)), *await _pdf_outs(deps, reply.tailored_resume_ids)]


# ---- daily digest ----------------------------------------------------------------------


async def digest_recipients(deps: Deps) -> list[tuple[int, int]]:
    """(user_id, chat_id) for every Telegram user with a resume."""
    async with deps.sessionmaker() as session:
        users = (
            await session.scalars(select(User).where(User.telegram_chat_id.is_not(None)))
        ).all()
        out = []
        for u in users:
            if await get_active_resume(session, u.id):
                out.append((u.id, u.telegram_chat_id))
    return out


async def digest_for(deps: Deps, user_id: int) -> list[Out]:
    outs = await jobs(deps, user_id, intro="☀️ <b>Good morning! Your top new matches:</b>")
    # Don't message people just to say there's nothing new.
    return outs if any(o.buttons for o in outs) else []


async def forget_matches(deps: Deps, outs: list[Out]) -> None:
    """Delivery failed: delete those matches so they count as never shown (and can be
    offered again) — "shown" must mean the user actually received it."""
    ids = [o.match_id for o in outs if o.match_id is not None]
    if ids:
        async with deps.sessionmaker() as session, session.begin():
            await session.execute(delete(Match).where(Match.id.in_(ids)))


def summarize(outs: list[Out]) -> dict[str, Any]:
    """For logs/tests: what would be sent."""
    return {
        "texts": [o.text for o in outs if o.text],
        "documents": [o.document[0] for o in outs if o.document],
    }
