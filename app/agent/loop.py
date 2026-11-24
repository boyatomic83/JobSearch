"""The agent loop, written by hand on top of native tool calling (no framework).

One user turn:
  1. Save the user message; load the last ~20 chat messages as short-term memory.
  2. Build the system prompt with the user's *current* preferences (long-term memory).
  3. Loop: call the LLM with the tools → run any tool calls → feed results back.
     Stop when the model answers in text, or after MAX_TOOL_CALLS tool calls — then
     ask for a final answer with tools disabled.
  4. Save every tool step and the final reply to `messages`. Each LLM call is already
     logged to `llm_calls` by LLMClient.
"""

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select

from app.agent.preferences import describe
from app.agent.tools import TOOLS, AgentContext, result_json, run_tool
from app.db.models import Company, Message
from app.db.queries import get_active_resume, get_or_create_preferences
from app.llm.client import LLMError

logger = logging.getLogger(__name__)

MAX_TOOL_CALLS = 6
HISTORY_MESSAGES = 20
REPLAY_CHARS = 600  # older replies are trimmed when replayed; keeps each turn's tokens low

SYSTEM_TEMPLATE = """You are JobScout, a concise job-search assistant chatting on Telegram.
Today is {today}.

The user's saved preferences (your long-term memory; always current):
{preferences}
Known industries (for exclusions): {industries}
Resume on file: {resume}

Rules:
- When the user states a lasting preference or dislike ("no more fintech", "remote
  only", "nothing below 40 LPA", "not Stripe", "I want ML roles"), you MUST call
  update_preferences before you reply. 1 LPA = 100,000 INR per year.
- Use search_jobs to find jobs. It already applies the saved preferences; use `filters`
  only for one-off variations the user asks for.
- Use run_sql_readonly for counts and statistics about the job data.
- Use explain_match for "why this job?" and tailor_resume for tailoring.
- Only call search_jobs when the user asks for jobs.
- Refer to jobs by the job_id given in tool results or in the "[jobs shown: ...]" notes
  of earlier replies — never by numbers taken from a URL.
- Never invent jobs, numbers, companies or links: report only what tools return.
- Never claim the user has a skill or experience that isn't in candidate_profile /
  their resume. Call it a gap instead.
- If a tool returns an error, say plainly what failed. Never promise to follow up later:
  you can only act when the user writes.
- Keep replies under ~120 words, skimmable. No tables or headings: Telegram can't
  render them. For jobs show title,
  company, location and the link."""


@dataclass
class AgentReply:
    text: str
    steps: list[dict[str, Any]] = field(default_factory=list)

    @property
    def tailored_resume_ids(self) -> list[int]:
        """PDFs the bot should send along with the text reply."""
        return [
            st["result"]["tailored_resume_id"]
            for st in self.steps
            if st["tool"] == "tailor_resume" and "tailored_resume_id" in st["result"]
        ]


async def _system_prompt(ctx: AgentContext) -> str:
    async with ctx.sessionmaker() as session, session.begin():
        prefs = await get_or_create_preferences(session, ctx.user_id)
        resume = await get_active_resume(session, ctx.user_id)
        industries = (
            await session.scalars(
                select(Company.industry).where(Company.industry.is_not(None)).distinct()
            )
        ).all()
        title = (resume.profile or {}).get("current_title") if resume else None
        return SYSTEM_TEMPLATE.format(
            today=datetime.now(UTC).date().isoformat(),
            preferences=result_json(describe(prefs)),
            industries=", ".join(sorted(industries)) or "none",
            resume=f"yes ({title})" if resume else "no — ask them to upload one with /resume",
        )


async def _history(ctx: AgentContext) -> list[dict[str, Any]]:
    """Last N user/assistant messages, oldest first.

    Tool steps are stored (for observability) but not replayed: their results are
    already summarised in the assistant replies, and replaying JSON would burn tokens.
    """
    async with ctx.sessionmaker() as session:
        rows = (
            await session.scalars(
                select(Message)
                .where(Message.user_id == ctx.user_id, Message.role.in_(["user", "assistant"]))
                .order_by(Message.created_at.desc(), Message.id.desc())
                .limit(HISTORY_MESSAGES)
            )
        ).all()
    return [{"role": m.role, "content": _with_job_note(m)} for m in reversed(rows)]


def _with_job_note(m: Message) -> str:
    """Replay which job_ids a past reply showed, so "the first one" resolves correctly.

    The user only saw titles and links; the model needs the ids to call explain_match.
    """
    content = m.content if len(m.content) <= REPLAY_CHARS else m.content[:REPLAY_CHARS] + "…"
    jobs = (m.tool_payload or {}).get("jobs_shown") if m.role == "assistant" else None
    if not jobs:
        return content
    note = "; ".join(f"{i}) job_id {j['job_id']} = {j['title']} @ {j['company']}"
                     for i, j in enumerate(jobs, 1))  # fmt: skip
    return f"{content}\n\n[jobs shown: {note}]"


def _jobs_shown(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"job_id": j["job_id"], "title": j["title"], "company": j["company"]}
        for st in steps
        if st["tool"] == "search_jobs"
        for j in st["result"].get("jobs", [])
    ]


async def _save(ctx: AgentContext, **fields: Any) -> None:
    async with ctx.sessionmaker() as session, session.begin():
        session.add(Message(user_id=ctx.user_id, **fields))


async def handle_message(ctx: AgentContext, text: str) -> AgentReply:
    history = await _history(ctx)
    await _save(ctx, role="user", content=text)
    system = await _system_prompt(ctx)
    specs = [t.spec() for t in TOOLS.values()]
    messages: list[dict[str, Any]] = [*history, {"role": "user", "content": text}]
    reply = AgentReply(text="")

    try:
        while True:
            budget_left = MAX_TOOL_CALLS - len(reply.steps)
            resp = await ctx.llm.chat(
                purpose="agent",
                system=system,
                messages=messages,
                tools=specs if budget_left > 0 else None,
                user_id=ctx.user_id,
            )
            if not resp.tool_calls or budget_left <= 0:
                reply.text = resp.text.strip()
                break
            messages.append(
                {"role": "assistant", "content": resp.text, "tool_calls": resp.tool_calls}
            )
            for call in resp.tool_calls:
                if len(reply.steps) >= MAX_TOOL_CALLS:
                    result = {"error": "tool-call limit reached for this turn; answer now"}
                else:
                    result = await run_tool(ctx, call)
                step = {"tool": call.name, "arguments": call.arguments, "result": result}
                reply.steps.append(step)
                logger.info("agent user=%s step %d: %s", ctx.user_id, len(reply.steps), call.name)
                await _save(ctx, role="tool", tool_name=call.name, tool_payload=step,
                            content="")  # fmt: skip
                messages.append({
                    "role": "tool", "tool_call_id": call.id, "name": call.name,
                    "content": result_json(result),
                })  # fmt: skip
    except LLMError:
        logger.exception("agent LLM failure")
        reply.text = "Sorry, I can't reach the AI service right now. Please try again in a minute."

    if not reply.text:
        reply.text = "Sorry, I couldn't come up with an answer. Could you rephrase?"
    jobs = _jobs_shown(reply.steps)
    await _save(ctx, role="assistant", content=reply.text,
                tool_payload={"jobs_shown": jobs} if jobs else None)  # fmt: skip
    return reply
