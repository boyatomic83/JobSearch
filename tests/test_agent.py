"""Agent loop with a scripted fake LLM: memory, tool execution, limits, logging."""

import json

from sqlalchemy import select

from app.agent.loop import MAX_TOOL_CALLS, handle_message
from app.agent.preferences import PreferenceChanges, apply_changes
from app.agent.tools import AgentContext
from app.db.models import LLMCall, Message, Preference, Resume, User
from tests.fakes import SAMPLE_PROFILE, FakeEmbedder, make_llm, tool_call


def test_apply_changes_merges_lists_case_insensitively():
    prefs = Preference(roles=["Backend Engineer"], locations=[], excluded_industries=["Crypto"],
                       excluded_companies=[], notes=[], remote_only=False,
                       salary_currency="USD")  # fmt: skip
    changed = apply_changes(
        prefs,
        PreferenceChanges(
            exclude_industries_add=["fintech", "crypto"],
            roles_add=["ML Engineer"], roles_remove=["backend engineer"],
            remote_only=True, min_salary=4_000_000, salary_currency="inr",
            note="no on-call",
        ),
    )  # fmt: skip
    assert prefs.excluded_industries == ["Crypto", "fintech"]
    assert prefs.roles == ["ML Engineer"]
    assert (prefs.remote_only, prefs.min_salary, prefs.salary_currency) == (True, 4_000_000, "INR")
    assert prefs.notes == ["no on-call"]
    assert any("remote_only" in c for c in changed)
    assert apply_changes(prefs, PreferenceChanges(remote_only=True)) == []  # no-op


async def _ctx(sessionmaker, engine, replies, *, with_resume=False):
    async with sessionmaker() as s, s.begin():
        user = User(name="agent test")
        s.add(user)
        await s.flush()
        if with_resume:
            s.add(Resume(user_id=user.id, raw_text="x", profile=SAMPLE_PROFILE))
    llm, provider = make_llm(sessionmaker, replies)
    return AgentContext(sessionmaker, engine, llm, FakeEmbedder(), user.id), provider


async def _messages(sessionmaker, user_id):
    async with sessionmaker() as s:
        rows = await s.scalars(
            select(Message).where(Message.user_id == user_id).order_by(Message.id)
        )
        return list(rows)


async def test_preference_statement_triggers_update_and_is_remembered(sessionmaker, engine):
    ctx, provider = await _ctx(
        sessionmaker,
        engine,
        [
            tool_call("update_preferences", exclude_industries_add=["fintech"], remote_only=True),
            "Got it — no more fintech, remote only.",
            "You excluded fintech and want remote roles.",  # second turn
        ],
    )
    reply = await handle_message(ctx, "no more fintech please, and remote only")
    assert reply.text == "Got it — no more fintech, remote only."
    assert reply.steps[0]["tool"] == "update_preferences"

    async with sessionmaker() as s:
        prefs = await s.scalar(select(Preference).where(Preference.user_id == ctx.user_id))
    assert prefs.excluded_industries == ["fintech"] and prefs.remote_only is True

    # The tool result went back to the model, and tools were offered.
    second_call = provider.calls[1]
    tool_msg = second_call["messages"][-1]
    assert tool_msg["role"] == "tool" and "fintech" in tool_msg["content"]
    assert {t.name for t in provider.calls[0]["tools"]} >= {"search_jobs", "run_sql_readonly"}

    # Next turn: saved preferences are in the system prompt, and chat history is replayed.
    await handle_message(ctx, "what did I ask for?")
    third = provider.calls[2]
    assert '"excluded_industries": ["fintech"]' in third["system"]
    assert [m["role"] for m in third["messages"]] == ["user", "assistant", "user"]

    roles = [m.role for m in await _messages(sessionmaker, ctx.user_id)]
    assert roles == ["user", "tool", "assistant", "user", "assistant"]


async def test_step_limit_forces_final_answer_without_tools(sessionmaker, engine):
    looping = [tool_call("get_preferences", call_id=f"c{i}") for i in range(MAX_TOOL_CALLS)]
    ctx, provider = await _ctx(sessionmaker, engine, [*looping, "Here is what I found."])
    reply = await handle_message(ctx, "loop forever")
    assert len(reply.steps) == MAX_TOOL_CALLS
    assert reply.text == "Here is what I found."
    assert provider.calls[-1]["tools"] is None  # final call can't call tools
    async with sessionmaker() as s:
        logged = (
            await s.scalars(select(LLMCall.purpose).where(LLMCall.user_id == ctx.user_id))
        ).all()
    assert logged == ["agent"] * (MAX_TOOL_CALLS + 1)  # every LLM call is logged


async def test_bad_tool_calls_are_reported_back_not_raised(sessionmaker, engine):
    ctx, provider = await _ctx(
        sessionmaker,
        engine,
        [
            tool_call("explain_match", job_id="not-a-number"),
            tool_call("does_not_exist", call_id="c2"),
            "Sorry, I couldn't find that job.",
        ],
    )
    reply = await handle_message(ctx, "explain job abc")
    assert "invalid arguments" in reply.steps[0]["result"]["error"]
    assert "unknown tool" in reply.steps[1]["result"]["error"]
    assert reply.text.startswith("Sorry")


async def test_search_without_resume_tells_model_to_ask_for_one(sessionmaker, engine):
    ctx, _ = await _ctx(sessionmaker, engine, [tool_call("search_jobs"), "Please upload."])
    reply = await handle_message(ctx, "find me jobs")
    assert "/resume" in reply.steps[0]["result"]["error"]


async def test_search_filters_are_one_off_and_not_saved(sessionmaker, engine):
    ctx, _ = await _ctx(
        sessionmaker,
        engine,
        [tool_call("search_jobs", filters={"remote_only": True}), "No jobs right now."],
        with_resume=True,
    )
    reply = await handle_message(ctx, "any remote jobs today?")
    assert reply.steps[0]["result"]["jobs"] == []
    async with sessionmaker() as s:
        prefs = await s.scalar(select(Preference).where(Preference.user_id == ctx.user_id))
    assert prefs.remote_only is False


async def test_llm_outage_gives_friendly_reply(sessionmaker, engine):
    ctx, _ = await _ctx(sessionmaker, engine, [RuntimeError("down")])
    reply = await handle_message(ctx, "hello")
    assert "try again" in reply.text


async def test_chat_endpoint(sessionmaker, engine):
    from httpx import ASGITransport, AsyncClient

    from app.db.session import get_sessionmaker
    from app.llm.client import get_llm_client
    from app.main import app
    from app.matching.embeddings import get_embedder

    ctx, provider = await _ctx(sessionmaker, engine, ["Hi! Upload your resume with /resume."])
    llm = ctx.llm
    from app.llm.client import get_agent_llm_client

    app.dependency_overrides.update({
        get_sessionmaker: lambda: sessionmaker, get_llm_client: lambda: llm,
        get_agent_llm_client: lambda: llm, get_embedder: lambda: FakeEmbedder(),
    })  # fmt: skip
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            resp = await c.post(
                f"/users/{ctx.user_id}/chat", json={"message": "hi"},
                headers={"X-Admin-Key": "test-admin-key"},
            )  # fmt: skip
    finally:
        app.dependency_overrides.clear()
    assert resp.status_code == 200
    assert json.loads(resp.text)["reply"].startswith("Hi!")


async def test_job_ids_shown_are_replayed_in_history(sessionmaker, engine):
    from app.db.models import Job

    async with sessionmaker() as s, s.begin():
        job = Job(source="t", source_id="1", company="GitLab", title="Staff Backend Engineer",
                  url="https://x/jobs/8775136002", embedding_text="x")  # fmt: skip
        s.add(job)
    ctx, provider = await _ctx(sessionmaker, engine, ["Here you go.", "It fits because…"])
    # Simulate a previous turn whose search showed this job.
    from app.agent.loop import _save

    await _save(ctx, role="assistant", content="1. Staff Backend Engineer — GitLab",
                tool_payload={"jobs_shown": [{"job_id": job.id, "title": job.title,
                                              "company": "GitLab"}]})  # fmt: skip
    await handle_message(ctx, "why is the first one a good fit?")
    replayed = provider.calls[0]["messages"][0]["content"]
    assert f"[jobs shown: 1) job_id {job.id} = Staff Backend Engineer @ GitLab]" in replayed


async def test_out_of_range_job_id_is_a_clean_tool_error(sessionmaker, engine):
    ctx, _ = await _ctx(sessionmaker, engine, [tool_call("explain_match", job_id=8775136002),
                                               "That id looks wrong."])  # fmt: skip
    reply = await handle_message(ctx, "explain")
    assert "invalid arguments" in reply.steps[0]["result"]["error"]


def test_tool_specs_have_no_pydantic_titles():
    import json as _json

    from app.agent.tools import TOOLS

    for tool in TOOLS.values():
        assert '"title":' not in _json.dumps(tool.spec().parameters)


async def test_explain_match_includes_the_real_profile(sessionmaker, engine):
    from app.db.models import Job

    async with sessionmaker() as s, s.begin():
        job = Job(source="t", source_id="1", company="Okta", title="SRE", url="https://x",
                  embedding_text="SRE\nOkta\n\nTerraform and Kubernetes.")  # fmt: skip
        s.add(job)
    ctx, _ = await _ctx(sessionmaker, engine, [tool_call("explain_match", job_id=job.id), "ok"],
                        with_resume=True)  # fmt: skip
    reply = await handle_message(ctx, "why this job?")
    result = reply.steps[0]["result"]
    assert result["candidate_profile"]["skills"][0] == "Python"
    assert "Terraform" in result["job"]["description"]
