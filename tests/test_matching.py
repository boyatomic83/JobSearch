"""End-to-end matching (retrieval + rerank + storage) and the user HTTP API.

LLM and embeddings are fakes; jobs and resumes are synthetic.
"""

import json

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from app.db.models import Job, Match, Resume, User
from app.db.session import get_sessionmaker
from app.llm.client import get_llm_client
from app.main import app
from app.matching.embeddings import get_embedder
from app.matching.service import NoResumeError, find_matches
from tests.fakes import SAMPLE_PROFILE, SAMPLE_RESUME, FakeEmbedder, make_llm, make_pdf

EMB = FakeEmbedder()


async def seed(sessionmaker, n_jobs=5) -> tuple[int, list[int]]:
    async with sessionmaker() as s, s.begin():
        user = User(name="u")
        s.add(user)
        await s.flush()
        s.add(
            Resume(
                user_id=user.id, raw_text=SAMPLE_RESUME, profile=SAMPLE_PROFILE,
                embedding=EMB.embed_query("x"),
            )
        )  # fmt: skip
        jobs = [
            Job(
                source="test", source_id=str(i), company="Acme", title=f"Python Engineer {i}",
                url=f"https://x/{i}", embedding_text=f"Python Engineer {i}\nAcme\n\npython {i}",
                embedding=EMB.embed_query(f"python engineer {i}"),
            )
            for i in range(n_jobs)
        ]  # fmt: skip
        s.add_all(jobs)
        await s.flush()
        return user.id, [j.id for j in jobs]


def rerank_reply(job_ids: list[int]) -> str:
    return json.dumps(
        {
            "matches": [
                {"job_id": jid, "fit_score": 90 - i, "reasons": [f"reason {jid}", "Python"],
                 "gap": "no Go"}
                for i, jid in enumerate(job_ids)
            ]
        }
    )  # fmt: skip


async def test_find_matches_reranks_stores_and_drops_hallucinated_ids(sessionmaker):
    user_id, job_ids = await seed(sessionmaker)
    llm, provider = make_llm(sessionmaker, [rerank_reply([job_ids[3], 999_999, job_ids[1]])])
    results = await find_matches(sessionmaker, llm, EMB, user_id)

    assert len(provider.calls) == 1  # exactly one LLM call per run
    prompt = provider.calls[0]["messages"][0]["content"]
    assert "Senior Backend Engineer" in prompt and f"job_id: {job_ids[0]}" in prompt
    assert [r.candidate.job_id for r in results][:2] == [job_ids[3], job_ids[1]]
    assert results[0].ranking.fit_score == 90 and results[0].ranking.reasons[1] == "Python"
    assert results[2].ranking is None  # 999_999 was invented → slot filled from retrieval

    async with sessionmaker() as s:
        stored = (await s.scalars(select(Match).order_by(Match.id))).all()
    assert [m.job_id for m in stored] == [r.candidate.job_id for r in results]
    assert stored[0].fit_score == 90 and stored[0].gap == "no Go"
    assert all(m.shown_at is not None for m in stored)


async def test_next_run_never_repeats_shown_jobs(sessionmaker):
    user_id, job_ids = await seed(sessionmaker, n_jobs=5)
    llm, _ = make_llm(sessionmaker, [rerank_reply(job_ids[:3]), rerank_reply(job_ids[3:])])
    first = await find_matches(sessionmaker, llm, EMB, user_id)
    second = await find_matches(sessionmaker, llm, EMB, user_id)
    assert {r.candidate.job_id for r in first}.isdisjoint(r.candidate.job_id for r in second)
    assert len(second) == 2  # only 2 jobs left


async def test_llm_failure_falls_back_to_retrieval_order(sessionmaker):
    user_id, _ = await seed(sessionmaker)
    llm, _ = make_llm(sessionmaker, [RuntimeError("provider down")])
    results = await find_matches(sessionmaker, llm, EMB, user_id)
    assert len(results) == 3 and all(r.ranking is None for r in results)


async def test_requires_resume(sessionmaker):
    async with sessionmaker() as s, s.begin():
        user = User(name="no resume")
        s.add(user)
    llm, _ = make_llm(sessionmaker)
    with pytest.raises(NoResumeError):
        await find_matches(sessionmaker, llm, EMB, user.id)


# ---- HTTP API ---------------------------------------------------------------------

ADMIN = {"X-Admin-Key": "test-admin-key"}


@pytest.fixture
def api(sessionmaker):
    llm, provider = make_llm(sessionmaker)
    app.dependency_overrides[get_sessionmaker] = lambda: sessionmaker
    app.dependency_overrides[get_llm_client] = lambda: llm
    app.dependency_overrides[get_embedder] = lambda: EMB
    yield provider
    app.dependency_overrides.clear()


async def _client():
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://t", headers=ADMIN)


async def test_user_api_flow(api, sessionmaker):
    async with await _client() as c:
        user = (await c.post("/users", json={"name": "Priya"})).json()
        prefs = await c.put(
            f"/users/{user['id']}/preferences",
            json={"roles": ["Backend Engineer"], "remote_only": False, "salary_currency": "inr"},
        )
        assert prefs.status_code == 200
        assert prefs.json()["roles"] == ["Backend Engineer"]
        assert prefs.json()["salary_currency"] == "INR"

        assert (await c.post(f"/users/{user['id']}/matches")).status_code == 409  # no resume

        api.replies.append(json.dumps(SAMPLE_PROFILE))
        files = {"file": ("cv.pdf", make_pdf(SAMPLE_RESUME), "application/pdf")}
        resp = await c.post(f"/users/{user['id']}/resume", files=files)
        assert resp.status_code == 200 and resp.json()["seniority"] == "senior"

        bad = {"file": ("cv.pdf", b"not a pdf", "application/pdf")}
        assert (await c.post(f"/users/{user['id']}/resume", files=bad)).status_code == 422

        _, job_ids = await seed(sessionmaker, n_jobs=3)
        api.replies.append(rerank_reply(job_ids))
        matches = (await c.post(f"/users/{user['id']}/matches")).json()
        assert [m["job_id"] for m in matches] == job_ids
        assert matches[0]["fit_score"] == 90 and matches[0]["url"].startswith("https://x/")


async def test_user_api_requires_admin_key(api):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        assert (await c.post("/users", json={})).status_code == 401
        assert (await c.post("/users/1/matches", headers=ADMIN)).status_code == 404
