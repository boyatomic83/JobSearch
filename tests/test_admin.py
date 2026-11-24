import pytest
from httpx import ASGITransport, AsyncClient

from app.api import admin
from app.config import get_settings
from app.ingest.service import IngestReport
from app.main import app


@pytest.fixture
def fake_ingest(monkeypatch):
    calls = []

    async def fake_run_ingest(sessionmaker, connectors):
        calls.append([c.label for c in connectors])
        from datetime import UTC, datetime

        return IngestReport(started_at=datetime(2026, 11, 24, tzinfo=UTC))

    async def fake_run_processing(sessionmaker, embedder):
        from app.matching.process import ProcessReport

        calls.append(["processing"])
        return ProcessReport()

    monkeypatch.setattr(admin, "run_ingest", fake_run_ingest)
    monkeypatch.setattr(admin, "run_processing", fake_run_processing)
    monkeypatch.setattr(admin, "get_embedder", lambda: None)
    return calls


async def _post(path: str, headers: dict | None = None):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        return await c.post(path, headers=headers or {})


async def test_ingest_requires_admin_key(fake_ingest):
    assert (await _post("/admin/ingest")).status_code == 401
    assert (await _post("/admin/ingest", {"X-Admin-Key": "wrong"})).status_code == 401
    assert fake_ingest == []


async def test_ingest_runs_with_valid_key(fake_ingest):
    resp = await _post("/admin/ingest?source=remoteok", {"X-Admin-Key": "test-admin-key"})
    assert resp.status_code == 200
    assert resp.json()["totals"]["failed_feeds"] == 0
    assert fake_ingest == [["remoteok"], ["processing"]]
    assert resp.json()["processing"]["embedded"] == 0


async def test_ingest_can_skip_processing(fake_ingest):
    resp = await _post(
        "/admin/ingest?source=lever&process=false", {"X-Admin-Key": "test-admin-key"}
    )
    assert resp.status_code == 200 and "processing" not in resp.json()
    assert len(fake_ingest) == 1  # ingest only, no processing call
    assert all(label.startswith("lever:") for label in fake_ingest[0])


async def test_ingest_rejects_unknown_source(fake_ingest):
    resp = await _post("/admin/ingest?source=monster", {"X-Admin-Key": "test-admin-key"})
    assert resp.status_code == 422


async def test_admin_disabled_when_no_key_configured(fake_ingest, monkeypatch):
    monkeypatch.setattr(get_settings(), "admin_api_key", "")
    resp = await _post("/admin/ingest", {"X-Admin-Key": ""})
    assert resp.status_code == 503


async def test_stats_reports_jobs_users_and_llm_cost(sessionmaker):
    from decimal import Decimal

    from app.db.models import Job, LLMCall, Match, User
    from app.db.session import get_sessionmaker

    async with sessionmaker() as s, s.begin():
        u = User(name="u", telegram_chat_id=9)
        s.add(u)
        await s.flush()
        s.add_all(
            [
                Job(source="greenhouse", source_id="1", company="A", title="t", url="x"),
                Job(
                    source="greenhouse",
                    source_id="2",
                    company="A",
                    title="t",
                    url="x",
                    ghost_score=0.9,
                ),
                Job(
                    source="lever", source_id="3", company="B", title="t", url="x", is_active=False
                ),
            ]
        )
        await s.flush()
        s.add(Match(user_id=u.id, job_id=(await s.get(Job, 1)).id, feedback=1))
        for cost in ("0.0010", "0.0030"):
            s.add(
                LLMCall(
                    user_id=u.id,
                    provider="groq",
                    model="m",
                    purpose="rerank",
                    prompt_tokens=1000,
                    completion_tokens=100,
                    cost_usd=Decimal(cost),
                )
            )
    app.dependency_overrides[get_sessionmaker] = lambda: sessionmaker
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            assert (await c.get("/admin/stats")).status_code == 401
            resp = await c.get("/admin/stats", headers={"X-Admin-Key": "test-admin-key"})
    finally:
        app.dependency_overrides.clear()
    data = resp.json()
    assert data["jobs"]["total"] == 3 and data["jobs"]["active"] == 2
    assert data["jobs"]["ghost_jobs"] == 1
    assert {r["source"]: r["active"] for r in data["by_source"]} == {"greenhouse": 2}
    assert data["users"] == {"total": 1, "telegram": 1, "with_resume": 0}
    assert data["matches"]["thumbs_up"] == 1
    [day] = data["per_user_per_day"]
    assert day["calls"] == 2 and day["tokens"] == 2200
    assert data["avg_est_cost_per_user_per_day_usd"] == pytest.approx(0.004)
