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
