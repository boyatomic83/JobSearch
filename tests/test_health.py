from httpx import ASGITransport, AsyncClient

from app.db.session import get_session
from app.main import app


class _BrokenSession:
    async def execute(self, *_args, **_kwargs):
        raise ConnectionError("db is down")


async def _client() -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def test_health_ok(migrated_db):
    async with await _client() as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "database": "up"}


async def test_health_reports_db_down():
    async def broken():
        yield _BrokenSession()

    app.dependency_overrides[get_session] = broken
    try:
        async with await _client() as client:
            resp = await client.get("/health")
    finally:
        app.dependency_overrides.clear()
    assert resp.status_code == 503
    assert resp.json()["database"] == "down"
