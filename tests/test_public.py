"""Landing page + public stats: works without auth and never exposes user data."""

from httpx import ASGITransport, AsyncClient

from app.api import public
from app.db.models import Job, User
from app.db.session import get_sessionmaker
from app.main import app


async def _get(path: str):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        return await c.get(path)


async def test_landing_page_is_served():
    resp = await _get("/")
    assert resp.status_code == 200 and resp.headers["content-type"].startswith("text/html")
    assert "JobScout" in resp.text and "https://t.me/Cosmo122bot" in resp.text


async def test_public_stats_are_aggregate_only_and_cached(sessionmaker):
    async with sessionmaker() as s, s.begin():
        s.add(User(name="secret person", telegram_chat_id=123))
        s.add_all(
            [
                Job(
                    source="greenhouse", source_id="1", company="A", title="t", url="x", remote=True
                ),
                Job(
                    source="lever", source_id="2", company="B", title="t", url="x", ghost_score=0.9
                ),
            ]
        )
    public._cache.update(at=0.0, data=None)
    app.dependency_overrides[get_sessionmaker] = lambda: sessionmaker
    try:
        first = (await _get("/public/stats")).json()
        async with sessionmaker() as s, s.begin():  # a change within the cache window…
            s.add(Job(source="lever", source_id="3", company="C", title="t", url="x"))
        second = (await _get("/public/stats")).json()
    finally:
        app.dependency_overrides.clear()
        public._cache.update(at=0.0, data=None)
    assert first["open_jobs"] == 1 and first["companies"] == 2 and first["remote_jobs"] == 1
    assert first["ghost_jobs_hidden"] == 1 and first["sources"] == {"greenhouse": 1, "lever": 1}
    assert second == first  # …is not visible: served from the 10-minute cache
    text = str(first).lower()
    assert "user" not in text and "secret" not in text and "123" not in text
