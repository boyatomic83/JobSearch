"""Embedding, dedup and ghost scoring against a real Postgres + pgvector.

All postings are synthetic test data; vectors come from FakeEmbedder.
"""

from datetime import UTC, datetime, timedelta

from sqlalchemy import select

from app.db.models import Job
from app.ingest.base import JobPosting
from app.ingest.store import upsert_jobs
from app.matching.process import (
    embed_pending_jobs,
    mark_duplicates,
    run_processing,
    update_ghost_scores,
)
from tests.fakes import FakeEmbedder

T0 = datetime(2026, 11, 1, tzinfo=UTC)
LONG = (
    "Responsibilities: build and operate Python services on Postgres, design APIs, "
    "own reliability and on-call, mentor engineers, write design docs. "
) * 15


def posting(sid, *, company="Acme", title="Backend Engineer", location="Bangalore",
            description=LONG, days_old=1) -> JobPosting:  # fmt: skip
    return JobPosting(
        source="greenhouse", source_id=sid, company=company, title=title,
        description=description, location=location, url=f"https://x/{sid}",
        posted_at=T0 - timedelta(days=days_old), salary_min=100_000, salary_max=150_000,
        salary_currency="USD",
    )  # fmt: skip


async def _jobs(sessionmaker) -> dict[str, Job]:
    async with sessionmaker() as s:
        return {j.source_id: j for j in (await s.scalars(select(Job))).all()}


async def _ingest(sessionmaker, postings, seen_at=T0):
    async with sessionmaker() as s, s.begin():
        await upsert_jobs(s, postings, seen_at=seen_at)


async def test_embeds_only_new_or_changed_jobs(sessionmaker):
    embedder = FakeEmbedder()
    await _ingest(sessionmaker, [posting("1"), posting("2")])
    assert await embed_pending_jobs(sessionmaker, embedder) == 2
    assert await embed_pending_jobs(sessionmaker, embedder) == 0  # nothing pending

    # Same text again → keeps its vector. Changed description → re-embedded.
    t1 = T0 + timedelta(days=1)
    await _ingest(sessionmaker, [posting("1"), posting("2", description="Now about Go.")], t1)
    jobs = await _jobs(sessionmaker)
    assert jobs["1"].embedding is not None and jobs["2"].embedding is None
    assert await embed_pending_jobs(sessionmaker, embedder) == 1
    assert "Now about Go." in embedder.calls[-1][0]
    assert embedder.calls[-1][0].startswith("Backend Engineer\nAcme")  # title + company first


async def test_dedup_marks_newer_copy_of_same_company_and_location(sessionmaker):
    await _ingest(
        sessionmaker,
        [
            posting("orig", days_old=10),
            posting("copy", days_old=2),  # same text, newer → duplicate of "orig"
            posting("copy2", days_old=1),  # also points at the oldest, not at "copy"
            posting("other-city", location="London", days_old=1),  # separate listing
            posting("other-co", company="Globex", days_old=1),  # different company
            posting("staff", title="Staff Backend Engineer", days_old=1),  # same text, new level
            posting("punct", title="backend engineer!", days_old=1),  # same title, normalised
            posting("diff-role", title="Chef", description="Cook pasta. " * 50),
        ],
    )
    await embed_pending_jobs(sessionmaker, FakeEmbedder())
    assert await mark_duplicates(sessionmaker) == 3

    jobs = await _jobs(sessionmaker)
    orig = jobs["orig"].id
    assert jobs["copy"].is_duplicate_of == orig
    assert jobs["copy2"].is_duplicate_of == orig
    assert jobs["punct"].is_duplicate_of == orig
    for sid in ("orig", "other-city", "other-co", "diff-role", "staff"):
        assert jobs[sid].is_duplicate_of is None

    # Idempotent: running again gives the same result.
    assert await mark_duplicates(sessionmaker) == 3


async def test_ingest_keeps_company_boilerplate_out_of_embedding_text(sessionmaker):
    intro = "Acme builds rockets for everyone and is growing fast across the globe."
    postings = [
        posting(str(i), title=f"Role {i}", description=f"{intro}\nYou will build thing {i}.")
        for i in range(5)
    ]
    await _ingest(sessionmaker, postings)
    jobs = await _jobs(sessionmaker)
    assert jobs["0"].embedding_text == "Role 0\nAcme\n\nYou will build thing 0."
    assert intro in jobs["0"].description  # the stored description is untouched


async def test_ghost_scores_are_stored_with_reasons(sessionmaker):
    await _ingest(
        sessionmaker,
        [
            posting("fresh"),
            posting(
                "ghost", title="General Application", description="Send us your CV.", days_old=150
            ),  # fmt: skip
        ],
    )
    scored, ghosts = await update_ghost_scores(sessionmaker, now=T0)
    assert (scored, ghosts) == (2, 1)
    jobs = await _jobs(sessionmaker)
    assert jobs["fresh"].ghost_score == 0.0 and jobs["fresh"].ghost_reasons == []
    assert jobs["ghost"].ghost_score == 1.0
    assert any("evergreen" in r for r in jobs["ghost"].ghost_reasons)


async def test_run_processing_end_to_end(sessionmaker):
    await _ingest(sessionmaker, [posting("a", days_old=5), posting("b")])
    report = await run_processing(sessionmaker, FakeEmbedder())
    assert report.embedded == 2 and report.duplicates == 1 and report.scored == 2
