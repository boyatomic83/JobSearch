"""Hybrid retrieval filters against real Postgres + pgvector (synthetic jobs)."""

from sqlalchemy import update

from app.db.models import Company, Job, Match, Preference, User
from app.matching.retrieval import retrieve_candidates
from tests.fakes import FakeEmbedder

EMB = FakeEmbedder()
QUERY = "python backend engineer postgres fastapi"


def job(sid, **kw) -> Job:
    title = kw.pop("title", "Python Backend Engineer")
    text = kw.pop("text", f"{title} python backend postgres fastapi apis")
    return Job(
        source="test", source_id=sid, company=kw.pop("company", "Acme"), title=title,
        url=f"https://x/{sid}", embedding_text=text,
        embedding=kw.pop("embedding", EMB.embed_query(text)),
        location=kw.pop("location", "Bangalore, India"), **kw,
    )  # fmt: skip


async def setup(sessionmaker, jobs, **pref_fields):
    async with sessionmaker() as s, s.begin():
        user = User(name="u")
        s.add(user)
        await s.flush()
        s.add_all(jobs)
        prefs = Preference(user_id=user.id, **pref_fields)
        s.add(prefs)
    return user.id, prefs


async def ids(sessionmaker, user_id, prefs) -> list[str]:
    async with sessionmaker() as s, s.begin():
        cands = await retrieve_candidates(s, EMB.embed_query(QUERY), prefs, user_id)
    by_id = {}
    async with sessionmaker() as s:
        for c in cands:
            by_id[c.job_id] = (await s.get(Job, c.job_id)).source_id
    return [by_id[c.job_id] for c in cands]


async def test_orders_by_similarity_and_excludes_hidden_jobs(sessionmaker):
    user_id, prefs = await setup(
        sessionmaker,
        [
            job("best"),
            job("weaker", title="Java Engineer", text="java spring backend"),
            job("dup", is_duplicate_of=None),
            job("ghost", ghost_score=0.9),
            job("closed", is_active=False),
            job("unembedded", embedding=None),
        ],
    )
    async with sessionmaker() as s, s.begin():  # make "dup" a duplicate of "best"
        await s.execute(
            update(Job).where(Job.source_id == "dup").values(is_duplicate_of=select_id("best"))
        )
    assert await ids(sessionmaker, user_id, prefs) == ["best", "weaker"]


def select_id(source_id):
    from sqlalchemy import select

    return select(Job.id).where(Job.source_id == source_id).scalar_subquery()


async def test_excludes_jobs_already_shown(sessionmaker):
    user_id, prefs = await setup(sessionmaker, [job("a"), job("b")])
    async with sessionmaker() as s, s.begin():
        s.add(Match(user_id=user_id, job_id=select_id("a")))
    assert await ids(sessionmaker, user_id, prefs) == ["b"]


async def test_remote_only_and_locations(sessionmaker):
    jobs = [
        job("blr"),
        job("london", location="London, UK"),
        job("remote-in", location="Remote - India", remote=True),
        job("remote-us", location="Remote - USA", remote=True),  # region-locked
        job("anywhere", location="Remote", remote=True),
        job("worldwide", location="Remote (Worldwide)", remote=True),
        job("remote-us-paren", location="Remote (U.S.)", remote=True),  # region-locked
    ]
    user_id, prefs = await setup(sessionmaker, jobs, locations=["bangalore", "India"])
    expected = {"blr", "remote-in", "anywhere", "worldwide"}
    assert set(await ids(sessionmaker, user_id, prefs)) == expected
    prefs.remote_only = True
    assert set(await ids(sessionmaker, user_id, prefs)) == {"remote-in", "anywhere", "worldwide"}
    prefs.locations = []  # no location preference: any remote job
    assert len(await ids(sessionmaker, user_id, prefs)) == 5


async def test_min_salary_only_compares_same_currency_and_keeps_unknown(sessionmaker):
    jobs = [
        job("low-usd", salary_min=50_000, salary_max=60_000, salary_currency="USD"),
        job("high-usd", salary_min=150_000, salary_max=200_000, salary_currency="USD"),
        job("inr", salary_min=2_000_000, salary_max=3_000_000, salary_currency="INR"),
        job("unknown"),
    ]
    user_id, prefs = await setup(sessionmaker, jobs, min_salary=100_000, salary_currency="USD")
    assert set(await ids(sessionmaker, user_id, prefs)) == {"high-usd", "inr", "unknown"}


async def test_excluded_companies_and_industries(sessionmaker):
    async with sessionmaker() as s, s.begin():
        s.add(Company(id=100, name="PayCo", slug="payco", ats="test", industry="fintech"))
    jobs = [job("pay", company="PayCo", company_id=100), job("globex", company="Globex"),
            job("acme")]  # fmt: skip
    user_id, prefs = await setup(
        sessionmaker, jobs, excluded_companies=["GLOBEX"], excluded_industries=["Fintech"]
    )
    assert await ids(sessionmaker, user_id, prefs) == ["acme"]


async def test_seniority_and_intern_filters(sessionmaker):
    jobs = [
        job("senior", title="Senior Python Engineer"),
        job("junior", title="Junior Python Engineer"),
        job("intern", title="Python Engineering Intern"),
        job("plain", title="Python Engineer"),
    ]
    user_id, prefs = await setup(sessionmaker, jobs, seniority="senior")
    assert set(await ids(sessionmaker, user_id, prefs)) == {"senior", "plain"}
    prefs.seniority = "junior"
    assert set(await ids(sessionmaker, user_id, prefs)) == {"junior", "plain"}
    prefs.seniority = "intern"
    assert await ids(sessionmaker, user_id, prefs) == ["intern"]


async def test_like_wildcards_in_preferences_are_escaped(sessionmaker):
    user_id, prefs = await setup(sessionmaker, [job("blr")], locations=["%"])
    assert await ids(sessionmaker, user_id, prefs) == []
