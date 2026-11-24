"""Text-to-SQL guardrails: the validator, plus the DB-level layers it's backed by."""

import importlib

import pytest
from sqlalchemy.exc import DBAPIError

from app.agent import text_to_sql
from app.agent.sql_guard import ALLOWED_COLUMNS, UnsafeSQL, validate_sql
from app.agent.text_to_sql import answer_question, run_readonly
from tests.fakes import make_llm

SAFE = [
    "SELECT count(*) FROM jobs WHERE remote AND title ILIKE '%python%' "
    "AND posted_at >= now() - interval '7 days'",
    "SELECT c.industry, count(*) AS n FROM jobs j JOIN companies c ON c.id = j.company_id "
    "GROUP BY c.industry ORDER BY n DESC",
    "SELECT title FROM jobs WHERE id IN (SELECT is_duplicate_of FROM jobs)",
]

MALICIOUS = {
    "drop": "DROP TABLE jobs",
    "update": "UPDATE jobs SET title = 'x'",
    "multiple statements": "SELECT 1 FROM jobs; DELETE FROM jobs",
    "non-whitelisted table": "SELECT raw_text FROM resumes",
    "non-whitelisted column": "SELECT embedding FROM jobs",
    "select star": "SELECT * FROM jobs",
    "system catalog": "SELECT usename FROM pg_catalog.pg_user",
    "dangerous function": "SELECT pg_read_file('/etc/passwd') FROM jobs",
    "sleep": "SELECT pg_sleep(30) FROM jobs",
    "union to other table": "SELECT title FROM jobs UNION SELECT raw_text FROM resumes",
    "subquery on other table": "SELECT title FROM jobs WHERE id = (SELECT count(*) FROM users)",
    "select into": "SELECT title INTO stolen FROM jobs",
    "cte": "WITH x AS (SELECT raw_text FROM resumes) SELECT title FROM jobs",
    "locking": "SELECT title FROM jobs FOR UPDATE",
}


@pytest.mark.parametrize("sql", SAFE)
def test_safe_queries_pass_with_limit(sql):
    assert validate_sql(sql).endswith("LIMIT 100")


@pytest.mark.parametrize("name", sorted(MALICIOUS))
def test_malicious_queries_are_blocked(name):
    with pytest.raises(UnsafeSQL):
        validate_sql(MALICIOUS[name])


def test_limit_is_capped_but_smaller_limits_kept():
    assert validate_sql("SELECT title FROM jobs LIMIT 5000").endswith("LIMIT 100")
    assert validate_sql("SELECT title FROM jobs LIMIT 5").endswith("LIMIT 5")


def test_migration_grants_match_the_whitelist():
    migration = importlib.import_module("app.db.migrations.versions.0005_readonly_sql_role")
    granted_jobs = {c.strip() for c in migration.JOBS_COLUMNS.split(",")}
    granted_companies = {c.strip() for c in migration.COMPANIES_COLUMNS.split(",")}
    assert granted_jobs == ALLOWED_COLUMNS["jobs"]
    assert granted_companies == ALLOWED_COLUMNS["companies"]


# ---- DB layers (validator bypassed on purpose, to prove the next layer holds) -------


async def test_runs_safe_query(engine, sessionmaker):
    rows = await run_readonly(engine, "SELECT count(*) AS n FROM jobs")
    assert rows == [{"n": 0}]


async def test_readonly_role_blocks_other_tables_even_if_validator_is_bypassed(
    engine, sessionmaker, monkeypatch
):
    monkeypatch.setattr(text_to_sql, "validate_sql", lambda sql: sql)
    with pytest.raises(DBAPIError, match="permission denied"):
        await run_readonly(engine, "SELECT raw_text FROM resumes")
    with pytest.raises(DBAPIError, match="permission denied"):
        await run_readonly(engine, "SELECT embedding FROM jobs")  # column-level grant


async def test_readonly_transaction_blocks_writes(engine, sessionmaker, monkeypatch):
    monkeypatch.setattr(text_to_sql, "validate_sql", lambda sql: sql)
    with pytest.raises(DBAPIError, match="read-only transaction|permission denied"):
        await run_readonly(engine, "DELETE FROM jobs")


async def test_statement_timeout(engine, sessionmaker, monkeypatch):
    monkeypatch.setattr(text_to_sql, "validate_sql", lambda sql: sql)
    monkeypatch.setattr(text_to_sql, "STATEMENT_TIMEOUT", "100ms")
    with pytest.raises(DBAPIError, match="statement timeout"):
        await run_readonly(engine, "SELECT pg_sleep(2)")


async def test_answer_question_lets_the_llm_fix_rejected_sql(engine, sessionmaker):
    llm, provider = make_llm(
        sessionmaker,
        ['{"sql": "SELECT * FROM jobs"}', '{"sql": "SELECT count(*) AS n FROM jobs"}'],
    )
    result = await answer_question(llm, engine, "how many jobs?")
    assert result["rows"] == [{"n": 0}] and result["sql"].endswith("LIMIT 100")
    assert "rejected by guardrails" in provider.calls[1]["messages"][0]["content"]


async def test_answer_question_gives_up_with_error(engine, sessionmaker):
    llm, _ = make_llm(sessionmaker, ['{"sql": "DROP TABLE jobs"}'] * 2)
    result = await answer_question(llm, engine, "drop everything")
    assert "rejected" in result["error"]
