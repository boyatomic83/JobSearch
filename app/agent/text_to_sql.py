"""Answer questions about the job data with LLM-written SQL, run behind guardrails.

Flow: question → LLM writes SQL (schema in the prompt) → validate_sql → execute in a
READ ONLY, 5 s-timeout transaction as the `jobscout_readonly` role → rows. If the SQL is
rejected or fails, the error goes back to the LLM once so it can fix its own query.
"""

import logging
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncEngine

from app.agent.sql_guard import MAX_ROWS, UnsafeSQL, validate_sql
from app.llm.client import LLMClient, LLMError

logger = logging.getLogger(__name__)

STATEMENT_TIMEOUT = "5s"
READONLY_ROLE = "jobscout_readonly"
MAX_CELL_CHARS = 300


class GeneratedSQL(BaseModel):
    """Either SQL, or a refusal (the model may decline e.g. "delete all jobs")."""

    sql: str | None = None
    refusal: str | None = None


SCHEMA_DOC = """PostgreSQL tables you may query (no others exist for you):

jobs(
  id int, source text -- 'greenhouse' | 'lever' | 'remoteok',
  company_id int -> companies.id (NULL for remoteok), company text, title text,
  description text -- plain text, long; use ILIKE for keyword search,
  location text -- free text, e.g. 'Bengaluru, India', 'Remote - USA', 'San Francisco, CA',
  remote bool, salary_min int, salary_max int, salary_currency text -- yearly, ISO code,
  posted_at timestamptz, url text, first_seen_at timestamptz, last_seen_at timestamptz,
  is_active bool -- false = no longer listed,
  times_reposted int, is_duplicate_of int -- NULL unless a duplicate,
  ghost_score float -- 0..1, > 0.7 = likely ghost job
)
companies(id int, name text, slug text, ats text, industry text -- e.g. 'fintech', 'ai')

Conventions:
- Unless the question says otherwise, only count open, real jobs:
  is_active AND is_duplicate_of IS NULL AND ghost_score <= 0.7
- Bangalore is also written 'Bengaluru': match both with ILIKE.
- "Python jobs" means title ILIKE '%python%' OR description ILIKE '%python%'.
- "this week" = posted_at >= now() - interval '7 days'.
- Never SELECT *; name columns. One SELECT statement only, no CTEs."""

SQL_SYSTEM = f"""You translate questions about a job database into one PostgreSQL SELECT.
{SCHEMA_DOC}
Return {{"sql": "..."}}, or {{"refusal": "<why>"}} if the request is not a read-only
question about jobs/companies (e.g. asks to modify data or read other tables).
Today is {{today}}."""


def _jsonable(value: Any) -> Any:
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, str) and len(value) > MAX_CELL_CHARS:
        return value[:MAX_CELL_CHARS] + "…"
    return value


async def _role_available(conn) -> bool:
    # Passing the role's oid (not its name) means this never errors if the role is absent.
    result = await conn.exec_driver_sql(
        f"SELECT count(*) FROM pg_roles r WHERE r.rolname = '{READONLY_ROLE}' "
        "AND pg_has_role(current_user, r.oid, 'MEMBER')"
    )
    return result.scalar_one() > 0


async def run_readonly(engine: AsyncEngine, sql: str) -> list[dict[str, Any]]:
    """Validate and execute. Raises UnsafeSQL or DBAPIError."""
    safe_sql = validate_sql(sql)
    async with engine.connect() as conn, conn.begin():
        await conn.exec_driver_sql("SET TRANSACTION READ ONLY")
        await conn.exec_driver_sql(f"SET LOCAL statement_timeout = '{STATEMENT_TIMEOUT}'")
        if await _role_available(conn):
            await conn.exec_driver_sql(f"SET LOCAL ROLE {READONLY_ROLE}")
        else:
            logger.warning("read-only role unavailable; relying on validator + READ ONLY")
        # exec_driver_sql: no SQLAlchemy ':param' parsing of LLM-written text.
        result = await conn.exec_driver_sql(safe_sql)
        rows = result.mappings().fetchmany(MAX_ROWS)
    return [{k: _jsonable(v) for k, v in row.items()} for row in rows]


async def answer_question(
    llm: LLMClient, engine: AsyncEngine, question: str, *, user_id: int | None = None
) -> dict[str, Any]:
    """Returns {"sql", "rows", "row_count"} or {"error", "sql"} — always JSON-able."""
    system = SQL_SYSTEM.replace("{today}", datetime.now(UTC).date().isoformat())
    prompt = f"Question: {question}"
    sql = ""
    for attempt in range(2):
        try:
            generated = await llm.structured(
                purpose="text_to_sql", system=system, prompt=prompt, schema=GeneratedSQL,
                user_id=user_id, use_cache=attempt == 0,
            )  # fmt: skip
        except LLMError as exc:
            return {"error": f"could not generate SQL: {exc}"[:300], "sql": sql}
        if not generated.sql:
            return {"error": f"refused: {generated.refusal or 'not a read-only question'}"}
        sql = generated.sql
        try:
            rows = await run_readonly(engine, sql)
            return {"sql": validate_sql(sql), "rows": rows, "row_count": len(rows)}
        except UnsafeSQL as exc:
            error = f"rejected by guardrails: {exc}"
        except DBAPIError as exc:
            error = f"database error: {str(exc.orig).splitlines()[0]}"
        logger.info("text_to_sql attempt %d failed: %s", attempt + 1, error)
        prompt = f"Question: {question}\nYour previous SQL:\n{sql}\nfailed: {error}\nFix it."
    return {"error": error, "sql": sql}
