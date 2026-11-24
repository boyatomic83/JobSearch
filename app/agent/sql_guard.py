"""Text-to-SQL guardrails: decide whether LLM-written SQL is safe to run.

Defense in depth — any one layer stopping an attack is enough:
1. This validator (sqlglot parse tree, not regexes): exactly one SELECT; only whitelisted
   tables/columns; no `*` except COUNT(*); no CTEs, SELECT INTO, locking clauses or
   dangerous functions; LIMIT forced to <= 100.
2. Execution in a READ ONLY transaction with a 5 s statement_timeout (see text_to_sql.py).
3. A Postgres role (`jobscout_readonly`, migration 0005) that can SELECT only the
   whitelisted columns — even a validator bug can't read or change anything else.
"""

import re

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

MAX_ROWS = 100

# Users can ask about jobs and companies — never users, resumes, messages or LLM logs.
# Embeddings are excluded too: useless in answers and huge.
ALLOWED_COLUMNS: dict[str, set[str]] = {
    "jobs": {
        "id", "source", "company_id", "company", "title", "description", "location", "remote",
        "salary_min", "salary_max", "salary_currency", "posted_at", "url", "first_seen_at",
        "last_seen_at", "is_active", "times_reposted", "is_duplicate_of", "ghost_score",
    },
    "companies": {"id", "name", "slug", "ats", "industry"},
}  # fmt: skip

# Functions that can read files, sleep, change settings or reach other databases.
_DENIED_FUNCTION = re.compile(
    r"^(pg_|lo_|dblink|set_config|current_setting|query_to_xml|table_to_xml|xpath|"
    r"txid_|inet_|version$)",
    re.IGNORECASE,
)


class UnsafeSQL(ValueError):
    """The SQL was rejected; the message says why (it's shown to the LLM to self-correct)."""


def _function_name(func: exp.Func) -> str:
    if isinstance(func, exp.Anonymous):
        return str(func.this)
    return func.sql_name()


def validate_sql(sql: str) -> str:
    """Return a safe, normalised version of `sql` (with LIMIT <= 100) or raise UnsafeSQL."""
    try:
        statements = [s for s in sqlglot.parse(sql, read="postgres") if s is not None]
    except SqlglotError as exc:
        raise UnsafeSQL(f"could not parse SQL: {exc}") from exc
    if len(statements) != 1:
        raise UnsafeSQL("exactly one statement is allowed")
    stmt = statements[0]
    if not isinstance(stmt, exp.Select):
        raise UnsafeSQL(f"only SELECT is allowed, got {stmt.key.upper()}")

    for forbidden, label in ((exp.With, "WITH/CTE"), (exp.Into, "SELECT INTO"),
                             (exp.Lock, "FOR UPDATE/SHARE")):  # fmt: skip
        if stmt.find(forbidden):
            raise UnsafeSQL(f"{label} is not allowed")
    if stmt.find(exp.Union, exp.Intersect, exp.Except):
        raise UnsafeSQL("UNION/INTERSECT/EXCEPT are not allowed")

    # Tables: whitelisted, in the default schema only (no pg_catalog.pg_user etc.).
    qualifiers: set[str] = set()
    for table in stmt.find_all(exp.Table):
        name = table.name.lower()
        if table.db and table.db.lower() != "public" or table.catalog:
            raise UnsafeSQL(f"schema-qualified table {table.sql()} is not allowed")
        if name not in ALLOWED_COLUMNS:
            raise UnsafeSQL(f"table {name!r} is not allowed (allowed: jobs, companies)")
        qualifiers.add(name)
        if table.alias:
            qualifiers.add(table.alias.lower())
    if not qualifiers:
        raise UnsafeSQL("query must read from jobs or companies")
    for sub in stmt.find_all(exp.Subquery):
        if sub.alias:
            qualifiers.add(sub.alias.lower())

    # Columns: whitelisted for the tables used, or an alias defined in this query.
    used_tables = {t.name.lower() for t in stmt.find_all(exp.Table)}
    allowed = set().union(*(ALLOWED_COLUMNS[t] for t in used_tables))
    aliases = {a.alias.lower() for a in stmt.find_all(exp.Alias) if a.alias}
    for col in stmt.find_all(exp.Column):
        if isinstance(col.this, exp.Star):
            raise UnsafeSQL("SELECT table.* is not allowed; name the columns")
        if col.table and col.table.lower() not in qualifiers:
            raise UnsafeSQL(f"unknown table or alias {col.table!r}")
        if col.name.lower() not in allowed | aliases:
            raise UnsafeSQL(f"column {col.name!r} is not allowed")

    for star in stmt.find_all(exp.Star):
        if not isinstance(star.parent, exp.Count):
            raise UnsafeSQL("SELECT * is not allowed; name the columns")

    for func in stmt.find_all(exp.Func):
        if _DENIED_FUNCTION.match(_function_name(func)):
            raise UnsafeSQL(f"function {_function_name(func)}() is not allowed")

    # Force a row cap. A non-literal LIMIT (e.g. a subquery) is replaced too.
    limit = stmt.args.get("limit")
    current = None
    if limit is not None and isinstance(limit.expression, exp.Literal):
        try:
            current = int(limit.expression.this)
        except ValueError:
            current = None
    if current is None or current > MAX_ROWS:
        stmt = stmt.limit(MAX_ROWS)

    return stmt.sql(dialect="postgres")
