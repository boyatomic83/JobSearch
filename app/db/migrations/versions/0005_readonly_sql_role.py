"""Read-only role for agent text-to-SQL: SELECT on whitelisted columns of jobs/companies.

Third layer of the text-to-SQL guardrails (after the sqlglot validator and the READ ONLY
transaction). Column-level grants mean even a validator bug can't read resumes, users,
messages or embeddings. If the DB user can't create roles (some hosted Postgres plans),
we log a notice and the other two layers still apply.

Revision ID: 0005
Revises: 0004
Create Date: 2026-11-24
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Frozen copy of app.agent.sql_guard.ALLOWED_COLUMNS (a test checks they match).
JOBS_COLUMNS = (
    "id, source, company_id, company, title, description, location, remote, salary_min, "
    "salary_max, salary_currency, posted_at, url, first_seen_at, last_seen_at, is_active, "
    "times_reposted, is_duplicate_of, ghost_score"
)
COMPANIES_COLUMNS = "id, name, slug, ats, industry"


def upgrade() -> None:
    op.execute(f"""
    DO $$
    BEGIN
        IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'jobscout_readonly') THEN
            CREATE ROLE jobscout_readonly NOLOGIN;
        END IF;
        GRANT USAGE ON SCHEMA public TO jobscout_readonly;
        GRANT SELECT ({JOBS_COLUMNS}) ON jobs TO jobscout_readonly;
        GRANT SELECT ({COMPANIES_COLUMNS}) ON companies TO jobscout_readonly;
        EXECUTE format('GRANT jobscout_readonly TO %I', current_user);
    EXCEPTION WHEN insufficient_privilege THEN
        RAISE NOTICE 'jobscout_readonly not created (insufficient privilege)';
    END $$;
    """)


def downgrade() -> None:
    # Roles are cluster-wide (shared by other databases), so only revoke this DB's grants.
    op.execute("""
    DO $$
    BEGIN
        IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'jobscout_readonly') THEN
            REVOKE ALL ON jobs, companies FROM jobscout_readonly;
            REVOKE USAGE ON SCHEMA public FROM jobscout_readonly;
        END IF;
    END $$;
    """)
