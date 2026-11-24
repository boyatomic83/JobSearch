# JobScout — conventions for Claude

`SPEC.md` is the source of truth. Build one phase at a time, in order.

## Workflow per phase
1. Implement only that phase's scope.
2. `ruff check . && ruff format --check .` and `pytest` must pass.
3. Commit with a clear message (`Phase N: <what>`), then post a 3–5 line summary + how to try it.
4. Ask the user before anything that uses their accounts or could cost money
   (Gemini calls with the real key, Neon, Render, Telegram webhook setup). 

## Commands
- Local DB: `docker compose up -d db` (pgvector/pg16 on localhost:5432, user/pass/db `jobscout`).
- Full stack: `docker compose up --build` → http://localhost:8000/health
- Migrations: `alembic upgrade head` / new one: `alembic revision -m "msg"` (hand-written, not autogenerate-only).
- Tests: `pytest` (DB tests use `TEST_DATABASE_URL`, default local docker DB; they skip if unreachable).
- Lint: `ruff check . --fix && ruff format .`
- Ingest (local DB): `DATABASE_URL=postgresql://jobscout:jobscout@localhost:5432/jobscout python -m app.ingest.run [--source greenhouse lever remoteok]`
- Admin API: `curl -X POST -H "X-Admin-Key: $KEY" localhost:8000/admin/ingest?source=lever`

## Code style
- Python 3.11+, type hints everywhere, async SQLAlchemy 2.0 (`Mapped[...]`, `mapped_column`).
- Pydantic v2 for all external data (API payloads, LLM structured output). 
- Keep it simple and explainable: no frameworks beyond the spec (no LangChain). 
  Short docstrings that explain *why*.
- Config only through `app.config.get_settings()`; never read `os.environ` elsewhere.
- Provider-agnostic code talks to `app/llm/` and the embeddings interface, never to SDKs directly.
- Logging via `logging.getLogger(__name__)`; no `print` in app code (CLIs may print results).

## Data & secrets
- Never commit `.env`, never print or log API keys / DB URLs with credentials.
- `.env` `DATABASE_URL` points to Neon (user's account). Local dev/tests use the docker DB.
- Connector tests use recorded real responses in `tests/fixtures/` (no network in tests).
- Salaries must be plausible *yearly* figures (see `plausible_annual_salary`); unknown beats wrong.
- No silent fake data. Synthetic data only in `tests/` and `evals/`, clearly labelled.
- Embedding dimension is 384 (`bge-small-en-v1.5`); Gemini embeddings must be requested
  at 384 dims (`output_dimensionality`) so the pgvector column never changes.

## Free-tier limits
- Gemini free tier: rate-limit + exponential backoff on 429, cache results in `llm_cache`,
  log every call to `llm_calls`, rerank in one call.
- Render free: low RAM — load the embedding model once, lazily, and keep it small.
