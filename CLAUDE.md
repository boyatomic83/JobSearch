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
- Embed + dedup + ghost score: same DATABASE_URL, `python -m app.matching.run [--reembed]` (~3 jobs/s on 2 CPUs)
- Admin API: `curl -X POST -H "X-Admin-Key: $KEY" localhost:8000/admin/ingest?source=lever`

## Dependencies
- New import → add it to `pyproject.toml` AND rebuild Docker (`docker compose up --build`)
  before committing; the venv can hide a missing dependency (happened with sqlglot).

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

## LLM
- Default provider: Groq `openai/gpt-oss-120b` (Gemini project is 403 free-tier-blocked;
  user chose Groq 2026-11-24). Free tier: 1000 req/day, 8000 tokens/min per model.
- All calls go through `LLMClient` (never a provider/SDK directly). Tests use
  `tests.fakes.make_llm` — no network in tests.
- Prompts must forbid outside knowledge: reasons/gaps grounded in shown text only.
- Agent model: LLM_AGENT_MODEL=openai/gpt-oss-20b. Don't switch to qwen: it fabricated a
  statistic instead of calling run_sql_readonly (tested 2026-11-24).
- Tailoring: keep BOTH the LLM verifier and the rule checks; each catches what the other
  misses (measured). PDF text must go through render._pdf_safe.
- Text-to-SQL: never loosen sql_guard without a test; the 0005 grants must match
  ALLOWED_COLUMNS (test enforces it).

## Bot
- Logic in app/bot/service.py (returns Out, testable); telegram_app.py only adapts.
- Use HTML parse mode + html.escape for anything user/LLM/job-provided.
- Only ONE process may poll a bot token (Telegram 409 Conflict): stop docker app before
  running the bot elsewhere.

## Free-tier limits
- Gemini free tier: rate-limit + exponential backoff on 429, cache results in `llm_cache`,
  log every call to `llm_calls`, rerank in one call.
- Render free: low RAM — load the embedding model once, lazily, and keep it small.
  Local embeddings use fastembed (ONNX), NOT sentence-transformers/torch (440 MB just to
  import; user approved the swap 2026-11-24). Keep `LocalEmbedder.BATCH_SIZE` small.
  Measured peak while embedding: ~440 MB with MALLOC_ARENA_MAX=2 — re-measure after
  adding anything heavy to the web process.
- Dedup = same company + location + normalised title + cosine > 0.95 (title/location
  guards are deliberate, backed by data in README). Don't loosen without re-measuring.
