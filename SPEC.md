# JobScout — Project Spec

A mini AI job agent, inspired by Elip AI (YC W25). A user chats with a Telegram bot,
uploads their resume, states preferences, and every morning gets the 3 best-matching
real jobs with reasons and a tailored resume.

This is a portfolio project for an AI engineer role (LLMs, agents, SQL, Python backend,
RAG, embeddings). Code quality, clarity and a working live demo matter more than
feature count.

---

## Tech stack (use exactly this unless something is impossible — then tell me why)

- Python 3.11+, FastAPI, Pydantic v2, SQLAlchemy 2.0 (async) + Alembic migrations
- PostgreSQL with the `pgvector` extension (hosted on Neon/Supabase; local via Docker)
- LLM behind a small provider interface (`app/llm/`) so the provider can be switched
  with an env var. **Default provider: Google Gemini (free tier)** using the official
  `google-genai` Python SDK. Use a current Gemini Flash model (check the Gemini API docs
  for the latest Flash model available on the free tier) and set it via `LLM_MODEL`.
  Use Gemini's native function calling for the agent and its JSON/structured output mode
  for extraction and reranking.
- The free tier has strict per-minute and per-day request limits, so: add a rate limiter
  and retry with exponential backoff on 429 errors, cache LLM results in Postgres, and
  keep LLM calls per user per day small (rerank in ONE call with all 20 candidates).
- **Embeddings run locally for free** with `sentence-transformers` and
  `BAAI/bge-small-en-v1.5` (384 dimensions, runs on CPU, no API key, no rate limits).
  Put this behind an embeddings interface too, with Gemini `gemini-embedding-001` as an
  optional alternative via `EMBEDDING_PROVIDER`. The pgvector column dimension must
  match the chosen model.
- Telegram bot via `python-telegram-bot` using **webhooks** in production, polling locally
- APScheduler for the nightly ingestion/matching job
- pytest for tests, ruff for linting
- Docker + docker-compose (app + postgres-with-pgvector) for local dev
- Deploy target: Render (Dockerfile-based web service)

## Environment variables (read from `.env` locally / Codespaces secrets; never hardcode)

```
DATABASE_URL=
GEMINI_API_KEY=
LLM_PROVIDER=gemini
LLM_MODEL=                # a current Gemini Flash model name
EMBEDDING_PROVIDER=local  # local (sentence-transformers) or gemini
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
TELEGRAM_BOT_TOKEN=
TELEGRAM_WEBHOOK_URL=     # set in production only
ADMIN_API_KEY=            # protects admin endpoints
```

Create `.env.example` with these keys and empty values. Add `.env` to `.gitignore`.

---

## Features — build in this order, one phase at a time

### Phase 1 — Skeleton & database
- Project layout:
  ```
  app/
    main.py            # FastAPI app
    config.py          # pydantic-settings
    db/                # models, session, migrations
    ingest/            # job source connectors
    matching/          # embeddings, retrieval, rerank
    agent/             # LLM agent + tools
    bot/               # Telegram handlers
    llm/               # provider interface (OpenAI / Anthropic)
    resume/            # parsing + tailoring
  evals/
  tests/
  ```
- Tables: `users`, `preferences`, `resumes`, `companies`, `jobs`, `job_embeddings`
  (or a vector column on `jobs`), `matches`, `messages` (chat history), `llm_calls`
  (tokens, cost, latency per call).
- `jobs` must include: source, source_id, company, title, description, location,
  remote (bool), salary_min/max (nullable), posted_at, url, first_seen_at, last_seen_at,
  times_reposted, is_duplicate_of (nullable FK), ghost_score (float).
- Unique constraint on (source, source_id). HNSW index on the embedding column.
- `GET /health` endpoint. docker-compose up must work.

### Phase 2 — Job ingestion
- Connectors (each a class with the same interface) for free public sources:
  - Greenhouse job board API (`boards-api.greenhouse.io/v1/boards/{company}/jobs?content=true`)
  - Lever postings API (`api.lever.co/v0/postings/{company}?mode=json`)
  - RemoteOK API (`remoteok.com/api`)
- A config file `app/ingest/companies.yaml` listing ~30 tech companies' Greenhouse/Lever slugs.
- Clean HTML from descriptions. Upsert jobs: update `last_seen_at`, bump `times_reposted`
  when a job disappears and reappears.
- CLI: `python -m app.ingest.run` and an admin endpoint `POST /admin/ingest`.
- Be polite: rate-limit, retry with backoff, timeouts, log failures without crashing.

### Phase 3 — Embeddings, dedup & ghost-job scoring
- Embed `title + company + cleaned description` (truncate sensibly). Embed in batches.
  Only embed new/changed jobs (store a content hash).
- Dedup: same company + cosine similarity > 0.95 → mark `is_duplicate_of` the oldest one.
- Ghost score (0–1), rule-based and explainable, from: age of posting, repost count,
  very short/vague description, missing salary/location. Store the score and the
  reasons. Document the heuristic in the README.

### Phase 4 — Resume + matching (RAG)
- Upload resume PDF (via Telegram and `POST /users/{id}/resume`). Extract text
  (pypdf), then have the LLM extract structured profile JSON: skills, years of
  experience, roles, education, location. Validate with Pydantic.
- Embed the resume profile.
- Hybrid retrieval: pgvector top-50 by cosine, filtered in SQL by user preferences
  (remote, location, min salary, excluded companies/industries, seniority), excluding
  duplicates and jobs with ghost_score > 0.7 and jobs already shown to the user.
- LLM rerank top-20 → top-3 with structured output: job_id, fit_score (0–100),
  2–3 bullet reasons, 1 gap/risk. Store in `matches`.

### Phase 5 — Agent + preference memory
- A tool-calling agent (write the loop yourself using the provider's native tool
  calling — no LangChain) with tools:
  - `search_jobs(query, filters)` — runs retrieval + rerank
  - `update_preferences(changes)` — writes structured prefs to SQL
  - `get_preferences()`
  - `explain_match(job_id)`
  - `tailor_resume(job_id)`
  - `run_sql_readonly(question)` — text-to-SQL over jobs ("how many remote Python jobs
    in Bangalore were posted this week?")
- Text-to-SQL guardrails: generate SQL with the schema in the prompt, parse it with
  `sqlglot`, allow only a single SELECT, only on whitelisted tables/columns, enforce
  LIMIT 100, run on a read-only DB role/transaction with a 5s statement timeout.
- Memory: when the user says things like "no more fintech" or "remote only", the agent
  must call `update_preferences`; preferences are always injected into the system prompt.
- Keep the last ~20 messages per user as conversation context.
- Max 6 tool-call steps per turn; log every step and every LLM call to `llm_calls`.

### Phase 6 — Resume tailoring
- For a given job, rewrite the resume summary + bullets to fit the job, using ONLY
  facts from the original resume. Add a verification pass: a second LLM call checks
  every claim in the tailored version against the original and flags anything invented;
  drop flagged claims.
- Render to PDF (use `weasyprint` or `reportlab`) and send it via Telegram.

### Phase 7 — Telegram bot
- Commands: `/start` (onboarding: ask role, location, remote, salary), `/resume`
  (upload), `/prefs`, `/jobs` (get matches now), `/help`. Free-text messages go to the agent.
- Daily digest: scheduled job at 08:00 IST sends each user their top 3 new matches with
  inline buttons: 👍 / 👎 / "Tailor resume". Store 👍/👎 as feedback and use 👎 reasons
  to update preferences.
- Polling mode when `TELEGRAM_WEBHOOK_URL` is empty, webhook route `POST /telegram/webhook`
  otherwise.

### Phase 8 — Evals & observability
- `evals/matching_eval.py`: a labelled set of ~50 (resume_profile, job_id, relevant: bool)
  pairs stored as JSON (create a small synthetic set from ingested jobs and 3 sample
  resumes in `evals/resumes/`). Report precision@3 and recall@20 for retrieval-only vs
  retrieval+rerank.
- `evals/sql_eval.py`: 15 natural-language questions with expected SQL results; report
  accuracy and confirm guardrails block 5 malicious inputs (DROP, UPDATE, multiple
  statements, non-whitelisted tables, etc.).
- `GET /admin/stats` (protected by ADMIN_API_KEY): jobs ingested, duplicates, ghost jobs,
  users, LLM tokens and estimated cost per user per day.

### Phase 9 — Tests, Docker, deploy, docs
- pytest tests for: connectors (with recorded fixtures, no network), dedup, ghost score,
  SQL guardrails, preference updates, the agent loop (with a fake LLM provider).
- Dockerfile (slim, non-root), `render.yaml`, Alembic migration runs on startup.
- Deploy to Render: guide me step by step through anything that needs my login, then
  set the Telegram webhook to the live URL and verify `/health` and the bot work.
- README.md: what it is, demo GIF placeholder, architecture diagram (Mermaid), how
  matching/dedup/ghost scoring/guardrails work, eval results table, how to run locally,
  how to deploy, cost per user, and "what I'd do next".

---

## How to work

- Start by writing a short plan and a `CLAUDE.md` with project conventions. Then build
  **one phase at a time**. At the end of each phase: run ruff + tests, fix failures,
  make a git commit with a clear message, and give me a 3–5 line summary of what was
  built and how to try it. Then continue to the next phase unless I say stop.
- Never commit secrets. Never print API keys.
- Keep code simple and readable — I must be able to explain every part in an interview.
  Add short docstrings explaining *why*, not just *what*.
- If a public API is down or changed, adapt and tell me; don't fake data silently.
  Synthetic data is fine only for evals and tests, and must be labelled as such.
- Everything must run on free tiers: Gemini free tier, local embeddings, Neon free
  Postgres, Render free web service, Telegram. Stay within free-tier rate limits.
- Render's free instances have little RAM, so keep the embedding model small, load it
  once at startup, and tell me if memory becomes a problem.
- Ask me before anything that costs money or needs my accounts (API keys, Render, Neon).