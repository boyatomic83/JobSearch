# JobScout

A mini AI job agent: chat with a Telegram bot, upload your resume, state preferences, and
every morning get the 3 best-matching real jobs with reasons and a tailored resume.

> Work in progress — built phase by phase (see `SPEC.md`). Full docs land in Phase 9.

## Job sources

Real, public job-board APIs only (no scraping, no API keys):

- **Greenhouse** and **Lever** boards for ~30 companies (`app/ingest/companies.yaml`)
- **[Remote OK](https://remoteok.com)** — jobs from Remote OK are linked back to their
  Remote OK page, as their API terms require.

## Embeddings

Jobs are embedded locally with **BAAI/bge-small-en-v1.5** (384-d), run with
**fastembed / ONNX Runtime** on CPU — free, no API key, no rate limits.

Why not sentence-transformers? It needs PyTorch, and measured in our Docker image,
importing torch + sentence-transformers alone used **440 MB**; embedding peaked at
**609 MB** even at batch size 4. That can't fit Render's 512 MB free tier. fastembed runs
the *same model* with identical vectors (cosine similarity 1.0000 vs sentence-transformers
on the same texts), peaks at ~410 MB, and shrinks the image from 2.4 GB to ~0.8 GB.

What we embed (`jobs.embedding_text`): `title + company + first 2000 chars of the cleaned,
boilerplate-free description`.

- **Boilerplate removal.** ~40% of all description text turned out to be per-company
  boilerplate (company intro, benefits, EEO statement). Left in, it fills the 2000-char
  window, so a resume gets compared mostly against "About Stripe". A line counts as
  boilerplate if it appears in ≥ 20% (and ≥ 3) of one company's postings; it's dropped from
  the embedded text only — the stored description is untouched.
- Title and company go first so they survive truncation; location is left out because it's
  filtered exactly in SQL.
- Each job stores a `content_hash` of that text; when it changes, ingest clears the vector
  and only those jobs are re-embedded.

## Duplicate detection

A job is marked `is_duplicate_of` the **oldest** active job with the **same company**,
**same location**, **same title** (ignoring case and punctuation) and an embedding with
**cosine similarity > 0.95**. Duplicates are hidden from matching. Links are recomputed
from scratch on every run, since an "original" can close.

The spec's rule was "same company + cosine > 0.95"; the two extra conditions came from
measuring it on real data:

- **Same title.** Embedding similarity alone flagged 1,504 of 7,479 jobs (20%), and 1,452 of
  those pairs had *different* titles: "Sr." vs "Staff Software Engineer", "Product
  Designer" vs "Brand Designer". Those roles share a description, so no threshold could
  separate them (in a 60-pair sample, 60/60 scored > 0.95, and 5/60 even > 0.98).
- **Same location.** Companies post one role per city as separate listings; collapsing
  those would hide the Bangalore opening behind the San Francisco one.

On the 7,481 live jobs ingested on 2026-11-24 this flags **96 duplicates (1.3%)**: e.g. a
Samsara SDR role in Atlanta re-posted two months later (similarity 1.000), or an Okta
"Senior Software Engineer, Toronto" posted twice on consecutive days (0.996).

## Ghost-job score

A rule-based score from 0 to 1; jobs scoring **above 0.7 are hidden** from matching. Every
point comes with a stored, human-readable reason (`jobs.ghost_reasons`), so it's always
clear why a job was hidden.

| Signal | Points |
|---|---|
| Evergreen title ("talent pool", "general application", …) | 0.50 |
| Posted > 90 days ago / > 45 days ago | 0.35 / 0.15 |
| Reposted (disappeared from the board and came back) | 0.15 per repost, max 0.30 |
| Description < 400 chars / < 1200 chars | 0.30 / 0.10 |
| No responsibilities / requirements section | 0.10 |
| No salary listed | 0.05 |
| No location listed | 0.10 |

No single *soft* signal (age, a repost, missing salary) can hide a job alone; stacked
signals (old + reposted + thin description) do. Example stored reasons:
`["posted 120 days ago (+0.35)", "reposted 2x (+0.30)", "very short description (300 chars) (+0.30)"]`.

On the same 7,481 jobs: **4 are hidden** (all genuine evergreen listings, e.g. Anthropic's
"[Expression of Interest] Research Engineer" posted 603 days ago, Okta's "Join Our Talent
Community"). The most common signals are age (4,303 jobs posted > 45 days ago — big
companies keep requisitions open for months) and missing salary (3,583). The weights are
deliberately conservative: hiding a real job costs the user more than showing a stale one.

## LLM provider

All LLM calls go through `app/llm/` (`LLMClient` + a small provider interface), so the
vendor is an env var: `LLM_PROVIDER=groq | gemini | xai`.

- **Default: Groq** (`openai/gpt-oss-120b`, free tier: 1,000 requests/day, 8,000
  tokens/minute). The spec's default was Gemini, but the Google project returned
  `403 "The Gemini API free tier is not available for this project"` on every model, and
  Google's only fix is enabling billing. The Gemini provider is still there as a drop-in.
- `LLMClient` adds, for every provider: a requests/minute limiter, exponential backoff on
  429s (honouring `Retry-After`), a Postgres response cache (`llm_cache`), Pydantic
  validation with one "repair your JSON" retry, and a row in `llm_calls` per attempt.

## Matching (RAG)

1. **Resume → profile.** PDF text (pypdf) → one LLM call extracts a Pydantic-validated
   profile (titles, skills, years, education, location). Cached, so re-uploads are free.
2. **Retrieve.** The profile (plus target roles from preferences) is embedded and pgvector
   returns the top 50 jobs by cosine, with hard filters applied *in the same SQL query*:
   location, remote-only, min salary (same currency only; unknown salary is kept),
   excluded companies/industries, seniority (by title), no duplicates, no ghost jobs
   (> 0.7), nothing already shown. pgvector's iterative HNSW scan keeps filtering from
   silently returning fewer than 50 rows.
3. **Rerank.** The top 20 go to the LLM in **one** call, which returns the best 3 with a
   0–100 fit score, 2–3 reasons and the biggest gap, grounded only in the text shown. Ids
   the model invents are dropped; if the LLM is down, the user still gets the top
   retrieval results, marked as unranked.

LLM cost per match run: 1 call, ~3.4k prompt + ~400 output tokens (measured).

Location detail worth knowing: 2,074 of 2,095 "remote" jobs are region-locked
("Remote - USA"). A remote job is only shown if its location mentions one of your
preferred locations (so put your country in preferences), unless it's truly unrestricted.

## Agent

A hand-written tool-calling loop (`app/agent/loop.py`, no framework) on the provider's
native function calling. Tools: `search_jobs`, `update_preferences`, `get_preferences`,
`explain_match`, `tailor_resume`, `run_sql_readonly`.

- **Long-term memory = the preferences row.** It is injected into the system prompt on
  every turn, and the prompt requires calling `update_preferences` whenever the user
  states a lasting preference ("no more fintech" → `exclude_industries_add: ["fintech"]`).
  Changes are structured add/remove lists validated by Pydantic, never free-form SQL.
- **Short-term memory = the last 20 chat messages.** Replies that showed jobs are replayed
  with a `[jobs shown: 1) job_id 4080 = …]` note, so "why is the first one a good fit?"
  resolves to the right id. (Found in testing: without it the model took a number from
  the job's URL.)
- **Limits:** max 6 tool calls per turn, then one final call with tools disabled. Every
  tool step goes to `messages`, every LLM call to `llm_calls`.
- **Grounding:** `explain_match` returns the user's real profile next to the job, and the
  prompt forbids claiming experience the profile doesn't contain (found in testing: the
  model invented "NLP projects" when it only had the job).
- **Models:** the agent's own turns run on `LLM_AGENT_MODEL=openai/gpt-oss-20b` (Groq
  rate-limits per model, so it gets its own tokens/minute budget); rerank, extraction and
  SQL use `openai/gpt-oss-120b`. `qwen/qwen3.8-27b` was faster still but answered "how many
  remote jobs?" with a made-up number (1,039; real: 2,068) instead of calling the SQL tool.

## Text-to-SQL guardrails

`run_sql_readonly("how many remote Python jobs in Bangalore were posted this week?")` →
the LLM writes SQL with the schema in its prompt, then three independent layers apply:

1. **Validator** (`app/agent/sql_guard.py`, sqlglot parse tree): exactly one `SELECT`;
   only `jobs` / `companies` and whitelisted columns; no `*` except `COUNT(*)`; no CTEs,
   `UNION`, `SELECT INTO`, `FOR UPDATE`, system catalogs or `pg_*`/`dblink` functions;
   `LIMIT` forced to ≤ 100. Rejections go back to the LLM once so it can fix its query.
2. **Transaction:** `READ ONLY` with `statement_timeout = 5s`.
3. **Role:** `SET LOCAL ROLE jobscout_readonly`, which has column-level `SELECT` grants on
   the whitelist only (migration 0005). Tests bypass the validator on purpose and show
   the role still blocks `SELECT raw_text FROM resumes` and `SELECT embedding FROM jobs`.

## Resume tailoring (no invented facts)

`tailor_resume(job_id)` rewrites the summary, skill order and bullets for one job, then
proves every claim came from the original resume:

1. **Tailor** (LLM): reorder / rephrase / emphasise only. No new employers, titles,
   dates, numbers, tools, skills, team sizes or achievements.
2. **Verify** (a second LLM call): the draft is split into claims (each summary
   sentence, skill, role line and bullet), and a strict fact-checker judges each one
   against the original text. A claim it doesn't return a verdict for counts as unverified.
3. **Rules** (deterministic): any number, skill or employer that doesn't literally
   appear in the original is dropped, whatever the LLM said.

Flagged claims are removed (an unverified employer removes the whole role) and listed back
to the user. Measured with the real model on a deliberately bad draft (6 invented +
6 genuine claims): the LLM verifier flagged all 6 inventions, including "Mentored
engineers" and "Built search", which have no number to catch; the rules independently
caught 4 of the 6 (team size 50, Rust, 45% vs 30%, Google). Neither layer flagged a genuine
claim.

The PDF is rendered with reportlab (pure Python; weasyprint needs system libraries).
Header, contact and education come straight from the original resume. Built-in PDF fonts
lack some Unicode (the model's non-breaking hyphens rendered as boxes), so text is
normalised to Windows-1252; resumes in non-Latin scripts would need an embedded TTF font.

## Telegram bot

`/start` (onboarding: role → location → remote → salary), `/resume` (send a PDF),
`/prefs`, `/jobs`, `/help`; anything else goes to the agent. Each match has 👍 / 👎 /
📝 Tailor resume buttons.

- **Feedback → preferences.** 👎 asks why: "Not this company" / "Not this industry" are
  saved as exclusions directly; seniority/location become notes; "Other" lets the user
  type a reason, which is stored and handed to the agent to update preferences.
- **Daily digest** at 08:00 IST (APScheduler) sends each user their top 3 *new* matches;
  ingest + embedding + dedup + ghost scoring run nightly at 02:30 IST. If a message can't
  be delivered, those matches are deleted again, so "shown" means actually received.
- **Polling locally, webhook in production** (`TELEGRAM_WEBHOOK_URL`). The webhook checks
  Telegram's `X-Telegram-Bot-Api-Secret-Token` (derived from the bot token) and queues the
  update, returning immediately so slow LLM turns never trigger Telegram retries.
- Bot logic lives in `app/bot/service.py` and returns plain `Out` messages, so it's
  tested without Telegram; `telegram_app.py` is a thin adapter. Salaries like "40 LPA" /
  "$150k" are parsed; "Bangalore" also matches "Bengaluru".
- **Render free tier sleeps** after ~15 idle minutes and a sleeping process can't run its
  scheduler, so `POST /admin/ingest` and `POST /admin/digest` exist for an external cron.

## Free-tier latency

Groq's free tier allows 8,000 tokens/minute per model, and one agent turn costs ~4k
(two calls of ~1.6k prompt tokens plus ~500 tokens of per-request overhead that Groq
bills but doesn't report). `LLMClient` keeps a token bucket that refills at the same
rate as Groq's and waits *before* a call would be rejected, so there are no 429 retries.
Measured: a first reply takes ~4 s; messages sent back-to-back without pause slow to
~30 s each once the bucket is empty. A paid tier removes the wait.

## Running locally

```bash
docker compose up -d db
python -m venv .venv && .venv/bin/pip install -e '.[dev]'
DATABASE_URL=postgresql://jobscout:jobscout@localhost:5432/jobscout .venv/bin/alembic upgrade head
DATABASE_URL=postgresql://jobscout:jobscout@localhost:5432/jobscout .venv/bin/python -m app.ingest.run
DATABASE_URL=postgresql://jobscout:jobscout@localhost:5432/jobscout .venv/bin/python -m app.matching.run
.venv/bin/pytest
```

Or the whole stack: `docker compose up --build` → http://localhost:8000/health
