# JobScout

A mini AI job agent on Telegram: upload your resume, say what you want ("remote only",
"no more fintech"), and every morning get the **3 best-matching real jobs** — with a fit
score, the reasons, the biggest gap, and a one-tap **tailored resume** that's
fact-checked against your original.

**Try it:** [@Cosmo122bot](https://t.me/Cosmo122bot) on Telegram · API health:
<!-- LIVE_URL --> (Render free tier)

![demo](docs/demo.gif) <!-- TODO: record a 20 s demo GIF -->

Built as a portfolio project for AI-engineering roles: LLM agents with tool calling,
RAG over pgvector, guarded text-to-SQL, evals, and observability — on free tiers only.

## Architecture

```mermaid
flowchart LR
    subgraph Sources["Public job APIs"]
        GH[Greenhouse] & LV[Lever] & RO[RemoteOK]
    end
    subgraph Nightly["Nightly pipeline (GitHub Actions)"]
        ING[Ingest + clean<br/>rate-limited, retries] --> EMB[Embed<br/>bge-small via ONNX] --> DD[Dedup + ghost score]
    end
    Sources --> ING
    DD --> DB[(Neon Postgres<br/>+ pgvector HNSW)]

    U((User)) <-->|Telegram| TG[Bot<br/>webhook]
    subgraph App["FastAPI on Render"]
        TG --> AG[Agent loop<br/>max 6 tool calls]
        AG -->|tools| T1[search_jobs] & T2[update_preferences] & T3[run_sql_readonly] & T4[tailor_resume] & T5[explain_match]
        T1 --> RET[SQL filters + vector top-50] --> RR[LLM rerank 20 → 3]
        T3 --> GUARD[sqlglot validator → read-only role, 5 s timeout]
        T4 --> TL[Rewrite → LLM fact-check → rule check → PDF]
    end
    RET & GUARD & AG --> DB
    AG & RR & TL <--> LLM[Groq<br/>gpt-oss-120b / 20b]
    CRON[GitHub Actions<br/>08:00 IST] -->|/admin/digest| App
```

| Layer | Choice | Why |
|---|---|---|
| API | FastAPI, Pydantic v2, SQLAlchemy 2 async, Alembic | typed, async, migrations run on startup |
| Data | Postgres + pgvector (Neon free) | one store for jobs, vectors, chat memory, LLM logs |
| Embeddings | BAAI/bge-small-en-v1.5 via fastembed (ONNX) | free, local, fits 512 MB RAM (see below) |
| LLM | Groq `openai/gpt-oss-120b` (tasks), `gpt-oss-20b` (agent) | free tier; provider is an env var |
| Bot | python-telegram-bot, webhooks in prod | |
| Jobs | GitHub Actions cron | Render free instances sleep |

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

## Evals

Run against a database with ingested jobs: `python -m evals.matching_eval`,
`python -m evals.sql_eval`. Results are written to `evals/results/`.

### Matching (`evals/matching_eval.py`)

3 **synthetic** resumes (`evals/resumes/`: senior backend in Bangalore, ML engineer in
India, data engineer in London) × real jobs ingested on 2026-11-24 → **76 hand-labelled
(resume, job, relevant) pairs** (`evals/data/matching_labels.json`). The labelling pool is
each resume's retrieval top-20 (so every possible top-3 job is labelled) plus up to 8
keyword matches outside it (so recall@20 can miss things); each resume has a written
relevance rule the labels follow.

| resume | relevant | P@3 retrieval | P@3 + rerank | P@3 upper bound* | R@20 |
|---|---|---|---|---|---|
| data_engineer_london | 1 | 0.33 | 0.33 | 0.33 | 1.00 |
| ml_engineer_india | 12 | 1.00 | 1.00 | 1.00 | 1.00 |
| senior_backend_blr | 15 | 1.00 | 1.00 | 1.00 | 0.73 |
| **mean** | | **0.78** | **0.78** | 0.78 | **0.91** |

\* the best any reranker could do with that top-20. R@20 is identical with and without
rerank (rerank only reorders the top 20).

What the eval taught us:
- **The first rerank prompt made things worse** (backend P@3 1.00 → 0.67): it put a
  "Senior SRE" job third because Kubernetes/AWS/Python overlapped — while itself writing
  "gap: primary focus is SRE, resume is backend". Adding one rule (*judge the role type
  first; a different discipline scores < 60*) fixed it.
- On this small set the rerank **ties** retrieval on precision; its value is the
  explanation (fit score, reasons, gap) and robustness when retrieval order is off. With
  3 resumes these numbers are indicative, not statistically strong.
- London data engineering has exactly 1 open role across our 31 companies, so 0.33 is
  the ceiling there — a coverage limit of the job sources, not of ranking.
- Building the pool surfaced a filter bug: "Remote (U.S.)" was treated as worldwide-remote.

### Text-to-SQL (`evals/sql_eval.py`)

15 questions with hand-written gold SQL, compared by **result** at run time (the data
changes daily), through the real guarded pipeline.

| check | result |
|---|---|
| Answer accuracy | **13 / 15 (87%)** |
| Malicious SQL blocked (DROP, UPDATE, multiple statements, other table, `pg_read_file`) | **5 / 5** |
| Prompt injections via natural language (delete rows, read resumes, "run DROP TABLE") | **3 / 3** refused, nothing leaked |
| `jobs` table intact after all attacks | **yes** (7,497 → 7,497) |

The 2 misses are interpretation differences, kept as failures: Databricks counted via the
`companies` join (880) vs by name (881 — one Databricks posting came from RemoteOK, which
has no company row); and "ML engineer jobs in India" also searched descriptions (6 vs 5).

## Cost per user

`GET /admin/stats` reports jobs, duplicates, ghost jobs, users, feedback and LLM tokens
with **estimated cost per user per day**. Estimates use Groq's paid list prices
(`app/llm/pricing.py`; gpt-oss-120b $0.15/$0.60, gpt-oss-20b $0.075/$0.30 per 1M tokens) —
on the free tier the bill is $0.

Measured average cost per call: rerank $0.00074 · agent turn (20b) $0.00014 · text-to-SQL
$0.00015 · resume extraction $0.00041 · tailoring + verification $0.00081.

| user type | LLM calls / day | est. cost / day | / month |
|---|---|---|---|
| digest only | 1 rerank | $0.0007 | $0.02 |
| active (digest, /jobs, ~4 chat turns, 1 SQL question, 1 tailoring) | ~13 | ~$0.0035 | ~$0.10 |

Free-tier capacity is ~1,000 requests/day per model, so roughly 100–200 active users/day.

## Free-tier latency

Groq's free tier allows 8,000 tokens/minute per model, and one agent turn costs ~4k
(two calls of ~1.6k prompt tokens plus ~500 tokens of per-request overhead that Groq
bills but doesn't report). `LLMClient` keeps a token bucket that refills at the same
rate as Groq's and waits *before* a call would be rejected, so there are no 429 retries.
Measured: a first reply takes ~4 s; messages sent back-to-back without pause slow to
~30 s each once the bucket is empty. A paid tier removes the wait.

## Running locally

```bash
cp .env.example .env   # fill in GROQ_API_KEY, TELEGRAM_BOT_TOKEN, ADMIN_API_KEY
docker compose up -d db
python -m venv .venv && .venv/bin/pip install -e '.[dev]'
DATABASE_URL=postgresql://jobscout:jobscout@localhost:5432/jobscout .venv/bin/alembic upgrade head
DATABASE_URL=postgresql://jobscout:jobscout@localhost:5432/jobscout .venv/bin/python -m app.ingest.run
DATABASE_URL=postgresql://jobscout:jobscout@localhost:5432/jobscout .venv/bin/python -m app.matching.run
.venv/bin/pytest
```

Or the whole stack: `docker compose up --build` → http://localhost:8000/health

With `TELEGRAM_BOT_TOKEN` set and no webhook URL, the bot runs in polling mode locally.
Only one process can poll a bot token at a time.

## Deploying (free tiers)

1. **Neon:** create a project and put its connection string in `DATABASE_URL`. Load data
   once from your machine: `alembic upgrade head && python -m app.ingest.run && python -m
   app.matching.run` (embedding ~7.5k jobs takes ~40 min on 2 CPUs).
2. **Render:** *New → Blueprint* → this repo. `render.yaml` defines the service; enter the
   four secrets (`DATABASE_URL`, `GROQ_API_KEY`, `TELEGRAM_BOT_TOKEN`, `ADMIN_API_KEY`).
   The app registers the Telegram webhook at `$RENDER_EXTERNAL_URL/telegram/webhook` on
   startup — no manual step.
3. **GitHub Actions secrets:** `DATABASE_URL`, `ADMIN_API_KEY`, `APP_URL` (the Render URL)
   for the nightly pipeline, the 08:00 IST digest and keep-alive pings.

Render's free instance has 512 MB RAM and 0.1 CPU: the app idles at ~310 MB with the
embedding model loaded, which is why heavy embedding runs in GitHub Actions instead.

## Tests

`pytest` — 170+ tests, no network: connectors on recorded API responses, dedup, ghost
score, SQL guardrails (including the DB-level role with the validator bypassed),
preference updates, the agent loop and tailoring with a scripted fake LLM, the bot logic
with a fake Telegram bot. CI runs ruff + pytest against pgvector on every push.

## What I'd do next

- **Bigger, independent eval set:** 3 resumes is enough to catch regressions (it caught
  two), not to claim numbers. Label ~30 resumes with 2 annotators and report agreement.
- **Learn from feedback:** use 👍/👎 as training signal — e.g. a lightweight learned
  re-ranker on (profile, job) features, or few-shot examples in the rerank prompt.
- **More sources:** Ashby, Workable, SmartRecruiters and Indian boards; coverage is the
  real ceiling (London data engineering had exactly one open role).
- **Better ghost detection:** track per-company repost patterns and time-to-close.
- **Streaming replies + a paid LLM tier:** the 8k tokens/minute free limit, not the code,
  sets latency when users send messages back-to-back.
- **Multi-language resumes:** embed a Unicode font in the PDF renderer.
