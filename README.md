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
