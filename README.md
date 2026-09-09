# LinkedIn Connections Intelligence

Upload your LinkedIn Connections CSV -> enrich every connection with Apify HarvestAPI
profile data -> search your network in plain English and get the **Top 20 matching
people you already know**, each tagged Exact / Possible / Near match with an
evidence-backed 0-100 match score, a grounded reason, exact supporting evidence, and a
**separate** data-confidence score.

## How the three layers divide the work

| Layer | Owns |
|---|---|
| **Apify** | Gets LinkedIn profile *facts* — employer, title, dates, education, location, skills. Runs ONLY during enrichment, never during search. |
| **Anthropic** | The primary intelligence layer — understands profile *meaning* (semantic enrichment) and *search intent* (query interpretation, ambiguous-case judging, a final correctness audit, display reasons). Never overrides a verified fact. |
| **Python (this backend)** | Owns factual truth, chronology, qualification, and the numeric score. Every LLM judgment is validated against real evidence before it can change a result — an invalid or hallucinated reference is rejected, not trusted. |

Anthropic is the primary provider once `ANTHROPIC_API_KEY` is set (see **Setup**
below) — it is a **paid** API. Optional free fallbacks (Groq, OpenRouter) and a fully
deterministic fallback exist for every LLM step, so the app still works with no LLM key
at all, just with less semantic nuance.

## Stack

| | |
|---|---|
| Backend | Python 3.11 · FastAPI · Pydantic v2 · SQLAlchemy 2 · SQLite (Postgres-ready via `DATABASE_URL`) |
| Frontend | Vite · React · TypeScript · Tailwind · TanStack Query |
| Profile data | Apify actor `harvestapi/linkedin-profile-scraper` (`LpVuK3Zozwuipa5bp`), *Profile details no email* — never enables email search, never switches to a costlier mode |
| LLM | Anthropic (primary, paid) -> Groq -> OpenRouter (`:free`) -> deterministic fallback, per step |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2`, local, numpy brute-force cosine — no key, no cost |

## Setup

```bash
git clone <this repo>
cd linkedin-connections-intelligence
cp .env.example backend/.env
```

Edit `backend/.env` and set, at minimum:

```
APIFY_API_TOKEN=...       # https://console.apify.com/account/integrations
ANTHROPIC_API_KEY=...     # https://console.anthropic.com/settings/keys
USE_FIXTURES=false        # false = real Apify enrichment
```

A normal Anthropic API key works as-is — `ANTHROPIC_WORKSPACE_ID` is only needed for
an identity-linked key tied to multiple workspaces, and `ENABLE_PAID_LLM` is a
deprecated no-op kept for old `.env` files (a configured `ANTHROPIC_API_KEY` **is** the
opt-in). `GROQ_API_KEY` / `OPENROUTER_API_KEY` are optional free fallbacks.

`backend/.env` is gitignored and must never be committed — the app reads secrets only
from that local file / real environment variables, never from source.

### Run it

```bash
.\run.ps1          # Windows: creates backend/.venv, installs both sides, starts both
```

or manually:

```bash
# backend  ->  http://localhost:8010   (docs at /docs)
cd backend
py -3.11 -m venv .venv
.venv/Scripts/pip install -r requirements.txt
.venv/Scripts/python -m uvicorn app.main:app --port 8010
```

```bash
# frontend  ->  http://localhost:5182   (proxies /api/* -> :8010)
cd frontend
npm install
npm run dev
```

Then open **http://localhost:5182**, upload your `Connections.csv`, click **Enrich**,
and search.

## Enrichment call flow (one-time per connection, cached)

```
CSV -> parse (skip export preamble) -> canonicalize URLs -> dedupe by public id
    -> dataset + people (is_connection=true)
    -> enrichment worker (resumable, batched):
         Apify -> raw_profiles (verbatim)
         -> deterministic normalize -> experiences/education/skills + completeness
         -> Anthropic semantic pass (role/industry meaning, leadership/mentoring
            signals, inferred skills w/ evidence) — cached by semantic_profile_version;
            re-run only if the raw profile materially changed or the version bumps.
            An Anthropic failure NEVER re-triggers Apify — the scrape stays, semantics
            are retried later.
         -> cached company classification (once per employer, reused by every
            person who worked there — never one call per person)
         -> local embedding
         -> READY
```

## Search call flow (every query, budget-aware)

```
query
  -> Anthropic query interpretation (or deterministic parser)
  -> full local scan of every connection (<= FULL_SCAN_MAX_CONNECTIONS)
  -> hard-fact viability gate (verified contradictions only)
  -> local pre-score: stored facts + cached company classification + stored
     ProfileSemantic (role/industry/leadership signals) resolve MOST candidates
     to TRUE/FALSE without any query-time LLM call
  -> Anthropic semantic judge — ONLY for candidates with a genuinely unresolved
     REQUIRED semantic criterion; batched, with adaptive splitting if a batch's
     response is truncated (never a blind identical retry)
  -> fact-consistency validator (every verdict checked against real evidence
     before it can change a result)
  -> deterministic rescore -> qualification (Exact / Possible / Not Match)
  -> cross-encoder rerank within tiers
  -> final Anthropic audit over the shown pool (batched) — can downgrade or
     remove, never invents a fact, never promotes Possible -> Exact
  -> ONE batched Anthropic call writes the display reasons for the top results
  -> persisted Top 20 + near matches
```

Search **never** calls Apify. A saved search reload **never** re-runs any LLM,
embedding, judge, or audit step — it replays the exact response that was first
returned.

Because most candidates are already decided from stored facts and semantics, a
~1,000-connection network does **not** turn a broad query into ~100 Anthropic calls —
only the genuinely ambiguous candidates reach the judge. An optional
`SEARCH_LLM_MAX_CALLS` soft budget can cap query-time LLM spend further; if it's hit,
deterministic results still stand and the UI marks verification as partial — it never
silently pretends a review was complete. `SEARCH_MAX_SECONDS` is the same idea for wall
time: a very broad or difficult query stops starting new judge/audit batches once the
deadline passes and returns partial-but-useful results instead of hanging.

## Cost

Apify: pay-per-event, ~`$0.004`/profile (`$4` per 1,000) on the *Profile details – no
email* tier — a one-time cost per connection, re-used until `PROFILE_TTL_DAYS` expires.
Fixture mode (`USE_FIXTURES=true`, what the automated tests always use) is $0.

Anthropic: paid, billed by Anthropic per the model in `ANTHROPIC_MODEL`. Actual spend
depends on network size and query breadth; see `SEARCH_LLM_MAX_CALLS` above to cap it,
and `backend/eval/pilot/` for an offline harness that estimates call counts before any
live run. Optional `GROQ_API_KEY` / `OPENROUTER_API_KEY` fallbacks are free-tier.

## Tests

```bash
cd backend && .venv/Scripts/python -m pytest
cd frontend && npm test -- --run
cd frontend && npx tsc --noEmit
```

No test suite makes a live Apify or LLM call — every provider is mocked or disabled.

## Resuming an interrupted enrichment run

Enrichment is resumable and batched. If the LLM quota / budget runs out mid-run, the
semantic step is deferred for the rest of the run — profiles are still scraped,
normalized, embedded, and marked READY. Click **Resume** later; it runs a
`backfill_semantics` pass for anyone still missing the current semantic version —
nothing is ever re-scraped just because semantics failed.

`scripts/reset_dataset.py <dataset_id>` wipes a dataset's derived data back to PENDING
(keeps the CSV rows) if a run went wrong.

## Config knobs (`.env`)

See `.env.example` for the full, commented list. Notable ones:

`USE_FIXTURES` · `ANTHROPIC_API_KEY` / `ANTHROPIC_MODEL` / `ANTHROPIC_WORKSPACE_ID` ·
`GROQ_API_KEY` / `OPENROUTER_API_KEY` (optional fallbacks) · `SEARCH_LLM_MAX_CALLS` ·
`SEARCH_MAX_SECONDS` ·
`SEMANTIC_JUDGE_MODE` · `FINAL_RESULT_AUDIT_ENABLED` · `TOP_CONNECTIONS` ·
`CANDIDATE_POOL_SIZE` · `MIN_MATCH_SCORE` · `EMBEDDINGS_ENABLED` · `PROFILE_TTL_DAYS`.
