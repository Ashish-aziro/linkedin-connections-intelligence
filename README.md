# LinkedIn Connections Intelligence

A local, single-user tool for searching **your own LinkedIn network** in plain
English. Upload your LinkedIn *Connections* CSV, enrich each connection with real
profile data, then ask questions like *"senior engineers in Atlanta"* or *"who
could help me raise funding for an AI startup"* and get a ranked shortlist of the
**people you already know**, each with an evidence-backed match score, a grounded
explanation, and a separate data-confidence score.

> ## ⚠️ Branch: `feature/full-sonnet-verification` — FULL SONNET VERIFICATION EXPERIMENT
>
> On this branch **every candidate that survives the deterministic hard-fact
> filter is reviewed by Claude Sonnet 4.6** against the whole search plan before
> a search can return successfully.
>
> - A successful search returns **only fully-reviewed candidates** whose every
>   *required* criterion verified **TRUE**. There is no *"some AI verification was
>   unavailable"* banner and no *"Possible / needs verification"* card.
> - A candidate whose review completes but lacks the evidence to establish a
>   required claim (`INSUFFICIENT_EVIDENCE`) is **excluded** from results, not
>   shown conservatively.
> - If full Sonnet verification **cannot complete** (API error, unrecoverable
>   truncation, missing output …) after bounded retry/split/chunk recovery, the
>   **whole search fails** with a retryable error (`HTTP 503`,
>   `{"error":"verification_incomplete","retryable":true}`) and the UI shows a
>   *"Full AI verification could not be completed — Retry search"* screen. It
>   never shows partially-verified candidates.
> - Facts stay authoritative: Sonnet verifies *meaning* against validated
>   evidence — it does not invent profile facts and does not choose the numeric
>   score (deterministic application code does).
>
> This is intentionally **higher accuracy / higher confidence** at the cost of
> **more Anthropic usage and higher latency** (see *Cost & latency* below). Turn
> it off with `FULL_LLM_VERIFICATION=false` to fall back to the legacy
> "semantic-judge-only-where-unresolved + conservative partial results" path.
>
> **The completed, stable phase is the branch/tag `phase-complete-2026-09-10`**
> (`main` is unchanged). Switch back any time:
> `git checkout phase-complete-2026-09-10`.

## What it does

```
Connections.csv
  → upload → dataset + Person rows
  → enrichment: Apify profile scrape → raw JSON (kept verbatim)
                → deterministic normalization (experiences / education / skills / …)
                → company classification (once per employer, cached)
                → semantic profile representation (Anthropic, cached by version)
                → local MiniLM embedding
                → READY / PARTIAL
  → dashboard
  → natural-language search
       → Anthropic understands the query → structured search plan
       → full scan of your connections → hard-fact viability gate
       → deterministic evidence-based pre-score
       → FULL SONNET VERIFICATION — every filtered candidate reviewed by Sonnet
         (batched, with retry / split / single-person / chunked recovery)
       → fact-consistency validation → deterministic rescore → Exact tier only
       → cross-encoder reranking → final grounded audit → grounded explanations
       → verified results, persisted   (or HTTP 503 verification_incomplete)
```

## Key capabilities

- Upload a LinkedIn *Connections* CSV (tolerant parser: skips the export
  preamble, handles reordered/missing columns, BOM, `;` delimiters)
- Enrich every connection with Apify HarvestAPI profile data
- **Resumable** enrichment — leave the page, come back, it picks up where it left off
- **Semantic backfill** — if the LLM was unavailable during a run, profiles are
  still scraped/normalized/embedded/marked READY and the semantic pass is
  retried later with **no Apify re-scrape**
- General natural-language network search (companies, schools, locations, roles,
  industries, seniority, AND/OR, exclusions, "might have X", cross-domain, …)
- **Exact / Possible** match tiers + a 0–100 match score
- A **separate** data-confidence score (how complete the profile is)
- Structured evidence per result (which experience / skill / etc. supports it)
- Grounded, one-line explanations
- Saved search history — reload a past search with **zero** external calls
- Excel export (Profiles + Experiences + Education + Skills + Certifications +
  Languages + Publications)
- Individual person refresh (respects a TTL unless forced)
- Dataset deletion (cascades to every derived row)

## Architecture

| Layer | Responsibility |
|---|---|
| **Apify** | Fetches LinkedIn profile **facts** — employer, title, dates, education, location, skills. Runs **only during enrichment**, never during search. |
| **Anthropic** | The only external LLM. Understands profile **meaning** (semantic enrichment) and search **intent** (query interpretation, the semantic judge, the final audit, grounded explanations). Never overrides a verified fact and cannot invent profile evidence. |
| **Python backend** | Owns factual truth, chronology, qualification tiers, and the deterministic numeric score. Every LLM judgment is validated against real stored evidence before it can change a result — an invalid or hallucinated reference is rejected. |
| **Local ML** | `sentence-transformers/all-MiniLM-L6-v2` embeddings + a `cross-encoder/ms-marco-MiniLM` reranker. No API, no cost. |
| **Frontend** | Vite + React + TypeScript + Tailwind + TanStack Query. |

Anthropic is **the only external LLM provider**. A configured `ANTHROPIC_API_KEY`
is the opt-in — there is no separate enable flag. With **no key** the app still
runs: LLM calls return nothing and the deterministic query parser + deterministic
scoring stand in (with less semantic nuance). There is **no Groq / OpenRouter
fallback**.

## General query understanding

The query-interpretation call uses a **two-stage** plan pipeline:

```
natural-language query
  → Anthropic  (raw JSON reply)
  → LenientSearchPlan / LenientSearchCriterion   (tolerant transport schema)
  → query_transport.repair_plan()               (STRUCTURAL repair only)
  → ParsedSearchQuery.model_validate()           (the STRICT internal schema)
  → deterministic fact validation                (explicit locations / companies / OR / NOT survive)
  → the search engine
```

**Why the tolerant transport layer exists.** Claude legitimately returns a
semantically correct plan with harmless representation quirks — `operator: null`
(it used `concept`/`values` instead), a missing `id`, `values` as a bare string,
a string `weight`, no `modality`. Feeding that straight into the strict schema
used to fail validation, retry the identical request three times, and then fall
back to the keyword-only parser — losing constraints the model had understood.
`repair_plan` now normalizes representation only (null operator → `ANY_OF`,
cross-fill `value`/`values`/`concept`, synthesize a missing id, numeric weight,
type-alias spelling) and **never invents a value the model did not supply**. The
strict `ParsedSearchQuery` still validates the repaired plan before it reaches
the search engine. A genuine failure (non-JSON, no criteria, provider error)
still falls back to the deterministic parser.

Explicit facts are casing-invariant: *"engineers in Atlanta"*, *"engineers in
atlanta"* and *"ENGINEERS IN ATLANTA"* produce the same required location
constraint (a generic profession/domain denylist, not a city list, keeps *"in
sales"* / *"in leadership"* from being read as places).

## Search behavior (this branch — full Sonnet verification)

- Query interpretation (Anthropic) → deterministic fact validation → full local
  scan of every connection (`<= FULL_SCAN_MAX_CONNECTIONS`)
- **Hard-fact viability gate** — rejects a candidate only on a *verified*
  contradiction (e.g. a required location that clearly conflicts with the known
  one). `UNKNOWN` is never treated as `FALSE`. Everyone else is a **filtered
  candidate**.
- Local deterministic pre-score (stored facts + cached company classification +
  stored semantic representation).
- **Full Sonnet verification** — a complete, evidence-referenced full-profile
  packet is built for **every filtered candidate** and Sonnet reviews the whole
  search plan for each. Batched; a truncated batch is split in half; a candidate
  omitted from a batch is retried alone; a candidate too large to fit is broken
  into per-required-criterion targeted calls. Per criterion the verdict is
  `TRUE` / `FALSE` / `INSUFFICIENT_EVIDENCE` (all *completed* reviews), and a
  separate technical `VERIFICATION_FAILED` when no usable answer was obtained.
- Fact-consistency validator — every verdict + evidence reference is checked
  against that person's exact packet and the locked deterministic facts before
  it can move a score. **Factual criteria (company / location / education /
  certification / language / chronology) stay backend-authoritative.**
- Deterministic rescore → **Exact / Possible / Not-Match** qualification →
  `MIN_MATCH_SCORE` filter → cross-encoder rerank **within** tiers.
- **Display eligibility:** only candidates whose review completed with every
  *required* criterion `TRUE` reach the results. A required
  `INSUFFICIENT_EVIDENCE` excludes the candidate (recorded in metadata, not
  shown).
- **Final audit** (Anthropic) still runs over the shown pool as a last grounded
  correctness check (keep / downgrade / remove only). Because every shown
  candidate is already fully verified it is advisory here — an incomplete audit
  does **not** fail the search.
- Grounded explanations, then the Top `TOP_CONNECTIONS` are persisted.

**If verification cannot complete** for any filtered candidate's required
criteria after bounded recovery, `run_connection_search` raises
`VerificationIncompleteError` → the `/search` endpoint returns
`HTTP 503 {"error":"verification_incomplete","retryable":true}` and **nothing is
persisted** — it is a retryable failure, never a partial result.

Observability (backend metadata / `judge_metadata` block): filtered vs verified
candidate counts, criteria reviewed, batch/split/single-retry/chunk/targeted-call
counts, total LLM calls, model, `excluded_false`, `excluded_insufficient_evidence`.
Invariant enforced at the orchestration layer: a *successful* full-verification
search has `filtered_candidate_count == sonnet_verified_candidate_count`.

## Result correctness

- **Stored facts are authoritative.** The LLM understands the *query*; it does
  not supply *profile* facts.
- **The LLM cannot invent evidence.** Every judge / audit reference is validated
  against the real profile; an invalid one is dropped.
- **`UNKNOWN` ≠ `FALSE`.** Missing evidence never becomes a fabricated match.
- **Exact** = every required criterion is verified true. **Possible** = no
  required criterion is false but at least one required *semantic* criterion is
  still uncertain.
- The **match score** is deterministic application logic. **Data confidence** is
  a separate signal (profile completeness), never mixed into the score.
- If AI verification is only **partial** (the judge or the final audit could not
  fully complete — e.g. no LLM key, provider error, time budget reached), the
  search still returns its conservative deterministic results with a
  *"some AI verification was unavailable; uncertain results are shown
  conservatively"* note. A search is **never** blanked out just because a review
  was incomplete.

## Search universe

Search covers **only the LinkedIn connections you uploaded**. It never scrapes
arbitrary LinkedIn users and it never calls Apify. Reloading a saved search
replays the exact stored response with **zero** LLM / embedding / Apify calls.

## Setup

**Requirements:** Python 3.11, Node + npm.

```bash
git clone <this repo>
cd linkedin-connections-intelligence
cp .env.example backend/.env
```

Edit `backend/.env` and set, at minimum:

```
APIFY_API_TOKEN=...       # https://console.apify.com/account/integrations
ANTHROPIC_API_KEY=...     # https://console.anthropic.com/settings/keys
ANTHROPIC_MODEL=claude-sonnet-4-6
USE_FIXTURES=false        # false = real Apify enrichment
```

`backend/.env` is gitignored and must never be committed — the app reads secrets
only from that local file / real environment variables, never from source.
`ANTHROPIC_WORKSPACE_ID` is only needed for an identity-linked key tied to
multiple workspaces.

### Run it

```bash
.\run.ps1          # Windows: creates backend/.venv, installs both sides, starts both
```

or manually:

```bash
# backend  ->  http://localhost:8010   (OpenAPI docs at /docs)
cd backend
py -3.11 -m venv .venv
.venv/Scripts/pip install -r requirements.txt
.venv/Scripts/python -m uvicorn app.main:app --port 8010
```

```bash
# frontend ->  http://localhost:5182   (proxies /api/* -> http://localhost:8010)
cd frontend
npm install
npm run dev
```

Then open **http://localhost:5182**, upload your `Connections.csv`, click
**Enrich**, and search.

## Required / notable environment variables

See `.env.example` for the full commented list. The ones that matter:

| Variable | Purpose |
|---|---|
| `APIFY_API_TOKEN` | Real profile enrichment (unused when `USE_FIXTURES=true`) |
| `ANTHROPIC_API_KEY` | The only external LLM. Empty = deterministic-only. |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-6` (FULL SONNET VERIFICATION EXPERIMENT) |
| `USE_FIXTURES` | `false` for real Apify; `true` uses local fixture profiles ($0) |
| `DATABASE_URL` | `sqlite:///./data/app.db` (swap for a Postgres DSN to move off SQLite) |
| `SEARCH_LLM_MAX_CALLS` | `0` = unlimited; a positive value soft-caps query-time LLM calls |
| `SEARCH_MAX_SECONDS` | wall-clock budget for the *optional* LLM search stages; `<= 0` disables it |
| `SEMANTIC_JUDGE_MODE` | `all_viable` (default) / `uncertain_only` / `off` |
| `FINAL_RESULT_AUDIT_ENABLED` | `true` (default) |
| `TOP_CONNECTIONS` | result count (default 20) |
| `CANDIDATE_POOL_SIZE`, `MIN_MATCH_SCORE`, `RERANKER_ENABLED`, `PROFILE_TTL_DAYS` | search / enrichment tuning |

## Cost

- **Apify:** pay-per-event, ~`$0.004`/profile (`$4` per 1,000) on the *Profile
  details – no email* tier — a one-time cost per connection, reused until
  `PROFILE_TTL_DAYS` expires. Fixture mode is `$0`.
- **Anthropic:** paid, billed per the model in `ANTHROPIC_MODEL`. Actual spend
  depends on network size and query breadth. `backend/eval/pilot/` has an
  offline harness that estimates call counts before a live run.
- **Local embeddings + reranker:** no API, no cost.

**Cost & latency on this branch.** Full Sonnet verification reviews *every*
filtered candidate. A search with *N* candidates past the hard-fact gate needs
roughly `ceil(N / FULL_VERIFICATION_BATCH_SIZE)` Sonnet calls on the happy path
(default batch size 6 → ~1 call per 6 candidates), plus one interpretation call,
plus a handful more for any batch that truncates/splits, any candidate retried
alone, and any oversized candidate's per-criterion targeted calls, plus the final
audit + one reason call. So a query that filters to ~30 candidates is ~6–12
Sonnet calls; a very broad query on a ~1,000-person network that leaves hundreds
viable can be **many dozens to a few hundred** Sonnet calls and take **several
minutes**. There is no per-query call cap and (with `FULL_VERIFICATION_MAX_SECONDS=0`)
no wall-clock cutoff — verification is mandatory. Use a narrower query, or
`FULL_LLM_VERIFICATION=false`, if that cost is not acceptable.

## Tests

```bash
cd backend  && python -m pytest          # 419 passing
cd frontend && npm test -- --run         # 13 passing
cd frontend && npx tsc --noEmit          # clean
cd frontend && npm run build             # succeeds
```

No test makes a live Apify or Anthropic call — every provider is mocked or the
key is cleared.

## Resuming an interrupted enrichment run

Enrichment is resumable and batched. If the LLM is unavailable mid-run, the
semantic step is deferred for the rest of the run — profiles are still scraped,
normalized, embedded, and marked READY. Click **Resume** (or re-`POST` to
`/datasets/{id}/enrich`) later; it runs a semantic-backfill pass for anyone
missing the current semantic version — **nothing is re-scraped just because
semantics failed**.

`python -m scripts.reset_dataset <dataset_id>` wipes a dataset's derived data
back to PENDING (keeps the CSV rows) if a run went wrong.

## Project structure

```
backend/
  app/
    routers/      FastAPI endpoints (datasets, enrich, people, search, health)
    services/
      llm/        Anthropic client + generic router (retries, circuit breaker)
      query_interpreter.py / query_transport.py / query_facts.py / query_intent.py
      candidate_gate.py / scoring.py / semantic_judge.py / judge_validator.py
      final_auditor.py / reranker.py / reason_generator.py / search_service.py
      enrichment_runner.py / normalize.py / semantic_enrich.py / embeddings.py
      apify_client.py / company_intel.py / export_service.py / …
    models.py / schemas.py / config.py / database.py / repositories.py
  tests/          pytest suite
  eval/pilot/     offline benchmark / call-count estimation harness
  fixtures/       sample CSV + hand-written profile JSON (USE_FIXTURES=true)
  scripts/        one-off maintenance scripts (reset / backfill)
frontend/
  src/pages/      Upload / Enrichment / Dashboard / Search / Results
  src/components/ ResultCard, shared UI
  src/api/        typed client + response types
```

## Safety / data behavior

- `backend/.env` is never committed; the app reads secrets only from it / real env vars.
- All raw and derived profile data is stored **locally** in `backend/data/app.db`.
- Search operates only on **your uploaded connections** — no arbitrary LinkedIn scraping.
- Search makes **zero** Apify calls. A saved-search reload makes **zero**
  external calls of any kind.

## Current limitations

- Local / single-user MVP: SQLite, no auth, no multi-tenancy.
- Windows-first tooling (`run.ps1`); the manual commands work on any OS.
- The Anthropic client uses the pre-Claude-5 request shape; `ANTHROPIC_MODEL` defaults to `claude-sonnet-4-6` on this branch. If your account/model needs the newer request shape, adjust `llm/anthropic_client.py`.
- A broad semantic query over a large network is not fast and can consume a
  meaningful number of Anthropic calls.
