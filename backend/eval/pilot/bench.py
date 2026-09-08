"""PHASE A — Anthropic model benchmark (EVAL-ONLY — never changes production).

Sweeps ``settings.anthropic_model`` over a fixed candidate set and runs the REAL
search pipeline (``recorder.run_query`` / ``run_connection_search``) unchanged,
Anthropic-only (Groq + OpenRouter keys are cleared for the duration so every
stage is measured on the model under test — never a fallback provider).

    python -m eval.pilot.bench all --max-calls 120 --max-cost 4.0

Three phases:
  1. query-interpretation probe   — all 3 candidates x 7 queries (cheap, isolates
     structured-output reliability: first-call valid JSON, null-field failures,
     retries, criteria shape)
  2. full-pipeline run            — haiku-4.5 + sonnet-5 x 7 queries against the
     40-profile pilot.db (judge / audit / reason latency, truncations, grounding)
  3. real-dataset regression      — the 2 realistic finalists x "nonprofit /
     Chicago" against a COPY of data/app.db (991 profiles, read only)

HARD budget guard: the moment cumulative live generation calls reach
``--max-calls`` OR estimated cost reaches ``--max-cost``, the next Anthropic
request raises ``BudgetExceeded``, the run stops, and everything gathered so far
is written to ``eval/pilot/results/bench_<ts>.json`` + ``.md``.

Nothing here scrapes, touches production data, calls Apify/Groq, or edits any
production module. The API key is never printed.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.config import settings

PILOT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PILOT_DIR / "results"
MAIN_DATASET_ID = "dataset_0ba27cae09d4"  # suraj_1000_connections

#: $/MTok (input, output) — Anthropic public pricing, fetched 2026-09-08.
PRICING: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-fable-5-1": (10.0, 50.0),
    "claude-opus-5": (5.0, 25.0),
}

FULL_PIPELINE_MODELS = ["claude-haiku-4-5-20251001", "claude-sonnet-5"]
INTERP_ONLY_MODELS = ["claude-fable-5-1"]
FINALIST_MODELS = ["claude-haiku-4-5-20251001", "claude-sonnet-5"]

BENCH_QUERIES: list[tuple[str, str]] = [
    ("bq1_nonprofit_chicago", "People with nonprofit experience in Chicago"),
    ("bq2_research_industry", "people with research plus industry experience"),
    ("bq3_eng_mentors_tech", "senior engineering mentors in tech"),
    ("bq4_ex_amazon_startups", "former Amazon employees now working at startups"),
    ("bq5_swe_fintech", "software engineers at fintech companies"),
    ("bq6_cyber_healthcare", "people with cybersecurity and healthcare experience"),
    ("bq7_backend_mgmt_mentor",
     "people who could mentor a backend engineer moving into management"),
]

REGRESSION_QUERY = "People with nonprofit experience in Chicago"
#: benchmark expectation ONLY — never used in application logic. Matched on the
#: last name token, case-insensitive.
REGRESSION_NAMES = ["Connor Donnelly", "Jack DelloStritto", "Guarocuy Frank Batista-Kunhardt"]


class BudgetExceeded(Exception):
    """Raised by the metering proxy when a hard limit would be crossed."""


# ─────────────────────────── metering ───────────────────────────


@dataclass
class Meter:
    max_calls: int
    max_cost: float
    calls: int = 0
    cost_usd: float = 0.0
    current_op: str = "?"
    current_model: str = "?"
    call_log: list[dict] = field(default_factory=list)          # every raw Anthropic call
    raw_by_op: dict[str, list[dict]] = field(default_factory=dict)   # op -> [{dict, stop_reason}]
    stopped_reason: str | None = None

    def price(self, model: str, t_in: int, t_out: int) -> float:
        p_in, p_out = PRICING.get(model, (2.0, 10.0))
        return t_in / 1e6 * p_in + t_out / 1e6 * p_out

    def before_call(self) -> None:
        if self.calls >= self.max_calls:
            self.stopped_reason = f"call cap reached ({self.calls}/{self.max_calls})"
            raise BudgetExceeded(self.stopped_reason)
        if self.cost_usd >= self.max_cost:
            self.stopped_reason = f"cost cap reached (${self.cost_usd:.3f}/${self.max_cost:.2f})"
            raise BudgetExceeded(self.stopped_reason)

    def note(self, t_in: int, t_out: int, ms: float, stop_reason: str | None, raw: dict | None) -> None:
        self.calls += 1
        c = self.price(self.current_model, t_in, t_out)
        self.cost_usd += c
        self.call_log.append({
            "n": self.calls, "op": self.current_op, "model": self.current_model,
            "in": t_in, "out": t_out, "ms": round(ms, 1), "cost": round(c, 6),
            "stop_reason": stop_reason,
        })
        self.raw_by_op.setdefault(self.current_op, []).append({"dict": raw, "stop_reason": stop_reason})


_METER: Meter | None = None


class _HttpxProxy:
    """Wraps the ``httpx`` module object used by ``anthropic_client`` — only
    ``.post`` is intercepted (for metering + the hard budget guard); every other
    attribute (``TimeoutException`` etc.) delegates to the real module."""

    def __init__(self, real):
        object.__setattr__(self, "_real", real)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_real"), name)

    def post(self, url, **kw):
        m = _METER
        if m is not None:
            m.before_call()
        t0 = time.perf_counter()
        resp = object.__getattribute__(self, "_real").post(url, **kw)
        dt = (time.perf_counter() - t0) * 1000.0
        if m is not None:
            t_in = t_out = 0
            stop = None
            raw = None
            try:
                body = resp.json()
                usage = body.get("usage") or {}
                t_in = int(usage.get("input_tokens") or 0)
                t_out = int(usage.get("output_tokens") or 0)
                stop = body.get("stop_reason")
                txt = "".join(
                    bl.get("text", "") for bl in body.get("content", []) if bl.get("type") == "text"
                )
                full = "{" + txt if not txt.lstrip().startswith("{") else txt
                try:
                    from app.services.llm.openai_compatible import _extract_json

                    raw = _extract_json(full)
                except Exception:  # noqa: BLE001
                    raw = {"_parse_error": True}
            except Exception:  # noqa: BLE001  — non-JSON error body etc.
                pass
            m.note(t_in, t_out, dt, stop, raw)
        return resp


@contextlib.contextmanager
def _meter_httpx():
    import app.services.llm.anthropic_client as ac

    real = ac.httpx
    ac.httpx = _HttpxProxy(real)
    try:
        yield
    finally:
        ac.httpx = real


_OP_RECORDS: list[dict] = []


@contextlib.contextmanager
def _op_capture():
    """Wrap ``generate_structured`` in every module that imported it, to record
    per-operation wall time + how many raw Anthropic calls it consumed (>1 ⇒ a
    retry happened)."""
    from app.services.llm.router import generate_structured as real_gs

    targets = []
    for name in (
        "app.services.llm.router", "app.services.query_interpreter",
        "app.services.semantic_judge", "app.services.semantic_llm",
        "app.services.reason_generator", "app.services.final_auditor",
    ):
        mod = sys.modules.get(name)
        if mod is not None and hasattr(mod, "generate_structured"):
            targets.append((mod, mod.generate_structured))

    def wrapper(*a, **kw):
        op = kw.get("operation", "unspecified")
        m = _METER
        prev = m.current_op if m else "?"
        if m:
            m.current_op = op
        c0 = m.calls if m else 0
        t0 = time.perf_counter()
        ok = False
        try:
            res = real_gs(*a, **kw)
            ok = res is not None and (res[0] is not None if isinstance(res, tuple) else True)
            return res
        finally:
            if m:
                _OP_RECORDS.append({
                    "op": op, "model": m.current_model,
                    "ms": round((time.perf_counter() - t0) * 1000.0, 1),
                    "calls": m.calls - c0, "ok": ok,
                })
                m.current_op = prev

    for mod, _orig in targets:
        mod.generate_structured = wrapper
    try:
        yield
    finally:
        for mod, orig in targets:
            mod.generate_structured = orig


@contextlib.contextmanager
def _anthropic_only():
    saved = (settings.groq_api_key, settings.openrouter_api_key)
    settings.groq_api_key = ""
    settings.openrouter_api_key = ""
    try:
        yield
    finally:
        settings.groq_api_key, settings.openrouter_api_key = saved


@contextlib.contextmanager
def _v5_client():
    """Route the Anthropic provider through the benchmark-only Claude 5 client
    (no prefill / no temperature). Production ``anthropic_client.py`` untouched."""
    import app.services.llm.anthropic_client as ac
    import app.services.llm.providers as pr

    from eval.pilot.bench_anthropic_v5 import messages_json_v5

    saved = (getattr(ac, "messages_json", None), getattr(pr, "messages_json", None))
    ac.messages_json = messages_json_v5
    pr.messages_json = messages_json_v5
    try:
        yield
    finally:
        if saved[0] is not None:
            ac.messages_json = saved[0]
        if saved[1] is not None:
            pr.messages_json = saved[1]


@contextlib.contextmanager
def _use_model(model: str):
    from app.services.llm import circuit

    saved = settings.anthropic_model
    settings.anthropic_model = model
    circuit.reset_all()
    if _METER:
        _METER.current_model = model
    try:
        yield
    finally:
        settings.anthropic_model = saved
        circuit.reset_all()


# ─────────────────────────── analysis helpers ───────────────────────────


def _classify_interpretation(raw_dicts: list[dict]) -> dict:
    """First-attempt structured-output reliability from the raw model JSON
    (before any router retry / backend repair)."""
    from pydantic import ValidationError

    from app.schemas import ParsedSearchQuery

    out = {
        "raw_attempts": len(raw_dicts), "first_call_valid_json": None,
        "null_field_failure": False, "parse_error": False, "schema_errors": None,
    }
    if not raw_dicts:
        return out
    d0 = raw_dicts[0]["dict"] or {}
    if d0.get("_parse_error"):
        out["parse_error"] = True
        out["first_call_valid_json"] = False
        return out
    try:
        ParsedSearchQuery.model_validate(d0)
        out["first_call_valid_json"] = True
    except ValidationError as e:
        out["first_call_valid_json"] = False
        errs = e.errors()
        out["schema_errors"] = "; ".join(
            f"{'.'.join(str(x) for x in er.get('loc', ()))}:{er.get('type')}" for er in errs[:8]
        )
        out["null_field_failure"] = any(
            er.get("input") is None
            and ("string_type" in str(er.get("type")) or str(er.get("loc", ("",))[-1]) == "value")
            for er in errs
        )
    except Exception:  # noqa: BLE001
        out["first_call_valid_json"] = False
        out["parse_error"] = True
    return out


_MEANING_TYPES = {"semantic_concept", "professional_concept", "industry_experience",
                  "role_function", "company_category", "concept", "leadership", "mentorship"}


def _dup_pairs_from_criteria(crits: list) -> list[str]:
    """Semantic-dimension overlap between two criteria (generic — reuses the
    production ``concept_overlap`` metric, no query words). Used on both the raw
    model output and the backend's post-repair plan."""
    from app.services.matching import concept_overlap

    texts = []
    for c in crits:
        if not isinstance(c, dict):
            continue
        t = str(c.get("type") or "").lower()
        if t in _MEANING_TYPES:
            txt = c.get("concept") or c.get("value") or " ".join(c.get("values") or [])
            texts.append((c.get("id") or t, t, str(txt)))
    pairs = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            ov = concept_overlap(texts[i][2], texts[j][2])
            if ov >= 0.6:
                pairs.append(f"{texts[i][0]}({texts[i][1]}) ~ {texts[j][0]}({texts[j][1]}) overlap={ov:.2f}")
    return pairs


def _raw_duplicate_pairs(raw_dicts: list[dict]) -> list[str]:
    if not raw_dicts:
        return []
    return _dup_pairs_from_criteria((raw_dicts[0]["dict"] or {}).get("criteria") or [])


def _slice_calls(before: dict[str, int]) -> dict[str, list[dict]]:
    m = _METER
    out: dict[str, list[dict]] = {}
    seen: dict[str, int] = {}
    for rec in m.call_log:
        seen[rec["op"]] = seen.get(rec["op"], 0) + 1
        if seen[rec["op"]] > before.get(rec["op"], 0):
            out.setdefault(rec["op"], []).append(rec)
    return out


def _op_counts_snapshot() -> dict[str, int]:
    m = _METER
    c: dict[str, int] = {}
    for rec in m.call_log:
        c[rec["op"]] = c.get(rec["op"], 0) + 1
    return c


def _stage_agg(recs: list[dict]) -> dict:
    return {
        "calls": len(recs),
        "ms": round(sum(r["ms"] for r in recs), 1),
        "in": sum(r["in"] for r in recs),
        "out": sum(r["out"] for r in recs),
        "truncated": sum(1 for r in recs if r["stop_reason"] == "max_tokens"),
    }


# ─────────────────────────── phase drivers ───────────────────────────


def _interp_probe(model: str, qid: str, query: str) -> dict:
    from app.services.query_interpreter import interpret_query

    m = _METER
    raw0 = len(m.raw_by_op.get("query_interpretation", []))
    before = _op_counts_snapshot()
    t0 = time.perf_counter()
    parsed, provider, model_id = interpret_query(query)
    wall = (time.perf_counter() - t0) * 1000.0
    raws = m.raw_by_op.get("query_interpretation", [])[raw0:]
    sliced = _slice_calls(before).get("query_interpretation", [])
    reliab = _classify_interpretation(raws)
    post_crits = [
        {"id": c.id, "type": c.type, "concept": c.concept, "value": c.value, "values": c.values}
        for c in parsed.criteria
    ]
    return {
        "model": model, "query_id": qid, "query": query,
        "provider": provider, "model_id": model_id,
        "fell_back_to_deterministic": "determ" in (provider or ""),
        "latency_ms": round(wall, 1),
        "anthropic_calls": len(sliced),
        "retries": max(0, len(sliced) - 1),
        "in_tokens": sum(r["in"] for r in sliced),
        "out_tokens": sum(r["out"] for r in sliced),
        "cost": round(sum(r["cost"] for r in sliced), 6),
        **reliab,
        "raw_duplicate_pairs": _raw_duplicate_pairs(raws),
        "post_repair_duplicate_pairs": _dup_pairs_from_criteria(post_crits),
        "intent": parsed.intent,
        "n_criteria": len(parsed.criteria),
        "n_required": sum(1 for c in parsed.criteria if c.required),
        "criteria": [
            {"id": c.id, "type": c.type, "value": c.value, "values": c.values,
             "concept": c.concept, "operator": c.operator, "scope": c.scope,
             "required": c.required, "modality": c.modality, "weight": round(c.weight, 1)}
            for c in parsed.criteria
        ],
    }


def _pipeline_run(db, model: str, qid: str, query: str) -> dict:
    from eval.pilot.recorder import run_query

    m = _METER
    before = _op_counts_snapshot()
    c0, cost0 = m.calls, m.cost_usd
    t0 = time.perf_counter()
    rec = run_query(db, {"id": qid, "query": query, "group": "bench"},
                    offline=False, reasons_enabled=True)
    wall = (time.perf_counter() - t0) * 1000.0
    d = rec.as_dict()
    sliced = _slice_calls(before)
    interp_raw = sliced.get("query_interpretation", [])
    raw0 = len(m.raw_by_op.get("query_interpretation", [])) - len(interp_raw)
    interp_reliab = _classify_interpretation(
        m.raw_by_op.get("query_interpretation", [])[max(0, raw0):]
    )
    results = d["results"]
    jm = d.get("judge_metadata") or {}
    am = d.get("audit_metadata") or {}
    jt = d.get("judge_trace") or {}
    return {
        "model": model, "query_id": qid, "query": query,
        "wall_ms": round(wall, 1),
        "total_anthropic_calls": m.calls - c0,
        "total_cost": round(m.cost_usd - cost0, 6),
        "stage": {op: _stage_agg(sliced.get(op, [])) for op in (
            "query_interpretation", "semantic_judge", "final_result_audit", "reason_generation")},
        "interpretation_reliability": interp_reliab,
        "interpretation": d["interpretation"],
        "raw_duplicate_pairs": _raw_duplicate_pairs(
            m.raw_by_op.get("query_interpretation", [])[max(0, raw0):]),
        "funnel": d["funnel"],
        "judge": {
            "status": jm.get("status"), "batch_count": jm.get("judge_batch_count"),
            "successful": jm.get("judge_successful_batches"), "failed": jm.get("judge_failed_batches"),
            "truncations": jm.get("truncations"), "adaptive_splits": jm.get("adaptive_splits"),
            "candidates": jm.get("judge_candidate_count"),
            "providers": jm.get("providers"), "models": jm.get("models"),
        },
        "judge_trace": {
            "people_judged": jt.get("people_judged"),
            "validator_downgrades": jt.get("validator_downgrades"),
            "grounding_downgrades": jt.get("grounding_downgrades"),
            "validator_dropped": jt.get("validator_dropped"),
            "invalid_evidence_refs": jt.get("invalid_evidence_refs"),
            "wrong_scope_refs": jt.get("wrong_scope_refs"),
            "missing_criterion_verdicts": jt.get("missing_criterion_verdicts"),
            "judgeable_criteria_expected": jt.get("judgeable_criteria_expected"),
            "required_semantic": {
                "true": jt.get("required_semantic_true"), "false": jt.get("required_semantic_false"),
                "unknown": jt.get("required_semantic_unknown"),
                "unknown_rate": jt.get("required_semantic_unknown_rate"),
            },
        },
        "audit": {
            "status": am.get("status"), "batch_count": am.get("batch_count"),
            "successful": am.get("successful_batches"), "failed": am.get("failed_batches"),
            "truncations": am.get("truncations"), "audited": am.get("audited_candidates"),
            "requested": am.get("requested_candidates"),
            "approved": am.get("approved"), "downgraded": am.get("downgraded"),
            "incorrect": am.get("incorrect"), "unknown": am.get("unknown"),
            "missing_required_reviews": am.get("missing_required_reviews"),
            "candidates_with_incomplete_reviews": am.get("candidates_with_incomplete_reviews"),
        },
        "audit_transitions": d.get("audit_changes"),
        "exact": d["funnel"].get("exact"), "possible": d["funnel"].get("possible"),
        "near": len(d.get("near_matches") or []),
        "n_results": len(results),
        "llm_verified_results": sum(1 for r in results if r.get("llm_verified")),
        "results": [
            {"rank": r["rank"], "name": r["name"], "person_id": r["person_id"],
             "qualification": r["qualification"], "match_score": r["match_score"],
             "llm_verified": r.get("llm_verified"), "audit_decision": r.get("audit_decision")}
            for r in results
        ],
        "llm_provider": d.get("llm_provider"), "llm_model": d.get("llm_model"),
    }


def _warm_local_models() -> dict:
    out = {}
    t0 = time.perf_counter()
    try:
        if settings.embeddings_enabled:
            from app.services.embeddings import _get_model as _em

            _em()
        out["embedding_cold_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
    except Exception as e:  # noqa: BLE001
        out["embedding_cold_ms"] = None
        out["embedding_error"] = type(e).__name__
    t0 = time.perf_counter()
    try:
        from app.services.reranker import _get_model as _rm

        _rm()
        out["reranker_cold_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
    except Exception as e:  # noqa: BLE001
        out["reranker_cold_ms"] = None
        out["reranker_error"] = type(e).__name__
    return out


def _dataset_regression(model: str, query: str) -> dict:
    src = settings.database_url.replace("sqlite:///", "", 1)
    src_path = (Path.cwd() / src).resolve() if not Path(src).is_absolute() else Path(src)
    tmp = RESULTS_DIR / f"_bench991_{model.replace('/', '_')}_{int(time.time())}.db"
    shutil.copy2(src_path, tmp)
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app.services.search_service import run_connection_search

    m = _METER
    before = _op_counts_snapshot()
    c0, cost0 = m.calls, m.cost_usd
    eng = create_engine(f"sqlite:///{tmp.as_posix()}", future=True)
    S = sessionmaker(bind=eng, autoflush=False, expire_on_commit=False, future=True)
    t0 = time.perf_counter()
    try:
        with S() as db:
            resp = run_connection_search(db, dataset_id=MAIN_DATASET_ID, query=query)
            db.rollback()
    finally:
        eng.dispose()
        with contextlib.suppress(OSError):
            tmp.unlink()
    wall = (time.perf_counter() - t0) * 1000.0
    r = resp.model_dump()
    results = r["connections"]["results"]
    sliced = _slice_calls(before)
    jm = r.get("judge_metadata") or {}
    am = r.get("audit_metadata") or {}
    lc = r.get("llm_calls") or {}

    def _find(name: str):
        last = name.split()[-1].lower()
        for x in results:
            if last in (x.get("name") or "").lower():
                return {"rank": x["rank"], "score": x["match_score"],
                        "qualification": x["qualification"], "llm_verified": x.get("llm_verified"),
                        "audit_decision": x.get("audit_decision")}
        for x in r["connections"].get("near_matches") or []:
            if last in (x.get("name") or "").lower():
                return {"near_match": True, "unmet": x.get("unmet_criteria")}
        return None

    anth_ms = sum(rr["ms"] for op in sliced.values() for rr in op)
    return {
        "model": model, "query": query,
        "wall_ms": round(wall, 1),
        "anthropic_latency_ms": round(anth_ms, 1),
        "anthropic_calls": m.calls - c0,
        "cost": round(m.cost_usd - cost0, 6),
        "stage_calls": {op: _stage_agg(v) for op, v in sliced.items()},
        "profile_timings_ms": (lc.get("profile") or {}).get("timings_ms"),
        "interpretation_provider": r.get("llm_provider"),
        "interpretation_model": r.get("llm_model"),
        "judge_status": jm.get("status"),
        "judge_batches": jm.get("judge_batch_count"),
        "judge_truncations": jm.get("truncations"),
        "judge_splits": jm.get("adaptive_splits"),
        "judge_candidates": jm.get("judge_candidate_count"),
        "audit_status": am.get("status"),
        "audit_completed": am.get("status") in ("full", "partial") and not am.get("deadline_reached"),
        "audit_missing_required_reviews": am.get("missing_required_reviews"),
        "deadline_reached": (lc.get("deadline") or {}).get("reached"),
        "exact": r["connections"].get("exact_match_count"),
        "possible": r["connections"].get("possible_match_count"),
        "returned": r["connections"].get("returned"),
        "near": len(r["connections"].get("near_matches") or []),
        "llm_verified_results": sum(1 for x in results if x.get("llm_verified")),
        "targets": {name: _find(name) for name in REGRESSION_NAMES},
        "top15": [
            {"rank": x["rank"], "name": x["name"], "score": x["match_score"],
             "qualification": x["qualification"], "llm_verified": x.get("llm_verified")}
            for x in results[:15]
        ],
    }


# ─────────────────────────── orchestration ───────────────────────────


def _checkpoint(doc: dict, ts: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    p = RESULTS_DIR / f"bench_{ts}.json"
    p.write_text(json.dumps(doc, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return p


def _summary_line(m: Meter) -> str:
    return (f"[budget] anthropic_calls={m.calls}/{m.max_calls}  "
            f"est_cost=${m.cost_usd:.3f}/${m.max_cost:.2f}"
            + (f"  STOPPED: {m.stopped_reason}" if m.stopped_reason else ""))


def cmd_all(args) -> None:
    global _METER
    _METER = Meter(max_calls=args.max_calls, max_cost=args.max_cost)
    ts = time.strftime("%Y%m%d_%H%M%S")
    pilot_db = Path(args.pilot_db)
    if not pilot_db.exists():
        print(f"pilot DB missing: {pilot_db}  — run `python -m eval.pilot.run_pilot isolate` first")
        return

    doc: dict = {
        "phase": "A",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "pricing_usd_per_mtok": PRICING,
        "hard_limits": {"max_calls": args.max_calls, "max_cost_usd": args.max_cost},
        "queries": [{"id": qid, "query": q} for qid, q in BENCH_QUERIES],
        "local_model_warmup": {},
        "interpretation_probe": [],
        "pipeline_runs": [],
        "regression_991": [],
        "budget_final": {},
        "call_log": [],
        "notes": [],
    }

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    eng = create_engine(f"sqlite:///{pilot_db.as_posix()}", future=True)
    Session = sessionmaker(bind=eng, autoflush=False, expire_on_commit=False, future=True)

    try:
        with _meter_httpx(), _op_capture(), _anthropic_only():
            doc["local_model_warmup"] = _warm_local_models()
            print(f"warmup: {doc['local_model_warmup']}")
            _checkpoint(doc, ts)

            # ── phase 1 — interpretation-only probe: Fable 5.1 (per the approved
            #    plan — Haiku/Sonnet interpretation reliability is extracted from
            #    their phase-3 pipeline runs, same mechanism, no extra calls) ──
            for model in INTERP_ONLY_MODELS:
                for qid, q in BENCH_QUERIES:
                    with _use_model(model):
                        try:
                            rec = _interp_probe(model, qid, q)
                        except BudgetExceeded as e:
                            doc["notes"].append(f"phase1 stopped at {model}/{qid}: {e}")
                            raise
                    doc["interpretation_probe"].append(rec)
                    print(f"  interp {model:28s} {qid:24s} "
                          f"prov={rec['provider']:15s} first_json={rec['first_call_valid_json']} "
                          f"retries={rec['retries']} {rec['latency_ms']:.0f}ms  {_summary_line(_METER)}")
                    _checkpoint(doc, ts)

            # ── phase 2 — real 991-profile regression (decision-critical: run
            #    BEFORE the pilot pipeline detail so budget risk truncates the
            #    least-important data, not this) ──
            for model in FINALIST_MODELS:
                with _use_model(model):
                    try:
                        rec = _dataset_regression(model, REGRESSION_QUERY)
                    except BudgetExceeded as e:
                        doc["notes"].append(f"phase2(991) stopped at {model}: {e}")
                        raise
                doc["regression_991"].append(rec)
                print(f"  991    {model:28s} exact={rec['exact']} poss={rec['possible']} "
                      f"calls={rec['anthropic_calls']} audit={rec['audit_status']} "
                      f"{rec['wall_ms']:.0f}ms  {_summary_line(_METER)}")
                _checkpoint(doc, ts)

            # ── phase 3 — full pilot pipeline (haiku + sonnet) ──
            with Session() as db:
                for model in FULL_PIPELINE_MODELS:
                    for qid, q in BENCH_QUERIES:
                        with _use_model(model):
                            try:
                                rec = _pipeline_run(db, model, qid, q)
                            except BudgetExceeded as e:
                                doc["notes"].append(f"phase3(pipeline) stopped at {model}/{qid}: {e}")
                                raise
                        doc["pipeline_runs"].append(rec)
                        print(f"  pipe   {model:28s} {qid:24s} "
                              f"exact={rec['exact']} poss={rec['possible']} "
                              f"jtrunc={rec['judge']['truncations']} "
                              f"verified={rec['llm_verified_results']}/{rec['n_results']} "
                              f"{rec['wall_ms']:.0f}ms  {_summary_line(_METER)}")
                        _checkpoint(doc, ts)

    except BudgetExceeded as e:
        doc["notes"].append(f"HARD BUDGET STOP: {e}")
        print(f"\n*** HARD BUDGET STOP: {e} — partial results saved ***")
    except KeyboardInterrupt:
        doc["notes"].append("interrupted by user")
        print("\ninterrupted — partial results saved")
    finally:
        eng.dispose()
        doc["budget_final"] = {
            "anthropic_calls": _METER.calls, "estimated_cost_usd": round(_METER.cost_usd, 4),
            "stopped_reason": _METER.stopped_reason,
        }
        doc["call_log"] = _METER.call_log
        doc["op_records"] = _OP_RECORDS
        p = _checkpoint(doc, ts)
        _write_md(doc, ts)
        print(f"\n{_summary_line(_METER)}")
        print(f"wrote {p}")
        print(f"wrote {p.with_suffix('.md')}")


def _write_md(doc: dict, ts: str) -> None:
    L: list[str] = [f"# Anthropic model benchmark — Phase A — {doc['started_at']}", ""]
    L.append(f"- hard limits: {doc['hard_limits']}")
    L.append(f"- budget final: {doc['budget_final']}")
    L.append(f"- local model warmup (cold-start): {doc['local_model_warmup']}")
    if doc.get("notes"):
        L.append(f"- notes: {doc['notes']}")
    L.append("")

    L.append("## Query interpretation probe")
    L.append("")
    L.append("| model | query | provider | 1st-call valid JSON | null-field fail | retries | latency ms | in tok | out tok | crit (req) | raw dup pairs |")
    L.append("|---|---|---|---|---|---|--:|--:|--:|---|---|")
    for r in doc["interpretation_probe"]:
        L.append(f"| {r['model']} | {r['query_id']} | {r['provider']} | {r['first_call_valid_json']} "
                 f"| {r['null_field_failure']} | {r['retries']} | {r['latency_ms']:.0f} "
                 f"| {r['in_tokens']} | {r['out_tokens']} | {r['n_criteria']} ({r['n_required']}) "
                 f"| {len(r['raw_duplicate_pairs'])} |")
    L.append("")
    for r in doc["interpretation_probe"]:
        L.append(f"### {r['model']} — {r['query_id']} — \"{r['query']}\"")
        if r.get("schema_errors"):
            L.append(f"- schema errors (1st call): `{r['schema_errors']}`")
        if r["raw_duplicate_pairs"]:
            L.append(f"- raw duplicate pairs: {r['raw_duplicate_pairs']}")
        L.append(f"- intent: `{r['intent']}`")
        for c in r["criteria"]:
            L.append(f"  - `{c['type']}` {c['concept'] or c['value']} "
                     f"| op={c['operator']} scope={c['scope']} required={c['required']} "
                     f"modality={c['modality']} w={c['weight']}")
        L.append("")

    L.append("## Full pipeline runs (pilot.db, 40 profiles)")
    L.append("")
    L.append("| model | query | exact | poss | near | verified/res | judge trunc/split | "
             "inval refs | wrong-scope | audit miss-req | audit status | wall ms | cost |")
    L.append("|---|---|--:|--:|--:|--:|---|--:|--:|--:|---|--:|--:|")
    for r in doc["pipeline_runs"]:
        jt = r["judge_trace"]
        L.append(f"| {r['model']} | {r['query_id']} | {r['exact']} | {r['possible']} | {r['near']} "
                 f"| {r['llm_verified_results']}/{r['n_results']} "
                 f"| {r['judge']['truncations']}/{r['judge']['adaptive_splits'] if 'adaptive_splits' in r['judge'] else r['judge'].get('adaptive_splits')} "
                 f"| {jt['invalid_evidence_refs']} | {jt['wrong_scope_refs']} "
                 f"| {r['audit']['missing_required_reviews']} | {r['audit']['status']} "
                 f"| {r['wall_ms']:.0f} | ${r['total_cost']:.4f} |")
    L.append("")
    for r in doc["pipeline_runs"]:
        L.append(f"### {r['model']} — {r['query_id']}")
        L.append(f"- stages: {json.dumps(r['stage'])}")
        L.append(f"- interpretation reliability: {json.dumps(r['interpretation_reliability'])}")
        if r["raw_duplicate_pairs"]:
            L.append(f"- raw duplicate criteria: {r['raw_duplicate_pairs']}")
        L.append(f"- judge: {json.dumps(r['judge'])}")
        L.append(f"- judge_trace: {json.dumps(r['judge_trace'])}")
        L.append(f"- audit: {json.dumps(r['audit'])}")
        L.append(f"- results:")
        for x in r["results"]:
            L.append(f"  {x['rank']}. {x['name']} — {x['qualification']} "
                     f"score={x['match_score']} verified={x['llm_verified']} audit={x['audit_decision']}")
        L.append("")

    L.append("## Real 991-profile regression — \"People with nonprofit experience in Chicago\"")
    L.append("")
    for r in doc["regression_991"]:
        L.append(f"### {r['model']}")
        L.append(f"- wall {r['wall_ms']:.0f}ms · anthropic {r['anthropic_latency_ms']:.0f}ms · "
                 f"calls {r['anthropic_calls']} · cost ${r['cost']:.4f}")
        L.append(f"- interpretation via {r['interpretation_provider']} ({r['interpretation_model']})")
        L.append(f"- judge {r['judge_status']} batches={r['judge_batches']} trunc={r['judge_truncations']} "
                 f"splits={r['judge_splits']} candidates={r['judge_candidates']}")
        L.append(f"- audit {r['audit_status']} completed={r['audit_completed']} "
                 f"missing_required={r['audit_missing_required_reviews']} deadline_reached={r['deadline_reached']}")
        L.append(f"- exact={r['exact']} possible={r['possible']} returned={r['returned']} near={r['near']} "
                 f"verified_results={r['llm_verified_results']}")
        L.append(f"- profile stage timings: {r['profile_timings_ms']}")
        for name, hit in r["targets"].items():
            L.append(f"  - **{name}**: {hit}")
        L.append(f"- top 15:")
        for x in r["top15"]:
            L.append(f"    {x['rank']}. {x['name']} — {x['qualification']} score={x['score']} "
                     f"verified={x['llm_verified']}")
        L.append("")

    (RESULTS_DIR / f"bench_{ts}.md").write_text("\n".join(L), encoding="utf-8")


def cmd_probe(args) -> None:
    """Validate the Claude 5 shim BEFORE spending the round-2 budget."""
    global _METER
    _METER = Meter(max_calls=args.max_calls, max_cost=args.max_cost)
    _METER.calls = args.start_calls
    _METER.cost_usd = args.start_cost
    model = "claude-sonnet-5"
    out: dict = {"model": model}
    with _meter_httpx(), _op_capture(), _anthropic_only(), _v5_client(), _use_model(model):
        from eval.pilot.bench_anthropic_v5 import messages_json_v5

        # 1 — raw shim call
        try:
            raw = messages_json_v5(
                api_key=settings.anthropic_api_key, model=model,
                system_prompt="You return JSON only.",
                user_prompt='Return exactly {"ok": true, "n": 3}.',
                max_tokens=100, timeout=30.0,
            )
            out["raw_call"] = {"http_200": True, "json_extracted": raw,
                               "prefill_sent": False, "temperature_sent": False}
        except Exception as e:  # noqa: BLE001
            out["raw_call"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            print(json.dumps(out, indent=2, default=str))
            print("\n*** SHIM VALIDATION FAILED — stopping, no benchmark budget spent ***")
            return
        out["raw_call"]["usage_captured"] = bool(_METER.call_log) and _METER.call_log[-1]["in"] > 0
        out["raw_call"]["last_call"] = _METER.call_log[-1] if _METER.call_log else None

        # 2 — one real interpretation
        try:
            rec = _interp_probe(model, "probe_nonprofit_chicago",
                                "People with nonprofit experience in Chicago")
            out["interpretation"] = rec
        except Exception as e:  # noqa: BLE001
            out["interpretation"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            print(json.dumps(out, indent=2, default=str))
            print("\n*** SHIM INTERPRETATION FAILED — stopping ***")
            return

    r = out["interpretation"]
    ok = (not r["fell_back_to_deterministic"]) and r["provider"].startswith("anthropic") and r["n_criteria"] >= 1
    out["VALIDATION_PASSED"] = bool(ok and out["raw_call"].get("json_extracted"))
    print(json.dumps(out, indent=2, default=str))
    print(f"\nVALIDATION_PASSED = {out['VALIDATION_PASSED']}   {_summary_line(_METER)}")


def cmd_round2(args) -> None:
    """Round 2 — Sonnet 5 full pipeline + Sonnet 5 on the 991 dataset, through
    the Claude 5 shim. Haiku 4.5 data is reused from round 1."""
    global _METER
    _METER = Meter(max_calls=args.max_calls, max_cost=args.max_cost)
    _METER.calls = args.start_calls
    _METER.cost_usd = args.start_cost
    ts = time.strftime("%Y%m%d_%H%M%S")
    pilot_db = Path(args.pilot_db)

    doc: dict = {
        "phase": "A-round2", "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "shim": "eval.pilot.bench_anthropic_v5 (no prefill, no temperature)",
        "round1_json": args.round1_json,
        "seed_budget": {"start_calls": args.start_calls, "start_cost_usd": args.start_cost},
        "hard_limits": {"max_calls": args.max_calls, "max_cost_usd": args.max_cost},
        "pricing_usd_per_mtok": PRICING,
        "local_model_warmup": {}, "pipeline_runs": [], "regression_991": [],
        "haiku_spotcheck_on_shim": [], "budget_final": {}, "call_log": [], "notes": [],
    }

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    eng = create_engine(f"sqlite:///{pilot_db.as_posix()}", future=True)
    Session = sessionmaker(bind=eng, autoflush=False, expire_on_commit=False, future=True)

    try:
        with _meter_httpx(), _op_capture(), _anthropic_only(), _v5_client():
            doc["local_model_warmup"] = _warm_local_models()
            print(f"warmup: {doc['local_model_warmup']}")
            _checkpoint(doc, ts)

            # ── Sonnet 5 — full pilot pipeline ──
            with Session() as db:
                for qid, q in BENCH_QUERIES:
                    with _use_model("claude-sonnet-5"):
                        try:
                            rec = _pipeline_run(db, "claude-sonnet-5", qid, q)
                        except BudgetExceeded as e:
                            doc["notes"].append(f"sonnet pipeline stopped at {qid}: {e}")
                            raise
                    doc["pipeline_runs"].append(rec)
                    print(f"  pipe   claude-sonnet-5 {qid:24s} exact={rec['exact']} poss={rec['possible']} "
                          f"jtrunc={rec['judge']['truncations']} jsplit={rec['judge']['adaptive_splits']} "
                          f"audit={rec['audit']['status']} verified={rec['llm_verified_results']}/{rec['n_results']} "
                          f"{rec['wall_ms']:.0f}ms  {_summary_line(_METER)}")
                    _checkpoint(doc, ts)

            # ── Sonnet 5 — 991-profile regression ──
            with _use_model("claude-sonnet-5"):
                try:
                    rec = _dataset_regression("claude-sonnet-5", REGRESSION_QUERY)
                    doc["regression_991"].append(rec)
                    print(f"  991    claude-sonnet-5 exact={rec['exact']} poss={rec['possible']} "
                          f"calls={rec['anthropic_calls']} judge={rec['judge_status']} audit={rec['audit_status']} "
                          f"{rec['wall_ms']:.0f}ms  {_summary_line(_METER)}")
                    _checkpoint(doc, ts)
                except BudgetExceeded as e:
                    doc["notes"].append(f"sonnet 991 stopped: {e}")
                    raise

            # ── Haiku 4.5 on the shim — 2 fast queries, confirm prefill/temp
            #    difference is immaterial (only if budget headroom) ──
            if _METER.calls < args.max_calls - 20 and _METER.cost_usd < args.max_cost - 1.0:
                with Session() as db:
                    for qid, q in [("bq1_nonprofit_chicago", BENCH_QUERIES[0][1]),
                                   ("bq4_ex_amazon_startups", BENCH_QUERIES[3][1])]:
                        with _use_model("claude-haiku-4-5-20251001"):
                            try:
                                rec = _pipeline_run(db, "claude-haiku-4-5-20251001", qid, q)
                            except BudgetExceeded as e:
                                doc["notes"].append(f"haiku shim spotcheck stopped at {qid}: {e}")
                                raise
                        doc["haiku_spotcheck_on_shim"].append(rec)
                        print(f"  haiku-shim {qid:24s} exact={rec['exact']} poss={rec['possible']} "
                              f"{rec['wall_ms']:.0f}ms  {_summary_line(_METER)}")
                        _checkpoint(doc, ts)
            else:
                doc["notes"].append("skipped haiku-on-shim spotcheck — budget headroom too small")

    except BudgetExceeded as e:
        doc["notes"].append(f"HARD BUDGET STOP: {e}")
        print(f"\n*** HARD BUDGET STOP: {e} — partial results saved ***")
    except KeyboardInterrupt:
        doc["notes"].append("interrupted by user")
    finally:
        eng.dispose()
        doc["budget_final"] = {
            "anthropic_calls_total": _METER.calls,
            "estimated_cost_usd_total": round(_METER.cost_usd, 4),
            "round2_calls": _METER.calls - args.start_calls,
            "round2_cost_usd": round(_METER.cost_usd - args.start_cost, 4),
            "stopped_reason": _METER.stopped_reason,
        }
        doc["call_log"] = _METER.call_log
        doc["op_records"] = _OP_RECORDS
        p = _checkpoint(doc, ts)
        print(f"\n{_summary_line(_METER)}\nwrote {p}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Phase A — Anthropic model benchmark (eval-only)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("all", "probe", "round2"):
        p = sub.add_parser(name)
        p.add_argument("--max-calls", type=int, default=120 if name == "all" else 180)
        p.add_argument("--max-cost", type=float, default=4.0 if name == "all" else 6.0)
        p.add_argument("--pilot-db", default=str(PILOT_DIR / "pilot.db"))
        p.add_argument("--start-calls", type=int, default=0)
        p.add_argument("--start-cost", type=float, default=0.0)
        p.add_argument("--round1-json", default="")
    args = ap.parse_args(argv)
    {"all": cmd_all, "probe": cmd_probe, "round2": cmd_round2}[args.cmd](args)


if __name__ == "__main__":
    main()
