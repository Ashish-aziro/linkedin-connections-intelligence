"""TASK 4 — reusable semantic-verdict cache for full verification.

Avoids re-asking Claude the same question about the same unchanged profile.
A cache entry is reusable only when EVERY one of these matches exactly
(``models.SemanticVerdictCache``'s unique key):

  * person_id
  * criterion_key    — normalized type + operator + scope + modality + the
                        actual value(s)/concept being judged
  * evidence_fingerprint — hash of the EXACT evidence packet content for that
                        person (full-profile packet, minus request-shaping
                        fields) — any fact change invalidates it naturally
  * model             — a different Claude model is a different judge
  * prompt_version    — bump when ``full_verification._TARGETED_SYSTEM`` /
                        ``semantic_judge._SYSTEM`` changes meaning
  * schema_version     — bump when the verdict dict's shape changes
  * context_key        — hash of the query context the verdict's meaning can
                        depend on (intent / context / target_person_context /
                        interpretation_summary); "" only for a criterion type
                        the codebase has decided is context-independent

Only ``store()`` may write a row, and only with an ALREADY-VALIDATED verdict
(the caller passes verdicts that already survived
``judge_validator.validate_person``) — this module never writes a raw,
unvalidated, truncated, or failed LLM response.
"""
from __future__ import annotations

import hashlib
import json
import logging

from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_upsert
from sqlalchemy.orm import Session

from app.config import settings
from app.models import SemanticVerdictCache

log = logging.getLogger("app.verdict_cache")

#: bump when ``full_verification._TARGETED_SYSTEM`` / the batched judge system
#: prompt (``semantic_judge._SYSTEM``) changes in a way that could change what
#: an identical packet+criterion decides.
PROMPT_VERSION = 1
#: bump when the verdict dict shape (``full_verification._unknown_verdict`` /
#: ``semantic_judge._expand_compact``'s output) changes.
SCHEMA_VERSION = 1

#: packet keys that shape the REQUEST, not the evidence itself — excluded from
#: the evidence fingerprint so re-stamping ``unresolved_criteria`` (which a
#: cache hit changes on every subsequent search) never busts the cache.
_REQUEST_SHAPING_KEYS = {"unresolved_criteria", "_truncated", "_packet_too_large"}

#: criterion types whose correct verdict is fully determined by the packet's
#: OWN evidence — the query's broader intent/context never changes what
#: "worked at Google" or "has a CS degree" means. Everything else (semantic
#: concepts, role functions, professional concepts, industry experience) is
#: interpretive enough that the query's framing genuinely can shift the
#: correct call, so those are ALWAYS scoped to the query context.
_CONTEXT_INDEPENDENT_TYPES = {"skill", "certification", "education", "language", "publication"}


def _stable_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def criterion_key(crit) -> str:
    """Normalized criterion identity — everything that changes what question
    is actually being asked, NEVER the query-assigned ``id`` (two searches
    that ask "the same question" should share a cache entry even though the
    criterion got a different generated id each time)."""
    return _stable_hash({
        "type": crit.type,
        "operator": crit.operator,
        "scope": crit.scope,
        "modality": crit.modality,
        "concept": crit.concept,
        "value": crit.value,
        "values": crit.values,
        "required": crit.required,
    })


def context_key(parsed, crit) -> str:
    if crit.type in _CONTEXT_INDEPENDENT_TYPES:
        return ""
    return _stable_hash({
        "intent": parsed.intent,
        "context": parsed.context,
        "target_person_context": parsed.target_person_context,
        "interpretation_summary": parsed.interpretation_summary,
    })


def evidence_fingerprint(packet: dict) -> str:
    return _stable_hash({k: v for k, v in packet.items() if k not in _REQUEST_SHAPING_KEYS})


def lookup(
    db: Session, packets_by_id: dict[str, dict], jcrits: list, parsed, *, model: str,
) -> dict[str, dict[str, dict]]:
    """Bulk cache lookup, ONE query, main thread only (never called from a
    worker thread — the caller must gather this before any concurrent batch
    work starts). Returns ``{person_id: {criterion_id: verdict_dict}}`` for
    every hit; a miss is simply absent, never a placeholder."""
    if not settings.semantic_verdict_cache_enabled or not packets_by_id or not jcrits:
        return {}

    fp_by_person = {pid: evidence_fingerprint(pkt) for pid, pkt in packets_by_id.items()}
    crit_key_by_id = {c.id: criterion_key(c) for c in jcrits}
    ctx_key_by_id = {c.id: context_key(parsed, c) for c in jcrits}
    # reverse map: (crit_key, ctx_key) -> [criterion_id, ...] (usually one, but
    # two differently-worded criteria that normalize identically legitimately
    # share a cache row — both get the SAME verdict, which is correct).
    ids_by_key: dict[tuple[str, str], list[str]] = {}
    for cid in crit_key_by_id:
        ids_by_key.setdefault((crit_key_by_id[cid], ctx_key_by_id[cid]), []).append(cid)

    rows = db.scalars(
        select(SemanticVerdictCache).where(
            SemanticVerdictCache.person_id.in_(fp_by_person),
            SemanticVerdictCache.model == model,
            SemanticVerdictCache.prompt_version == PROMPT_VERSION,
            SemanticVerdictCache.schema_version == SCHEMA_VERSION,
        )
    ).all()

    hits: dict[str, dict[str, dict]] = {}
    for row in rows:
        if row.evidence_fingerprint != fp_by_person.get(row.person_id):
            continue  # evidence changed since this row was written — stale, skip
        for cid in ids_by_key.get((row.criterion_key, row.context_key), []):
            hits.setdefault(row.person_id, {})[cid] = row.verdict_json
    return hits


def store(
    db: Session,
    validated_by_person: dict[str, dict[str, dict]],
    packets_by_id: dict[str, dict],
    jcrits: list,
    parsed,
    *,
    model: str,
    skip_keys: set[tuple[str, str]] | None = None,
) -> int:
    """Bulk upsert, main thread only, called AFTER every worker thread has
    finished (search_service's ``verification_validation`` stage, post-gather).
    Writes ONLY verdicts the caller confirms are already validated —
    ``full_verification``/``judge_validator`` decide that, this module trusts
    it. ``skip_keys`` — (person_id, criterion_id) pairs that were themselves
    cache HITS this run — re-writing them would be a harmless no-op upsert,
    skipped purely to keep the write count meaningful for the cache-hit report.
    Returns the number of rows written."""
    if not settings.semantic_verdict_cache_enabled or not validated_by_person:
        return 0
    skip_keys = skip_keys or set()
    crit_by_id = {c.id: c for c in jcrits}

    rows: list[dict] = []
    for pid, crit_verdicts in validated_by_person.items():
        packet = packets_by_id.get(pid)
        if packet is None:
            continue
        fp = evidence_fingerprint(packet)
        for cid, verdict in crit_verdicts.items():
            if (pid, cid) in skip_keys:
                continue
            crit = crit_by_id.get(cid)
            if crit is None or not isinstance(verdict, dict):
                continue
            # never cache a placeholder — only a verdict the model actually
            # returned (not the "judge_missing" / omitted fill-in).
            if verdict.get("judge_missing"):
                continue
            rows.append({
                "id": None,  # set per-row below (gen_id has no bulk form)
                "person_id": pid,
                "criterion_key": criterion_key(crit),
                "evidence_fingerprint": fp,
                "model": model,
                "prompt_version": PROMPT_VERSION,
                "schema_version": SCHEMA_VERSION,
                "context_key": context_key(parsed, crit),
                "verdict_json": verdict,
            })
    if not rows:
        return 0

    from app.models_base import gen_id

    for r in rows:
        r["id"] = gen_id("vcache")

    stmt = sqlite_upsert(SemanticVerdictCache).values(rows)
    stmt = stmt.on_conflict_do_update(
        index_elements=[
            "person_id", "criterion_key", "evidence_fingerprint",
            "model", "prompt_version", "schema_version", "context_key",
        ],
        set_={"verdict_json": stmt.excluded.verdict_json},
    )
    try:
        db.execute(stmt)
    except Exception:  # noqa: BLE001 — the cache is an optimization, never a search failure
        log.exception("verdict cache write failed — continuing without caching this batch")
        return 0
    return len(rows)
