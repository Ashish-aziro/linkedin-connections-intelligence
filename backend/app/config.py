"""Application configuration.

All values load from environment / ``backend/.env``. Secrets never have a
non-empty default and are never logged (see ``logging_config.redact``).
"""
from __future__ import annotations

import logging
from functools import lru_cache

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_log = logging.getLogger("app.config")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── Apify ────────────────────────────────────────────────
    apify_api_token: str = ""
    apify_actor_id: str = "LpVuK3Zozwuipa5bp"
    apify_profile_scraper_mode: str = "Profile details no email ($4 per 1k)"

    # ── LLM (Anthropic is the only external provider) ────────
    #: A CONFIGURED api key is itself the opt-in. With no key the app still runs
    #: — LLM calls return None and the deterministic parser / scoring stands in.
    anthropic_api_key: str = ""
    anthropic_workspace_id: str = ""  # only needed for identity-linked keys
    #: FULL SONNET VERIFICATION EXPERIMENT — Sonnet is now the default. An env
    #: value (ANTHROPIC_MODEL) still overrides this.
    anthropic_model: str = "claude-sonnet-4-6"
    llm_max_retries: int = 2
    #: DEPRECATED — kept so an old .env file doesn't error. Does not gate anything.
    enable_paid_llm: bool = False
    anthropic_first: bool = True

    # Provider circuit breaker — process-local, never persisted.
    llm_provider_cooldown_seconds: float = 90.0               # transient (429/5xx/transport)
    anthropic_config_failure_cooldown_seconds: float = 900.0  # auth / bad workspace / bad model

    # ── Embeddings (local) ───────────────────────────────────
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    embeddings_enabled: bool = True  # False -> deterministic hash vectors (tests / no-torch)

    # ── Database ─────────────────────────────────────────────
    database_url: str = "sqlite:///./data/app.db"

    # ── Enrichment / cost controls ───────────────────────────
    environment: str = "development"
    use_fixtures: bool = True
    development_batch_size: int = 5
    production_batch_size: int = 50
    apify_profile_batch_size: int = 50
    max_apify_retries: int = 2
    apify_max_charge_usd: float = 0.0  # 0 = no cap; e.g. 2.0 to hard-stop a run at $2
    profile_ttl_days: int = 30

    # ── Semantic enrichment ──────────────────────────────────
    semantic_enabled: bool = True  # False -> skip the LLM profile pass entirely
    #: v2 added semantic_assertions[]; v3 (V4 §16) adds experience_semantics[] +
    #: source-id-grounded assertions (role_function vs employer_industry). Bumping
    #: forces a backfill (no Apify re-scrape — see
    #: repositories.people_missing_semantics(current_version=...)). v2 rows stay
    #: searchable until backfilled.
    semantic_profile_version: int = 3

    # ── Search ───────────────────────────────────────────────
    candidate_pool_size: int = 60
    top_connections: int = 20
    min_match_score: float = 12.0          # drop incidental / weak matches (spec §38)
    llm_query_interpretation: bool = True  # False -> deterministic query parser only
    #: hardening PART 17 — query interpretation is foundational (never metered
    #: by SEARCH_LLM_MAX_CALLS) so it must not be allowed to hold a search
    #: hostage on a slow-but-not-failing provider; a bounded timeout falls
    #: through to the next provider (then the deterministic parser) quickly.
    query_interpretation_timeout_seconds: float = 20.0
    llm_reason_generation: bool = True     # False -> deterministic reason templates
    llm_reason_top_n: int = 8              # LLM reasons for the top N; template for the rest
    #: hardening PART 6 — soft cap on LLM calls AFTER query interpretation (judge
    #: + audit + reason). 0 = unlimited. Never causes an incorrect result: once
    #: hit, remaining optional work is skipped, deterministic results stand, and
    #: unresolved conditions stay UNKNOWN with judge/audit status PARTIAL.
    search_llm_max_calls: int = 0
    #: hardening PART 14 — wall-clock budget for a search's OPTIONAL LLM work
    #: (judge + audit + reason generation). Deterministic scoring is never
    #: skipped for time. <= 0 disables the deadline (unlimited).
    #: NOTE: ignored for the verification stage when FULL_LLM_VERIFICATION is on
    #: (verification is mandatory there, not optional).
    search_max_seconds: float = 45.0

    # ── FULL SONNET VERIFICATION EXPERIMENT ─────────────────────
    #: When True, EVERY candidate that survives the hard-fact gate is reviewed by
    #: Claude Sonnet before a search can succeed. A candidate whose review does
    #: not complete (API error / truncation that cannot be recovered / missing
    #: output) fails the whole SEARCH — partial/unverified results are never
    #: returned. The legacy "semantic judge only where unresolved + conservative
    #: partial results" path stays available with False.
    full_llm_verification: bool = True
    #: per-search wall-clock ceiling for the WHOLE full-verification search.
    #: <= 0 (default) = no application-level deadline — verification is mandatory
    #: and runs to completion. Individual HTTP calls still have their own timeout.
    full_verification_max_seconds: float = 0.0
    #: bounded recovery for a candidate omitted / incomplete in a batch: retry
    #: the candidate alone this many times before falling back to chunked review.
    full_verification_single_retries: int = 2
    #: split an oversized candidate's evidence into sections, review each, then
    #: aggregate — instead of leaving them unverified.
    full_verification_chunk_oversized: bool = True
    #: larger per-packet budget for the full-profile verification packet.
    full_verification_max_packet_chars: int = 14000
    full_verification_batch_size: int = 6

    # ── Search quality v2 ────────────────────────────────────
    relevance_weight: float = 20.0         # points reserved for whole-profile relevance
    reranker_enabled: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    rerank_pool: int = 40                  # candidates re-scored by the cross-encoder
    rerank_blend: float = 0.6             # cross-encoder share within the relevance component
    llm_rerank_enabled: bool = False       # opt-in LLM reorder of the final top 20
    recency_weighting_enabled: bool = True
    company_id_matching: bool = True

    # ── Search quality v3 — semantic concepts ────────────────
    #: below this many connections, score EVERYONE — no SQL prefilter at all
    #: (spec §10/§31: correctness over shaving milliseconds on a ~1k network).
    full_scan_max_connections: int = 5000
    company_classification_enabled: bool = True
    #: master switch — False disables the semantic judge entirely.
    semantic_judge_enabled: bool = True
    #: V4 PART 3 §8 — off | uncertain_only | all_viable.
    #:   all_viable      -> EVERY candidate past the hard-fact gate is judged
    #:                      (the correct setting for a ~1k network).
    #:   uncertain_only  -> only ambiguity-band candidates (cheaper deployments).
    #:   off             -> deterministic scoring only.
    semantic_judge_mode: str = "all_viable"
    #: V4 PART 3 §9 — hard cap on judged candidates AFTER hard-fact gating.
    #: 0 = NO artificial cap. A positive value is respected and the response
    #: reports the run was capped (never a silent cap).
    semantic_judge_max_candidates: int = 0
    semantic_judge_batch_size: int = 10     # candidates per LLM judge request (never 1/candidate)
    #: V4 PART 3 §34 — prompt-size guards. A batch whose packets exceed these is
    #: split into smaller batches rather than truncated into meaninglessness.
    semantic_judge_max_packet_chars: int = 7000
    semantic_judge_max_batch_chars: int = 48000
    #: uncertain_only mode only — the ambiguity band + pool cap (ignored by all_viable).
    semantic_judge_pool: int = 60
    semantic_judge_low: float = 0.15        # below this concept strength: skip judging, already a clear miss
    semantic_judge_high: float = 0.75       # above this: skip judging, already a clear hit
    #: a cached company classification below this confidence is treated as UNKNOWN
    #: (V4 §7) — a low-confidence TRUE cannot produce an EXACT_MATCH
    company_category_confidence_min: float = 0.6

    # ── V4 PART 5 — final result correctness audit ──────────
    #: ONE grounded LLM review of the candidates about to be shown. Downgrade /
    #: remove only — it can never upgrade POSSIBLE->EXACT or invent a score.
    final_result_audit_enabled: bool = True
    #: DEPRECATED (V4 PART 5.5 §20) — ``top_connections`` is the ONE authoritative
    #: user-facing result count. Kept only so old .env files don't error; a value
    #: that disagrees with ``top_connections`` is ignored (with a startup log
    #: warning). The audit pool is ``top_connections + final_result_audit_buffer``.
    final_result_audit_top_n: int = 20
    final_result_audit_buffer: int = 10
    final_result_audit_batch_size: int = 10   # candidates per audit request (never 1/candidate)
    final_result_audit_max_packet_chars: int = 6000
    final_result_audit_max_batch_chars: int = 44000

    # ── Current-user profile context (V4 PART 2 §3) ──────────
    #: Used ONLY to resolve relational queries like "anyone in my field" /
    #: "a mentor in my space". Left empty by default: when a query says "my
    #: field" and this is blank, the interpreter marks the field UNRESOLVED and
    #: lowers confidence — it never guesses the searcher's field.
    user_field: str = ""
    user_current_role: str = ""
    user_goal: str = ""

    # ── App ──────────────────────────────────────────────────
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    log_level: str = "INFO"

    @model_validator(mode="after")
    def _warn_audit_top_n_drift(self) -> "Settings":
        if self.final_result_audit_top_n != self.top_connections:
            _log.warning(
                "FINAL_RESULT_AUDIT_TOP_N=%d disagrees with TOP_CONNECTIONS=%d — "
                "TOP_CONNECTIONS is authoritative; FINAL_RESULT_AUDIT_TOP_N is deprecated and ignored.",
                self.final_result_audit_top_n, self.top_connections,
            )
        return self

    @property
    def is_development(self) -> bool:
        return self.environment.lower().startswith("dev")

    @property
    def effective_batch_size(self) -> int:
        """Batch size used by the enrichment worker for Apify calls."""
        return self.development_batch_size if self.is_development else self.apify_profile_batch_size

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
