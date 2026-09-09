"""SQLAlchemy ORM models — the normalized relational layer (spec §13–§19).

Design notes
------------
* One row per experience / education / skill (never ``experience_1_company`` …).
* Deferred profile sections (certifications, projects, publications, …) are NOT
  dropped — the full Apify item is kept verbatim in ``RawProfile.raw_json`` so
  those tables can be added later with a pure backfill, no re-scrape.
* Foreign keys use ``ondelete="CASCADE"``; SQLite enforces this because
  ``PRAGMA foreign_keys=ON`` is set on every connection (see database.py).
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.constants import DatasetStatus, EnrichmentState, JobStatus
from app.database import Base
from app.models_base import gen_id, utcnow


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: gen_id("dataset"))
    name: Mapped[str] = mapped_column(String, default="My LinkedIn Network")
    connection_count: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String, default=DatasetStatus.READY_FOR_ENRICHMENT)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)


class Person(Base):
    __tablename__ = "people"
    __table_args__ = (UniqueConstraint("dataset_id", "public_identifier", name="uq_person_dataset_pubid"),)

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: gen_id("person"))
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"), index=True)
    is_connection: Mapped[bool] = mapped_column(Boolean, default=True, index=True)

    # identity
    linkedin_id: Mapped[str | None] = mapped_column(String, nullable=True)
    public_identifier: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    linkedin_url: Mapped[str] = mapped_column(String, index=True)

    first_name: Mapped[str | None] = mapped_column(String, nullable=True)
    last_name: Mapped[str | None] = mapped_column(String, nullable=True)
    full_name: Mapped[str | None] = mapped_column(String, nullable=True)
    headline: Mapped[str | None] = mapped_column(Text, nullable=True)
    about: Mapped[str | None] = mapped_column(Text, nullable=True)

    # location
    location_text: Mapped[str | None] = mapped_column(String, nullable=True)
    city: Mapped[str | None] = mapped_column(String, nullable=True)
    state: Mapped[str | None] = mapped_column(String, nullable=True)
    country: Mapped[str | None] = mapped_column(String, nullable=True)
    country_code: Mapped[str | None] = mapped_column(String, nullable=True)

    # current role (denormalized for fast filtering)
    current_title: Mapped[str | None] = mapped_column(String, nullable=True)
    current_company: Mapped[str | None] = mapped_column(String, nullable=True)
    current_company_linkedin_url: Mapped[str | None] = mapped_column(String, nullable=True)
    current_start_month: Mapped[int | None] = mapped_column(Integer, nullable=True)
    current_start_year: Mapped[int | None] = mapped_column(Integer, nullable=True)

    connections_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    followers_count: Mapped[int | None] = mapped_column(Integer, nullable=True)

    open_to_work: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    hiring: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    premium: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    verified: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    influencer: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    creator: Mapped[bool | None] = mapped_column(Boolean, nullable=True)

    registered_at: Mapped[str | None] = mapped_column(String, nullable=True)
    profile_picture_url: Mapped[str | None] = mapped_column(String, nullable=True)

    # CSV-sourced fields (present before enrichment)
    csv_company: Mapped[str | None] = mapped_column(String, nullable=True)
    csv_position: Mapped[str | None] = mapped_column(String, nullable=True)
    connected_on: Mapped[str | None] = mapped_column(String, nullable=True)

    # scoring-adjacent
    profile_completeness: Mapped[int] = mapped_column(Integer, default=0)
    completeness_detail: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # pipeline state
    enrichment_state: Mapped[str] = mapped_column(String, default=EnrichmentState.PENDING, index=True)
    enrichment_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    apify_attempts: Mapped[int] = mapped_column(Integer, default=0)
    semantic_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    raw_hash: Mapped[str | None] = mapped_column(String, nullable=True)

    last_scraped_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)


class Connection(Base):
    """CSV import record — 1:1 with a Person that has ``is_connection = true``."""

    __tablename__ = "connections"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: gen_id("conn"))
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"), index=True)
    person_id: Mapped[str] = mapped_column(ForeignKey("people.id", ondelete="CASCADE"), index=True)
    csv_first_name: Mapped[str | None] = mapped_column(String, nullable=True)
    csv_last_name: Mapped[str | None] = mapped_column(String, nullable=True)
    csv_email: Mapped[str | None] = mapped_column(String, nullable=True)
    csv_company: Mapped[str | None] = mapped_column(String, nullable=True)
    csv_position: Mapped[str | None] = mapped_column(String, nullable=True)
    connected_on: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class RawProfile(Base):
    """Verbatim Apify item — stored before any transformation (spec §9). Never discard."""

    __tablename__ = "raw_profiles"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: gen_id("raw"))
    person_id: Mapped[str] = mapped_column(ForeignKey("people.id", ondelete="CASCADE"), index=True)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"), index=True)
    source: Mapped[str] = mapped_column(String)
    actor_id: Mapped[str | None] = mapped_column(String, nullable=True)
    apify_run_id: Mapped[str | None] = mapped_column(String, nullable=True)
    apify_dataset_id: Mapped[str | None] = mapped_column(String, nullable=True)
    raw_hash: Mapped[str | None] = mapped_column(String, nullable=True)
    scraped_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    raw_json: Mapped[dict] = mapped_column(JSON)


class Experience(Base):
    __tablename__ = "experiences"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: gen_id("exp"))
    person_id: Mapped[str] = mapped_column(ForeignKey("people.id", ondelete="CASCADE"), index=True)
    position: Mapped[str | None] = mapped_column(String, nullable=True)
    company_name: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    company_linkedin_url: Mapped[str | None] = mapped_column(String, nullable=True)
    company_id: Mapped[str | None] = mapped_column(String, nullable=True)
    company_universal_name: Mapped[str | None] = mapped_column(String, nullable=True)
    employment_type: Mapped[str | None] = mapped_column(String, nullable=True)
    workplace_type: Mapped[str | None] = mapped_column(String, nullable=True)
    location: Mapped[str | None] = mapped_column(String, nullable=True)
    start_month: Mapped[int | None] = mapped_column(Integer, nullable=True)
    start_year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    start_text: Mapped[str | None] = mapped_column(String, nullable=True)
    end_month: Mapped[int | None] = mapped_column(Integer, nullable=True)
    end_year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    end_text: Mapped[str | None] = mapped_column(String, nullable=True)
    is_current: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    duration_text: Mapped[str | None] = mapped_column(String, nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    skills_json: Mapped[list | None] = mapped_column(JSON, nullable=True)
    order_index: Mapped[int] = mapped_column(Integer, default=0)


class Education(Base):
    __tablename__ = "education"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: gen_id("edu"))
    person_id: Mapped[str] = mapped_column(ForeignKey("people.id", ondelete="CASCADE"), index=True)
    school_name: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    school_linkedin_url: Mapped[str | None] = mapped_column(String, nullable=True)
    school_id: Mapped[str | None] = mapped_column(String, nullable=True)
    degree: Mapped[str | None] = mapped_column(String, nullable=True)
    field_of_study: Mapped[str | None] = mapped_column(String, nullable=True)
    start_year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    end_year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    period: Mapped[str | None] = mapped_column(String, nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    skills_json: Mapped[list | None] = mapped_column(JSON, nullable=True)
    order_index: Mapped[int] = mapped_column(Integer, default=0)


class Skill(Base):
    __tablename__ = "skills"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: gen_id("skill"))
    person_id: Mapped[str] = mapped_column(ForeignKey("people.id", ondelete="CASCADE"), index=True)
    skill_name: Mapped[str] = mapped_column(String, index=True)
    skill_name_norm: Mapped[str] = mapped_column(String, index=True)
    source: Mapped[str] = mapped_column(String)
    is_inferred: Mapped[bool] = mapped_column(Boolean, default=False)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    evidence: Mapped[str | None] = mapped_column(Text, nullable=True)


class Certification(Base):
    __tablename__ = "certifications"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: gen_id("cert"))
    person_id: Mapped[str] = mapped_column(ForeignKey("people.id", ondelete="CASCADE"), index=True)
    name: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    issuer: Mapped[str | None] = mapped_column(String, nullable=True)
    issuer_url: Mapped[str | None] = mapped_column(String, nullable=True)
    issued_at: Mapped[str | None] = mapped_column(String, nullable=True)
    credential_id: Mapped[str | None] = mapped_column(String, nullable=True)
    url: Mapped[str | None] = mapped_column(String, nullable=True)
    order_index: Mapped[int] = mapped_column(Integer, default=0)


class Publication(Base):
    __tablename__ = "publications"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: gen_id("pub"))
    person_id: Mapped[str] = mapped_column(ForeignKey("people.id", ondelete="CASCADE"), index=True)
    title: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)
    publisher: Mapped[str | None] = mapped_column(String, nullable=True)
    published_at: Mapped[str | None] = mapped_column(String, nullable=True)
    url: Mapped[str | None] = mapped_column(String, nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    order_index: Mapped[int] = mapped_column(Integer, default=0)


class Patent(Base):
    __tablename__ = "patents"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: gen_id("pat"))
    person_id: Mapped[str] = mapped_column(ForeignKey("people.id", ondelete="CASCADE"), index=True)
    title: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)
    number: Mapped[str | None] = mapped_column(String, nullable=True)
    status: Mapped[str | None] = mapped_column(String, nullable=True)
    issued_at: Mapped[str | None] = mapped_column(String, nullable=True)
    url: Mapped[str | None] = mapped_column(String, nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    order_index: Mapped[int] = mapped_column(Integer, default=0)


class Language(Base):
    __tablename__ = "languages"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: gen_id("lang"))
    person_id: Mapped[str] = mapped_column(ForeignKey("people.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String, index=True)
    name_norm: Mapped[str] = mapped_column(String, index=True)
    proficiency: Mapped[str | None] = mapped_column(String, nullable=True)
    order_index: Mapped[int] = mapped_column(Integer, default=0)


class Volunteering(Base):
    __tablename__ = "volunteering"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: gen_id("vol"))
    person_id: Mapped[str] = mapped_column(ForeignKey("people.id", ondelete="CASCADE"), index=True)
    role: Mapped[str | None] = mapped_column(String, nullable=True)
    organization: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    cause: Mapped[str | None] = mapped_column(String, nullable=True)
    start_year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    end_year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    order_index: Mapped[int] = mapped_column(Integer, default=0)


class Recommendation(Base):
    __tablename__ = "recommendations"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: gen_id("rec"))
    person_id: Mapped[str] = mapped_column(ForeignKey("people.id", ondelete="CASCADE"), index=True)
    recommender_name: Mapped[str | None] = mapped_column(String, nullable=True)
    recommender_headline: Mapped[str | None] = mapped_column(String, nullable=True)
    relationship: Mapped[str | None] = mapped_column(String, nullable=True)
    text: Mapped[str | None] = mapped_column(Text, nullable=True)
    given_at: Mapped[str | None] = mapped_column(String, nullable=True)
    order_index: Mapped[int] = mapped_column(Integer, default=0)


class CompanySemantic(Base):
    """Company-level classification, cached once and reused across every person
    who worked there (spec §4). Keyed by LinkedIn ``company_id`` when known,
    else a normalized company name. Never re-classified per-person.

    ``is_startup`` / ``is_big_tech`` / ``is_technology_company`` are tri-state:
    ``True``/``False`` only when the classifier is confident; ``None`` (=
    UNKNOWN) when evidence is insufficient — UNKNOWN is never treated as False.
    """

    __tablename__ = "company_semantics"
    __table_args__ = (UniqueConstraint("company_key", name="uq_company_semantics_key"),)

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: gen_id("cosem"))
    #: "id:<linkedin company_id>" or "name:<normalized name>" — see company_intel.py
    company_key: Mapped[str] = mapped_column(String, index=True)
    company_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    display_name: Mapped[str | None] = mapped_column(String, nullable=True)

    industries: Mapped[list | None] = mapped_column(JSON, nullable=True)
    categories: Mapped[list | None] = mapped_column(JSON, nullable=True)
    is_technology_company: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    is_startup: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    is_big_tech: Mapped[bool | None] = mapped_column(Boolean, nullable=True)

    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    provenance: Mapped[str] = mapped_column(String, default="unknown")
    llm_provider: Mapped[str | None] = mapped_column(String, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)


class ProfileSemantic(Base):
    """LLM-derived interpretation of one profile (spec §26). Cached by version."""

    __tablename__ = "profile_semantics"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: gen_id("sem"))
    person_id: Mapped[str] = mapped_column(
        ForeignKey("people.id", ondelete="CASCADE"), unique=True, index=True
    )
    version: Mapped[int] = mapped_column(Integer, default=1)
    llm_provider: Mapped[str | None] = mapped_column(String, nullable=True)
    llm_model: Mapped[str | None] = mapped_column(String, nullable=True)
    data: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)


class ProfileEmbedding(Base):
    __tablename__ = "profile_embeddings"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: gen_id("emb"))
    person_id: Mapped[str] = mapped_column(
        ForeignKey("people.id", ondelete="CASCADE"), unique=True, index=True
    )
    model: Mapped[str] = mapped_column(String)
    dim: Mapped[int] = mapped_column(Integer)
    vector: Mapped[bytes] = mapped_column(LargeBinary)  # float32 little-endian
    search_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class EnrichmentJob(Base):
    __tablename__ = "enrichment_jobs"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: gen_id("job"))
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"), index=True)
    actor_id: Mapped[str | None] = mapped_column(String, nullable=True)
    apify_run_id: Mapped[str | None] = mapped_column(String, nullable=True)
    apify_dataset_id: Mapped[str | None] = mapped_column(String, nullable=True)
    requested_profiles: Mapped[int] = mapped_column(Integer, default=0)
    completed_profiles: Mapped[int] = mapped_column(Integer, default=0)
    failed_profiles: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String, default=JobStatus.QUEUED)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class SearchQuery(Base):
    __tablename__ = "search_queries"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: gen_id("search"))
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"), index=True)
    query_text: Mapped[str] = mapped_column(Text)
    interpreted_query_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    llm_provider: Mapped[str | None] = mapped_column(String, nullable=True)
    llm_model: Mapped[str | None] = mapped_column(String, nullable=True)
    total_candidates: Mapped[int] = mapped_column(Integer, default=0)
    external_searched: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class SearchResult(Base):
    __tablename__ = "search_results"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: gen_id("sr"))
    search_id: Mapped[str] = mapped_column(ForeignKey("search_queries.id", ondelete="CASCADE"), index=True)
    person_id: Mapped[str] = mapped_column(ForeignKey("people.id", ondelete="CASCADE"), index=True)
    #: connection | connection_near | external — near matches live in their own
    #: bucket so ``load_search`` can rebuild ``connections.near_matches`` without
    #: ever mixing them into the main results (V4 PART 7 §4).
    bucket: Mapped[str] = mapped_column(String, default="connection")
    rank: Mapped[int] = mapped_column(Integer)
    match_score: Mapped[float] = mapped_column(Float)
    data_confidence: Mapped[int] = mapped_column(Integer)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    payload: Mapped[dict] = mapped_column(JSON)  # full result object as returned to the client
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class SearchRunState(Base):
    """Search-level snapshot of one completed search (V4 PART 7).

    A NEW table (never a column added to ``search_queries`` / ``search_results``)
    so ``Base.metadata.create_all()`` can add it to an existing SQLite ``app.db``
    with no migration framework — ``create_all`` only creates missing tables, it
    never alters existing ones.

    Holds the FINAL validated response-level metadata (captured AFTER
    ``final_auditor.finalize()``) so ``load_search`` can rebuild the exact
    response first returned, with zero LLM / embedding / judge / audit / reason
    re-runs.
    """

    __tablename__ = "search_run_states"
    __table_args__ = (UniqueConstraint("search_id", name="uq_search_run_state_search"),)

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: gen_id("srs"))
    search_id: Mapped[str] = mapped_column(
        ForeignKey("search_queries.id", ondelete="CASCADE"), unique=True, index=True
    )
    #: stored-response format version — lets a future format change be detected
    #: without a migration framework (V4 PART 7 §9).
    response_version: Mapped[int] = mapped_column(Integer, default=1)

    exact_match_count: Mapped[int] = mapped_column(Integer, default=0)
    possible_match_count: Mapped[int] = mapped_column(Integer, default=0)
    returned_count: Mapped[int] = mapped_column(Integer, default=0)
    near_match_count: Mapped[int] = mapped_column(Integer, default=0)
    total_candidates: Mapped[int] = mapped_column(Integer, default=0)
    external_searched: Mapped[bool] = mapped_column(Boolean, default=False)

    #: FINAL validated observability blocks (``JudgeMetadata.as_dict()`` /
    #: ``AuditMetadata.as_dict()``); null when that phase did not run.
    judge_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    audit_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
