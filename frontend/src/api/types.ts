export interface DatasetSummary {
  dataset_id: string;
  name: string;
  connection_count: number;
  status: string;
  created_at: string;
  updated_at: string;
}

export interface UploadReport {
  dataset: DatasetSummary;
  total_rows: number;
  imported: number;
  duplicates_removed: number;
  skipped_no_url: number;
  skipped: Array<Record<string, unknown>>;
  duplicates: Array<Record<string, unknown>>;
}

export interface DatasetStatus {
  dataset_id: string;
  name: string;
  status: string;
  connections: number;
  counts: Record<string, number>;
  ready: number;
  partial: number;
  failed: number;
  pending: number;
  waiting_for_llm: number;
  progress_done: number;
  progress_total: number;
  progress_pct: number;
  last_updated: string | null;
  job: null | {
    job_id: string;
    status: string;
    requested_profiles: number;
    completed_profiles: number;
    failed_profiles: number;
    error: string | null;
  };
}

export interface PersonListItem {
  person_id: string;
  full_name: string | null;
  headline: string | null;
  current_title: string | null;
  current_company: string | null;
  linkedin_url: string;
  profile_completeness: number;
  enrichment_state: string;
}

/**
 * type distinguishes provenance (spec §19, §38):
 *  experience | education | skill | certification | language | publication | location
 *      -> verified LinkedIn data
 *  semantic          -> AI inferred from the profile
 *  company_inference -> AI classification of the employer
 *  semantic_relevance-> whole-profile / cross-encoder similarity signal
 */
export interface EvidenceItem {
  type: string;
  text: string;
  detail: Record<string, unknown>;
}

export type EvidenceProvenance = "linkedin" | "ai_inferred" | "company_inference" | "relevance";

export function evidenceProvenance(type: string): EvidenceProvenance {
  if (type === "company_inference") return "company_inference";
  if (type === "semantic_relevance" || type === "relevance") return "relevance";
  if (type === "semantic") return "ai_inferred";
  return "linkedin";
}

export interface ScoreComponent {
  criterion: string;
  criterion_id: string;
  type: string;
  weight: number;
  match_strength: number;
  score: number;
  required: boolean;
  evidence: EvidenceItem[];
}

export interface ExperienceOut {
  position: string | null;
  company_name: string | null;
  company_linkedin_url: string | null;
  start_year: number | null;
  end_year: number | null;
  is_current: boolean;
  duration_text: string | null;
  description: string | null;
  location: string | null;
}

export interface EducationOut {
  school_name: string | null;
  degree: string | null;
  field_of_study: string | null;
  start_year: number | null;
  end_year: number | null;
}

export interface SkillOut {
  skill_name: string;
  source: string;
  is_inferred: boolean;
  confidence: number;
  evidence: string | null;
}

export interface SearchResultItem {
  rank: number;
  person_id: string;
  name: string | null;
  linkedin_url: string;
  profile_picture_url: string | null;
  current_title: string | null;
  current_company: string | null;
  location: string | null;
  is_connection: boolean;
  match_score: number;
  data_confidence: number;
  reason: string;
  // V4 §22-25 — exact_match / possible_match / not_match (near-match rows only).
  qualification: "exact_match" | "possible_match" | "not_match";
  // required criteria still uncertain — why this is possible_match, not exact.
  uncertain_criteria: string[];
  // required criteria this person FAILS — near-match rows only.
  unmet_criteria: string[];
  matched_criteria: string[];
  score_breakdown: ScoreComponent[];
  evidence: EvidenceItem[];
  relevant_experience: ExperienceOut[];
  relevant_skills: SkillOut[];
  relevant_education: EducationOut[];
  // V4 PART 5 §26 — final-audit outcome (optional, backward-compatible).
  audit_decision?: "approved" | "downgrade" | "incorrect" | "unknown" | null;
  audit_confidence?: number | null;
  audit_reason?: string | null;
  audit_issues?: string[];
  llm_verified?: boolean;
}

export interface SearchResponse {
  search_id: string;
  query: string;
  interpreted_query: {
    intent?: string;
    criteria: Array<{
      id: string;
      type: string;
      value: string;
      values?: string[];
      operator?: "ANY_OF" | "ALL_OF" | "NOT";
      scope?: string | null;
      concept?: string | null;
      weight: number;
      required: boolean;
      // V4 PART 2 §5 — "certain" (default) vs "possible" ("might have X")
      modality?: "certain" | "possible";
    }>;
    context?: Record<string, string>;
    // V4 PART 2 §3 — the mentee / searcher a candidate must be able to help.
    // Never a search phrase.
    target_person_context?: Record<string, string>;
    // context keys the interpreter refused to guess (e.g. "field" for "my field")
    unresolved?: string[];
    interpretation_summary?: string;
    interpretation_confidence?: number;
  };
  connections: {
    total_candidates: number;
    returned: number;
    results: SearchResultItem[];
    // V4 §22-26 — tier counts + candidates that miss exactly one required
    // criterion (clearly labelled, never mixed into `results`).
    exact_match_count: number;
    possible_match_count: number;
    near_matches: SearchResultItem[];
  };
  external: {
    searched: boolean;
    total_candidates: number;
    returned: number;
    results: SearchResultItem[];
  };
  llm_provider: string | null;
  llm_model: string | null;
  // V4 PART 3 §32 — observability for the exhaustive semantic-judge run.
  // Absent on searches that did not run the judge; frontend rendering comes later.
  judge_metadata?: {
    mode: "off" | "uncertain_only" | "all_viable";
    status: "full" | "partial" | "not_used" | "unavailable";
    network_size: number;
    candidate_pool_size: number;
    hard_fact_rejected_count: number;
    viable_candidate_count: number;
    judge_candidate_count: number;
    judge_batch_count: number;
    judge_successful_batches: number;
    judge_failed_batches: number;
    capped: boolean;
    omitted_people: number;
    omitted_criteria: number;
    // V4 PART 3.6 §8 — packets too large to judge (subset of omitted_people)
    oversized_packets?: number;
    providers: Record<string, number>;
    models: string[];
  } | null;
  // V4 PART 5 §25 — observability for the final result audit. Live-response
  // only for now (persistence + UI are PART 7-9).
  audit_metadata?: {
    enabled: boolean;
    status: "full" | "partial" | "not_used" | "unavailable";
    requested_candidates: number;
    audited_candidates: number;
    batch_count: number;
    successful_batches: number;
    failed_batches: number;
    oversized_packets: number;
    approved: number;
    downgraded: number;
    incorrect: number;
    unknown: number;
    // V4 PART 5.5 §4 — required-criterion reviews the auditor omitted
    missing_required_reviews: number;
    candidates_with_incomplete_reviews: number;
    providers: Record<string, number>;
    models: string[];
  } | null;
}
