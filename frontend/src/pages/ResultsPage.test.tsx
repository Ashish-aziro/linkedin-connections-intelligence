import { describe, expect, it } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import ResultsPage from "./ResultsPage";
import type { SearchResponse, SearchResultItem } from "../api/types";

function item(over: Partial<SearchResultItem> = {}): SearchResultItem {
  return {
    rank: over.rank ?? 1,
    person_id: over.person_id ?? "p1",
    name: over.name ?? "Person",
    linkedin_url: "https://www.linkedin.com/in/x",
    profile_picture_url: null,
    current_title: "Engineer",
    current_company: "Acme",
    location: null,
    is_connection: true,
    match_score: 80,
    data_confidence: 70,
    reason: "reason",
    qualification: "exact_match",
    uncertain_criteria: [],
    unmet_criteria: [],
    matched_criteria: [],
    score_breakdown: [],
    evidence: [],
    relevant_experience: [],
    relevant_skills: [],
    relevant_education: [],
    ...over,
  };
}

function response(over: Partial<SearchResponse> = {}): SearchResponse {
  return {
    search_id: "s1",
    query: "who could mentor a backend engineer moving into management?",
    interpreted_query: { criteria: [] },
    connections: {
      total_candidates: 10,
      returned: 1,
      results: [item()],
      exact_match_count: 1,
      possible_match_count: 0,
      near_matches: [],
    },
    external: { searched: false, total_candidates: 0, returned: 0, results: [] },
    llm_provider: "anthropic",
    llm_model: "claude-x",
    judge_metadata: null,
    audit_metadata: null,
    ...over,
  };
}

function renderPage(res: SearchResponse) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter
        initialEntries={[{ pathname: `/datasets/d1/search/${res.search_id}`, state: res }]}
      >
        <Routes>
          <Route path="/datasets/:datasetId/search/:searchId" element={<ResultsPage />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe("ResultsPage", () => {
  it("never shows the internal interpretation panel or criterion chips", () => {
    renderPage(
      response({
        interpreted_query: {
          criteria: [
            { id: "loc", type: "location", value: "Chicago", weight: 31, required: true },
            { id: "inv", type: "professional_concept", value: "investor", weight: 46, required: true },
          ],
          interpretation_summary:
            "People with engineering-management experience and mentoring evidence.",
          interpretation_confidence: 0.82,
          unresolved: ["field"],
        },
      }),
    );
    expect(screen.queryByText("How we interpreted your search")).not.toBeInTheDocument();
    expect(screen.queryByText(/82% confidence/)).not.toBeInTheDocument();
    expect(screen.queryByText(/engineering-management experience/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Some context could not be resolved/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Chicago/)).not.toBeInTheDocument();
    expect(screen.queryByText(/investor/)).not.toBeInTheDocument();
    // the query itself is still shown — original query -> results, nothing else
    expect(screen.getByText(/who could mentor a backend engineer/)).toBeInTheDocument();
  });

  it("shows Exact / Possible counts", () => {
    renderPage(
      response({
        connections: {
          total_candidates: 20,
          returned: 18,
          results: [item()],
          exact_match_count: 12,
          possible_match_count: 6,
          near_matches: [],
        },
      }),
    );
    expect(screen.getByText("12 Exact")).toBeInTheDocument();
    expect(screen.getByText("6 Possible")).toBeInTheDocument();
    expect(screen.getByText("18 shown")).toBeInTheDocument();
  });

  it("renders a Near matches section when there are no main results but near matches exist", () => {
    renderPage(
      response({
        connections: {
          total_candidates: 5,
          returned: 0,
          results: [],
          exact_match_count: 0,
          possible_match_count: 0,
          near_matches: [
            item({ person_id: "n1", name: "Near One", qualification: "not_match", unmet_criteria: ["CXO-level seniority"] }),
            item({ person_id: "n2", name: "Near Two", qualification: "not_match", unmet_criteria: ["healthcare experience"] }),
          ],
        },
      }),
    );
    expect(screen.getByText("No exact/possible matches were found.")).toBeInTheDocument();
    expect(screen.getByText("Near matches")).toBeInTheDocument();
    expect(screen.getByText(/Missing: CXO-level seniority/)).toBeInTheDocument();
    expect(
      screen.getByText(/don.t fully match every requirement but may/),
    ).toBeInTheDocument();
    // internal near-match concepts must never leak to the user
    expect(screen.queryByText(/relaxed_criterion_id/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/hard gate/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/semantic judge/i)).not.toBeInTheDocument();
  });

  it("shows a reliable near-match relation label but not an unreliable one", () => {
    renderPage(
      response({
        connections: {
          total_candidates: 2,
          returned: 0,
          results: [],
          exact_match_count: 0,
          possible_match_count: 0,
          near_matches: [
            item({
              person_id: "n1", name: "Nearby VC", qualification: "not_match",
              unmet_criteria: ["Fernbrook"], near_relation_type: "geographic_adjacent",
              near_match_confidence: 0.85,
            }),
            item({
              person_id: "n2", name: "Unlabeled", qualification: "not_match",
              unmet_criteria: ["Fernbrook"], near_relation_type: "other_relevant",
            }),
          ],
        },
      }),
    );
    expect(screen.getByText("Nearby location")).toBeInTheDocument();
    expect(screen.queryByText(/^other_relevant$/)).not.toBeInTheDocument();
  });

  it("warns when final audit verification was partial", () => {
    renderPage(
      response({
        audit_metadata: {
          enabled: true,
          status: "partial",
          requested_candidates: 5,
          audited_candidates: 3,
          batch_count: 2,
          successful_batches: 1,
          failed_batches: 1,
          oversized_packets: 0,
          approved: 2,
          downgraded: 1,
          incorrect: 0,
          unknown: 0,
          missing_required_reviews: 0,
          candidates_with_incomplete_reviews: 0,
          providers: {},
          models: [],
        },
      }),
    );
    expect(
      screen.getByText(/Some AI verification was unavailable; uncertain results are shown conservatively/),
    ).toBeInTheDocument();
  });

  it("does not warn when verification was full", () => {
    renderPage(
      response({
        judge_metadata: {
          mode: "all_viable",
          status: "full",
          network_size: 100,
          candidate_pool_size: 50,
          hard_fact_rejected_count: 10,
          viable_candidate_count: 40,
          judge_candidate_count: 40,
          judge_batch_count: 8,
          judge_successful_batches: 8,
          judge_failed_batches: 0,
          capped: false,
          omitted_people: 0,
          omitted_criteria: 0,
          providers: {},
          models: [],
        },
      }),
    );
    expect(screen.queryByText(/Some AI verification was unavailable/)).not.toBeInTheDocument();
    fireEvent.click(screen.getByText(/Search quality details/));
    expect(screen.getByText("Semantic review")).toBeInTheDocument();
  });

  // Bug report: a search that completed full Sonnet verification for every
  // candidate (mode: "full_verification", status: "complete") rendered
  // "Semantic review: Not used" — status "complete" fell through
  // verificationLabel()'s switch to its catch-all. This must show as
  // completed, under a header that reflects which review actually ran.
  it("shows full verification as complete, not 'Not used'", () => {
    renderPage(
      response({
        judge_metadata: {
          mode: "full_verification",
          status: "complete",
          network_size: 987,
          candidate_pool_size: 987,
          hard_fact_rejected_count: 971,
          judge_candidate_count: 16,
          judge_batch_count: 6,
          judge_successful_batches: 6,
          judge_failed_batches: 0,
          omitted_criteria: 0,
          providers: { "anthropic:paid": 6 },
          models: ["claude-haiku-4-5-20251001"],
          filtered_candidate_count: 16,
          sonnet_verified_candidate_count: 16,
          total_llm_calls: 6,
          cache_hits: 0,
        },
      }),
    );
    expect(screen.queryByText(/Some AI verification was unavailable/)).not.toBeInTheDocument();
    fireEvent.click(screen.getByText(/Search quality details/));
    expect(screen.getByText("Full verification")).toBeInTheDocument();
    expect(screen.getByText("Complete")).toBeInTheDocument();
    expect(screen.queryByText("Not used")).not.toBeInTheDocument();
    expect(screen.getByText("Candidates verified: 16")).toBeInTheDocument();
    expect(screen.getByText("Reused cached verdicts: 0")).toBeInTheDocument();
    expect(screen.getByText("New verification calls: 6")).toBeInTheDocument();
  });

  // Bug report TASK 6 — a repeat search served entirely from the verdict
  // cache (0 new Sonnet calls) must show that composition explicitly, not
  // "Batches: 0/0 ok" (which reads as if verification never ran at all).
  it("distinguishes cached verdicts from new verification calls on a fully-cached repeat search", () => {
    renderPage(
      response({
        judge_metadata: {
          mode: "full_verification",
          status: "complete",
          network_size: 987,
          candidate_pool_size: 987,
          hard_fact_rejected_count: 971,
          judge_candidate_count: 16,
          judge_batch_count: 0,
          judge_successful_batches: 0,
          judge_failed_batches: 0,
          omitted_criteria: 0,
          providers: {},
          models: ["claude-haiku-4-5-20251001"],
          filtered_candidate_count: 16,
          sonnet_verified_candidate_count: 16,
          total_llm_calls: 0,
          cache_hits: 16,
          cache_fully_cached_candidates: 16,
        },
      }),
    );
    fireEvent.click(screen.getByText(/Search quality details/));
    expect(screen.getByText("Complete")).toBeInTheDocument();
    expect(screen.getByText("Candidates verified: 16")).toBeInTheDocument();
    expect(screen.getByText("Reused cached verdicts: 16")).toBeInTheDocument();
    expect(screen.getByText("New verification calls: 0")).toBeInTheDocument();
    expect(screen.queryByText(/Batches: 0\/0 ok/)).not.toBeInTheDocument();
  });

  it("shows an unrecognized status verbatim instead of guessing", () => {
    renderPage(
      response({
        judge_metadata: {
          mode: "full_verification",
          status: "some_future_status" as never,
          network_size: 10,
          candidate_pool_size: 10,
          hard_fact_rejected_count: 0,
          judge_candidate_count: 10,
          judge_batch_count: 1,
          judge_successful_batches: 1,
          judge_failed_batches: 0,
          omitted_criteria: 0,
          providers: {},
          models: [],
        },
      }),
    );
    fireEvent.click(screen.getByText(/Search quality details/));
    expect(screen.getByText("Unknown (some_future_status)")).toBeInTheDocument();
    expect(screen.queryByText("Not used")).not.toBeInTheDocument();
    expect(screen.queryByText("Complete")).not.toBeInTheDocument();
  });
});
