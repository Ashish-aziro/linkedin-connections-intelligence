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
  it("shows the interpretation summary and confidence percentage", () => {
    renderPage(
      response({
        interpreted_query: {
          criteria: [],
          interpretation_summary:
            "People with engineering-management experience and mentoring evidence.",
          interpretation_confidence: 0.82,
        },
      }),
    );
    expect(screen.getByText("How we interpreted your search")).toBeInTheDocument();
    expect(screen.getByText(/82% confidence/)).toBeInTheDocument();
    expect(screen.getByText(/engineering-management experience/)).toBeInTheDocument();
  });

  it("surfaces unresolved context", () => {
    renderPage(
      response({
        interpreted_query: {
          criteria: [],
          interpretation_summary: "s",
          interpretation_confidence: 0.5,
          unresolved: ["field"],
        },
      }),
    );
    expect(screen.getByText(/Some context could not be resolved: field/)).toBeInTheDocument();
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
});
