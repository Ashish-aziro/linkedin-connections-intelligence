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

  it("notes a partial verification when the deadline was hit mid-verification", () => {
    renderPage(response({ search_status: "success", verification_status: "incomplete" }));
    expect(screen.getByText(/AI verification finished only partially/)).toBeInTheDocument();
  });

  it("shows a 'Verified by' banner and no partial note when verification was complete", () => {
    renderPage(
      response({ search_status: "success", verification_status: "complete", ai_model: "Claude Sonnet 5" }),
    );
    expect(screen.getByText(/Verified by Claude Sonnet 5/)).toBeInTheDocument();
    expect(screen.queryByText(/AI verification finished only partially/)).not.toBeInTheDocument();
  });

  it("renders NO candidate cards when AI verification could not be completed", () => {
    renderPage(
      response({
        search_status: "verification_incomplete",
        verification_status: "incomplete",
        connections: {
          total_candidates: 16,
          returned: 0,
          results: [],
          exact_match_count: 0,
          possible_match_count: 0,
          near_matches: [],
        },
      }),
    );
    expect(screen.getByText(/AI verification could not be completed/)).toBeInTheDocument();
    expect(screen.getByText(/No unverified results were returned/)).toBeInTheDocument();
    expect(screen.getByText("Retry Search")).toBeInTheDocument();
    expect(screen.queryByText("Person")).not.toBeInTheDocument();
    expect(screen.queryByText(/Search quality details/)).not.toBeInTheDocument();
  });

  it("shows the AI-unavailable state distinctly", () => {
    renderPage(response({ search_status: "ai_unavailable", verification_status: "incomplete" }));
    expect(screen.getByText(/AI search is temporarily unavailable/)).toBeInTheDocument();
    expect(screen.getByText("Retry Search")).toBeInTheDocument();
  });

  it("labels a fallback-verified search", () => {
    renderPage(response({ search_status: "success_with_fallback" }));
    expect(screen.getByText(/Verified by fallback AI provider/)).toBeInTheDocument();
  });

  const _judgeFull = {
    mode: "all_viable" as const,
    status: "full" as const,
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
  };

  it("does not show the partial note when verification was full", () => {
    renderPage(response({ judge_metadata: _judgeFull, verification_status: "complete" }));
    expect(screen.queryByText(/AI verification finished only partially/)).not.toBeInTheDocument();
    fireEvent.click(screen.getByText(/Search quality details/));
    expect(screen.getByText("Semantic review")).toBeInTheDocument();
  });
});
