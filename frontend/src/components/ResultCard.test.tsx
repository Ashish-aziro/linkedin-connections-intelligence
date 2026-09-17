import { describe, expect, it } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { ResultCard } from "./ResultCard";
import type { SearchResultItem } from "../api/types";

const base: SearchResultItem = {
  rank: 1,
  person_id: "p1",
  name: "Jane Smith",
  linkedin_url: "https://www.linkedin.com/in/jane-smith",
  profile_picture_url: null,
  current_title: "Senior Software Engineer",
  current_company: "Google",
  location: "Seattle, WA",
  is_connection: true,
  match_score: 94,
  data_confidence: 78,
  reason: "Jane previously worked at Amazon and lists AWS on her profile.",
  qualification: "exact_match",
  uncertain_criteria: [],
  unmet_criteria: [],
  matched_criteria: ["Amazon", "AWS"],
  score_breakdown: [
    {
      criterion: "Previously at Amazon",
      criterion_id: "past_amazon",
      type: "past_company",
      weight: 60,
      match_strength: 1,
      score: 60,
      required: true,
      evidence: [],
    },
    {
      criterion: "AWS",
      criterion_id: "aws",
      type: "skill",
      weight: 40,
      match_strength: 0.85,
      score: 34,
      required: false,
      evidence: [],
    },
  ],
  evidence: [
    { type: "experience", text: "SDE II at Amazon (2021–2023)", detail: {} },
    { type: "semantic", text: "Microservices — inferred from ECS work", detail: {} },
  ],
  relevant_experience: [],
  relevant_skills: [],
  relevant_education: [],
};

function make(over: Partial<SearchResultItem>): SearchResultItem {
  return { ...base, ...over };
}

describe("ResultCard", () => {
  it("shows the match score as a percentage, not a X/100 fraction", () => {
    render(<ResultCard item={base} />);
    expect(screen.getByText("Jane Smith")).toBeInTheDocument();
    expect(screen.getByText("Connection")).toBeInTheDocument();
    expect(screen.getAllByText("94%").length).toBeGreaterThan(0);
    expect(screen.queryByText(/94\/100/)).not.toBeInTheDocument();
  });

  it("expands to reveal score breakdown, evidence and fact/inference badges", () => {
    render(<ResultCard item={base} />);
    fireEvent.click(screen.getByText("Jane Smith"));

    expect(screen.getByText(/previously worked at Amazon/i)).toBeInTheDocument();
    expect(screen.getByText("Score breakdown")).toBeInTheDocument();
    // points are kept (they explain how the total was calculated) but each
    // row also shows its own percentage: 60/60 -> 100%, 34/40 -> 85%.
    expect(screen.getByText("60/60 (100%)")).toBeInTheDocument();
    expect(screen.getByText("34/40 (85%)")).toBeInTheDocument();
    // "94%" appears twice while expanded: the header Match meter and the
    // expanded Match score meter both show the same match_score.
    expect(screen.getAllByText("94%").length).toBeGreaterThanOrEqual(2);
    expect(screen.queryByText(/94\/100/)).not.toBeInTheDocument();

    expect(screen.getByText("Data confidence")).toBeInTheDocument();
    expect(screen.getByText("78%")).toBeInTheDocument();

    expect(screen.getByText("LinkedIn data")).toBeInTheDocument();
    expect(screen.getByText("AI inferred")).toBeInTheDocument();
  });

  it("shows a criterion's percentage as 0% instead of NaN/Infinity when its weight is 0", () => {
    render(
      <ResultCard
        item={make({
          score_breakdown: [
            {
              criterion: "Edge case",
              criterion_id: "edge",
              type: "skill",
              weight: 0,
              match_strength: 0,
              score: 0,
              required: false,
              evidence: [],
            },
          ],
        })}
      />,
    );
    fireEvent.click(screen.getByText("Jane Smith"));
    expect(screen.getByText("0/0 (0%)")).toBeInTheDocument();
  });

  it("shows an Exact match badge for an exact candidate", () => {
    render(<ResultCard item={make({ qualification: "exact_match" })} />);
    expect(screen.getByText("Exact match")).toBeInTheDocument();
    expect(screen.queryByText("Possible match")).not.toBeInTheDocument();
    expect(screen.queryByText(/Needs verification/)).not.toBeInTheDocument();
  });

  it("shows Possible match + Needs verification with the uncertain criterion", () => {
    render(
      <ResultCard
        item={make({
          qualification: "possible_match",
          uncertain_criteria: ["Current employer startup classification", "Mentoring experience"],
        })}
      />,
    );
    expect(screen.getByText("Possible match")).toBeInTheDocument();
    expect(screen.getByText(/Needs verification/)).toBeInTheDocument();
    expect(screen.getByText(/startup classification/)).toBeInTheDocument();
  });

  it("shows the Final audit verified badge only when llm_verified is true", () => {
    const { rerender } = render(<ResultCard item={make({ llm_verified: true })} />);
    expect(screen.getByText("Final audit verified")).toBeInTheDocument();

    rerender(<ResultCard item={make({ llm_verified: false })} />);
    expect(screen.queryByText("Final audit verified")).not.toBeInTheDocument();
  });

  it("never shows audit doubt text next to an Exact match badge", () => {
    // hardening PART 12: the badge is the FINAL applied qualification. Even
    // when the audit's own decision label is "unknown" (e.g. because facts
    // alone already proved every required criterion), the doubt text must not
    // be shown next to an Exact match badge — badge and audit text must never
    // visibly disagree.
    render(
      <ResultCard
        item={make({
          qualification: "exact_match",
          audit_decision: "unknown",
          audit_issues: ["some validator note"],
        })}
      />,
    );
    fireEvent.click(screen.getByText("Jane Smith"));
    expect(screen.getByText("Exact match")).toBeInTheDocument();
    expect(screen.queryByText("Uncertain")).not.toBeInTheDocument();
    expect(screen.queryByText("some validator note")).not.toBeInTheDocument();
  });

  it("renders a near match with its missing requirement and no verified badge", () => {
    render(
      <ResultCard
        variant="near"
        item={make({
          qualification: "not_match",
          llm_verified: true,
          unmet_criteria: ["CXO-level seniority"],
        })}
      />,
    );
    expect(screen.getByText("Near match")).toBeInTheDocument();
    expect(screen.getByText(/Missing: CXO-level seniority/)).toBeInTheDocument();
    expect(screen.queryByText("Final audit verified")).not.toBeInTheDocument();
  });
});
