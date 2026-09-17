import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { ScoreMeter } from "./ui";

describe("ScoreMeter", () => {
  it("displays the score as a rounded percentage, not a X/max fraction", () => {
    render(<ScoreMeter label="Match" value={85} />);
    expect(screen.getByText("85%")).toBeInTheDocument();
    expect(screen.queryByText(/\/100/)).not.toBeInTheDocument();
  });

  it("computes the percentage against a non-default max and rounds it", () => {
    render(<ScoreMeter label="Custom" value={17} max={40} />);
    // 17/40 = 42.5% -> rounds to 43%
    expect(screen.getByText("43%")).toBeInTheDocument();
  });

  it("sets the progress bar width from the computed percentage", () => {
    const { container } = render(<ScoreMeter label="Match" value={85} />);
    const bar = container.querySelector<HTMLDivElement>(".rounded-full.bg-emerald-500, .rounded-full.bg-amber-500, .rounded-full.bg-slate-400, .rounded-full.bg-slate-300");
    expect(bar).not.toBeNull();
    expect(bar?.style.width).toBe("85%");
  });

  it("shows 0% instead of NaN/Infinity when max is 0", () => {
    render(<ScoreMeter label="Edge case" value={5} max={0} />);
    expect(screen.getByText("0%")).toBeInTheDocument();
  });
});
