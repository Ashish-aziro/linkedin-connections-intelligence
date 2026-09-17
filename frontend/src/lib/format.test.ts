import { describe, expect, it } from "vitest";
import { formatPercent } from "./format";

describe("formatPercent", () => {
  it("rounds value/max*100 to the nearest whole percent", () => {
    expect(formatPercent(85, 100)).toBe(85);
    expect(formatPercent(34, 40)).toBe(85);
    expect(formatPercent(60, 60)).toBe(100);
    expect(formatPercent(17, 40)).toBe(43); // 42.5 -> rounds up
  });

  it("defaults max to 100", () => {
    expect(formatPercent(92)).toBe(92);
  });

  it("returns 0 instead of NaN/Infinity when max is 0 or negative", () => {
    expect(formatPercent(5, 0)).toBe(0);
    expect(formatPercent(5, -10)).toBe(0);
  });
});
