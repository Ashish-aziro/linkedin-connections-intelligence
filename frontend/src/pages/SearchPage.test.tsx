import { describe, expect, it, vi, afterEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import SearchPage from "./SearchPage";
import { ApiError } from "../api/client";
import * as clientModule from "../api/client";

function renderPage() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <MemoryRouter initialEntries={["/datasets/d1/search"]}>
        <Routes>
          <Route path="/datasets/:datasetId/search" element={<SearchPage />} />
          <Route path="/datasets/:datasetId/search/:searchId" element={<div>RESULTS PAGE</div>} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

afterEach(() => vi.restoreAllMocks());

describe("SearchPage — full Sonnet verification failure", () => {
  it("shows a retryable error screen instead of partial results when verification is incomplete", async () => {
    vi.spyOn(clientModule.api, "searchHistory").mockResolvedValue([]);
    const search = vi
      .spyOn(clientModule.api, "search")
      .mockRejectedValue(
        new ApiError(
          503,
          { error: "verification_incomplete", message: "Full Sonnet verification could not be completed. Please retry.", retryable: true },
          "503: Full Sonnet verification could not be completed. Please retry.",
        ),
      );

    renderPage();
    fireEvent.change(screen.getByPlaceholderText(/previously worked at Amazon/i), {
      target: { value: "people with healthcare experience" },
    });
    fireEvent.click(screen.getByRole("button", { name: /^Search$/i }));

    await waitFor(() =>
      expect(screen.getByText(/Full AI verification could not be completed/i)).toBeInTheDocument(),
    );
    // no results are rendered — only the error + a retry button
    expect(screen.queryByText("RESULTS PAGE")).not.toBeInTheDocument();
    const retry = screen.getByRole("button", { name: /Retry search/i });
    fireEvent.click(retry);
    await waitFor(() => expect(search).toHaveBeenCalledTimes(2));
  });
});
