import type {
  DatasetStatus,
  DatasetSummary,
  PersonListItem,
  SearchResponse,
  UploadReport,
} from "./types";

const BASE = "/api";

export class ApiError extends Error {
  status: number;
  detail: unknown;
  constructor(status: number, detail: unknown, message: string) {
    super(message);
    this.status = status;
    this.detail = detail;
  }
  /** FastAPI `detail` may be a string or a structured object. */
  get code(): string | undefined {
    return typeof this.detail === "object" && this.detail !== null
      ? (this.detail as Record<string, unknown>).error as string | undefined
      : undefined;
  }
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(BASE + path, init);
  if (!res.ok) {
    let detail: unknown = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? body;
    } catch {
      /* ignore */
    }
    const msg =
      typeof detail === "object" && detail !== null && "message" in (detail as object)
        ? String((detail as Record<string, unknown>).message)
        : typeof detail === "string"
          ? detail
          : JSON.stringify(detail);
    throw new ApiError(res.status, detail, `${res.status}: ${msg}`);
  }
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

export const api = {
  health: () => req<Record<string, unknown>>("/health"),

  listDatasets: () => req<DatasetSummary[]>("/datasets"),

  getDataset: (id: string) => req<DatasetSummary>(`/datasets/${id}`),

  uploadCsv: (file: File, name?: string) => {
    const fd = new FormData();
    fd.append("file", file);
    if (name) fd.append("name", name);
    return req<UploadReport>("/datasets", { method: "POST", body: fd });
  },

  deleteDataset: (id: string) => req<void>(`/datasets/${id}`, { method: "DELETE" }),

  datasetStatus: (id: string) => req<DatasetStatus>(`/datasets/${id}/status`),

  startEnrichment: (id: string) =>
    req<{ started: boolean; job_id?: string; pending: number; message?: string }>(
      `/datasets/${id}/enrich`,
      { method: "POST" },
    ),

  listPeople: (id: string) => req<PersonListItem[]>(`/datasets/${id}/people`),

  search: (datasetId: string, query: string) =>
    req<SearchResponse>("/search", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ dataset_id: datasetId, query }),
    }),

  getSearch: (searchId: string) => req<SearchResponse>(`/search/${searchId}`),

  exportUrl: (datasetId: string) => `${BASE}/datasets/${datasetId}/export`,

  searchHistory: (datasetId: string) =>
    req<
      Array<{
        search_id: string;
        query: string;
        created_at: string;
        total_candidates: number;
        llm_provider: string | null;
      }>
    >(`/datasets/${datasetId}/searches`),
};
