import { useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { useMutation, useQuery } from "@tanstack/react-query";
import { AlertTriangle } from "lucide-react";
import { api, ApiError } from "../api/client";
import { Button, Card } from "../components/ui";

const EXAMPLES = [
  "People who previously worked at Amazon",
  "People currently at Google who know Java",
  "Senior Java backend engineers",
  "People with AWS and distributed systems experience",
  "People who studied at Georgia Tech",
  "Who should I reach out to for AWS architecture advice?",
];

export default function SearchPage() {
  const { datasetId = "" } = useParams();
  const nav = useNavigate();
  const [q, setQ] = useState("");

  const run = useMutation({
    mutationFn: () => api.search(datasetId, q),
    onSuccess: (res) => nav(`/datasets/${datasetId}/search/${res.search_id}`, { state: res }),
  });

  const history = useQuery({
    queryKey: ["searches", datasetId],
    queryFn: () => api.searchHistory(datasetId),
  });

  return (
    <div className="mx-auto max-w-2xl">
      <h1 className="text-2xl font-semibold">Search your professional network</h1>
      <p className="mt-2 text-sm text-ink-soft">
        Ask in plain English. Results rank the people you already know, backed by evidence from their profiles.
      </p>

      <Card className="mt-6 p-4">
        <textarea
          value={q}
          onChange={(e) => setQ(e.target.value)}
          rows={3}
          placeholder="People who previously worked at Amazon and know AWS…"
          className="w-full resize-none rounded-lg border border-slate-200 p-3 text-sm outline-none focus:border-accent focus:ring-2 focus:ring-accent-ring"
        />
        <div className="mt-3 flex justify-end">
          <Button type="submit" onClick={() => run.mutate()} disabled={q.trim().length < 3 || run.isPending}>
            {run.isPending ? "Searching…" : "Search"}
          </Button>
        </div>
      </Card>

      {run.error &&
        (run.error instanceof ApiError && run.error.code === "verification_incomplete" ? (
          <Card className="mt-4 border-amber-200 bg-amber-50 px-5 py-6 text-center">
            <AlertTriangle size={20} className="mx-auto text-amber-600" />
            <p className="mt-2 text-sm font-medium text-amber-900">
              Full AI verification could not be completed.
            </p>
            <p className="mx-auto mt-1 max-w-md text-xs text-amber-800">
              Every candidate is reviewed by Claude Sonnet before results are shown. That review
              did not finish — no partial results are displayed.
            </p>
            <div className="mt-4">
              <Button onClick={() => run.mutate()} disabled={run.isPending}>
                {run.isPending ? "Retrying…" : "Retry search"}
              </Button>
            </div>
          </Card>
        ) : (
          <p className="mt-3 text-sm text-red-600">{String(run.error)}</p>
        ))}

      <div className="mt-6 flex flex-wrap gap-2">
        {EXAMPLES.map((e) => (
          <button
            key={e}
            onClick={() => setQ(e)}
            className="rounded-full border border-slate-200 bg-white px-3 py-1.5 text-xs text-ink-soft hover:border-accent hover:text-accent"
          >
            {e}
          </button>
        ))}
      </div>

      {history.data && history.data.length > 0 && (
        <div className="mt-10">
          <h2 className="text-xs font-semibold uppercase tracking-wide text-ink-faint">Recent searches</h2>
          <ul className="mt-2 divide-y divide-slate-100 rounded-xl border border-slate-200 bg-card">
            {history.data.map((h) => (
              <li key={h.search_id}>
                <Link
                  to={`/datasets/${datasetId}/search/${h.search_id}`}
                  className="flex items-center justify-between px-4 py-2.5 text-sm hover:bg-slate-50"
                >
                  <span className="truncate">{h.query}</span>
                  <span className="ml-3 shrink-0 text-xs text-ink-faint">
                    {new Date(h.created_at).toLocaleDateString()} · {h.total_candidates} matched
                  </span>
                </Link>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
