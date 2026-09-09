import { useState } from "react";
import { Link, useLocation, useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { AlertTriangle } from "lucide-react";
import { api } from "../api/client";
import type { SearchResponse } from "../api/types";
import { ResultCard } from "../components/ResultCard";
import { Badge, Card } from "../components/ui";

type JudgeMeta = NonNullable<SearchResponse["judge_metadata"]>;
type AuditMeta = NonNullable<SearchResponse["audit_metadata"]>;

const DEGRADED = new Set(["partial", "unavailable"]);

function InterpretationPanel({ res }: { res: SearchResponse }) {
  const iq = res.interpreted_query ?? {};
  const summary = iq.interpretation_summary;
  const confidence = iq.interpretation_confidence;
  const tpc = iq.target_person_context ?? {};
  const goal = tpc.goal || [tpc.current_role, tpc.field].filter(Boolean).join(" → ");
  const unresolved = iq.unresolved ?? [];

  if (!summary && confidence == null && !goal && unresolved.length === 0) return null;

  return (
    <Card className="mt-4 px-5 py-4">
      <div className="flex items-baseline justify-between">
        <div className="text-xs font-semibold uppercase tracking-wide text-ink-faint">
          How we interpreted your search
        </div>
        {typeof confidence === "number" && (
          <span className="font-mono text-xs text-ink-faint">{Math.round(confidence * 100)}% confidence</span>
        )}
      </div>
      {summary && <p className="mt-1.5 text-sm text-ink">{summary}</p>}
      <div className="mt-2 flex flex-wrap gap-2">
        {iq.intent && <Badge>Goal: {iq.intent.replace(/_/g, " ")}</Badge>}
        {goal && <Badge tone="accent">{goal}</Badge>}
      </div>
      {unresolved.length > 0 && (
        <div className="mt-2 flex items-start gap-1.5 text-xs text-amber-700">
          <AlertTriangle size={13} className="mt-0.5 shrink-0" />
          <span>Some context could not be resolved: {unresolved.join(", ")}</span>
        </div>
      )}
    </Card>
  );
}

function verificationLabel(status?: string): string {
  if (!status) return "Unavailable";
  if (status === "full") return "Full";
  if (status === "partial") return "Partial";
  if (status === "unavailable") return "Unavailable";
  return "Not used";
}

function SearchQualityDetails({
  judge,
  audit,
}: {
  judge: JudgeMeta | null | undefined;
  audit: AuditMeta | null | undefined;
}) {
  const [open, setOpen] = useState(false);
  if (!judge && !audit) return null;

  return (
    <section className="mt-8">
      <button
        onClick={() => setOpen((v) => !v)}
        className="text-xs font-semibold uppercase tracking-wide text-ink-faint hover:text-ink"
      >
        {open ? "▾" : "▸"} Search quality details
      </button>
      {open && (
        <div className="mt-3 grid gap-2 text-sm sm:grid-cols-2">
          {judge && (
            <Card className="px-4 py-3">
              <div className="text-xs font-semibold text-ink-faint">Semantic review</div>
              <div className="mt-1">{verificationLabel(judge.status)}</div>
              <dl className="mt-2 space-y-0.5 text-xs text-ink-soft">
                <div>Candidates reviewed: {judge.judge_candidate_count}</div>
                <div>Batches: {judge.judge_successful_batches}/{judge.judge_batch_count} ok</div>
                {judge.judge_failed_batches > 0 && <div>Failed batches: {judge.judge_failed_batches}</div>}
                {judge.omitted_criteria > 0 && <div>Missing required reviews: {judge.omitted_criteria}</div>}
              </dl>
            </Card>
          )}
          {audit && (
            <Card className="px-4 py-3">
              <div className="text-xs font-semibold text-ink-faint">Final audit</div>
              <div className="mt-1">{audit.enabled ? verificationLabel(audit.status) : "Disabled"}</div>
              <dl className="mt-2 space-y-0.5 text-xs text-ink-soft">
                <div>Audit candidates: {audit.audited_candidates}</div>
                <div>Batches: {audit.successful_batches}/{audit.batch_count} ok</div>
                {audit.failed_batches > 0 && <div>Failed batches: {audit.failed_batches}</div>}
                {audit.missing_required_reviews > 0 && (
                  <div>Missing required reviews: {audit.missing_required_reviews}</div>
                )}
              </dl>
            </Card>
          )}
        </div>
      )}
    </section>
  );
}

export default function ResultsPage() {
  const { datasetId = "", searchId = "" } = useParams();
  const location = useLocation();
  const preloaded = location.state as SearchResponse | undefined;

  const q = useQuery({
    queryKey: ["search", searchId],
    queryFn: () => api.getSearch(searchId),
    initialData: preloaded && preloaded.search_id === searchId ? preloaded : undefined,
  });

  const res = q.data;

  if (!res) {
    return (
      <div>
        <Link to={`/datasets/${datasetId}/search`} className="text-sm text-accent hover:underline">
          &larr; New search
        </Link>
        <p className="mt-6 text-ink-faint">Loading…</p>
      </div>
    );
  }

  const conn = res.connections;
  const nearMatches = conn.near_matches ?? [];
  const verificationDegraded =
    DEGRADED.has(res.judge_metadata?.status ?? "") ||
    (res.audit_metadata?.enabled === true && DEGRADED.has(res.audit_metadata?.status ?? ""));

  return (
    <div>
      <Link to={`/datasets/${datasetId}/search`} className="text-sm text-accent hover:underline">
        &larr; New search
      </Link>

      <h1 className="mt-3 text-xl font-semibold">&ldquo;{res.query}&rdquo;</h1>

      <InterpretationPanel res={res} />

      {res.interpreted_query?.criteria?.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-2">
          {res.interpreted_query.criteria.map((c) => {
            const label =
              c.concept ||
              (c.values && c.values.length > 1
                ? c.values.join(c.operator === "ALL_OF" ? " AND " : " or ")
                : c.value);
            return (
              <Badge key={c.id} tone={c.required ? "accent" : "default"}>
                {c.operator === "NOT" ? "NOT " : ""}
                {label} · {c.weight.toFixed(0)}
                {c.required ? " · required" : ""}
              </Badge>
            );
          })}
        </div>
      )}

      {verificationDegraded && (
        <div className="mt-4 flex items-start gap-2 rounded-lg bg-amber-50 px-4 py-3 text-sm text-amber-800 ring-1 ring-amber-200">
          <AlertTriangle size={15} className="mt-0.5 shrink-0" />
          <span>Some AI verification was unavailable; uncertain results are shown conservatively.</span>
        </div>
      )}

      <section className="mt-8">
        <div className="flex flex-wrap items-baseline gap-3">
          <h2 className="text-sm font-semibold uppercase tracking-wide text-ink-faint">
            Your connections
          </h2>
          <span className="flex items-center gap-2 text-xs text-ink-faint">
            <Badge tone="fact">{conn.exact_match_count ?? 0} Exact</Badge>
            <Badge tone="warn">{conn.possible_match_count ?? 0} Possible</Badge>
            <span>{conn.returned} shown</span>
          </span>
        </div>

        <div className="mt-4 space-y-3">
          {conn.results.map((item) => (
            <ResultCard key={item.person_id} item={item} />
          ))}
          {conn.results.length === 0 && (
            <Card className="px-5 py-8 text-center text-sm text-ink-faint">
              {nearMatches.length > 0
                ? "No exact/possible matches were found."
                : "No connections matched this query."}
            </Card>
          )}
        </div>
      </section>

      {nearMatches.length > 0 && (
        <section className="mt-8">
          <h2 className="text-sm font-semibold uppercase tracking-wide text-ink-faint">Near matches</h2>
          <p className="mt-1 text-xs text-ink-faint">
            These people were relevant but failed one required condition.
          </p>
          <div className="mt-4 space-y-3">
            {nearMatches.map((item) => (
              <ResultCard key={`near-${item.person_id}`} item={item} variant="near" />
            ))}
          </div>
        </section>
      )}

      <SearchQualityDetails judge={res.judge_metadata} audit={res.audit_metadata} />
    </div>
  );
}
