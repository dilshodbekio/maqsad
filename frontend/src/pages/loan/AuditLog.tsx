import { useCallback, useEffect, useState } from "react";
import { api, type AuditEntry } from "../../api";
import { actorName, aiRecommendationText, decisionName, formatLocalDateTime } from "../../format";
import { t } from "../../i18n";
import { BlockError, BlockLoading, type Load } from "./evidence";

/** Append-only audit log of this loan, as seen from the caller's sandbox. `version` changes -> reload. */
export function AuditLog({ loanId, version }: { loanId: string; version: number }) {
  const [state, setState] = useState<Load<AuditEntry[]>>({ kind: "loading" });

  const load = useCallback(() => {
    setState((s) => (s.kind === "ok" ? s : { kind: "loading" })); // keep rows visible while refreshing
    api
      .audit(loanId)
      .then((data) => setState({ kind: "ok", data }))
      .catch((e: Error) => setState({ kind: "error", message: e.message }));
  }, [loanId]);

  useEffect(load, [load, version]);

  return (
    <section className="card" aria-labelledby="h-audit">
      <h2 id="h-audit">{t("audit.title")}</h2>
      <p className="muted small">{t("audit.intro")}</p>
      {state.kind === "loading" && <BlockLoading text={t("audit.loading")} />}
      {state.kind === "error" && <BlockError message={t("audit.error", { message: state.message })} onRetry={load} />}
      {state.kind === "ok" && state.data.length === 0 && <p className="muted">{t("audit.none")}</p>}
      {state.kind === "ok" && state.data.length > 0 && (
        <ol className="audit-list">
          {[...state.data].reverse().map((r) => (
            <li key={r.inputs_hash + r.ts} className="audit-row">
              <div className="audit-top">
                <strong>{decisionName(r.action)}</strong>
                <span className="muted small">{formatLocalDateTime(r.ts)}</span>
              </div>
              <dl className="kv small">
                <dt>{t("audit.who")}</dt>
                <dd>{actorName(r.actor)}</dd>
                <dt>{t("audit.aiRec")}</dt>
                <dd>{aiRecommendationText(r.ai_recommendation)}</dd>
                <dt>{t("audit.model")}</dt>
                <dd>{r.model_version ?? "—"}</dd>
                <dt>{t("audit.hash")}</dt>
                <dd className="hash" title={r.inputs_hash}>
                  {r.inputs_hash.slice(0, 12)}…
                </dd>
              </dl>
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}
