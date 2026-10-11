import { useEffect, useState } from "react";
import { api, type DecisionAction, type LoanDetail, type Session } from "../../api";
import { COLOR_ICON, colorName, decisionName, formatRisk } from "../../format";
import { t } from "../../i18n";
import { tn } from "../../i18n/jsx";
import { BlockError, BlockLoading, type Load } from "./evidence";

const ACTIONS: { action: DecisionAction; cls: string }[] = [
  { action: "approve", cls: "btn-approve" },
  { action: "reject", cls: "btn-reject" },
  { action: "visit", cls: "btn-visit" },
];

type Sent = { kind: "idle" } | { kind: "sending"; action: DecisionAction } | { kind: "done"; action: DecisionAction; auditId: number } | { kind: "error"; message: string };

/** The officer's decision. The AI only recommends (shown here); the decision is stored next to it in the audit log. */
export function Decision({ detail, onSaved }: { detail: LoanDetail; onSaved: () => void }) {
  const [session, setSession] = useState<Load<Session>>({ kind: "loading" });
  const [comment, setComment] = useState("");
  const [sent, setSent] = useState<Sent>({ kind: "idle" });

  const loadSession = () => {
    setSession({ kind: "loading" });
    api
      .session()
      .then((data) => setSession({ kind: "ok", data }))
      .catch((e: Error) => setSession({ kind: "error", message: e.message }));
  };
  useEffect(loadSession, []);

  const submit = (action: DecisionAction) => {
    if (session.kind !== "ok") return;
    setSent({ kind: "sending", action });
    api
      .decision({ loan_id: detail.loan.loan_id, officer_id: session.data.officer_id, action, comment: comment.trim() })
      .then((r) => {
        setSent({ kind: "done", action, auditId: r.audit_id });
        setComment("");
        onSaved();
      })
      .catch((e: Error) => setSent({ kind: "error", message: e.message }));
  };

  const busy = sent.kind === "sending";
  return (
    <section className="card" aria-labelledby="h-decision">
      <h2 id="h-decision">{t("decision.title")}</h2>
      <p className="small">
        {tn("decision.aiRec", {
          color: (
            <>
              {COLOR_ICON[detail.color]} <strong>{colorName(detail.color)}</strong>
            </>
          ),
          risk: formatRisk(detail.risk),
        })}
      </p>
      {session.kind === "loading" && <BlockLoading text={t("decision.sessionLoading")} />}
      {session.kind === "error" && (
        <BlockError message={t("decision.sessionError", { message: session.message })} onRetry={loadSession} />
      )}
      {session.kind === "ok" && (
        <p className="muted small">
          {t("decision.officer")} <strong>{session.data.officer_id}</strong> {t("decision.officerDemo")}
        </p>
      )}
      <label className="comment-field">
        <span>{t("decision.comment")}</span>
        <textarea
          value={comment}
          onChange={(e) => setComment(e.target.value)}
          maxLength={2000}
          rows={3}
          placeholder={t("decision.placeholder")}
        />
      </label>
      <div className="decision-buttons">
        {ACTIONS.map((a) => (
          <button
            key={a.action}
            type="button"
            className={a.cls}
            onClick={() => submit(a.action)}
            disabled={busy || session.kind !== "ok"}
          >
            {sent.kind === "sending" && sent.action === a.action ? t("decision.saving") : t(`decision.btn.${a.action}` as const)}
          </button>
        ))}
      </div>
      {sent.kind === "done" && (
        <p className="ok decision-done" role="status">
          {t("decision.saved", { action: decisionName(sent.action), id: sent.auditId })}
        </p>
      )}
      {sent.kind === "error" && <BlockError message={t("decision.saveError", { message: sent.message })} />}
    </section>
  );
}
