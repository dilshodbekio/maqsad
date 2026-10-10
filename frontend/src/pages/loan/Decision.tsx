import { useEffect, useState } from "react";
import { api, type DecisionAction, type LoanDetail, type Session } from "../../api";
import { COLOR_ICON, COLOR_UZ, DECISION_UZ, formatRisk } from "../../format";
import { BlockError, BlockLoading, type Load } from "./evidence";

const ACTIONS: { action: DecisionAction; label: string; cls: string }[] = [
  { action: "approve", label: "Tasdiqlash", cls: "btn-approve" },
  { action: "reject", label: "Rad etish", cls: "btn-reject" },
  { action: "visit", label: "Tashrif belgilash", cls: "btn-visit" },
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
      <h2 id="h-decision">Qaror</h2>
      <p className="small">
        AI tavsiyasi: {COLOR_ICON[detail.color]} <strong>{COLOR_UZ[detail.color]}</strong>, xavf bahosi{" "}
        {formatRisk(detail.risk)}. Yakuniy qarorni siz qabul qilasiz.
      </p>
      {session.kind === "loading" && <BlockLoading text="Xodim ma'lumoti yuklanmoqda…" />}
      {session.kind === "error" && (
        <BlockError message={`Xodim aniqlanmadi, qaror yuborib bo'lmaydi: ${session.message}`} onRetry={loadSession} />
      )}
      {session.kind === "ok" && (
        <p className="muted small">
          Xodim: <strong>{session.data.officer_id}</strong> (demo: brauzeringiz sandbox'idan)
        </p>
      )}
      <label className="comment-field">
        <span>Izoh</span>
        <textarea
          value={comment}
          onChange={(e) => setComment(e.target.value)}
          maxLength={2000}
          rows={3}
          placeholder="Masalan: joyiga borib uskunani ko'rish kerak"
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
            {sent.kind === "sending" && sent.action === a.action ? "Saqlanmoqda…" : a.label}
          </button>
        ))}
      </div>
      {sent.kind === "done" && (
        <p className="ok decision-done" role="status">
          ✓ Qaror saqlandi: {DECISION_UZ[sent.action]} (audit yozuvi #{sent.auditId})
        </p>
      )}
      {sent.kind === "error" && <BlockError message={`Qaror saqlanmadi: ${sent.message}`} />}
    </section>
  );
}
