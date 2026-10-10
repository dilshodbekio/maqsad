import { useCallback, useEffect, useMemo, useState } from "react";
import { api, ApiError, type LoanDetail } from "../../api";
import { COLOR_ICON, COLOR_UZ, escalationLabel, formatDate, formatRisk, formatUzs, purposeName } from "../../format";
import { lastQueueUrl, navigate } from "../../router";
import { AuditLog } from "./AuditLog";
import { Decision } from "./Decision";
import { DraftReportBlock } from "./DraftReportBlock";
import { anchorId, BlockError, BlockLoading, FocusContext, type FocusApi, type Load } from "./evidence";
import { InvoiceBlock } from "./InvoiceBlock";
import { MoneyTrail } from "./MoneyTrail";
import { Photos } from "./Photos";
import { Reasons } from "./Reasons";
import "./loan.css";

/** /loan/:id — "why is this loan red": reasons with evidence, money trail, photos, invoice, report, decision, audit. */
export function LoanPage({ loanId, ready }: { loanId: string; ready: boolean }) {
  const [state, setState] = useState<Load<LoanDetail>>({ kind: "loading" });
  const [activeRef, setActiveRef] = useState<string | null>(null);
  const [auditVersion, setAuditVersion] = useState(0);

  const load = useCallback(() => {
    setState({ kind: "loading" });
    api
      .loan(loanId)
      .then((data) => setState({ kind: "ok", data }))
      .catch((e: Error) => setState({ kind: "error", message: e.message, status: e instanceof ApiError ? e.status : 0 }));
  }, [loanId]);

  useEffect(() => {
    if (ready) load();
  }, [ready, load]);

  useEffect(() => {
    window.scrollTo(0, 0); // newer browsers return a Promise here: never return it as an effect cleanup
  }, [loanId]);

  const evidence = useMemo(
    () => new Map(state.kind === "ok" ? state.data.evidence.map((e) => [e.ref, e] as const) : []),
    [state],
  );

  const focus: FocusApi = useMemo(
    () => ({
      activeRef,
      evidence,
      focusRef: (ref, scroll = true) => {
        setActiveRef(ref);
        const id = anchorId(evidence.get(ref));
        if (!scroll || !id) return;
        requestAnimationFrame(() =>
          document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "center" }),
        );
      },
    }),
    [activeRef, evidence],
  );

  return (
    <div className="loan-page">
      <BackLink />
      {!ready && <BlockLoading text="Model tayyorlanmoqda… Kredit bir necha soniyadan keyin ochiladi." />}
      {ready && state.kind === "loading" && (
        <section className="card">
          <h2>Kredit {loanId}</h2>
          <BlockLoading text="Kredit ma'lumotlari yuklanmoqda…" />
        </section>
      )}
      {ready && state.kind === "error" && (
        <section className="card">
          <h2>Kredit {loanId}</h2>
          {state.status === 404 ? (
            <BlockError message={state.message} />
          ) : (
            <BlockError message={`Kredit ma'lumotlari yuklanmadi: ${state.message}`} onRetry={load} />
          )}
        </section>
      )}
      {ready && state.kind === "ok" && (
        <FocusContext.Provider value={focus}>
          <LoanHeader detail={state.data} />
          <div className="loan-layout">
            <div className="loan-main">
              <Reasons detail={state.data} />
              <MoneyTrail detail={state.data} />
              <Photos detail={state.data} />
              <InvoiceBlock detail={state.data} />
            </div>
            <div className="loan-side">
              <Decision detail={state.data} onSaved={() => setAuditVersion((v) => v + 1)} />
              <DraftReportBlock loanId={loanId} />
              <AuditLog loanId={loanId} version={auditVersion} />
            </div>
          </div>
        </FocusContext.Provider>
      )}
    </div>
  );
}

function BackLink() {
  const href = lastQueueUrl(); // the queue with the filters it had
  return (
    <a
      href={href}
      className="back-link"
      onClick={(e) => {
        if (e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
        e.preventDefault();
        navigate(href);
      }}
    >
      ← Navbatga qaytish
    </a>
  );
}

function LoanHeader({ detail }: { detail: LoanDetail }) {
  const { loan } = detail;
  const esc = escalationLabel(detail);
  return (
    <section className={`card loan-header border-top-${detail.color}`}>
      <div className="lh-top">
        <span className={`color-pill pill-${detail.color}`}>
          <span aria-hidden="true">{COLOR_ICON[detail.color]}</span> {COLOR_UZ[detail.color]}
        </span>
        <span className="lh-risk">
          Xavf bahosi <strong>{formatRisk(detail.risk)}</strong>
        </span>
      </div>
      <h1 className="lh-title">
        Kredit {loan.loan_id}
        {loan.borrower_name_masked && <span className="lh-name"> · {loan.borrower_name_masked}</span>}
      </h1>
      <dl className="lh-facts">
        <div>
          <dt>Maqsad</dt>
          <dd>{loan.purpose ? purposeName(loan.purpose) : "—"}</dd>
        </div>
        <div>
          <dt>Summa</dt>
          <dd>{typeof loan.amount_uzs === "number" ? formatUzs(loan.amount_uzs) : "—"}</dd>
        </div>
        <div>
          <dt>Hudud</dt>
          <dd>{loan.region ?? "—"}</dd>
        </div>
        <div>
          <dt>Berilgan sana</dt>
          <dd>{formatDate(loan.disbursed_at)}</dd>
        </div>
      </dl>
      {esc && (
        <div className="lh-escalation" role="note">
          <strong>
            ⬆ Rang ko'tarildi: {COLOR_ICON[detail.base_color]} {COLOR_UZ[detail.base_color]} → {COLOR_ICON[detail.color]}{" "}
            {COLOR_UZ[detail.color]}
          </strong>
          <ul className="small">
            {detail.escalations
              .filter((e) => e.raises_color)
              .map((e) => (
                <li key={e.code}>{e.text_uz}</li>
              ))}
          </ul>
          <p className="small">
            Pul izi bo'yicha rang — {COLOR_UZ[detail.base_color]}. Xavf bahosi o'zgarmaydi, faqat navbatdagi o'rin
            ko'tariladi.
          </p>
        </div>
      )}
      <p className="lh-model muted">
        Model: {detail.model_version} · Xavf bahosi — tartiblash uchun model bahosi (0..1). Yakuniy qarorni bank xodimi
        qabul qiladi.
      </p>
    </section>
  );
}
