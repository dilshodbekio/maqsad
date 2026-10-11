import { useCallback, useEffect, useMemo, useState } from "react";
import { api, ApiError, type LoanDetail } from "../../api";
import { COLOR_ICON, colorName, escalationLabel, escalationText, formatDate, formatRisk, formatUzs, purposeName, regionName } from "../../format";
import { t } from "../../i18n";
import { useLang } from "../../i18n";
import { lastQueueUrl, navigate, onLinkClick } from "../../router";
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
  const lang = useLang();

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
      {!ready && <BlockLoading text={t("loan.waiting")} />}
      {ready && state.kind === "loading" && (
        <section className="card">
          <h2>{t("loan.title", { id: loanId })}</h2>
          <BlockLoading text={t("loan.loading")} />
        </section>
      )}
      {ready && state.kind === "error" && (
        <section className="card">
          <h2>{t("loan.title", { id: loanId })}</h2>
          {state.status === 404 ? (
            <BlockError message={state.message} />
          ) : (
            <BlockError message={t("loan.loadError", { message: state.message })} onRetry={load} />
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
              <DraftReportBlock key={lang} loanId={loanId} />
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
      ← {t("app.backToQueue")}
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
          <span aria-hidden="true">{COLOR_ICON[detail.color]}</span> {colorName(detail.color)}
        </span>
        <span className="lh-risk">
          {t("loan.riskScore")} <strong>{formatRisk(detail.risk)}</strong>
        </span>
      </div>
      <h1 className="lh-title">
        {t("loan.title", { id: loan.loan_id })}
        {loan.borrower_name_masked && <span className="lh-name"> · {loan.borrower_name_masked}</span>}
      </h1>
      <a
        className="capture-link"
        href={`/capture?loan=${encodeURIComponent(loan.loan_id)}`}
        onClick={(e) => onLinkClick(e, `/capture?loan=${encodeURIComponent(loan.loan_id)}`)}
      >
        <span aria-hidden="true">📷</span> {t("loan.uploadPhoto")}
      </a>
      <dl className="lh-facts">
        <div>
          <dt>{t("loan.purpose")}</dt>
          <dd>{loan.purpose ? purposeName(loan.purpose) : "—"}</dd>
        </div>
        <div>
          <dt>{t("loan.amount")}</dt>
          <dd>{typeof loan.amount_uzs === "number" ? formatUzs(loan.amount_uzs) : "—"}</dd>
        </div>
        <div>
          <dt>{t("loan.region")}</dt>
          <dd>{regionName(loan.region)}</dd>
        </div>
        <div>
          <dt>{t("loan.disbursed")}</dt>
          <dd>{formatDate(loan.disbursed_at)}</dd>
        </div>
      </dl>
      {esc && (
        <div className="lh-escalation" role="note">
          <strong>
            ⬆{" "}
            {t("loan.raised", {
              from: `${COLOR_ICON[detail.base_color]} ${colorName(detail.base_color)}`,
              to: `${COLOR_ICON[detail.color]} ${colorName(detail.color)}`,
            })}
          </strong>
          <ul className="small">
            {detail.escalations
              .filter((e) => e.raises_color)
              .map((e) => (
                <li key={e.code}>{escalationText(e)}</li>
              ))}
          </ul>
          <p className="small">
            {t("loan.raisedNote", { base: colorName(detail.base_color) })}
          </p>
        </div>
      )}
      <p className="lh-model muted">
        {t("loan.modelNote", { version: detail.model_version })}
      </p>
    </section>
  );
}
