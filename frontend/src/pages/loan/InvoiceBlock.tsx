import type { LoanDetail } from "../../api";
import { evidenceSummary, formatUzs } from "../../format";
import { t } from "../../i18n";
import { field, useFocus } from "./evidence";

/** Invoice (supplier, OKED, amount) + what the supplier graph evidence says about the supplier. */
export function InvoiceBlock({ detail }: { detail: LoanDetail }) {
  const { activeRef, evidence } = useFocus();
  const inv = detail.invoice;
  const invEv = detail.evidence.find((e) => e.kind === "invoice");
  const graphEv = detail.evidence.find((e) => e.kind === "graph");
  // OKED <-> purpose: the evidence item when the model cites it, otherwise the invoice itself
  const matches = invEv ? field<boolean | null>(invEv, "oked_matches_purpose") : inv?.oked_matches_purpose;
  const invActive = activeRef !== null && evidence.get(activeRef)?.kind === "invoice";
  const graphActive = activeRef !== null && evidence.get(activeRef)?.kind === "graph";
  const loanAmount = detail.loan.amount_uzs;

  return (
    <section className="card" aria-labelledby="h-invoice">
      <h2 id="h-invoice">{t("invoice.title")}</h2>
      {!inv ? (
        <p className="muted">{t("invoice.none")}</p>
      ) : (
        <div id="ev-invoice" className={`invoice${invActive ? " is-active" : ""}`}>
          <dl className="kv">
            <dt>{t("invoice.supplier")}</dt>
            <dd>{inv.supplier_name ?? "—"}</dd>
            <dt>OKED</dt>
            <dd>{inv.supplier_oked ?? "—"}</dd>
            <dt>{t("invoice.amount")}</dt>
            <dd>{typeof inv.invoice_amount_uzs === "number" ? formatUzs(inv.invoice_amount_uzs) : "—"}</dd>
            {typeof loanAmount === "number" && (
              <>
                <dt>{t("invoice.loanAmount")}</dt>
                <dd>{formatUzs(loanAmount)}</dd>
              </>
            )}
          </dl>
          <p className={`oked-badge ${matches === true ? "ok" : matches === false ? "warn" : "muted"}`}>
            {matches === true ? t("invoice.okedOk") : matches === false ? t("invoice.okedBad") : t("invoice.okedUnknown")}
          </p>
          {invEv && evidenceSummary(invEv) && <p className="small muted">{evidenceSummary(invEv)}</p>}
        </div>
      )}
      {graphEv && (
        <div id="ev-graph" className={`supplier-graph${graphActive ? " is-active" : ""}`}>
          <h3 className="sub-h">{t("invoice.graphTitle")}</h3>
          <p className="small">{evidenceSummary(graphEv) || graphEv.ref}</p>
        </div>
      )}
    </section>
  );
}
