import type { LoanDetail } from "../../api";
import { formatUzs } from "../../format";
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
      <h2 id="h-invoice">Hisob-faktura</h2>
      {!inv ? (
        <p className="muted">Bu kredit bo'yicha hisob-faktura yo'q.</p>
      ) : (
        <div id="ev-invoice" className={`invoice${invActive ? " is-active" : ""}`}>
          <dl className="kv">
            <dt>Ta'minotchi</dt>
            <dd>{inv.supplier_name ?? "—"}</dd>
            <dt>OKED</dt>
            <dd>{inv.supplier_oked ?? "—"}</dd>
            <dt>Faktura summasi</dt>
            <dd>{typeof inv.invoice_amount_uzs === "number" ? formatUzs(inv.invoice_amount_uzs) : "—"}</dd>
            {typeof loanAmount === "number" && (
              <>
                <dt>Kredit summasi</dt>
                <dd>{formatUzs(loanAmount)}</dd>
              </>
            )}
          </dl>
          <p className={`oked-badge ${matches === true ? "ok" : matches === false ? "warn" : "muted"}`}>
            {matches === true
              ? "✓ Ta'minotchining faoliyat turi (OKED) kredit maqsadiga mos"
              : matches === false
                ? "⚠ Ta'minotchining faoliyat turi (OKED) kredit maqsadiga mos emas"
                : "OKED va kredit maqsadi mosligi noma'lum"}
          </p>
          {invEv?.summary_uz && <p className="small muted">{invEv.summary_uz}</p>}
        </div>
      )}
      {graphEv && (
        <div id="ev-graph" className={`supplier-graph${graphActive ? " is-active" : ""}`}>
          <h3 className="sub-h">Ta'minotchi haqida (to'lovlar tarmog'i)</h3>
          <p className="small">{graphEv.summary_uz ?? graphEv.ref}</p>
        </div>
      )}
    </section>
  );
}
