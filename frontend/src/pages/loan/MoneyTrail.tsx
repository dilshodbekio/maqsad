import type { LoanDetail, Transaction } from "../../api";
import {
  counterpartyName,
  daysBetween,
  formatCount,
  formatDate,
  formatDateTime,
  formatUzs,
  INFLOW_TYPES,
  txnTypeName,
} from "../../format";
import { useFocus } from "./evidence";

const TYPE_ORDER = [
  "disbursement",
  "cash_withdrawal",
  "supplier_payment",
  "p2p_out",
  "external_transfer",
  "p2p_in",
  "income",
];

/** Money trail: every transaction on a day axis from the disbursement, cited ones (evidence) highlighted. */
export function MoneyTrail({ detail }: { detail: LoanDetail }) {
  const { activeRef, focusRef } = useFocus();
  const { loan } = detail;
  const start = loan.disbursed_at ?? detail.transactions.find((t) => t.type === "disbursement")?.ts ?? null;
  const txns = [...detail.transactions].sort((a, b) => a.ts.localeCompare(b.ts));
  const day = (t: Transaction) => daysBetween(start, t.ts);
  const snapshotDays = daysBetween(start, loan.snapshot_date);
  const axisMax = Math.max(1, snapshotDays ?? 0, ...txns.map((t) => day(t) ?? 0));

  // transactions cited as evidence -> the reasons citing them (the "why" next to the row)
  const cited = new Map<string, string[]>();
  for (const r of detail.reasons) {
    for (const ref of r.evidence_refs) {
      if (detail.transactions.some((t) => t.txn_id === ref)) cited.set(ref, [...(cited.get(ref) ?? []), r.text_uz]);
    }
  }
  for (const e of detail.evidence) {
    if (e.kind === "txn" && !cited.has(e.ref)) cited.set(e.ref, []);
  }

  const totals = new Map<string, { sum: number; n: number }>();
  for (const t of txns) {
    const s = totals.get(t.type) ?? { sum: 0, n: 0 };
    totals.set(t.type, { sum: s.sum + t.amount_uzs, n: s.n + 1 });
  }
  const totalTypes = [...totals.keys()].sort(
    (a, b) => (TYPE_ORDER.indexOf(a) + 1 || 99) - (TYPE_ORDER.indexOf(b) + 1 || 99),
  );

  return (
    <section className="card" aria-labelledby="h-trail">
      <h2 id="h-trail">Pul izi</h2>
      <p className="small trail-intro">
        Kredit tushgan sana: <strong>{formatDate(start)}</strong>
        {snapshotDays !== null && (
          <>
            {" "}
            · holat sanasi {formatDate(loan.snapshot_date)} — <strong>{formatCount(snapshotDays)} kun</strong> o'tgan
          </>
        )}
      </p>

      {txns.length === 0 ? (
        <p className="muted">Tranzaksiyalar yo'q.</p>
      ) : (
        <>
          <div className="trail-axis" aria-hidden="true">
            {txns.map((t) => {
              const d = day(t);
              if (d === null) return null;
              return (
                <span
                  key={t.txn_id}
                  className={`axis-dot t-${t.type}${cited.has(t.txn_id) ? " cited" : ""}${activeRef === t.txn_id ? " active" : ""}`}
                  style={{ left: `${(Math.max(0, d) / axisMax) * 100}%` }}
                  title={`+${d} kun · ${txnTypeName(t.type)} · ${formatUzs(t.amount_uzs)}`}
                />
              );
            })}
            <span className="axis-label axis-start">0</span>
            <span className="axis-label axis-end">{formatCount(axisMax)} kun</span>
          </div>

          <ul className="type-totals small" aria-label="Tur bo'yicha jami">
            {totalTypes.map((type) => {
              const s = totals.get(type)!;
              return (
                <li key={type}>
                  <span className={`type-dot t-${type}`} aria-hidden="true" />
                  {txnTypeName(type)}: <strong>{formatUzs(s.sum)}</strong> ({s.n} ta)
                </li>
              );
            })}
          </ul>

          <ol className="trail-list">
            {txns.map((t) => {
              const d = day(t);
              const inflow = INFLOW_TYPES.has(t.type);
              const why = cited.get(t.txn_id);
              const cp = counterpartyName(t.counterparty_kind);
              return (
                <li
                  key={t.txn_id}
                  id={`txn-${t.txn_id}`}
                  className={`trail-row${why ? " cited" : ""}${activeRef === t.txn_id ? " is-active" : ""}`}
                >
                  <span className="trail-day">{d === null ? "—" : `+${formatCount(d)} kun`}</span>
                  <div className="trail-main">
                    <div className="trail-line">
                      <span className={`type-dot t-${t.type}`} aria-hidden="true" />
                      <span className="trail-type">{txnTypeName(t.type)}</span>
                      <span className={`trail-amount ${inflow ? "in" : "out"}`}>
                        {inflow ? "+" : "−"}
                        {formatUzs(t.amount_uzs)}
                      </span>
                    </div>
                    <div className="trail-sub small muted">
                      {formatDateTime(t.ts)}
                      {cp && ` · ${cp}`}
                    </div>
                    {why && (
                      <button type="button" className="trail-why small" onClick={() => focusRef(t.txn_id, false)}>
                        ⚑ Dalil{why.length ? `: ${why.join("; ")}` : ""}
                      </button>
                    )}
                  </div>
                </li>
              );
            })}
          </ol>
        </>
      )}
    </section>
  );
}
