import { useState } from "react";
import type { LoanDetail, LoanReason } from "../../api";
import { anchorId, EvidenceChip, useFocus } from "./evidence";

/** Order: policy reasons that raised the colour (they explain it), then money-trail reasons by weight, then the rest. */
function sortReasons(d: LoanDetail): LoanReason[] {
  const raising = new Set(d.escalations.filter((e) => e.raises_color).map((e) => e.code));
  const group = (r: LoanReason) => (r.weight === null ? (raising.has(r.code) ? 0 : 2) : 1);
  return d.reasons
    .map((r, i) => ({ r, i }))
    .sort((a, b) => group(a.r) - group(b.r) || (b.r.weight ?? 0) - (a.r.weight ?? 0) || a.i - b.i)
    .map((x) => x.r);
}

export function Reasons({ detail }: { detail: LoanDetail }) {
  const reasons = sortReasons(detail);
  const raising = new Set(detail.escalations.filter((e) => e.raises_color).map((e) => e.code));
  const maxWeight = Math.max(0, ...reasons.map((r) => r.weight ?? 0));

  return (
    <section className="card" aria-labelledby="h-reasons">
      {detail.color === "green" ? (
        <>
          <h2 id="h-reasons">Xavf bahosiga ta'sir qilgan omillar</h2>
          <p className="muted small">Yashil: navbat bo'yicha birinchi navbatda tekshirish talab etilmaydi.</p>
        </>
      ) : (
        <h2 id="h-reasons">Nega birinchi navbatda tekshirish kerak</h2>
      )}
      {reasons.length === 0 ? (
        <p className="muted">Sezilarli xavf belgisi aniqlanmadi.</p>
      ) : (
        <ol className="reasons">
          {reasons.map((r, i) => (
            <ReasonItem
              key={`${r.code}-${i}`}
              reason={r}
              raisedColor={r.weight === null && raising.has(r.code)}
              maxWeight={maxWeight}
            />
          ))}
        </ol>
      )}
      <p className="muted small">
        Og'irlik — sababning xavf bahosini oshirishdagi ulushi (0..1). Rasm siyosati sabablari xavf bahosini
        o'zgartirmaydi, faqat navbatdagi rangni ko'taradi.
      </p>
    </section>
  );
}

function ReasonItem({ reason, raisedColor, maxWeight }: { reason: LoanReason; raisedColor: boolean; maxWeight: number }) {
  const { evidence, focusRef } = useFocus();
  const [open, setOpen] = useState<string | null>(null);

  const toggle = (ref: string) => {
    const next = open === ref ? null : ref;
    setOpen(next);
    if (next) focusRef(next, false); // highlight it everywhere, without leaving the reasons
  };
  const ev = open ? evidence.get(open) : undefined;
  const target = anchorId(ev);

  return (
    <li className={`reason-item${reason.weight === null ? " policy" : ""}`}>
      <div className="reason-text">{reason.text_uz}</div>
      <div className="reason-meta small">
        {reason.weight === null ? (
          <span className="policy-tag">
            {reason.code.startsWith("PHOTO_") ? "Rasm siyosati" : "Dalil siyosati"}
            {raisedColor && " — rangni ko'tardi"}
          </span>
        ) : (
          <span className="weight">
            <span className="weight-bar" aria-hidden="true">
              <span style={{ width: `${maxWeight > 0 ? (reason.weight / maxWeight) * 100 : 0}%` }} />
            </span>
            og'irligi {reason.weight.toFixed(2)}
          </span>
        )}
      </div>
      {reason.evidence_refs.length > 0 ? (
        <div className="chips">
          <span className="muted small">Dalil:</span>
          {reason.evidence_refs.map((ref) => (
            <EvidenceChip key={ref} refId={ref} onClick={() => toggle(ref)} />
          ))}
        </div>
      ) : (
        <p className="muted small no-refs">Alohida dalil havolasi yo'q (pul izi bo'yicha umumiy ko'rsatkich).</p>
      )}
      {open && ev && (
        <div className="ev-open">
          <p>{ev.summary_uz ?? open}</p>
          {target && (
            <button type="button" className="link-button" onClick={() => focusRef(open)}>
              Sahifada ko'rsatish ↓
            </button>
          )}
        </div>
      )}
    </li>
  );
}
