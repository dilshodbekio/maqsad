import { useState } from "react";
import { api, ApiError, type DraftReport } from "../../api";
import { BlockError, BlockLoading, EvidenceChip } from "./evidence";

type State =
  | { kind: "idle" }
  | { kind: "loading" }
  | { kind: "unavailable"; detail: string }
  | { kind: "error"; message: string }
  | { kind: "ok"; data: DraftReport };

/** LLM draft of the inspection report. Every fact is linked to its evidence; unverified facts are kept, marked. */
export function DraftReportBlock({ loanId }: { loanId: string }) {
  const [state, setState] = useState<State>({ kind: "idle" });

  const load = () => {
    setState({ kind: "loading" });
    api
      .draftReport(loanId)
      .then((data) => setState({ kind: "ok", data }))
      .catch((e: Error) =>
        setState(
          e instanceof ApiError && e.status === 503
            ? { kind: "unavailable", detail: e.message }
            : { kind: "error", message: e.message },
        ),
      );
  };

  return (
    <section className="card" aria-labelledby="h-report">
      <h2 id="h-report">Dalolatnoma</h2>
      {state.kind !== "ok" && (
        <button type="button" onClick={load} disabled={state.kind === "loading"}>
          {state.kind === "loading" ? "Tayyorlanmoqda…" : "Dalolatnoma qoralamasi"}
        </button>
      )}
      {state.kind === "idle" && (
        <p className="muted small">AI faqat shu kreditning dalillari asosida qoralama yozadi; har bir fakt kodda tekshiriladi.</p>
      )}
      {state.kind === "loading" && <BlockLoading text="Qoralama tayyorlanmoqda, bu bir necha soniya olishi mumkin…" />}
      {state.kind === "unavailable" && (
        <div className="notice" role="status">
          <p>
            <strong>Qoralama hozircha mavjud emas.</strong>
          </p>
          <p className="muted small">{state.detail}</p>
        </div>
      )}
      {state.kind === "error" && <BlockError message={`Qoralama tayyorlanmadi: ${state.message}`} onRetry={load} />}
      {state.kind === "ok" && <Report data={state.data} onRefresh={load} />}
    </section>
  );
}

function Report({ data, onRefresh }: { data: DraftReport; onRefresh: () => void }) {
  const verified = data.facts.filter((f) => !f.unverified);
  const unverified = data.facts.filter((f) => f.unverified);
  return (
    <div className="report">
      <p className="report-badge small">AI qoralamasi — yakuniy qarorni bank xodimi qabul qiladi</p>
      <div className="report-text">
        {data.text_uz
          .split(/\n+/)
          .filter((p) => p.trim())
          .map((p, i) => (
            <p key={i}>{p}</p>
          ))}
      </div>

      {verified.length > 0 && (
        <>
          <h3 className="sub-h">Dalil bilan tasdiqlangan faktlar</h3>
          <ul className="facts">
            {verified.map((f, i) => (
              <li key={i}>
                <span>{f.claim}</span> <EvidenceChip refId={f.evidence_ref} />
              </li>
            ))}
          </ul>
        </>
      )}
      {unverified.length > 0 && (
        <>
          <h3 className="sub-h warn">⚠ Tasdiqlanmagan faktlar ({unverified.length})</h3>
          <p className="muted small">
            Bu faktlar keltirilgan dalilda topilmadi (havola yo'q yoki raqam mos emas). Ularga tayanmang.
          </p>
          <ul className="facts unverified">
            {unverified.map((f, i) => (
              <li key={i}>
                <span>{f.claim}</span> {f.evidence_ref ? <EvidenceChip refId={f.evidence_ref} /> : null}
              </li>
            ))}
          </ul>
        </>
      )}
      {data.facts.length === 0 && <p className="muted small">Qoralamada alohida fakt ro'yxati yo'q.</p>}

      <p className="report-meta">
        Model: {data.model} · kechikish {Math.round(data.latency_ms)} ms · narx{" "}
        {data.cost_usd === null ? "noma'lum" : `$${data.cost_usd.toFixed(4)}`}
      </p>
      <button type="button" className="secondary" onClick={onRefresh}>
        Qayta yaratish
      </button>
    </div>
  );
}
