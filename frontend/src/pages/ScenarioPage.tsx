import { useEffect, useRef, useState, type FormEvent } from "react";
import { api, type Color, type Scenario, type ScenarioPreset, type ScenarioScore, type ScenarioTx } from "../api";
import { COLOR_ICON, COLOR_UZ, formatRisk, purposeName, txnTypeName } from "../format";
import { onLinkClick } from "../router";
import "./scenario.css";

const PURPOSES = ["tikuv_dastgohi", "chorva", "issiqxona", "parranda", "asalarichilik"];
const TX_TYPES = ["cash_withdrawal", "supplier_payment", "p2p_out", "p2p_in", "external_transfer", "income"];
const MAX_TX = 60;

type Row = { key: number; day: string; type: string; amount: string; kind: string | null };

type Form = {
  purpose: string;
  amount: string;
  snapshotDay: string;
  hasInvoice: boolean;
  invoice: string;
  region: string;
  supplierMode: "existing" | "new";
  supplierId: string;
  oked: string;
  registeredDays: string;
  rows: Row[];
};

let rowKey = 0;
const newRow = (r: Partial<Row> = {}): Row => ({ key: ++rowKey, day: "1", type: "cash_withdrawal", amount: "", kind: null, ...r });

const EMPTY: Form = {
  purpose: "chorva",
  amount: "",
  snapshotDay: "60",
  hasInvoice: false,
  invoice: "",
  region: "",
  supplierMode: "new",
  supplierId: "",
  oked: "",
  registeredDays: "0",
  rows: [],
};

function fromScenario(s: Scenario): Form {
  const sup = s.supplier;
  return {
    purpose: s.purpose,
    amount: String(s.amount_uzs),
    snapshotDay: String(s.snapshot_day),
    hasInvoice: s.invoice_amount_uzs != null,
    invoice: s.invoice_amount_uzs != null ? String(s.invoice_amount_uzs) : "",
    region: s.region ?? "",
    supplierMode: sup.mode,
    supplierId: sup.mode === "existing" ? sup.supplier_id : "",
    oked: sup.mode === "new" ? sup.oked : "",
    registeredDays: sup.mode === "new" ? String(sup.registered_days_before) : "0",
    rows: s.transactions.map((t) =>
      newRow({ day: String(t.day), type: t.type, amount: String(t.amount_uzs), kind: t.counterparty_kind ?? null }),
    ),
  };
}

const digits = (v: string) => v.replace(/[\s']/g, "");

function toInt(v: string, what: string): number {
  const d = digits(v);
  if (!/^[0-9]+$/.test(d)) throw new Error(`${what}: butun son kiriting.`);
  return Number(d);
}

function toScenario(f: Form): Scenario {
  const rows: ScenarioTx[] = f.rows.map((r, i) => {
    const t: ScenarioTx = {
      day: toInt(r.day, `${i + 1}-tranzaksiya kuni`),
      type: r.type,
      amount_uzs: toInt(r.amount, `${i + 1}-tranzaksiya summasi`),
    };
    if (r.kind) t.counterparty_kind = r.kind;
    return t;
  });
  const supplier: Scenario["supplier"] =
    f.supplierMode === "existing"
      ? { mode: "existing", supplier_id: f.supplierId.trim() }
      : { mode: "new", oked: f.oked.trim(), registered_days_before: toInt(f.registeredDays, "Ta'minotchi ro'yxatdan o'tganiga (kun)") };
  const s: Scenario = {
    purpose: f.purpose,
    amount_uzs: toInt(f.amount, "Kredit summasi"),
    snapshot_day: toInt(f.snapshotDay, "Baholash kuni"),
    supplier,
    transactions: rows,
  };
  if (f.hasInvoice) s.invoice_amount_uzs = toInt(f.invoice, "Hisob-faktura summasi");
  if (f.region) s.region = f.region;
  return s;
}

type Load<T> = { kind: "loading" } | { kind: "error"; message: string } | { kind: "ok"; data: T };

export function ScenarioPage({ ready }: { ready: boolean }) {
  const [presets, setPresets] = useState<Load<ScenarioPreset[]>>({ kind: "loading" });
  const [regions, setRegions] = useState<string[]>([]);
  const [form, setForm] = useState<Form>(EMPTY);
  const [presetId, setPresetId] = useState("");
  const [run, setRun] = useState<Load<ScenarioScore> | { kind: "idle" }>({ kind: "idle" });
  const seq = useRef(0);

  const loadPresets = () => {
    setPresets({ kind: "loading" });
    api
      .scenarioPresets()
      .then((data) => setPresets({ kind: "ok", data }))
      .catch((e: Error) => setPresets({ kind: "error", message: e.message }));
  };

  useEffect(() => {
    if (!ready) return;
    loadPresets();
    api
      .queueFacets()
      .then((f) => setRegions(f.regions))
      .catch(() => undefined); // the region field is optional: without the list it stays "not given"
  }, [ready]);

  const patch = (p: Partial<Form>) => setForm((f) => ({ ...f, ...p }));
  const patchRow = (key: number, p: Partial<Row>) =>
    setForm((f) => ({ ...f, rows: f.rows.map((r) => (r.key === key ? { ...r, ...p } : r)) }));

  const choosePreset = (id: string) => {
    setPresetId(id);
    setRun({ kind: "idle" });
    if (presets.kind !== "ok") return;
    const p = presets.data.find((x) => x.id === id);
    if (p) setForm(fromScenario(p.scenario));
  };

  const selected = presets.kind === "ok" ? presets.data.find((p) => p.id === presetId) : undefined;
  if (form.region && !regions.includes(form.region)) regions.unshift(form.region);

  const submit = (e: FormEvent) => {
    e.preventDefault();
    let scenario: Scenario;
    try {
      scenario = toScenario(form);
    } catch (err) {
      setRun({ kind: "error", message: (err as Error).message });
      return;
    }
    const id = ++seq.current;
    setRun({ kind: "loading" });
    api
      .scoreScenario(scenario)
      .then((data) => id === seq.current && setRun({ kind: "ok", data }))
      .catch((err: Error) => id === seq.current && setRun({ kind: "error", message: err.message }));
  };

  return (
    <div className="scenario">
      <a className="back-link" href="/" onClick={(e) => onLinkClick(e, "/")}>
        ← Navbatga qaytish
      </a>
      <section className="card">
        <h2>🧪 Ssenariy sinash</h2>
        <p className="scn-note">Bu gipotetik ssenariy, portfelga yozilmaydi.</p>
        <p className="muted small">
          Taxminiy kredit va pul harakatlarini kiriting: model uni xuddi portfeldagi kredit kabi baholaydi. Natija —
          birinchi navbatda tekshirish uchun belgi, yakuniy qarorni xodim qabul qiladi.
        </p>

        {!ready && <p className="muted">Model tayyorlanmoqda…</p>}
        {ready && presets.kind === "loading" && <p className="muted">Namunalar yuklanmoqda…</p>}
        {ready && presets.kind === "error" && (
          <div className="error">
            <p>Namunalar yuklanmadi: {presets.message}</p>
            <button onClick={loadPresets}>Qayta urinish</button>
          </div>
        )}
        {presets.kind === "ok" && (
          <label className="capture-field">
            <span>Tayyor namuna</span>
            <select value={presetId} onChange={(e) => choosePreset(e.target.value)}>
              <option value="">— o'zingiz to'ldiring —</option>
              {presets.data.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.title_uz}
                </option>
              ))}
            </select>
          </label>
        )}
        {selected && <p className="muted small scn-desc">{selected.description_uz}</p>}
      </section>

      <form className="card scn-form" onSubmit={submit}>
        <h3>Kredit</h3>
        <label className="capture-field">
          <span>Kredit maqsadi</span>
          <select value={form.purpose} onChange={(e) => patch({ purpose: e.target.value })}>
            {PURPOSES.map((p) => (
              <option key={p} value={p}>
                {purposeName(p)}
              </option>
            ))}
          </select>
        </label>
        <label className="capture-field">
          <span>Kredit summasi (so'm)</span>
          <input inputMode="numeric" value={form.amount} onChange={(e) => patch({ amount: e.target.value })} required />
        </label>
        <label className="capture-field">
          <span>Baholash kuni (kredit berilganidan necha kun keyin, 2–364)</span>
          <input
            inputMode="numeric"
            value={form.snapshotDay}
            onChange={(e) => patch({ snapshotDay: e.target.value })}
            required
          />
        </label>
        <label className="capture-field">
          <span>Hudud (ixtiyoriy)</span>
          <select value={form.region} onChange={(e) => patch({ region: e.target.value })}>
            <option value="">Ko'rsatilmagan</option>
            {regions.map((r) => (
              <option key={r} value={r}>
                {r}
              </option>
            ))}
          </select>
        </label>
        <label className="scn-check">
          <input type="checkbox" checked={form.hasInvoice} onChange={(e) => patch({ hasInvoice: e.target.checked })} />
          <span>Hisob-faktura bor</span>
        </label>
        {form.hasInvoice && (
          <label className="capture-field">
            <span>Hisob-faktura summasi (so'm)</span>
            <input inputMode="numeric" value={form.invoice} onChange={(e) => patch({ invoice: e.target.value })} required />
          </label>
        )}

        <h3>Ta'minotchi</h3>
        <label className="capture-field">
          <span>Ta'minotchi turi</span>
          <select
            value={form.supplierMode}
            onChange={(e) => patch({ supplierMode: e.target.value as "existing" | "new" })}
          >
            <option value="new">Yangi ta'minotchi</option>
            <option value="existing">Portfeldagi ta'minotchi</option>
          </select>
        </label>
        {form.supplierMode === "existing" ? (
          <label className="capture-field">
            <span>Ta'minotchi kodi (masalan S1500747)</span>
            <input value={form.supplierId} onChange={(e) => patch({ supplierId: e.target.value })} required />
          </label>
        ) : (
          <>
            <label className="capture-field">
              <span>Faoliyat kodi (OKED, NN.NN)</span>
              <input value={form.oked} onChange={(e) => patch({ oked: e.target.value })} placeholder="01.41" required />
            </label>
            <label className="capture-field">
              <span>Ro'yxatdan o'tganiga (kun, kreditdan oldin)</span>
              <input
                inputMode="numeric"
                value={form.registeredDays}
                onChange={(e) => patch({ registeredDays: e.target.value })}
                required
              />
            </label>
          </>
        )}

        <h3>Pul harakatlari</h3>
        <p className="muted small">
          Kredit summasi 0-kuni hisobga tushadi (ro'yxatga yozilmaydi). Naqd yechish va ta'minotchiga to'lovning kuni va
          summasini o'zgartirib ko'ring.
        </p>
        {form.rows.length === 0 && <p className="muted small">Hozircha harakat yo'q.</p>}
        <ul className="scn-rows">
          {form.rows.map((r, i) => (
            <li key={r.key} className="scn-row">
              <strong className="scn-row-n">{i + 1}.</strong>
              <label className="capture-field">
                <span>Turi</span>
                <select value={r.type} onChange={(e) => patchRow(r.key, { type: e.target.value })}>
                  {TX_TYPES.map((t) => (
                    <option key={t} value={t}>
                      {txnTypeName(t)}
                    </option>
                  ))}
                </select>
              </label>
              <label className="capture-field">
                <span>Kun</span>
                <input inputMode="numeric" value={r.day} onChange={(e) => patchRow(r.key, { day: e.target.value })} required />
              </label>
              <label className="capture-field">
                <span>Summa (so'm)</span>
                <input
                  inputMode="numeric"
                  value={r.amount}
                  onChange={(e) => patchRow(r.key, { amount: e.target.value })}
                  required
                />
              </label>
              <button
                type="button"
                className="secondary"
                onClick={() => setForm((f) => ({ ...f, rows: f.rows.filter((x) => x.key !== r.key) }))}
              >
                O'chirish
              </button>
            </li>
          ))}
        </ul>
        <button
          type="button"
          className="secondary"
          disabled={form.rows.length >= MAX_TX}
          onClick={() => setForm((f) => ({ ...f, rows: [...f.rows, newRow()] }))}
        >
          + Harakat qo'shish
        </button>

        <button type="submit" className="shutter" disabled={run.kind === "loading" || !ready}>
          {run.kind === "loading" ? "Baholanmoqda…" : "Baholash"}
        </button>
      </form>

      <Result run={run} />
    </div>
  );
}

function Result({ run }: { run: Load<ScenarioScore> | { kind: "idle" } }) {
  if (run.kind === "idle") return null;
  if (run.kind === "loading") {
    return (
      <section className="card" aria-live="polite">
        <p className="muted">Baholanmoqda…</p>
      </section>
    );
  }
  if (run.kind === "error") {
    return (
      <section className="card" role="alert">
        <p className="error">{run.message}</p>
      </section>
    );
  }
  const r = run.data;
  const c: Color = r.color;
  return (
    <section className="card scn-result" aria-live="polite">
      <p className="scn-note">Gipotetik ssenariy natijasi — portfelga yozilmadi.</p>
      <div className={`scn-score border-${c}`}>
        <span className="scn-color">
          {COLOR_ICON[c]} {COLOR_UZ[c]}
        </span>
        <span>
          Xavf bahosi: <strong>{formatRisk(r.risk)}</strong>
        </span>
      </div>
      <p className="muted small">
        Portfeldagi o'rni (persentil): {r.percentile.toFixed(1)}. Xavf bahosi — 0 dan 1 gacha, kreditlarni
        tartiblash uchun ball.
      </p>
      <h3>Nega shunday?</h3>
      {r.reasons.length === 0 ? (
        <p>Sezilarli xavf belgisi aniqlanmadi.</p>
      ) : (
        <ul className="scn-reasons">
          {r.reasons.map((x) => (
            <li key={x.code}>{x.text_uz}</li>
          ))}
        </ul>
      )}
      <p className="muted small">
        Bu kredit birinchi navbatda tekshirilishi kerakligini ko'rsatadi, hech kimni ayblamaydi. Model: {r.model_version}
        , hisoblash {Math.round(r.latency_ms)} ms.
      </p>
    </section>
  );
}
