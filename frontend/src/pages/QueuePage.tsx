import { useCallback, useEffect, useMemo, useRef, useState, type MouseEvent } from "react";
import {
  api,
  type Color,
  type ColorLegend,
  type QueueFacets,
  type QueueFilters,
  type QueueItem,
} from "../api";
import {
  COLOR_ICON,
  COLOR_UZ,
  COLORS,
  escalationLabel,
  formatCount,
  formatRisk,
  formatShare,
  formatUzs,
  purposeName,
} from "../format";
import { navigate, onLinkClick, rememberQueueUrl } from "../router";

type Load<T> = { kind: "loading" } | { kind: "error"; message: string } | { kind: "ok"; data: T };

type ListState = {
  kind: "loading" | "ok" | "error";
  error?: string;
  items: QueueItem[];
  total: number;
  counts: Record<Color, number> | null;
  more: "idle" | "loading" | "error";
  moreError?: string;
};

const EMPTY_LIST: ListState = { kind: "loading", items: [], total: 0, counts: null, more: "idle" };

function parseFilters(search: string): QueueFilters {
  const p = new URLSearchParams(search);
  const color = p.get("color");
  return {
    color: COLORS.includes(color as Color) ? (color as Color) : undefined,
    region: p.get("region") || undefined,
    purpose: p.get("purpose") || undefined,
  };
}

function filtersToUrl(f: QueueFilters): string {
  const p = new URLSearchParams();
  if (f.color) p.set("color", f.color);
  if (f.region) p.set("region", f.region);
  if (f.purpose) p.set("purpose", f.purpose);
  const s = p.toString();
  return s ? `/?${s}` : "/";
}

const loanHref = (id: string) => `/loan/${encodeURIComponent(id)}`;

export function QueuePage({ search, ready }: { search: string; ready: boolean }) {
  const filters = useMemo(() => parseFilters(search), [search]);
  const filterKey = filtersToUrl(filters);
  const [list, setList] = useState<ListState>(EMPTY_LIST);
  const [facets, setFacets] = useState<Load<QueueFacets>>({ kind: "loading" });
  const [legend, setLegend] = useState<Load<ColorLegend>>({ kind: "loading" });
  const seq = useRef(0); // ignores responses of outdated requests (filters changed meanwhile)

  const setFilters = (next: QueueFilters) => navigate(filtersToUrl(next), { replace: true });

  useEffect(() => {
    rememberQueueUrl(filterKey); // the loan page's back link returns here
  }, [filterKey]);

  const loadFirst = useCallback(() => {
    const id = ++seq.current;
    setList((s) => ({ ...EMPTY_LIST, counts: s.counts }));
    api
      .queue(filters, 0)
      .then((page) => {
        if (id !== seq.current) return;
        setList({ kind: "ok", items: page.items, total: page.total, counts: page.counts, more: "idle" });
      })
      .catch((e: Error) => {
        if (id !== seq.current) return;
        setList((s) => ({ ...s, kind: "error", error: e.message }));
      });
  }, [filterKey]); // filterKey identifies the filters; `filters` is derived from it

  const loadMore = () => {
    const id = seq.current;
    setList((s) => ({ ...s, more: "loading" }));
    api
      .queue(filters, list.items.length)
      .then((page) => {
        if (id !== seq.current) return;
        setList((s) => {
          const seen = new Set(s.items.map((i) => i.loan_id)); // the queue may change between pages
          return {
            ...s,
            items: [...s.items, ...page.items.filter((i) => !seen.has(i.loan_id))],
            total: page.total,
            counts: page.counts,
            more: "idle",
          };
        });
      })
      .catch((e: Error) => {
        if (id !== seq.current) return;
        setList((s) => ({ ...s, more: "error", moreError: e.message }));
      });
  };

  useEffect(() => {
    if (ready) loadFirst();
  }, [ready, loadFirst]);

  const loadSide = useCallback(() => {
    setFacets({ kind: "loading" });
    api
      .queueFacets()
      .then((data) => setFacets({ kind: "ok", data }))
      .catch((e: Error) => setFacets({ kind: "error", message: e.message }));
    setLegend({ kind: "loading" });
    api
      .queueLegend()
      .then((data) => setLegend({ kind: "ok", data }))
      .catch((e: Error) => setLegend({ kind: "error", message: e.message }));
  }, []);

  useEffect(() => {
    if (ready) loadSide();
  }, [ready, loadSide]);

  const anyFilter = Boolean(filters.color || filters.region || filters.purpose);
  const regions = facets.kind === "ok" ? [...facets.data.regions] : [];
  const purposes = facets.kind === "ok" ? [...facets.data.purposes] : [];
  // keep a value from the URL selectable even before / without facets
  if (filters.region && !regions.includes(filters.region)) regions.unshift(filters.region);
  if (filters.purpose && !purposes.includes(filters.purpose)) purposes.unshift(filters.purpose);

  return (
    <>
      <section className="card">
        <div className="queue-head">
          <div>
            <h2>Xavf navbati</h2>
            <p className="muted small">
              Birinchi navbatda tekshirilishi kerak bo'lgan kreditlar yuqorida. Yakuniy qarorni bank xodimi qabul
              qiladi.
            </p>
          </div>
        </div>

        <div className="counts" role="group" aria-label="Rang bo'yicha soni">
          {COLORS.map((c) => (
            <button
              key={c}
              className={`count-chip chip-${c}${filters.color === c ? " active" : ""}`}
              aria-pressed={filters.color === c}
              onClick={() => setFilters({ ...filters, color: filters.color === c ? undefined : c })}
              disabled={!list.counts}
            >
              <span aria-hidden="true">{COLOR_ICON[c]}</span>
              <strong>{list.counts ? formatCount(list.counts[c]) : "…"}</strong>
              <span className="chip-label">{COLOR_UZ[c]}</span>
            </button>
          ))}
        </div>

        <Legend state={legend} onRetry={loadSide} />

        <div className="filters">
          <label>
            <span>Rang</span>
            <select
              value={filters.color ?? ""}
              onChange={(e) => setFilters({ ...filters, color: (e.target.value || undefined) as Color | undefined })}
            >
              <option value="">Hammasi</option>
              {COLORS.map((c) => (
                <option key={c} value={c}>
                  {COLOR_ICON[c]} {COLOR_UZ[c]}
                </option>
              ))}
            </select>
          </label>
          <label>
            <span>Hudud</span>
            <select
              value={filters.region ?? ""}
              onChange={(e) => setFilters({ ...filters, region: e.target.value || undefined })}
              disabled={facets.kind === "loading"}
            >
              <option value="">Hammasi</option>
              {regions.map((r) => (
                <option key={r} value={r}>
                  {r}
                </option>
              ))}
            </select>
          </label>
          <label>
            <span>Maqsad</span>
            <select
              value={filters.purpose ?? ""}
              onChange={(e) => setFilters({ ...filters, purpose: e.target.value || undefined })}
              disabled={facets.kind === "loading"}
            >
              <option value="">Hammasi</option>
              {purposes.map((p) => (
                <option key={p} value={p}>
                  {purposeName(p)}
                </option>
              ))}
            </select>
          </label>
          {anyFilter && (
            <button className="secondary" onClick={() => setFilters({})}>
              Filtrlarni tozalash
            </button>
          )}
        </div>
        {facets.kind === "error" && (
          <p className="error small">
            Hudud va maqsad ro'yxati yuklanmadi: {facets.message}{" "}
            <button className="link-button" onClick={loadSide}>
              Qayta urinish
            </button>
          </p>
        )}
      </section>

      <section className="card list-card">
        {!ready && <p className="muted">Model tayyorlanmoqda… Navbat bir necha soniyadan keyin chiqadi.</p>}
        {ready && list.kind === "loading" && <p className="muted">Navbat yuklanmoqda…</p>}
        {ready && list.kind === "error" && (
          <div className="error">
            <p>Navbat yuklanmadi: {list.error}</p>
            <button onClick={loadFirst}>Qayta urinish</button>
          </div>
        )}
        {ready && list.kind === "ok" && (
          <>
            <p className="muted small summary">
              {list.total === 0
                ? "Bu filtrlarga mos kredit yo'q."
                : `${formatCount(list.total)} ta kreditdan ${formatCount(list.items.length)} tasi ko'rsatilmoqda`}
            </p>
            {list.items.length > 0 && <QueueTable items={list.items} />}
            {list.items.length > 0 && <QueueCards items={list.items} />}
            {list.items.length < list.total && (
              <div className="more">
                {list.more === "error" && <p className="error small">Yuklanmadi: {list.moreError}</p>}
                <button onClick={loadMore} disabled={list.more === "loading"}>
                  {list.more === "loading" ? "Yuklanmoqda…" : "Yana yuklash"}
                </button>
              </div>
            )}
          </>
        )}
      </section>
    </>
  );
}

function Legend({ state, onRetry }: { state: Load<ColorLegend>; onRetry: () => void }) {
  if (state.kind === "loading") return <p className="legend muted small">Rang qoidasi yuklanmoqda…</p>;
  if (state.kind === "error") {
    return (
      <p className="legend error small">
        Rang qoidasi yuklanmadi: {state.message}{" "}
        <button className="link-button" onClick={onRetry}>
          Qayta urinish
        </button>
      </p>
    );
  }
  const l = state.data;
  const escalated = l.n_escalated_to_red + l.n_escalated_to_yellow;
  return (
    <div className="legend small">
      <p>
        Rang navbatdagi o'rin bo'yicha: eng yuqori {formatShare(l.n_red, l.n_loans)}% — qizil, keyingi{" "}
        {formatShare(l.n_yellow, l.n_loans)}% — sariq, qolganlari — yashil.
      </p>
      <p className="muted">
        Xavf bahosi chegaralari: qizil ≥ {formatRisk(l.red)}, sariq ≥ {formatRisk(l.yellow)}. Xavf bahosi — tartiblash
        uchun model bahosi.
        {escalated > 0 &&
          ` Rasm dalillari tufayli yana ${formatCount(l.n_escalated_to_red)} ta kredit qizilga, ` +
            `${formatCount(l.n_escalated_to_yellow)} ta kredit sariqqa ko'tarilgan (xavf bahosi o'zgarmaydi).`}
      </p>
    </div>
  );
}

function ColorMark({ color }: { color: Color }) {
  return (
    <span className={`color-mark mark-${color}`} title={COLOR_UZ[color]}>
      <span aria-hidden="true">{COLOR_ICON[color]}</span>
      <span className="sr-only">{COLOR_UZ[color]}</span>
    </span>
  );
}

function EscalationBadge({ item }: { item: QueueItem }) {
  const esc = escalationLabel(item);
  if (!esc) return null;
  return (
    <span className="esc-badge" title={esc.detail}>
      ⬆ {esc.label}
    </span>
  );
}

function QueueTable({ items }: { items: QueueItem[] }) {
  const open = (e: MouseEvent, id: string) => {
    if ((e.target as HTMLElement).closest("a")) return; // the link handles its own click
    navigate(loanHref(id));
  };
  return (
    <div className="table-wrap">
      <table className="queue-table">
        <thead>
          <tr>
            <th>Rang</th>
            <th>Kredit</th>
            <th>Qarz oluvchi</th>
            <th>Maqsad</th>
            <th className="num">Summa</th>
            <th>Hudud</th>
            <th className="num">Xavf bahosi</th>
            <th>Asosiy sabab</th>
          </tr>
        </thead>
        <tbody>
          {items.map((it) => (
            <tr key={it.loan_id} onClick={(e) => open(e, it.loan_id)}>
              <td>
                <ColorMark color={it.color} />
              </td>
              <td>
                <a href={loanHref(it.loan_id)} onClick={(e) => onLinkClick(e, loanHref(it.loan_id))}>
                  {it.loan_id}
                </a>
              </td>
              <td>{it.borrower_name_masked}</td>
              <td>{purposeName(it.purpose)}</td>
              <td className="num">{formatUzs(it.amount_uzs)}</td>
              <td>{it.region}</td>
              <td className="num">{formatRisk(it.risk)}</td>
              <td className="reason">
                {it.top_reason}
                <EscalationBadge item={it} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function QueueCards({ items }: { items: QueueItem[] }) {
  return (
    <ul className="queue-cards">
      {items.map((it) => (
        <li key={it.loan_id}>
          <a
            className={`queue-card border-${it.color}`}
            href={loanHref(it.loan_id)}
            onClick={(e) => onLinkClick(e, loanHref(it.loan_id))}
          >
            <div className="qc-top">
              <ColorMark color={it.color} />
              <span className="qc-id">{it.loan_id}</span>
              <span className="qc-risk">xavf bahosi {formatRisk(it.risk)}</span>
            </div>
            <div className="qc-name">{it.borrower_name_masked}</div>
            <div className="qc-meta">
              {purposeName(it.purpose)} · {it.region}
            </div>
            <div className="qc-amount">{formatUzs(it.amount_uzs)}</div>
            <div className="qc-reason">{it.top_reason}</div>
            <EscalationBadge item={it} />
          </a>
        </li>
      ))}
    </ul>
  );
}
