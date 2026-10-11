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
  colorName,
  COLORS,
  escalationLabel,
  formatCount,
  formatRisk,
  formatShare,
  formatUzs,
  purposeName,
  regionName,
  topReasonText,
} from "../format";
import { t } from "../i18n";
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
      <div className="top-links">
        <a className="capture-link" href="/capture" onClick={(e) => onLinkClick(e, "/capture")}>
          <span aria-hidden="true">📷</span> {t("queue.capture")}
        </a>
        <a className="capture-link" href="/scenario" onClick={(e) => onLinkClick(e, "/scenario")}>
          <span aria-hidden="true">🧪</span> {t("queue.scenario")}
        </a>
      </div>
      <section className="card">
        <div className="queue-head">
          <div>
            <h2>{t("queue.title")}</h2>
            <p className="muted small">{t("queue.intro")}</p>
          </div>
        </div>

        <div className="counts" role="group" aria-label={t("queue.countsAria")}>
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
              <span className="chip-label">{colorName(c)}</span>
            </button>
          ))}
        </div>

        <Legend state={legend} onRetry={loadSide} />

        <div className="filters">
          <label>
            <span>{t("queue.filterColor")}</span>
            <select
              value={filters.color ?? ""}
              onChange={(e) => setFilters({ ...filters, color: (e.target.value || undefined) as Color | undefined })}
            >
              <option value="">{t("common.all")}</option>
              {COLORS.map((c) => (
                <option key={c} value={c}>
                  {COLOR_ICON[c]} {colorName(c)}
                </option>
              ))}
            </select>
          </label>
          <label>
            <span>{t("queue.filterRegion")}</span>
            <select
              value={filters.region ?? ""}
              onChange={(e) => setFilters({ ...filters, region: e.target.value || undefined })}
              disabled={facets.kind === "loading"}
            >
              <option value="">{t("common.all")}</option>
              {regions.map((r) => (
                <option key={r} value={r}>
                  {regionName(r)}
                </option>
              ))}
            </select>
          </label>
          <label>
            <span>{t("queue.filterPurpose")}</span>
            <select
              value={filters.purpose ?? ""}
              onChange={(e) => setFilters({ ...filters, purpose: e.target.value || undefined })}
              disabled={facets.kind === "loading"}
            >
              <option value="">{t("common.all")}</option>
              {purposes.map((p) => (
                <option key={p} value={p}>
                  {purposeName(p)}
                </option>
              ))}
            </select>
          </label>
          {anyFilter && (
            <button className="secondary" onClick={() => setFilters({})}>
              {t("queue.clearFilters")}
            </button>
          )}
        </div>
        {facets.kind === "error" && (
          <p className="error small">
            {t("queue.facetsError", { message: facets.message })}{" "}
            <button className="link-button" onClick={loadSide}>
              {t("common.retry")}
            </button>
          </p>
        )}
      </section>

      <section className="card list-card">
        {!ready && <p className="muted">{t("queue.warming")}</p>}
        {ready && list.kind === "loading" && <p className="muted">{t("queue.loading")}</p>}
        {ready && list.kind === "error" && (
          <div className="error">
            <p>{t("queue.loadError", { message: list.error ?? "" })}</p>
            <button onClick={loadFirst}>{t("common.retry")}</button>
          </div>
        )}
        {ready && list.kind === "ok" && (
          <>
            <p className="muted small summary">
              {list.total === 0
                ? t("queue.none")
                : t("queue.shown", { total: formatCount(list.total), shown: formatCount(list.items.length) })}
            </p>
            {list.items.length > 0 && <QueueTable items={list.items} />}
            {list.items.length > 0 && <QueueCards items={list.items} />}
            {list.items.length < list.total && (
              <div className="more">
                {list.more === "error" && <p className="error small">{t("queue.moreError", { message: list.moreError ?? "" })}</p>}
                <button onClick={loadMore} disabled={list.more === "loading"}>
                  {list.more === "loading" ? t("queue.moreLoading") : t("queue.more")}
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
  if (state.kind === "loading") return <p className="legend muted small">{t("legend.loading")}</p>;
  if (state.kind === "error") {
    return (
      <p className="legend error small">
        {t("legend.error", { message: state.message })}{" "}
        <button className="link-button" onClick={onRetry}>
          {t("common.retry")}
        </button>
      </p>
    );
  }
  const l = state.data;
  const escalated = l.n_escalated_to_red + l.n_escalated_to_yellow;
  return (
    <div className="legend small">
      <p>{t("legend.rule", { red: formatShare(l.n_red, l.n_loans), yellow: formatShare(l.n_yellow, l.n_loans) })}</p>
      <p className="muted">
        {t("legend.thresholds", { red: formatRisk(l.red), yellow: formatRisk(l.yellow) })}
        {escalated > 0 &&
          t("legend.escalated", { red: formatCount(l.n_escalated_to_red), yellow: formatCount(l.n_escalated_to_yellow) })}
      </p>
    </div>
  );
}

function ColorMark({ color }: { color: Color }) {
  return (
    <span className={`color-mark mark-${color}`} title={colorName(color)}>
      <span aria-hidden="true">{COLOR_ICON[color]}</span>
      <span className="sr-only">{colorName(color)}</span>
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
            <th>{t("th.color")}</th>
            <th>{t("th.loan")}</th>
            <th>{t("th.borrower")}</th>
            <th>{t("th.purpose")}</th>
            <th className="num">{t("th.amount")}</th>
            <th>{t("th.region")}</th>
            <th className="num">{t("th.risk")}</th>
            <th>{t("th.reason")}</th>
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
              <td>{regionName(it.region)}</td>
              <td className="num">{formatRisk(it.risk)}</td>
              <td className="reason">
                {topReasonText(it)}
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
              <span className="qc-risk">{t("card.risk", { risk: formatRisk(it.risk) })}</span>
            </div>
            <div className="qc-name">{it.borrower_name_masked}</div>
            <div className="qc-meta">
              {purposeName(it.purpose)} · {regionName(it.region)}
            </div>
            <div className="qc-amount">{formatUzs(it.amount_uzs)}</div>
            <div className="qc-reason">{topReasonText(it)}</div>
            <EscalationBadge item={it} />
          </a>
        </li>
      ))}
    </ul>
  );
}
