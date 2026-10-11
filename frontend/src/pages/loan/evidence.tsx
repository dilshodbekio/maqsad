// Evidence links shared by every block of the loan page: a reason / report fact cites evidence[].ref;
// clicking it highlights the evidence and scrolls to where it is shown (money-trail row, photo, invoice).
import { createContext, useContext } from "react";
import type { Duplicate, Evidence } from "../../api";
import { evidenceSummary, formatDate, formatUzs } from "../../format";
import { t } from "../../i18n";

export type Load<T> = { kind: "loading" } | { kind: "error"; message: string; status?: number } | { kind: "ok"; data: T };

export type FocusApi = {
  activeRef: string | null;
  evidence: Map<string, Evidence>;
  focusRef: (ref: string, scroll?: boolean) => void;
};

export const FocusContext = createContext<FocusApi>({
  activeRef: null,
  evidence: new Map(),
  focusRef: () => undefined,
});

export const useFocus = () => useContext(FocusContext);

const isRecord = (x: unknown): x is Record<string, unknown> => typeof x === "object" && x !== null;

export function field<T>(ev: Evidence | undefined, key: string): T | undefined {
  return isRecord(ev?.data) ? (ev!.data as Record<string, unknown>)[key] as T | undefined : undefined;
}

/** Sandbox uploads are cited as photo:sandbox:<loan_id>:<n>; the page shows them as one group. */
export const isSandboxPhotoRef = (ref: string) => ref.startsWith("photo:sandbox:");

/** DOM id of the place on the page that shows this evidence (null: nothing to scroll to). */
export function anchorId(ev: Evidence | undefined): string | null {
  if (!ev) return null;
  switch (ev.kind) {
    case "txn":
      return `txn-${ev.ref}`;
    case "invoice":
      return "ev-invoice";
    case "graph":
      return "ev-graph";
    case "photo":
      return isSandboxPhotoRef(ev.ref) ? "photos-uploads" : `photo-${ev.ref.slice("photo:".length)}`;
    default:
      return null;
  }
}

/** Short chip label for an evidence item (all values from the API). */
export function evidenceLabel(ev: Evidence | undefined, ref: string): string {
  if (!ev) return ref;
  switch (ev.kind) {
    case "txn": {
      const ts = field<string>(ev, "ts");
      const amount = field<number>(ev, "amount_uzs");
      return [ts ? formatDate(ts) : ev.ref, typeof amount === "number" ? formatUzs(amount) : null]
        .filter(Boolean)
        .join(" · ");
    }
    case "invoice":
      return t("ev.invoice");
    case "graph":
      return t("ev.supplier", { name: field<string>(ev, "supplier_name") ?? ev.ref.slice("graph:".length) });
    case "photo":
      return isSandboxPhotoRef(ev.ref) ? t("ev.photoUpload") : t("ev.photoPortfolio");
    case "period": {
      const start = field<string>(ev, "start");
      const end = field<string>(ev, "end");
      return start && end ? t("ev.period", { from: formatDate(start), to: formatDate(end) }) : t("ev.periodShort");
    }
    default:
      return ref;
  }
}

const KIND_ICON: Record<string, string> = { txn: "💸", invoice: "🧾", graph: "🏪", photo: "📷", period: "🗓" };

/** A clickable evidence reference. A ref missing from evidence[] is shown, never hidden. */
export function EvidenceChip({ refId, onClick }: { refId: string; onClick?: () => void }) {
  const { activeRef, evidence, focusRef } = useFocus();
  const ev = evidence.get(refId);
  if (!ev) {
    return (
      <span className="ev-chip ev-missing" title={refId}>
        {t("ev.missing", { ref: refId })}
      </span>
    );
  }
  return (
    <button
      type="button"
      className={`ev-chip${activeRef === refId ? " active" : ""}`}
      onClick={onClick ?? (() => focusRef(refId))}
      title={evidenceSummary(ev) || refId}
    >
      <span aria-hidden="true">{KIND_ICON[ev.kind] ?? "•"}</span> {evidenceLabel(ev, refId)}
    </button>
  );
}

export function duplicatesOf(ev: Evidence | undefined): Duplicate[] {
  const d = field<unknown>(ev, "duplicate_of");
  return Array.isArray(d) ? (d.filter((x) => isRecord(x) && typeof x.loan_id === "string") as Duplicate[]) : [];
}

export function BlockError({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div className="error block-error" role="alert">
      <p>{message}</p>
      {onRetry && (
        <button type="button" className="secondary" onClick={onRetry}>
          {t("common.retry")}
        </button>
      )}
    </div>
  );
}

export function BlockLoading({ text }: { text: string }) {
  return (
    <p className="muted block-loading" role="status">
      <span className="spinner" aria-hidden="true" /> {text}
    </p>
  );
}
