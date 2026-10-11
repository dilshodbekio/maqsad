// Display helpers (language-aware: uz / en / ru). Only labels live here — every number comes from the API.
import type { Color, QueueItem } from "./api";
import { getLang, localized, t, type Key, type Lang } from "./i18n";

// generator purpose codes (ml/generator/config.py PURPOSES) -> names; unknown code -> shown as is
const PURPOSES = new Set(["tikuv_dastgohi", "chorva", "issiqxona", "parranda", "asalarichilik"]);

export function purposeName(code: string): string {
  return PURPOSES.has(code) ? t(`purpose.${code}` as Key) : code.replace(/_/g, " ");
}

/** Region names are data (Uzbek spelling); shown in the selected language, unknown ones as they are. */
export function regionName(region: string | null | undefined): string {
  if (!region) return "—";
  const key = `region.${region}` as Key;
  const name = t(key);
  return name === key ? region : name;
}

export const COLOR_ICON: Record<Color, string> = { red: "🔴", yellow: "🟡", green: "🟢" };
export const COLORS: Color[] = ["red", "yellow", "green"];

export function colorName(c: Color): string {
  return t(`color.${c}` as Key);
}

const GROUP_SEP: Record<Lang, string> = { uz: " ", en: ",", ru: " " }; // non-breaking space: the number never wraps

function group(digits: string): string {
  return digits.replace(/\B(?=(\d{3})+(?!\d))/g, GROUP_SEP[getLang()]);
}

/** 12500000 -> "12 500 000 so'm" / "12,500,000 UZS" / "12 500 000 сум". */
export function formatUzs(amount: number): string {
  return `${group(Math.round(amount).toString())} ${t("unit.som")}`;
}

export function formatCount(n: number): string {
  return group(n.toString());
}

/** "213" days -> "213 kun" / "213 d" / "213 дн." */
export function formatDays(n: number): string {
  return t("unit.days", { n: formatCount(n) });
}

/** Risk is a ranking score, not a probability: a plain number, never "%". */
export function formatRisk(risk: number): string {
  return risk.toFixed(2);
}

/** Share of the portfolio (for the colour legend), e.g. 500 / 5000 -> "10". */
export function formatShare(part: number, whole: number): string {
  if (whole <= 0) return "0";
  const pct = (part / whole) * 100;
  return Number.isInteger(Math.round(pct * 10) / 10) ? Math.round(pct).toString() : pct.toFixed(1);
}

/** The text of an escalation / reason / evidence item in the selected language (text: {uz,en,ru}; legacy text_uz as the fallback). */
export const escalationText = (e: { text?: Partial<Record<Lang, string>>; text_uz: string }) => localized(e.text, e.text_uz);
export const reasonText = escalationText;
export const evidenceSummary = (e: { summary?: Partial<Record<Lang, string>> | null; summary_uz?: string | null }) =>
  localized(e.summary, e.summary_uz);
export const topReasonText = (it: Pick<QueueItem, "top_reason" | "top_reason_text">) => localized(it.top_reason_text, it.top_reason);

/** "Rasm tufayli ko'tarildi" when the colour was raised by the evidence-fusion policy (base_color != color). */
export function escalationLabel(
  item: Pick<QueueItem, "base_color" | "color" | "escalations">,
): { label: string; detail: string } | null {
  if (item.base_color === item.color) return null;
  const raising = item.escalations.filter((e) => e.raises_color);
  const shown = raising.length ? raising : item.escalations;
  const byPhoto = shown.length > 0 && shown.every((e) => e.code.startsWith("PHOTO_"));
  return {
    label: t(byPhoto ? "esc.byPhoto" : "esc.byEvidence", { from: colorName(item.base_color), to: colorName(item.color) }),
    detail: shown.map(escalationText).join(" "),
  };
}

// ---------- loan page ----------

const TXN_TYPES = new Set(["disbursement", "cash_withdrawal", "supplier_payment", "p2p_out", "p2p_in", "external_transfer", "income"]);

/** Money coming into the borrower's account; everything else is an outflow. */
export const INFLOW_TYPES = new Set(["disbursement", "p2p_in", "income"]);

export function txnTypeName(type: string): string {
  return TXN_TYPES.has(type) ? t(`txn.${type}` as Key) : type.replace(/_/g, " ");
}

const COUNTERPARTIES = new Set(["person", "business", "external", "supplier"]);

/** counterparty_kind -> name; "none" / empty -> null (nothing to show). */
export function counterpartyName(kind: string | null | undefined): string | null {
  if (!kind || kind === "none") return null;
  return COUNTERPARTIES.has(kind) ? t(`cp.${kind}` as Key) : kind;
}

const EN_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

function dateParts(y: string, mo: string, d: string): string {
  // uz / ru: 31.12.2025; en: 31 Dec 2025 (unambiguous day-month order)
  return getLang() === "en" ? `${+d} ${EN_MONTHS[+mo - 1]} ${y}` : `${d}.${mo}.${y}`;
}

/** "2025-12-31T01:04:48" -> "31.12.2025" (uz, ru) / "31 Dec 2025" (en): the date part of the ISO string, no time-zone shift. */
export function formatDate(iso: string | null | undefined): string {
  const m = iso?.match(/^(\d{4})-(\d{2})-(\d{2})/);
  return m ? dateParts(m[1], m[2], m[3]) : iso ?? "—";
}

/** "...T01:04:48" -> "31.12.2025 01:04". A timestamp with a zone (server time, UTC) -> viewer's local time. */
export function formatDateTime(iso: string | null | undefined): string {
  if (iso && /(Z|[+-]\d{2}:\d{2})$/i.test(iso)) return formatLocalDateTime(iso).slice(0, -3);
  const m = iso?.match(/^(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2})/);
  return m ? `${dateParts(m[1], m[2], m[3])} ${m[4]}:${m[5]}` : formatDate(iso);
}

/** Audit timestamps are UTC with an offset: show them in the viewer's local time (with seconds). */
export function formatLocalDateTime(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  const p = (n: number) => n.toString().padStart(2, "0");
  const date = dateParts(String(d.getFullYear()), p(d.getMonth() + 1), p(d.getDate()));
  return `${date} ${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
}

const DAY_MS = 24 * 3600 * 1000;

function calendarDay(iso: string): number | null {
  const m = iso.match(/^(\d{4})-(\d{2})-(\d{2})/);
  return m ? Date.UTC(+m[1], +m[2] - 1, +m[3]) / DAY_MS : null;
}

/** Calendar days from `from` to `to` (date parts of ISO strings, as shown on the page); null if missing. */
export function daysBetween(from: string | null | undefined, to: string | null | undefined): number | null {
  if (!from || !to) return null;
  const a = calendarDay(from);
  const b = calendarDay(to);
  return a === null || b === null ? null : b - a;
}

export function decisionName(action: string): string {
  return ["approve", "reject", "visit", "photo_upload"].includes(action) ? t(`decision.${action}` as Key) : action;
}

export function actorName(actor: string): string {
  return actor === "borrower" ? t("actor.borrower") : actor;
}

/** Backend stores the AI recommendation as "red (risk=0.592)" or, when a photo raised the colour,
 * "red (risk=0.592, base=yellow)" -> "red, risk score 0.592 (by money trail: yellow)" in the selected language. */
export function aiRecommendationText(rec: string | null): string {
  if (!rec) return "—";
  const m = rec.match(/^(red|yellow|green) \(risk=([0-9.]+)(?:, base=(red|yellow|green))?\)$/);
  if (!m) return rec;
  const base = m[3] ? t("ai.base", { color: colorName(m[3] as Color) }) : "";
  return `${COLOR_ICON[m[1] as Color]} ${t("ai.rec", { color: colorName(m[1] as Color), risk: m[2], base })}`;
}

/** Similarity of two photos, 0..1 (not a probability): a plain number like the risk score. */
export function formatSimilarity(s: number): string {
  return s.toFixed(2);
}
