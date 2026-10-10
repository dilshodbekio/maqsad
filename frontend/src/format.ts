// Display helpers (Uzbek, Latin). Only labels live here — every number comes from the API.
import type { Color, QueueItem } from "./api";

// generator purpose codes (ml/generator/config.py PURPOSES) -> Uzbek names; unknown code -> shown as is
const PURPOSE_UZ: Record<string, string> = {
  tikuv_dastgohi: "Tikuv dastgohi",
  chorva: "Chorvachilik",
  issiqxona: "Issiqxona",
  parranda: "Parrandachilik",
  asalarichilik: "Asalarichilik",
};

export function purposeName(code: string): string {
  return PURPOSE_UZ[code] ?? code.replace(/_/g, " ");
}

export const COLOR_UZ: Record<Color, string> = { red: "qizil", yellow: "sariq", green: "yashil" };
export const COLOR_ICON: Record<Color, string> = { red: "🔴", yellow: "🟡", green: "🟢" };
export const COLORS: Color[] = ["red", "yellow", "green"];

/** 12500000 -> "12 500 000 so'm" (non-breaking spaces, so the number never wraps). */
export function formatUzs(amount: number): string {
  const digits = Math.round(amount).toString();
  return `${digits.replace(/\B(?=(\d{3})+(?!\d))/g, " ")} so'm`;
}

export function formatCount(n: number): string {
  return n.toString().replace(/\B(?=(\d{3})+(?!\d))/g, " ");
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

/** "Rasm tufayli ko'tarildi" when the colour was raised by the evidence-fusion policy (base_color != color). */
export function escalationLabel(
  item: Pick<QueueItem, "base_color" | "color" | "escalations">,
): { label: string; detail: string } | null {
  if (item.base_color === item.color) return null;
  const raising = item.escalations.filter((e) => e.raises_color);
  const shown = raising.length ? raising : item.escalations;
  const byPhoto = shown.length > 0 && shown.every((e) => e.code.startsWith("PHOTO_"));
  return {
    label: `${byPhoto ? "Rasm tufayli" : "Qo'shimcha dalil tufayli"} ko'tarildi (${COLOR_UZ[item.base_color]} → ${COLOR_UZ[item.color]})`,
    detail: shown.map((e) => e.text_uz).join(" "),
  };
}

// ---------- loan page ----------

export const TXN_TYPE_UZ: Record<string, string> = {
  disbursement: "Kredit tushishi",
  cash_withdrawal: "Naqd yechish",
  supplier_payment: "Ta'minotchiga to'lov",
  p2p_out: "P2P o'tkazma (chiqim)",
  p2p_in: "P2P o'tkazma (kirim)",
  external_transfer: "Boshqa bankka o'tkazma",
  income: "Kirim (tushum)",
};

/** Money coming into the borrower's account; everything else is an outflow. */
export const INFLOW_TYPES = new Set(["disbursement", "p2p_in", "income"]);

export function txnTypeName(type: string): string {
  return TXN_TYPE_UZ[type] ?? type.replace(/_/g, " ");
}

const COUNTERPARTY_UZ: Record<string, string> = {
  person: "jismoniy shaxs",
  business: "korxona",
  external: "boshqa bank",
  supplier: "ta'minotchi",
};

/** counterparty_kind -> Uzbek; "none" / empty -> null (nothing to show). */
export function counterpartyName(kind: string | null | undefined): string | null {
  if (!kind || kind === "none") return null;
  return COUNTERPARTY_UZ[kind] ?? kind;
}

/** "2025-12-31T01:04:48" -> "31.12.2025" (the date part of the ISO string, no time-zone shift). */
export function formatDate(iso: string | null | undefined): string {
  const m = iso?.match(/^(\d{4})-(\d{2})-(\d{2})/);
  return m ? `${m[3]}.${m[2]}.${m[1]}` : iso ?? "—";
}

/** "...T01:04:48" -> "31.12.2025 01:04". A timestamp with a zone (server time, UTC) -> viewer's local time. */
export function formatDateTime(iso: string | null | undefined): string {
  if (iso && /(Z|[+-]\d{2}:\d{2})$/i.test(iso)) return formatLocalDateTime(iso).slice(0, 16);
  const m = iso?.match(/^(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2})/);
  return m ? `${m[3]}.${m[2]}.${m[1]} ${m[4]}:${m[5]}` : formatDate(iso);
}

/** Audit timestamps are UTC with an offset: show them in the viewer's local time. */
export function formatLocalDateTime(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  const p = (n: number) => n.toString().padStart(2, "0");
  return `${p(d.getDate())}.${p(d.getMonth() + 1)}.${d.getFullYear()} ${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
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

export const DECISION_UZ: Record<string, string> = {
  approve: "Tasdiqlandi",
  reject: "Rad etildi",
  visit: "Tashrif belgilandi",
  photo_upload: "Rasm yuklandi",
};

export function actorName(actor: string): string {
  return actor === "borrower" ? "Qarz oluvchi" : actor;
}

/** Backend stores the AI recommendation as "red (risk=0.592)" or, when a photo raised the colour,
 * "red (risk=0.592, base=yellow)" -> "qizil, xavf bahosi 0.592 (pul izi bo'yicha: sariq)". */
export function aiRecommendationText(rec: string | null): string {
  if (!rec) return "—";
  const m = rec.match(/^(red|yellow|green) \(risk=([0-9.]+)(?:, base=(red|yellow|green))?\)$/);
  if (!m) return rec;
  const base = m[3] ? ` (pul izi bo'yicha: ${COLOR_UZ[m[3] as Color]})` : "";
  return `${COLOR_ICON[m[1] as Color]} ${COLOR_UZ[m[1] as Color]}, xavf bahosi ${m[2]}${base}`;
}

/** Similarity of two photos, 0..1 (not a probability): a plain number like the risk score. */
export function formatSimilarity(s: number): string {
  return s.toFixed(2);
}
