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
export function escalationLabel(item: QueueItem): { label: string; detail: string } | null {
  if (item.base_color === item.color) return null;
  const raising = item.escalations.filter((e) => e.raises_color);
  const shown = raising.length ? raising : item.escalations;
  const byPhoto = shown.length > 0 && shown.every((e) => e.code.startsWith("PHOTO_"));
  return {
    label: `${byPhoto ? "Rasm tufayli" : "Qo'shimcha dalil tufayli"} ko'tarildi (${COLOR_UZ[item.base_color]} → ${COLOR_UZ[item.color]})`,
    detail: shown.map((e) => e.text_uz).join(" "),
  };
}
