// UI language (uz / en / ru): module-level state + a hook that re-renders the app when it changes.
// The choice is remembered in localStorage (wrapped in try/catch: it may be unavailable) and sets <html lang>.
import { useSyncExternalStore } from "react";
import { en } from "./en";
import { ru } from "./ru";
import { uz, type Dict, type Key } from "./uz";

export type Lang = "uz" | "en" | "ru";
export const LANGS: Lang[] = ["uz", "en", "ru"];
export type { Key };

const DICTS: Record<Lang, Dict> = { uz, en, ru };
const STORAGE_KEY = "maqsad:lang";

function read(): Lang {
  try {
    const v = window.localStorage.getItem(STORAGE_KEY);
    if (v === "uz" || v === "en" || v === "ru") return v;
  } catch {
    // storage unavailable: default language
  }
  return "uz";
}

let current: Lang = typeof window === "undefined" ? "uz" : read();
const listeners = new Set<() => void>();

export const getLang = (): Lang => current;

export function applyHtmlLang(): void {
  if (typeof document !== "undefined") document.documentElement.lang = current;
}

export function setLang(lang: Lang): void {
  if (lang === current) return;
  current = lang;
  try {
    window.localStorage.setItem(STORAGE_KEY, lang);
  } catch {
    // ignore
  }
  applyHtmlLang();
  listeners.forEach((l) => l());
}

const subscribe = (cb: () => void) => {
  listeners.add(cb);
  return () => listeners.delete(cb);
};

/** Subscribe a component to language changes (call it once near the root: the tree below re-renders). */
export function useLang(): Lang {
  return useSyncExternalStore(subscribe, getLang, () => "uz" as Lang);
}

export function fill(template: string, params?: Record<string, string | number>): string {
  if (!params) return template;
  return template.replace(/\{(\w+)\}/g, (m, k: string) => (k in params ? String(params[k]) : m));
}

/** UI string in the current language. */
export function t(key: Key, params?: Record<string, string | number>): string {
  return fill(DICTS[current][key] ?? uz[key] ?? key, params);
}

/** A {uz, en, ru} text from the API in the current language; falls back to uz, then to the plain legacy field. */
export function localized(text: Partial<Record<Lang, string>> | null | undefined, fallback?: string | null): string {
  return text?.[current] || text?.uz || fallback || "";
}

export const LOCALES: Record<Lang, string> = { uz: "uz-Latn-UZ", en: "en-GB", ru: "ru-RU" };
