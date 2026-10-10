// Minimal client-side router (no extra dependency): pathname + search from window.location,
// updated on pushState / popstate. FastAPI serves index.html for every non-/api path.
import { useEffect, useState, type MouseEvent } from "react";

const EVENT = "maqsad:navigate";

export type Location = { path: string; search: string };

function current(): Location {
  return { path: window.location.pathname, search: window.location.search };
}

export function navigate(to: string, opts: { replace?: boolean } = {}): void {
  const state = { inApp: true };
  if (opts.replace) window.history.replaceState(state, "", to);
  else window.history.pushState(state, "", to);
  window.dispatchEvent(new Event(EVENT));
}

/** Go back if the previous page is ours (keeps the queue filters), otherwise open `fallback`. */
export function goBack(fallback: string): void {
  if ((window.history.state as { inApp?: boolean } | null)?.inApp) window.history.back();
  else navigate(fallback);
}

export function useLocation(): Location {
  const [loc, setLoc] = useState(current);
  useEffect(() => {
    const update = () => setLoc(current());
    window.addEventListener("popstate", update);
    window.addEventListener(EVENT, update);
    return () => {
      window.removeEventListener("popstate", update);
      window.removeEventListener(EVENT, update);
    };
  }, []);
  return loc;
}

/** For <a href> clicks: plain left click -> in-app navigation; ctrl/cmd/middle click -> browser default. */
export function onLinkClick(e: MouseEvent, to: string): void {
  if (e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
  e.preventDefault();
  navigate(to);
}

// The last queue URL (with its filters), so "← Navbatga qaytish" keeps them even after loan -> loan links
// or a page reload. sessionStorage may be unavailable (private mode): then the plain queue opens.
const QUEUE_URL_KEY = "maqsad:queue-url";

export function rememberQueueUrl(url: string): void {
  try {
    window.sessionStorage.setItem(QUEUE_URL_KEY, url);
  } catch {
    // ignore
  }
}

export function lastQueueUrl(): string {
  try {
    const url = window.sessionStorage.getItem(QUEUE_URL_KEY);
    return url && url.startsWith("/") && !url.startsWith("//") ? url : "/";
  } catch {
    return "/";
  }
}
