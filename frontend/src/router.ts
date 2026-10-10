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
