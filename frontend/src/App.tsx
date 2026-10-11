import { useCallback, useEffect, useState } from "react";
import { api } from "./api";
import { HealthStatus, type HealthState } from "./components/HealthStatus";
import { LangSwitch } from "./components/LangSwitch";
import { StubBanner } from "./components/StubBanner";
import { CapturePage } from "./pages/CapturePage";
import { LoanPage } from "./pages/loan/LoanPage";
import { ScenarioPage } from "./pages/ScenarioPage";
import { QueuePage } from "./pages/QueuePage";
import { t, useLang } from "./i18n";
import { onLinkClick, useLocation } from "./router";

const WARMUP_POLL_MS = 2000;

export default function App() {
  const [state, setState] = useState<HealthState>({ kind: "loading" });
  const loc = useLocation();
  useLang(); // re-render the whole tree when the UI language changes

  const load = useCallback(() => {
    setState({ kind: "loading" });
    api
      .health()
      .then((health) => setState({ kind: "ok", health }))
      .catch((e: Error) => setState({ kind: "error", message: e.message }));
  }, []);

  useEffect(load, [load]);

  // while ml builds its portfolio in the background, re-check health until it is ready
  const warming = state.kind === "ok" && (state.health.warmup === "pending" || state.health.warmup === "running");
  useEffect(() => {
    if (!warming) return;
    const t = setTimeout(() => {
      api
        .health()
        .then((health) => setState({ kind: "ok", health }))
        .catch(() => undefined);
    }, WARMUP_POLL_MS);
    return () => clearTimeout(t);
  }, [warming, state]);

  // the queue / loan is requested once the model is warm (or health is unknown: then the queue shows its own error)
  const ready = state.kind === "error" || (state.kind === "ok" && !warming);
  const loanMatch = loc.path.match(/^\/loan\/([^/]+)\/?$/);

  return (
    <>
      <StubBanner health={state.kind === "ok" ? state.health : null} />
      <header className="topbar">
        <div className="topbar-inner">
          <a className="brand" href="/" onClick={(e) => onLinkClick(e, "/")}>
            Maqsad
          </a>
          <LangSwitch />
        </div>
        <p className="subtitle">{t("app.subtitle")}</p>
      </header>
      <HealthStatus state={state} onRetry={load} />
      <main className="container">
        {loanMatch ? (
          <LoanPage key={loanMatch[1]} loanId={safeDecode(loanMatch[1])} ready={ready} />
        ) : loc.path === "/" ? (
          <QueuePage search={loc.search} ready={ready} />
        ) : loc.path === "/capture" || loc.path === "/capture/" ? (
          <CapturePage search={loc.search} ready={ready} />
        ) : loc.path === "/scenario" || loc.path === "/scenario/" ? (
          <ScenarioPage ready={ready} />
        ) : (
          <section className="card">
            <h2>{t("app.notFound")}</h2>
            <a href="/" onClick={(e) => onLinkClick(e, "/")}>
              {t("app.backToQueue")}
            </a>
          </section>
        )}
      </main>
      {state.kind === "ok" && (
        <footer className="footer muted small">{t("app.footerModel", { version: state.health.model_version })}</footer>
      )}
    </>
  );
}

function safeDecode(s: string): string {
  try {
    return decodeURIComponent(s);
  } catch {
    return s;
  }
}
