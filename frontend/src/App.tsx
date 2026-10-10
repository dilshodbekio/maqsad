import { useCallback, useEffect, useState } from "react";
import { api } from "./api";
import { HealthStatus, type HealthState } from "./components/HealthStatus";
import { StubBanner } from "./components/StubBanner";
import { CapturePage } from "./pages/CapturePage";
import { LoanPage } from "./pages/loan/LoanPage";
import { ScenarioPage } from "./pages/ScenarioPage";
import { QueuePage } from "./pages/QueuePage";
import { onLinkClick, useLocation } from "./router";

const WARMUP_POLL_MS = 2000;

export default function App() {
  const [state, setState] = useState<HealthState>({ kind: "loading" });
  const loc = useLocation();

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
          <HealthStatus state={state} onRetry={load} />
        </div>
        <p className="subtitle">Imtiyozli kreditlarning maqsadli ishlatilishini masofadan tekshirish</p>
      </header>
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
            <h2>Sahifa topilmadi</h2>
            <a href="/" onClick={(e) => onLinkClick(e, "/")}>
              Navbatga qaytish
            </a>
          </section>
        )}
      </main>
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
