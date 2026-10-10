import { useCallback, useEffect, useState } from "react";
import { api, type FunctionStatus, type Health } from "./api";
import { StubBanner } from "./components/StubBanner";

type State =
  | { kind: "loading" }
  | { kind: "error"; message: string }
  | { kind: "ok"; health: Health };

const STATUS_UZ: Record<FunctionStatus, string> = {
  real: "ulangan",
  stub: "stub",
  not_ready: "hali ulanmagan",
  missing: "mavjud emas",
};

const WARMUP_UZ: Record<Health["warmup"], string> = {
  pending: "kutilmoqda",
  running: "model tayyorlanmoqda…",
  done: "tayyor",
  failed: "xato (stub ishlatilmoqda)",
};

const WARMUP_POLL_MS = 2000;

export default function App() {
  const [state, setState] = useState<State>({ kind: "loading" });

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

  return (
    <>
      <StubBanner health={state.kind === "ok" ? state.health : null} />
      <header className="topbar">
        <h1>Maqsad</h1>
        <p className="subtitle">Imtiyozli kreditlarning maqsadli ishlatilishini masofadan tekshirish</p>
      </header>
      <main className="container">
        <section className="card">
          <h2>Tizim holati</h2>
          {state.kind === "loading" && <p className="muted">Yuklanmoqda…</p>}
          {state.kind === "error" && (
            <div className="error">
              <p>{state.message}</p>
              <button onClick={load}>Qayta urinish</button>
            </div>
          )}
          {state.kind === "ok" && (
            <>
              <dl className="kv">
                <dt>ML rejimi</dt>
                <dd>{state.health.ml_mode}</dd>
                <dt>Model versiyasi</dt>
                <dd>{state.health.model_version}</dd>
                <dt>Model holati</dt>
                <dd>{WARMUP_UZ[state.health.warmup]}</dd>
              </dl>
              <h3>Funksiyalar</h3>
              <ul className="fn-list">
                {Object.entries(state.health.functions).map(([name, status]) => (
                  <li key={name}>
                    <code>{name}</code>
                    <span className={`fn-status fn-${status}`}>{STATUS_UZ[status]}</span>
                  </li>
                ))}
              </ul>
            </>
          )}
        </section>
      </main>
    </>
  );
}
