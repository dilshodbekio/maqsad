import { useCallback, useEffect, useState } from "react";
import { api, type Health } from "./api";
import { StubBanner } from "./components/StubBanner";

type State =
  | { kind: "loading" }
  | { kind: "error"; message: string }
  | { kind: "ok"; health: Health };

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
            <dl className="kv">
              <dt>ML rejimi</dt>
              <dd>{state.health.ml_mode}</dd>
              <dt>Model versiyasi</dt>
              <dd>{state.health.model_version}</dd>
            </dl>
          )}
        </section>
      </main>
    </>
  );
}
