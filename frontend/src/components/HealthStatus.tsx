import type { Health } from "../api";

export type HealthState =
  | { kind: "loading" }
  | { kind: "error"; message: string }
  | { kind: "ok"; health: Health };

const WARMUP_UZ: Record<Health["warmup"], string> = {
  pending: "kutilmoqda",
  running: "tayyorlanmoqda…",
  done: "tayyor",
  failed: "xato (stub)",
};

/** Small status line in the top bar: model version + warm-up state. */
export function HealthStatus({ state, onRetry }: { state: HealthState; onRetry: () => void }) {
  if (state.kind === "loading") return <span className="health muted-on-brand">Holat tekshirilmoqda…</span>;
  if (state.kind === "error") {
    return (
      <span className="health health-bad">
        Server holati noma'lum{" "}
        <button className="link-button on-brand" onClick={onRetry}>
          qayta tekshirish
        </button>
      </span>
    );
  }
  const h = state.health;
  const ok = h.ml_mode === "real" && h.warmup === "done";
  return (
    <span className={`health ${ok ? "health-ok" : "health-warn"}`} title={`ML rejimi: ${h.ml_mode}`}>
      <span className="health-dot" aria-hidden="true" />
      Model: {h.model_version} · {WARMUP_UZ[h.warmup]}
    </span>
  );
}
