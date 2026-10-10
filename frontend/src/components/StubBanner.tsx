import type { Health } from "../api";

// Mandatory (frontend/CLAUDE.md): visible on every page while the real model is not connected.
export function StubBanner({ health }: { health: Health | null }) {
  if (!health) return null;
  if (health.ml_mode === "stub") {
    return (
      <div className="stub-banner" role="alert">
        STUB — haqiqiy model ulanmagan
      </div>
    );
  }
  if (health.warmup === "pending" || health.warmup === "running") {
    return (
      <div className="partial-banner" role="status">
        Model tayyorlanmoqda… Navbat bir necha soniyadan keyin chiqadi.
      </div>
    );
  }
  const notReady = Object.entries(health.functions)
    .filter(([, status]) => status === "not_ready" || status === "missing")
    .map(([name]) => name);
  if (notReady.length === 0) return null;
  return (
    <div className="partial-banner" role="status">
      Hali ulanmagan funksiyalar: {notReady.join(", ")}
    </div>
  );
}
