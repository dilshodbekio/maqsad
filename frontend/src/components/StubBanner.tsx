import type { Health } from "../api";
import { t } from "../i18n";

// Mandatory (frontend/CLAUDE.md): visible on every page while the real model is not connected.
export function StubBanner({ health }: { health: Health | null }) {
  if (!health) return null;
  if (health.ml_mode === "stub") {
    return (
      <div className="stub-banner" role="alert">
        {t("banner.stub")}
      </div>
    );
  }
  if (health.warmup === "pending" || health.warmup === "running") {
    return (
      <div className="partial-banner" role="status">
        {t("banner.warming")}
      </div>
    );
  }
  const notReady = Object.entries(health.functions)
    .filter(([, status]) => status === "not_ready" || status === "missing")
    .map(([name]) => name);
  if (notReady.length === 0) return null;
  return (
    <div className="partial-banner" role="status">
      {t("banner.notReady", { names: notReady.join(", ") })}
    </div>
  );
}
