import type { Health } from "../api";
import { t } from "../i18n";

export type HealthState =
  | { kind: "loading" }
  | { kind: "error"; message: string }
  | { kind: "ok"; health: Health };

/** Banner shown only when the server state is unknown (the model version is in the footer, the stub / warm-up banners are separate). */
export function HealthStatus({ state, onRetry }: { state: HealthState; onRetry: () => void }) {
  if (state.kind !== "error") return null;
  return (
    <div className="partial-banner" role="status">
      {t("health.unknown")}{" "}
      <button className="link-button" onClick={onRetry}>
        {t("health.recheck")}
      </button>
    </div>
  );
}
