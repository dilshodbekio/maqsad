import type { Health } from "../api";

// Mandatory (frontend/CLAUDE.md): visible on every page while the real model is not connected.
export function StubBanner({ health }: { health: Health | null }) {
  if (health?.ml_mode !== "stub") return null;
  return (
    <div className="stub-banner" role="alert">
      STUB — haqiqiy model ulanmagan
    </div>
  );
}
