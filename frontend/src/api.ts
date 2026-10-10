// All requests are relative (/api/...); the same FastAPI service serves the API and this build.
// Contract: docs/api_contract.md + docs/api_contract_changes.md.

export type Health = {
  ml_mode: "real" | "stub";
  model_version: string;
};

export class ApiError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
  }
}

async function getJson<T>(path: string): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, { headers: { Accept: "application/json" } });
  } catch {
    throw new ApiError("Server bilan aloqa yo'q. Internetni tekshirib, qayta urinib ko'ring.", 0);
  }
  let body: unknown = null;
  try {
    body = await res.json();
  } catch {
    // non-JSON body (e.g. proxy error page)
  }
  if (!res.ok) {
    const detail = (body as { detail?: unknown } | null)?.detail;
    throw new ApiError(typeof detail === "string" ? detail : `Server xatosi (${res.status}).`, res.status);
  }
  return body as T;
}

export const api = {
  health: () => getJson<Health>("/api/health"),
};
