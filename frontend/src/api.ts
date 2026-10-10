// All requests are relative (/api/...); the same FastAPI service serves the API and this build.
// Contract: docs/api_contract.md (v1.3).

export type FunctionStatus = "real" | "stub" | "not_ready" | "missing";

export type Health = {
  ml_mode: "real" | "stub";
  model_version: string;
  warmup: "pending" | "running" | "done" | "failed";
  functions: Record<string, FunctionStatus>;
};

export type Color = "red" | "yellow" | "green";

export type Escalation = {
  code: string;
  min_color: Color;
  text_uz: string;
  evidence_refs: string[];
  raises_color: boolean;
};

export type QueueItem = {
  loan_id: string;
  borrower_name_masked: string;
  purpose: string;
  amount_uzs: number;
  region: string;
  // model score 0..1 for ranking, NOT a probability: shown as "xavf bahosi", never as %
  risk: number;
  color: Color;
  top_reason: string;
  base_color: Color;
  priority: number;
  escalations: Escalation[];
};

export type QueuePage = {
  items: QueueItem[];
  total: number;
  counts: Record<Color, number>;
};

export type QueueFilters = {
  color?: Color;
  region?: string;
  purpose?: string;
};

export type QueueFacets = {
  regions: string[];
  purposes: string[];
};

export type ColorLegend = {
  red: number;
  yellow: number;
  n_red: number;
  n_yellow: number;
  n_loans: number;
  rule: string;
  n_escalated_to_red: number;
  n_escalated_to_yellow: number;
  n_red_total: number;
  n_yellow_total: number;
  note: string;
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

function query(params: Record<string, string | number | undefined>): string {
  const q = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== "") q.set(k, String(v));
  }
  const s = q.toString();
  return s ? `?${s}` : "";
}

export const QUEUE_PAGE_SIZE = 50;

export const api = {
  health: () => getJson<Health>("/api/health"),
  queue: (filters: QueueFilters, offset = 0, limit = QUEUE_PAGE_SIZE) =>
    getJson<QueuePage>(`/api/queue${query({ ...filters, offset, limit })}`),
  queueFacets: () => getJson<QueueFacets>("/api/queue/facets"),
  queueLegend: () => getJson<ColorLegend>("/api/queue/legend"),
};
