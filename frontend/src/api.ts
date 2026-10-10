// All requests are relative (/api/...); the same FastAPI service serves the API and this build.
// Contract: docs/api_contract.md (v1.5).

export type FunctionStatus = "real" | "stub" | "not_ready" | "missing";

export type Health = {
  ml_mode: "real" | "stub";
  model_version: string;
  warmup: "pending" | "running" | "done" | "failed";
  functions: Record<string, FunctionStatus>;
  clip_loaded?: boolean;
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

// ---------- GET /api/loan/{id} (= ml.api.get_loan + backend photo urls) ----------

export type Loan = {
  loan_id: string;
  borrower_id?: string | null;
  borrower_name_masked?: string | null;
  purpose?: string | null;
  amount_uzs?: number | null;
  term_months?: number | null;
  disbursed_at?: string | null;
  snapshot_date?: string | null;
  region?: string | null;
  branch_id?: string | null;
  supplier_id?: string | null;
  invoice_amount_uzs?: number | null;
};

export type TxnType =
  | "disbursement"
  | "cash_withdrawal"
  | "supplier_payment"
  | "p2p_out"
  | "p2p_in"
  | "external_transfer"
  | "income";

export type Transaction = {
  txn_id: string;
  ts: string;
  type: TxnType | string;
  amount_uzs: number;
  counterparty_id?: string | null;
  counterparty_kind?: string | null;
};

export type Duplicate = { loan_id: string; similarity: number };

// CLIP asset check (v1.5); null when CLIP is unavailable. score is a model score, not shown as %.
export type AssetMatch = { label: string; score: number; matches_purpose: boolean; label_uz: string };

export type CheckPhotoResult = {
  asset_match: AssetMatch | null;
  duplicate_of: Duplicate[];
  geo_ok: boolean | null;
  time_ok: boolean | null;
  faces_blurred: boolean;
  faces_count?: number | null;
  latency_ms: number;
};

export type Photo = {
  photo_id: string;
  path?: string | null;
  purpose?: string | null;
  captured_at?: string | null;
  lat?: number | null;
  lon?: number | null;
  duplicate_of: Duplicate[];
  asset_match?: AssetMatch | null;
  url?: string | null;
  source: "portfolio" | "borrower_upload";
  client_captured_at?: string | null;
  check?: CheckPhotoResult | null;
};

export type Invoice = {
  supplier_name?: string | null;
  supplier_oked?: string | null;
  oked_matches_purpose?: boolean | null;
  invoice_amount_uzs?: number | null;
};

export type EvidenceKind = "txn" | "invoice" | "graph" | "photo";

export type Evidence = {
  ref: string;
  kind: EvidenceKind;
  summary_uz?: string | null;
  data?: unknown;
};

export type LoanReason = {
  code: string;
  text_uz: string;
  weight: number | null; // null for evidence-fusion policy reasons
  evidence_refs: string[];
  source?: "fusion_policy" | null;
};

export type LoanDetail = {
  loan: Loan;
  transactions: Transaction[];
  photos: Photo[];
  invoice?: Invoice | null;
  risk: number;
  color: Color;
  base_color: Color;
  priority: number;
  escalations: Escalation[];
  reasons: LoanReason[];
  evidence: Evidence[];
  model_version: string;
};

export type Fact = { claim: string; evidence_ref: string; unverified: boolean };

export type DraftReport = {
  text_uz: string;
  facts: Fact[];
  model: string;
  latency_ms: number;
  cost_usd: number | null;
  input_tokens?: number | null;
  output_tokens?: number | null;
  generated_at?: string | null; // UTC ISO time the draft was generated
  cached?: boolean; // served from the cache: no LLM call, latency_ms is the lookup time
};

export type DraftJob = {
  job_id: string;
  status: "queued" | "running" | "done" | "error";
  position: number; // 1 = next; 0 when running / finished
  eta_s: number | null; // estimate, null when unknown
  result?: DraftReport | null;
  error?: string | null;
};

export type DecisionAction = "approve" | "reject" | "visit";

export type AuditEntry = {
  ts: string;
  actor: string;
  action: string;
  ai_recommendation: string | null;
  model_version: string | null;
  inputs_hash: string;
};

export type Session = { officer_id: string };

// ---------- scenario (contract v1.6): hypothetical, never stored ----------

export type ScenarioSupplier =
  | { mode: "existing"; supplier_id: string }
  | { mode: "new"; oked: string; registered_days_before: number };

export type ScenarioTx = { day: number; type: string; amount_uzs: number; counterparty_kind?: string | null };

export type Scenario = {
  purpose: string;
  amount_uzs: number;
  invoice_amount_uzs?: number | null;
  region?: string | null;
  snapshot_day: number;
  supplier: ScenarioSupplier;
  transactions: ScenarioTx[];
};

export type ScenarioPreset = { id: string; title_uz: string; description_uz: string; scenario: Scenario };

export type ScenarioScore = {
  risk: number; // model score 0..1, "xavf bahosi"
  color: Color;
  reasons: { code: string; text_uz: string; weight: number }[];
  model_version: string;
  latency_ms: number;
  percentile: number; // 0..100, rank among the demo portfolio's loans
};

export class ApiError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, { ...init, headers: { Accept: "application/json", ...init?.headers } });
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

const getJson = <T>(path: string) => request<T>(path);

const postJson = <T>(path: string, payload: unknown) =>
  request<T>(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

function query(params: Record<string, string | number | undefined>): string {
  const q = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== "") q.set(k, String(v));
  }
  const s = q.toString();
  return s ? `?${s}` : "";
}

export const QUEUE_PAGE_SIZE = 50;

async function postForm<T>(path: string, form: FormData): Promise<T> {
  return request<T>(path, { method: "POST", body: form });
}

export const api = {
  health: () => getJson<Health>("/api/health"),
  queue: (filters: QueueFilters, offset = 0, limit = QUEUE_PAGE_SIZE) =>
    getJson<QueuePage>(`/api/queue${query({ ...filters, offset, limit })}`),
  queueFacets: () => getJson<QueueFacets>("/api/queue/facets"),
  queueLegend: () => getJson<ColorLegend>("/api/queue/legend"),
  loan: (loanId: string) => getJson<LoanDetail>(`/api/loan/${encodeURIComponent(loanId)}`),
  session: () => getJson<Session>("/api/session"),
  audit: (loanId: string) => getJson<AuditEntry[]>(`/api/audit${query({ loan_id: loanId })}`),
  scenarioPresets: () => getJson<ScenarioPreset[]>("/api/scenario_presets"),
  scoreScenario: (scenario: Scenario) => postJson<ScenarioScore>("/api/score_scenario", scenario),
  checkPhoto: (form: FormData) => postForm<CheckPhotoResult>("/api/check_photo", form),
  // 200 = a cached DraftReport; 202 = a DraftJob to poll
  draftReport: (loanId: string, regenerate = false) =>
    postJson<DraftReport | DraftJob>("/api/draft_report", { loan_id: loanId, regenerate }),
  draftJob: (jobId: string) => getJson<DraftJob>(`/api/draft_report/job/${encodeURIComponent(jobId)}`),
  decision: (body: { loan_id: string; officer_id: string; action: DecisionAction; comment: string }) =>
    postJson<{ audit_id: number }>("/api/decision", body),
};
