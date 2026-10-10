# API contract (Dilshodbek ↔ Oybek) — v1.1

Source: `docs/brief.md` §5. Any change requires agreement from both sides (via PR).
During development the backend may fall back to a visible STUB; **in the demo only the real model runs**.
Code: HTTP schemas in `backend/app/schemas.py` (1:1 with this file), Python interface in `ml/api.py`.

## HTTP (all endpoints under `/api`)

```
GET  /api/health
  → {ml_mode: "real|stub", model_version, warmup: "pending|running|done|failed",
     functions: {get_queue|get_loan|score_loan|check_photo|reset_sandbox: "real|stub|not_ready|missing"}}

GET  /api/queue
  → [{loan_id, borrower_name_masked, purpose, amount_uzs, region, risk, color, top_reason}]
     sorted by risk, descending

GET  /api/loan/{loan_id}
  → {loan, transactions[], photos[], invoice?,
     risk, color, reasons:[{code, text_uz, weight, evidence_refs[]}], evidence[], model_version}

POST /api/score_loan        {loan_id}
  → {risk: 0..1, color: "green|yellow|red", reasons:[{code, text_uz, weight}], model_version, latency_ms}

POST /api/check_photo       multipart: image (file) + loan_id, lat?, lon?, captured_at?
                            (fallback JSON: {loan_id, image_base64, lat?, lon?, captured_at?})
  → {asset_match: {label, score} | null, duplicate_of: [{loan_id, similarity}],
     geo_ok: bool|null, time_ok: bool|null, faces_blurred: bool, faces_count?: int, latency_ms}

POST /api/draft_report      {loan_id}
  → {text_uz, facts:[{claim, evidence_ref, unverified: bool}], model, latency_ms, cost_usd|null}

POST /api/decision          {loan_id, officer_id, action: "approve|reject|visit", comment}
  → {audit_id}

GET  /api/audit?loan_id=...
  → [{ts, actor, action, ai_recommendation, model_version, inputs_hash}]

POST /api/admin/reset       header X-Admin-Token  → {ok: true}
GET  /api/photo/{photo_id}  → blurred JPEG of a borrower upload (owner sandbox only)
GET  /api/asset/{path}      → portfolio photo from ml/assets/ (photos[].url)
```

**`risk` is a model score in 0..1 for ranking, not a probability.** The UI calls it "xavf bahosi" and never shows it as %.

### Inner shapes of `GET /api/loan/{loan_id}` (= `ml.api.get_loan` output)

| Field | Shape |
|---|---|
| `loan` | `{loan_id, borrower_id, borrower_name_masked, purpose, amount_uzs, term_months, disbursed_at, snapshot_date, region, branch_id, supplier_id, invoice_amount_uzs}` |
| `transactions[]` | `{txn_id, ts, type, amount_uzs, counterparty_id, counterparty_kind}`; `type` ∈ `disbursement, cash_withdrawal, supplier_payment, p2p_out, p2p_in, external_transfer, income` |
| `photos[]` | ml: `{photo_id, path, purpose, captured_at, lat, lon, duplicate_of[]}` (`path` repo-relative, under `ml/assets/`); backend adds `url`, `source: "portfolio"`. Borrower uploads of the caller's sandbox: `{photo_id, url, captured_at (server time), client_captured_at, lat, lon, source: "borrower_upload", check}` |
| `invoice?` | `{supplier_name, supplier_oked, oked_matches_purpose, invoice_amount_uzs}` |
| `reasons[]` | `{code, text_uz, weight, evidence_refs[]}`; every `evidence_refs` value is an `evidence[].ref` |
| `evidence[]` | `{ref, kind: "txn|invoice|graph", summary_uz, data}`; `ref` = `txn_id`, `invoice:…` or `graph:…` |

### Errors
Every error → `{detail: "<Uzbek message>"}`:
`400` bad file/input (incl. `InvalidImageError`) · `404` unknown `loan_id` · `413` file too large (> 8 MB) ·
`422` validation · `429` rate limit · `502` ML/LLM failure ·
`503` function not connected yet (`NotImplementedError`: "Bu funksiya hali ulanmagan") or LLM not configured.

## Python interface (backend imports `ml/` directly, only via `backend/app/ml_adapter.py`)

| Endpoint | Backend calls | Owner |
|---|---|---|
| `GET /api/queue` | `ml.api.get_queue()` | ml |
| `GET /api/loan/{id}` | `ml.api.get_loan(loan_id)` | ml |
| `POST /api/score_loan` | `ml.api.score_loan(loan_id)` | ml |
| `POST /api/check_photo` | `ml.api.check_photo(loan_id, image_bytes, lat, lon, captured_at, sandbox_id)` → also returns internal `blurred_image: bytes` (stored by backend, never sent over HTTP) | ml |
| `POST /api/admin/reset` | `ml.api.reset_sandbox(sandbox_id)` for each sandbox + `reset_sandbox(None)` (= all) | ml |
| `POST /api/draft_report` | backend (LLM) | backend |
| `POST /api/decision`, `GET /api/audit` | backend (audit log) | backend |

Errors: `ml.errors.LoanNotFoundError` (KeyError) → 404 · `InvalidImageError` (ValueError) → 400 / 413 ·
`NotImplementedError` → 503 · anything else → 502.
Warm-up: backend calls `get_queue()` once in a background thread at start; if it fails the backend serves the STUB
(or, with `REQUIRE_REAL_ML=1`, the process exits).

## v1.1 o'zgarishlar (v1.0 → v1.1)

1. Hamma endpoint `/api` ostida (frontend yo'llari `/`, `/loan/:id`, `/capture` bilan to'qnashmasligi uchun).
2. `/draft_report`: `facts[].unverified` qo'shildi; `evidence_ref` faqat `evidence[].ref` qiymatlaridan biri bo'lishi mumkin
   (kodda tekshiriladi; o'tmagan fakt yashirilmaydi, `unverified: true` bo'ladi). `cost_usd` narx noma'lum bo'lsa `null`.
3. `/check_photo`: `captured_at` ixtiyoriy (server vaqti asosiy); `asset_match` `null` bo'lishi mumkin; `faces_count` qo'shildi;
   kirish: asosiy `multipart/form-data`, zaxira JSON + `image_base64`.
4. `ml.api.check_photo` ga `sandbox_id` (takror qidiruv = portfel indeksi + o'z sandbox'i) va ichki `blurred_image` qo'shildi
   (Dilshodbek bilan kelishilgan, ml'da amalga oshirilgan).
5. `ml.api.reset_sandbox(sandbox_id)` qo'shildi; `/api/admin/reset` uni chaqiradi. Audit jurnali append-only, reset'da saqlanadi.
6. Ichki shakllar ml chiqishiga moslashtirildi (yuqoridagi jadval): `counterparty_id/_kind`, `supplier_oked`, `invoice_amount_uzs`,
   `reasons[].evidence_refs`, `evidence.kind` ∈ `txn|invoice|graph`, `photos[].path`.
7. Xizmat endpointlari: `/api/health` (funksiya bo'yicha holat va warm-up), `/api/admin/reset`, `/api/photo/{id}`, `/api/asset/{path}`.
   Har javobda `X-ML-Mode`, `X-Model-Version` headerlari. `/check_photo` audit'ga `actor: "borrower", action: "photo_upload"` yozadi.
8. Xato shakli `{detail}` va status kodlari belgilandi (yuqorida); `NotImplementedError` → 503.
9. `risk` ehtimollik emas — UI'da "xavf bahosi", "%" ishlatilmaydi.

## Open points
- `color` thresholds (green/yellow/red): set by ml (`ml/artifacts/money_trail_v1.json`); values in `docs/metrics.md`.
- `reset_sandbox(None)` = "all sandboxes" is agreed, but `ml/photo/checker.py` currently treats `None` as one key
  (no-op). The backend therefore also resets every sandbox it knows by id.
