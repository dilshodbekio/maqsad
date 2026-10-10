# Proposed changes to `docs/api_contract.md`

Status: **proposal**. Agreed by Oybek; items marked ⚠️ need Dilshodbek's approval (they change the `ml.api` signature).
The backend already implements all items below; items 3–4 are handled as optional so the backend works either way.

---

### 1. `/draft_report` — mark unverified facts
- **Before:** `facts:[{claim, evidence_ref}]`
- **After:** `facts:[{claim, evidence_ref, unverified: bool = false}]`
- Why: facts that fail the code-side check (missing `evidence_ref`, number not in evidence) are flagged, not hidden.

### 2. `/check_photo` — `captured_at` is optional
- **Before:** `{loan_id, image, lat?, lon?, captured_at}`
- **After:** `{loan_id, image, lat?, lon?, captured_at?}`
- Server time is the primary timestamp (recorded by backend); client `captured_at` is only compared against it.

### 3. ⚠️ `ml.api.check_photo` — sandbox scope (**Dilshodbek tasdig'i kerak**)
- **Before:** `check_photo(loan_id, image_bytes, lat=None, lon=None, captured_at=None)`
- **After:** `check_photo(loan_id, image_bytes, lat=None, lon=None, captured_at=None, sandbox_id=None)`
- Duplicate search = portfolio index + photos uploaded in the same `sandbox_id` only.
- Backend fallback until approved: if `ml.api.check_photo` does not accept `sandbox_id`, it is called without it.

### 4. ⚠️ `ml.api.check_photo` — return the blurred image (**Dilshodbek tasdig'i kerak**)
- **Before:** returns `{asset_match, duplicate_of, geo_ok, time_ok, faces_blurred, latency_ms}`
- **After:** same + internal `blurred_image: bytes | None` (Python only, **not** exposed over HTTP)
- The original image is never written to disk; only `blurred_image` may be stored.
- Backend fallback until approved: if `blurred_image` is absent, no image is stored at all.

### 5. Inner shapes (temporary — replaced by the generator's sample JSON when available)
- **Before:** `loan`, `transactions[]`, `photos[]`, `invoice?`, `evidence[]`, `top_reason` — shape not defined
- **After (temporary):**
  - `loan: {loan_id, borrower_name_masked, purpose, amount_uzs, disbursed_at, region, branch_id, supplier_id}`
  - `transactions[]: {txn_id, ts, type, amount_uzs, counterparty}` — `type` ∈ `disbursement | cash_withdrawal | supplier_payment | p2p_out | p2p_in | external_transfer | income`
  - `photos[]: {photo_id, url?, captured_at?, source}`
  - `invoice?: {supplier_id, supplier_name, oked_code, oked_matches_purpose, amount_uzs}`
  - `evidence[]: {ref, kind: "txn|photo|invoice|graph", summary_uz, data}`
  - `top_reason: string` (= `reasons[0].text_uz`)
- Single place in code: `backend/app/schemas.py`. When the generator's JSON arrives, its shape wins.

### 6. `/api` prefix
- **Before:** `GET /queue`, `GET /loan/{id}`, …
- **After:** `GET /api/queue`, `GET /api/loan/{id}`, … (all endpoints under `/api`)
- Why: frontend routes (`/`, `/loan/:id`, `/capture`) are served from `/` by the same service.

### 7. Service endpoints (backend-only)
- **Before:** not in contract
- **After:**
  - `GET /api/health → {ml_mode: "real|stub", model_version}`
  - `POST /api/admin/reset` (header `X-Admin-Token`) `→ {ok: true}` — clears sandbox photos; audit log is kept (append-only)
  - `GET /api/photo/{photo_id}` → stored blurred image (own sandbox only; exists only if ml returns `blurred_image`)
  - every response carries headers `X-ML-Mode` and `X-Model-Version` (so stub mode is visible even where the body has no `model_version`)
  - `/check_photo` also writes an audit row: `actor: "borrower", action: "photo_upload"`

### 8. Error shape
- **Before:** not defined
- **After:** every error → `{detail: "<Uzbek message>"}` with status
  `400` bad file/input · `404` unknown `loan_id` · `413` file too large · `422` validation · `429` rate limit ·
  `502` ML/LLM failure · `503` LLM not configured (`/draft_report` only; the loan page works without it)

### 9. `/check_photo` input format
- **Before:** "base64 or multipart"
- **After:** primary `multipart/form-data` (`image` file + form fields) — used by the frontend;
  fallback `application/json` with `image_base64`.
