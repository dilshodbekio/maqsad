# API contract (Dilshodbek ↔ Oybek)

Source: `docs/brief.md` §5. Any change requires agreement from both sides (via PR).
During development the backend may return mock responses; **in the demo only the real model runs**.
`ml/` serves the **demo** portfolio (generator split `demo`, seed 7), generated in memory on the first call
(~6 s including imports; warm the service at start-up by calling `get_queue()` once).

```
GET  /queue
  → [{loan_id, borrower_name_masked, purpose, amount_uzs, region, risk, color, top_reason}]
     sorted by risk, descending

GET  /loan/{loan_id}
  → {loan, transactions[], photos[], invoice?,
     risk, color, reasons:[{code, text_uz, weight, evidence_refs}], evidence:[...], model_version}

POST /score_loan        {loan_id}
  → {risk: 0..1, color: "green|yellow|red", reasons:[{code, text_uz, weight}], model_version, latency_ms}

POST /check_photo       {loan_id, image (base64 or multipart), lat?, lon?, captured_at, sandbox_id?}
  → {asset_match: {label, score} | null, duplicate_of: [{loan_id, similarity}],
     geo_ok: bool | null, time_ok: bool | null, faces_blurred: bool, faces_count: int, latency_ms}
     (the blurred JPEG is returned by `ml` as `blurred_image: bytes` for the backend to store; it is NOT part of the HTTP response)

POST /draft_report      {loan_id}
  → {text_uz, facts:[{claim, evidence_ref}], model, latency_ms, cost_usd}

POST /decision          {loan_id, officer_id, action: "approve|reject|visit", comment}
  → {audit_id}

GET  /audit?loan_id=...
  → [{ts, actor, action, ai_recommendation, model_version, inputs_hash}]
```

## Python interface (backend imports `ml/` directly)

| Endpoint | Backend calls | Owner |
|---|---|---|
| `GET /queue` | `ml.api.get_queue()` | ml |
| `GET /loan/{id}` | `ml.api.get_loan(loan_id)` | ml |
| `POST /score_loan` | `ml.api.score_loan(loan_id)` | ml |
| `POST /check_photo` | `ml.api.check_photo(loan_id, image_bytes, lat, lon, captured_at, sandbox_id)` | ml |
| demo "reset" button | `ml.api.reset_sandbox(sandbox_id)` (`None` = every sandbox) | ml |
| colour legend | `ml.api.get_color_thresholds()` | ml |
| `POST /draft_report` | backend (LLM) | backend |
| `POST /decision`, `GET /audit` | backend (audit log) | backend |

Exceptions: `ml.api.LoanNotFoundError` (unknown or invalid `loan_id`, any type → HTTP 404) and
`ml.api.InvalidImageError` (empty / not an image / corrupted / > 10 MB → HTTP 400/413).

## Colour rule (by queue rank)

`color` is assigned by rank in the **served portfolio**: `red` = top 10 % by risk, `yellow` = next 10 %, `green` = the rest,
so red + yellow = the 20 % inspection budget. The risk scores at those ranks are available from
`ml.api.get_color_thresholds()` → `{red, yellow, n_red, n_yellow, n_loans, rule}` and are listed in `docs/metrics.md`.
`risk` is a model score in 0..1, **not** a calibrated probability.

## Response shapes

`reasons[]` (top 3 risk-raising facts, plain Uzbek, never accusatory; at most one per group of equivalent facts):
`{code, text_uz, weight, evidence_refs}`. `weight` = share of the reason in the loan's total risk-raising push (0..1).
`evidence_refs` is present in `get_loan` only (not in `score_loan`) and lists `ref`s of `evidence[]`.

`evidence[]` item: `{ref, kind: "txn" | "invoice" | "graph", summary_uz, data}`.
- `txn`: `ref` = transaction id; `data` = `{txn_id, ts, type, amount_uzs, counterparty_id}`.
- `invoice`: `ref` = `invoice:<loan_id>`; `data` = `{supplier_name, supplier_oked, oked_matches_purpose, invoice_amount_uzs}`.
- `graph`: `ref` = `graph:<supplier_id>`; `data` = `{supplier_name, in_degree, cash_share, exit_share, active_span_days, employee_link}`.
- (`photo` evidence is added with the photo ↔ risk combination.)
`ref` is what `/draft_report` cites as `evidence_ref`.

`invoice`: `{supplier_name, supplier_oked, oked_matches_purpose, invoice_amount_uzs}`.

`photos[]` (only for the subset of loans that have a photo, otherwise `[]`):
`{photo_id, path, purpose, captured_at, lat, lon, duplicate_of: [{loan_id, similarity}], source: "portfolio"}`;
`path` is relative to the repository root (the backend serves the file).

## check_photo details

- `sandbox_id`: the demo visitor's sandbox. Duplicate search covers the portfolio index **and only photos uploaded in this
  sandbox**; sandbox photos never enter the portfolio index and live in memory only. `None` = production mode (nothing is
  remembered). `reset_sandbox(sandbox_id)` forgets one sandbox, `reset_sandbox(None)` all of them; the portfolio index is untouched.
- The original image is never written anywhere: faces are blurred, EXIF is dropped by re-encoding, only the blurred JPEG
  (`blurred_image`, internal) and a perceptual hash are kept.
- `faces_count` = faces found; `faces_blurred` = `faces_count > 0` (faces were found and blurred).
- `asset_match` is `null` until the CLIP step is added. `geo_ok` / `time_ok` are `null` when the input is missing.
- The photo result does **not** change the money-trail `risk` / `color` (combination comes later).

## Open points (to agree)
- Combination of photo signals with the money-trail risk (later step).
