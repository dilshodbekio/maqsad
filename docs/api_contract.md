# API contract (Dilshodbek ↔ Oybek)

Source: `docs/brief.md` §5. Any change requires agreement from both sides (via PR).
During development the backend may return mock responses; **in the demo only the real model runs**.
`ml/` serves the **demo** portfolio (generator split `demo`, seed 7), generated in memory on the first call
(~6 s including imports; warm the service at start-up by calling `get_queue()` once).

```
GET  /queue?sandbox_id=...            (sandbox_id optional)
  → [{loan_id, borrower_name_masked, purpose, amount_uzs, region, risk, color, top_reason,
      base_color, priority, escalations[]}]
     sorted by priority, descending (priority = colour rank + risk)

GET  /loan/{loan_id}?sandbox_id=...   (sandbox_id optional)
  → {loan, transactions[], photos[], invoice?,
     risk, color, base_color, priority, escalations[],
     reasons:[{code, text_uz, weight, evidence_refs}], evidence:[...], model_version}

POST /score_loan        {loan_id}
  → {risk: 0..1, color: "green|yellow|red", reasons:[{code, text_uz, weight}], model_version, latency_ms}

POST /score_scenario    {purpose, amount_uzs, region?, supplier, transactions[], snapshot_day}
  → {risk, color, reasons:[{code, text_uz, weight}], model_version, latency_ms, percentile, features{}}
     invalid input → HTTP 400 with the Uzbek message of `ScenarioError`

GET  /scenario_presets
  → [{id, title_uz, description_uz, scenario}]     (inputs only; scores are always computed live)

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
| `GET /queue` | `ml.api.get_queue(sandbox_id=None)` | ml |
| `GET /loan/{id}` | `ml.api.get_loan(loan_id, sandbox_id=None)` | ml |
| `POST /score_loan` | `ml.api.score_loan(loan_id)` | ml |
| `POST /score_scenario` | `ml.api.score_scenario(scenario)` | ml |
| `GET /scenario_presets` | `ml.api.get_scenario_presets()` | ml |
| `POST /check_photo` | `ml.api.check_photo(loan_id, image_bytes, lat, lon, captured_at, sandbox_id)` | ml |
| demo "reset" button | `ml.api.reset_sandbox(sandbox_id)` (`None` = every sandbox) | ml |
| colour legend | `ml.api.get_color_thresholds()` | ml |
| `POST /draft_report` | backend (LLM) | backend |
| `POST /decision`, `GET /audit` | backend (audit log) | backend |

Exceptions: `ml.api.LoanNotFoundError` (unknown or invalid `loan_id`, any type → HTTP 404),
`ml.api.InvalidImageError` (empty / not an image / corrupted / > 10 MB → HTTP 400/413) and
`ml.api.ScenarioError` (a `ValueError`; invalid scenario, message in Uzbek → HTTP 400).

## Colour rule (by queue rank)

`color` is assigned by rank in the **served portfolio**: `red` = top 10 % by risk, `yellow` = next 10 %, `green` = the rest,
so red + yellow = the 20 % inspection budget. The risk scores at those ranks are available from
`ml.api.get_color_thresholds()` → `{red, yellow, n_red, n_yellow, n_loans, rule}` and are listed in `docs/metrics.md`.
`risk` is a model score in 0..1, **not** a calibrated probability. After evidence fusion (below) the queue / loan `color` can be
higher than this money-trail colour (`base_color`), never lower; `score_loan` and `score_scenario` return the money-trail colour only.

## Evidence fusion (photo → queue)

A documented, configurable **policy** (`ml/fusion.py`), not a learned model (no labelled photo data). Escalation only raises
priority and never changes `risk`:

| Code | Minimum colour | Triggered when |
|---|---|---|
| `PHOTO_DUPLICATE` | red | the loan's photo duplicates another loan's photo (portfolio index, or the same sandbox) |
| `PHOTO_FAR` | yellow | `geo_ok == false` |
| `PHOTO_TIME` | yellow | `time_ok == false` |
| `PHOTO_ASSET_MISMATCH` | yellow | asset match below a threshold (inactive until the CLIP step exists) |

- `priority` = colour rank (green 0, yellow 1, red 2) + `risk`: the queue sort key.
- `escalations[]` item: `{code, min_color, text_uz, evidence_refs, raises_color}` (`raises_color` = it raised the colour).
- `get_loan().reasons` lists the escalation reasons first: `{code, text_uz, weight: null, evidence_refs, source: "fusion_policy"}`;
  texts are factual (e.g. "Yuklangan rasm L00xxxx kreditidagi rasm bilan bir xil") and never accusatory.
- The demo portfolio's own photos are evaluated with the same policy in the default view.
- `sandbox_id` on `get_queue` / `get_loan`: the view of that demo sandbox, including the escalations caused by photos uploaded
  there with `check_photo(..., sandbox_id=...)`. Other sandboxes and the default view (`None`) are unaffected; `reset_sandbox`
  removes the effect. Both parameters are optional, so existing callers keep working.

## Response shapes

`reasons[]` (top 3 risk-raising facts, plain Uzbek, never accusatory; at most one per group of equivalent facts):
`{code, text_uz, weight, evidence_refs}`. `weight` = share of the reason in the loan's total risk-raising push (0..1).
`evidence_refs` is present in `get_loan` only (not in `score_loan`) and lists `ref`s of `evidence[]`.

`evidence[]` item: `{ref, kind: "txn" | "invoice" | "graph" | "photo", summary_uz, data}`.
- `txn`: `ref` = transaction id; `data` = `{txn_id, ts, type, amount_uzs, counterparty_id}`.
- `invoice`: `ref` = `invoice:<loan_id>`; `data` = `{supplier_name, supplier_oked, oked_matches_purpose, invoice_amount_uzs}`.
- `graph`: `ref` = `graph:<supplier_id>`; `data` = `{supplier_name, in_degree, cash_share, exit_share, active_span_days, employee_link}`.
- `photo`: `ref` = `photo:<photo_id>` (portfolio) or `photo:sandbox:<loan_id>:<n>` (upload); `data` = `{photo, uploaded, duplicate_of, geo_ok, distance_km, time_ok}`.
`ref` is what `/draft_report` cites as `evidence_ref`.

`invoice`: `{supplier_name, supplier_oked, oked_matches_purpose, invoice_amount_uzs}`.

`photos[]` (only for the subset of loans that have a photo, otherwise `[]`):
`{photo_id, path, purpose, captured_at, lat, lon, duplicate_of: [{loan_id, similarity}], source: "portfolio"}`;
`path` is relative to the repository root (the backend serves the file).

## Scenario scoring

`score_scenario(scenario)` lets a user build a loan and its transactions and get a **live** score from the same model, features and
reason templates as a portfolio loan (changing one transaction changes the score). The scenario is evaluated as one extra loan
inside a copy of the demo portfolio's context; the cached portfolio is never modified.

```
{purpose, amount_uzs, region?, snapshot_day,
 supplier: {mode: "existing", supplier_id} | {mode: "new", oked: "NN.NN", registered_days_before: int},
 transactions: [{day, type, amount_uzs, counterparty_kind?}]}
```

- `purpose`: `tikuv_dastgohi | chorva | issiqxona | parranda | asalarichilik`; `region` (optional): a name from the generator's region list.
- `snapshot_day`: days from the disbursement to the evaluation date (0..730). The loan amount arrives on day 0 (not listed in `transactions`).
- `transactions[]` (at most 60): `day` in `[0, snapshot_day]`; `type` one of `cash_withdrawal | supplier_payment | p2p_out | p2p_in | external_transfer | income`;
  `amount_uzs` > 0; `counterparty_kind` optional (`person | business | external | supplier | none`). `supplier_payment` goes to the
  scenario's supplier. Outflows may not exceed the available balance (inflows of the same day are applied first): such a scenario is rejected,
  never capped.
- Assumptions: the invoice amount equals the loan amount; a new supplier has no outflows of its own; for an existing supplier its real demo
  ledger and loans are the context for the supplier-graph features (in-degree includes the scenario loan).
- Returns the `score_loan` shape + `percentile` (0-100: share of demo-portfolio loans with a lower-or-equal risk) + `features` (the 29 computed
  values). The colour is the money-trail colour by the demo portfolio's rank thresholds. Invalid input → `ScenarioError` (HTTP 400).
- `get_scenario_presets()` → 4 editable starting points with Uzbek titles: `honest_purchase`, `fast_cash_out`, `supplier_ring`
  (uses an existing demo supplier with many borrowers and high downstream cash share), `asset_resale`. They are inputs only.

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
