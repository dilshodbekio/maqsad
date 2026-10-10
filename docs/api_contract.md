# API contract (Dilshodbek ↔ Oybek) — v1.2

Source: `docs/brief.md` §5. Any change requires agreement from both sides (via PR).
During development the backend may fall back to a visible STUB; **in the demo only the real model runs**.
Code: HTTP schemas in `backend/app/schemas.py` (1:1 with this file), Python interface in `ml/api.py`.

**Demo portfolio.** `ml/` serves the **demo** portfolio (generator split `demo`, seed 7), generated in memory on the first call
(~6 s including imports). The backend warms it at start-up by calling `get_queue()` once in a background thread, so
`/api/health` answers immediately (`warmup: "running"` → `"done"`).

## HTTP (all endpoints under `/api`)

```
GET  /api/health
  → {ml_mode: "real|stub", model_version, warmup: "pending|running|done|failed",
     functions: {get_queue|get_loan|score_loan|check_photo|reset_sandbox: "real|stub|not_ready|missing"}}

GET  /api/queue                       (view of the caller's sandbox: sandbox_id comes from the cookie)
  → [{loan_id, borrower_name_masked, purpose, amount_uzs, region, risk, color, top_reason,
      base_color, priority, escalations[]}]
     sorted by priority, descending (priority = colour rank + risk)

GET  /api/loan/{loan_id}              (view of the caller's sandbox)
  → {loan, transactions[], photos[], invoice?,
     risk, color, base_color, priority, escalations[],
     reasons:[{code, text_uz, weight, evidence_refs[], source?}], evidence[], model_version}

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

-- agreed in ml, NOT yet exposed by the backend (next step) --
POST /api/score_scenario    {purpose, amount_uzs, invoice_amount_uzs?, region?, supplier, transactions[], snapshot_day}
  → {risk, color, reasons:[{code, text_uz, weight}], model_version, latency_ms, percentile, features{}}
GET  /api/scenario_presets  → [{id, title_uz, description_uz, scenario}]
```

**`risk` is a model score in 0..1 for ranking, not a probability.** The UI calls it "xavf bahosi" and never shows it as %.

### Inner shapes of `GET /api/loan/{loan_id}` (= `ml.api.get_loan` output)

| Field | Shape |
|---|---|
| `loan` | `{loan_id, borrower_id, borrower_name_masked, purpose, amount_uzs, term_months, disbursed_at, snapshot_date, region, branch_id, supplier_id, invoice_amount_uzs}` |
| `transactions[]` | `{txn_id, ts, type, amount_uzs, counterparty_id, counterparty_kind}`; `type` ∈ `disbursement, cash_withdrawal, supplier_payment, p2p_out, p2p_in, external_transfer, income` |
| `photos[]` | ml: `{photo_id, path, purpose, captured_at, lat, lon, duplicate_of[], source: "portfolio"}` (only for loans that have a photo; `path` repo-relative, under `ml/assets/`); backend adds `url`. Borrower uploads of the caller's sandbox: `{photo_id, url, captured_at (server time), client_captured_at, lat, lon, source: "borrower_upload", check}` |
| `invoice?` | `{supplier_name, supplier_oked, oked_matches_purpose, invoice_amount_uzs}` |
| `reasons[]` | top 3 money-trail reasons `{code, text_uz, weight, evidence_refs[]}` (`weight` = share of the loan's risk-raising push, 0..1) + policy reasons `{code, text_uz, weight: null, evidence_refs[], source: "fusion_policy"}`; plain Uzbek, never accusatory; every `evidence_refs` value is an `evidence[].ref`. `score_loan` reasons have no `evidence_refs` |
| `escalations[]` | `{code, min_color, text_uz, evidence_refs[], raises_color}` (see "Evidence fusion") |
| `evidence[]` | `{ref, kind, summary_uz, data}`: `txn` (`ref` = txn id; `data` = `{txn_id, ts, type, amount_uzs, counterparty_id}`), `invoice` (`invoice:<loan_id>`; `data` = invoice), `graph` (`graph:<supplier_id>`; `data` = `{supplier_name, in_degree, cash_share, exit_share, active_span_days, employee_link}`), `photo` (`photo:<photo_id>` or `photo:sandbox:<loan_id>:<n>`; `data` = `{photo, uploaded, duplicate_of, geo_ok, distance_km, time_ok}`). `ref` is what `/draft_report` cites as `evidence_ref` |

### Colour rule (by queue rank)

`color` is assigned by rank in the **served portfolio**: `red` = top 10 % by risk, `yellow` = next 10 %, `green` = the rest,
so red + yellow = the 20 % inspection budget. The risk scores at those ranks come from `ml.api.get_color_thresholds()`
→ `{red, yellow, n_red, n_yellow, n_loans, rule, n_escalated_to_red, n_escalated_to_yellow, n_red_total, n_yellow_total, note}`
(listed in `docs/metrics.md`). The photo policy can raise a loan's colour, so the budget is exceeded by exactly the escalations
reported there (the thresholds themselves stay 10 % / 10 % by risk). `color` (after fusion) ≥ `base_color` (money-trail colour),
never lower; `score_loan` and `score_scenario` return the money-trail colour only.

### Evidence fusion (photo → queue)

A documented, configurable **policy** (`ml/fusion.py`), not a learned model (no labelled photo data). Escalation only raises
priority and never changes `risk`:

| Code | Minimum colour | Triggered when |
|---|---|---|
| `PHOTO_DUPLICATE` | red | the loan's photo duplicates another loan's photo (portfolio index, or the same sandbox) |
| `PHOTO_FAR` | yellow | `geo_ok == false` |
| `PHOTO_TIME` | yellow | `time_ok == false` |
| `PHOTO_ASSET_MISMATCH` | yellow | asset match below a threshold (inactive until the CLIP step exists) |

- `priority` = colour rank (green 0, yellow 1, red 2) + `risk`: the queue sort key.
- One escalation per code; if several photos trigger the same code, their texts are joined and all evidence refs are kept.
- Reason order (`get_loan().reasons`, and `top_reason` == `reasons[0].text_uz`, or "Sezilarli xavf belgisi aniqlanmadi"):
  policy reasons that **raised** the colour, then money-trail reasons, then policy reasons that did not raise it.
- Sandbox view: uploads made with `check_photo(..., sandbox_id=...)` escalate only that sandbox's `get_queue` / `get_loan`;
  other sandboxes and the default view are unaffected; `reset_sandbox` removes the effect.

### Errors
Every error → `{detail: "<Uzbek message>"}`:
`400` bad file/input (incl. `InvalidImageError`, `ScenarioError`) · `404` unknown `loan_id` · `413` file too large (> 8 MB) ·
`422` validation · `429` rate limit · `502` ML/LLM failure ·
`503` function not connected yet (`NotImplementedError`: "Bu funksiya hali ulanmagan") or LLM not configured.

## Python interface (backend imports `ml/` directly, only via `backend/app/ml_adapter.py`)

| Endpoint | Backend calls | Owner |
|---|---|---|
| `GET /api/queue` | `ml.api.get_queue(sandbox_id)` | ml |
| `GET /api/loan/{id}` | `ml.api.get_loan(loan_id, sandbox_id)` | ml |
| `POST /api/score_loan` | `ml.api.score_loan(loan_id)` | ml |
| `POST /api/check_photo` | `ml.api.check_photo(loan_id, image_bytes, lat, lon, captured_at, sandbox_id)` → also returns internal `blurred_image: bytes` (stored by backend, never sent over HTTP) | ml |
| `POST /api/admin/reset` | `ml.api.reset_sandbox(None)` (`None` = every sandbox) | ml |
| colour legend (not exposed yet) | `ml.api.get_color_thresholds()` | ml |
| `POST /api/score_scenario` (not exposed yet) | `ml.api.score_scenario(scenario)` | ml |
| `GET /api/scenario_presets` (not exposed yet) | `ml.api.get_scenario_presets()` | ml |
| `POST /api/draft_report` | backend (LLM) | backend |
| `POST /api/decision`, `GET /api/audit` | backend (audit log) | backend |

### Exceptions

| `ml` raises | Meaning | HTTP |
|---|---|---|
| `ml.api.LoanNotFoundError` (KeyError) | unknown or invalid `loan_id` (any type) | 404 |
| `ml.api.InvalidImageError` (ValueError) | empty / not an image / corrupted image | 400 |
| `ml.api.InvalidImageError` "image larger than …" | > 10 MB (the backend already rejects > 8 MB) | 413 |
| `ml.api.ScenarioError` (ValueError) | invalid scenario, message in Uzbek | 400 |
| `NotImplementedError` | function not wired yet | 503 |
| anything else | ml failure | 502 |

Warm-up: if `get_queue()` fails at start, the backend serves the STUB (or, with `REQUIRE_REAL_ML=1`, the process exits).

### check_photo details

- `sandbox_id`: the demo visitor's sandbox. Duplicate search covers the portfolio index **and only photos uploaded in this
  sandbox**; sandbox photos never enter the portfolio index and live in memory only. `None` = production mode (nothing is
  remembered). `reset_sandbox(sandbox_id)` forgets one sandbox, `reset_sandbox(None)` all of them; the portfolio index is untouched.
- The original image is never written anywhere: faces are blurred, EXIF is dropped by re-encoding; only the blurred JPEG
  (`blurred_image`, internal) and a perceptual hash are kept.
- `faces_count` = faces found; `faces_blurred` = `faces_count > 0`.
- `asset_match` is `null` until the CLIP step is added. `geo_ok` / `time_ok` are `null` when the input is missing.
- The photo result does **not** change `risk`; it can only raise colour / priority through the evidence-fusion policy.

### Scenario scoring (ml; HTTP endpoint not exposed yet)

`score_scenario(scenario)` builds a loan and its transactions and returns a **live** score from the same model, features and
reason templates as a portfolio loan. The scenario is evaluated as one extra loan inside a copy of the demo portfolio's context;
the cached portfolio is never modified.

```
{purpose, amount_uzs, invoice_amount_uzs?, region?, snapshot_day,
 supplier: {mode: "existing", supplier_id} | {mode: "new", oked: "NN.NN", registered_days_before: int},
 transactions: [{day, type, amount_uzs, counterparty_kind?}]}
```

- `purpose`: `tikuv_dastgohi | chorva | issiqxona | parranda | asalarichilik`; `region` (optional): a name from the generator's region list.
- `snapshot_day`: days from the disbursement to the evaluation date, **2..364**; outside this range → `ScenarioError`.
  The loan amount arrives on day 0 (not listed in `transactions`).
- `invoice_amount_uzs` (optional): without it there is no invoice (invoice-gap feature = `-1`, no invoice-gap reason).
- Unknown keys anywhere → `ScenarioError` listing them. Absurd inputs → `ScenarioError`: any single amount above 10 × `amount_uzs`,
  or total inflows (`p2p_in` + `income`) above 20 × `amount_uzs`.
- `transactions[]` (at most 60): `day` in `[0, snapshot_day]`; `type` one of `cash_withdrawal | supplier_payment | p2p_out | p2p_in |
  external_transfer | income`; `amount_uzs` > 0; `counterparty_kind` optional (`person | business | external | supplier | none`).
  Outflows may not exceed the available balance (same-day inflows first): such a scenario is rejected, never capped.
- Reasons only state facts the user entered or that follow from them (no supplier cash-out / exit / activity reasons for a new
  supplier, no invoice-gap reason without an invoice); hiding a reason never changes the score.
- Returns the `score_loan` shape + `percentile` (0-100, share of demo-portfolio loans with lower-or-equal risk) + `features`
  (the 29 computed values). The colour is the money-trail colour by the demo portfolio's thresholds.
- `get_scenario_presets()` → 4 editable starting points with Uzbek titles: `honest_purchase`, `fast_cash_out`, `supplier_ring`,
  `asset_resale` (inputs only; scores are always computed live).

## v1.1 o'zgarishlar (v1.0 → v1.1)

1. Hamma endpoint `/api` ostida (frontend yo'llari `/`, `/loan/:id`, `/capture` bilan to'qnashmasligi uchun).
2. `/draft_report`: `facts[].unverified` qo'shildi; `evidence_ref` faqat `evidence[].ref` qiymatlaridan biri bo'lishi mumkin
   (kodda tekshiriladi; o'tmagan fakt yashirilmaydi, `unverified: true` bo'ladi). `cost_usd` narx noma'lum bo'lsa `null`.
3. `/check_photo`: `captured_at` ixtiyoriy (server vaqti asosiy); `asset_match` `null` bo'lishi mumkin; `faces_count` qo'shildi;
   kirish: asosiy `multipart/form-data`, zaxira JSON + `image_base64`.
4. `ml.api.check_photo` ga `sandbox_id` va ichki `blurred_image` qo'shildi.
5. `ml.api.reset_sandbox(sandbox_id)` qo'shildi; `/api/admin/reset` uni chaqiradi. Audit jurnali append-only, reset'da saqlanadi.
6. Ichki shakllar ml chiqishiga moslashtirildi: `counterparty_id/_kind`, `supplier_oked`, `invoice_amount_uzs`,
   `reasons[].evidence_refs`, `photos[].path`.
7. Xizmat endpointlari: `/api/health`, `/api/admin/reset`, `/api/photo/{id}`, `/api/asset/{path}`;
   `X-ML-Mode`, `X-Model-Version` headerlari; `/check_photo` audit'ga `photo_upload` yozadi.
8. Xato shakli `{detail}` va status kodlari belgilandi; `NotImplementedError` → 503.
9. `risk` ehtimollik emas — UI'da "xavf bahosi", "%" ishlatilmaydi.

## v1.2 o'zgarishlar (model v2)

1. Demo portfel (`demo` split, seed 7), warm-up ~6 s; backend uni fonda isitadi.
2. Rang qoidasi: navbatdagi o'rin bo'yicha 10 % qizil / 10 % sariq; `ml.api.get_color_thresholds()`.
3. Evidence fusion: queue / loan'da `base_color`, `priority`, `escalations[]`; navbat `priority` bo'yicha tartiblanadi;
   policy sabablari `weight: null`, `source: "fusion_policy"`; `evidence.kind` ga `photo` qo'shildi.
4. `get_queue` / `get_loan` ixtiyoriy `sandbox_id` qabul qiladi (backend cookie'dagi sandbox'ni uzatadi).
5. `reset_sandbox(None)` = hamma sandbox'lar (ml v2 da amalga oshirilgan); backend faqat shuni chaqiradi.
6. `score_scenario`, `get_scenario_presets`, `ScenarioError` (→ 400) — ml'da tayyor, backend HTTP endpointi keyingi qadam.
7. Istisnolar jadvali (yuqorida).

## Open points
- Photo asset match (CLIP) is not wired yet; `PHOTO_ASSET_MISMATCH` stays inactive until it is.
- Backend: expose `/api/score_scenario`, `/api/scenario_presets` and a colour legend from `get_color_thresholds()`.
