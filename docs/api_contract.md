# API contract (Dilshodbek ↔ Oybek) — v1.5

Source: `docs/brief.md` §5. Any change requires agreement from both sides (via PR).
During development the backend may fall back to a visible STUB; **in the demo only the real model runs**.
Code: HTTP schemas in `backend/app/schemas.py` (1:1 with this file), Python interface in `ml/api.py`.

**Demo portfolio.** `ml/` serves the **demo** portfolio (generator split `demo`, seed 7), generated in memory on the first call
(~6-7 s including imports, measured). The backend warms it at start-up by calling `get_queue()` once in a background thread, so
`/api/health` answers immediately (`warmup: "running"` → `"done"`).
**ML warm-up (v1.5): the recommended start-up call is `ml.api.warm_up()`**, not `get_queue()` alone: it loads the demo portfolio, the face
detector **and CLIP** (measured on the development machine, see `docs/metrics.md` section 4: ~16 s and ~1.4 GB in total) and returns
`{portfolio_s, face_detector_s, clip_s, total_s, clip_loaded, clip_error}`. If only `get_queue()` is called, CLIP is not preloaded and the
first photo upload pays for loading it (~11 s measured). If CLIP cannot be loaded (`clip_loaded: false`, reason in `clip_error`), the service keeps
working: `check_photo` returns `asset_match: null` and no asset escalation is raised (never retried, nothing is downloaded at request time).
Build-time weight download and the `MAQSAD_*` environment variables: `ml/README.md`.

## HTTP (all endpoints under `/api`)

```
GET  /api/health
  → {ml_mode: "real|stub", model_version, warmup: "pending|running|done|failed",
     functions: {get_queue|get_loan|score_loan|check_photo|reset_sandbox|get_color_thresholds:
                 "real|stub|not_ready|missing"}}

GET  /api/queue?limit=50&offset=0&color=&region=&purpose=
                                      (view of the caller's sandbox: sandbox_id comes from the cookie)
  → {items: [{loan_id, borrower_name_masked, purpose, amount_uzs, region, risk, color, top_reason,
              base_color, priority, escalations[]}],
     total,                            items matching the filters (before limit/offset)
     counts: {red, yellow, green}}     the whole queue of the sandbox, BEFORE filters
     items in ml order: priority descending (priority = colour rank + risk); the backend never re-ranks.
     limit 1..200 (default 50), offset ≥ 0; color ∈ red|yellow|green; region / purpose: exact match,
     empty = no filter, unknown value = empty page (200); bad limit/offset/color → 422.

GET  /api/queue/facets                → {regions: [], purposes: []}   distinct values in the queue, sorted
GET  /api/queue/legend                → ml.api.get_color_thresholds() 1:1:
     {red, yellow, n_red, n_yellow, n_loans, rule, n_escalated_to_red, n_escalated_to_yellow,
      n_red_total, n_yellow_total, note}   (default view; `rule` / `note` are English, the UI writes its own
      Uzbek text from the numbers; not connected → 503)

GET  /api/loan/{loan_id}              (view of the caller's sandbox)
  → {loan, transactions[], photos[], invoice?,
     risk, color, base_color, priority, escalations[],
     reasons:[{code, text_uz, weight, evidence_refs[], source?}], evidence[], model_version}

POST /api/score_loan        {loan_id}
  → {risk: 0..1, color: "green|yellow|red", reasons:[{code, text_uz, weight}], model_version, latency_ms}

POST /api/check_photo       multipart: image (file) + loan_id, lat?, lon?, captured_at?
                            (fallback JSON: {loan_id, image_base64, lat?, lon?, captured_at?})
  → {asset_match: {label, score, matches_purpose, label_uz} | null, duplicate_of: [{loan_id, similarity}],
     geo_ok: bool|null, time_ok: bool|null, faces_blurred: bool, faces_count?: int, latency_ms}

POST /api/draft_report      {loan_id}
  → {text_uz, facts:[{claim, evidence_ref, unverified: bool}], model, latency_ms, cost_usd|null}

POST /api/decision          {loan_id, officer_id, action: "approve|reject|visit", comment}
  → {audit_id}

GET  /api/audit?loan_id=...
  → [{ts, actor, action, ai_recommendation, model_version, inputs_hash}]
                                      (view of the caller's sandbox; the hash chain itself is global)

GET  /api/session                     → {officer_id}   demo: "Xodim-XXXX", stable per sandbox cookie
                                      (hash of sandbox_id, never the id itself); production: from the bank's SSO

POST /api/admin/reset       header X-Admin-Token  → {ok: true}
GET  /api/photo/{photo_id}  → blurred JPEG of a borrower upload (owner sandbox only)
GET  /api/asset/{path}      → portfolio photo from ml/assets/ (photos[].url)

-- scenario (v1.6): hypothetical loan, scored live, never stored --
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
| `photos[]` | ml: `{photo_id, path, purpose, captured_at, lat, lon, duplicate_of[], source: "portfolio"}` (no `asset_match` here: the asset result of a portfolio photo is in-sample, the probe was trained on these images; it only shows up as `evidence[].data.asset_match` when it escalates) (only for loans that have a photo; `path` repo-relative, under `ml/assets/`); backend adds `url`. Borrower uploads of the caller's sandbox: `{photo_id, url, captured_at (server time), client_captured_at, lat, lon, source: "borrower_upload", check}` |
| `invoice?` | `{supplier_name, supplier_oked, oked_matches_purpose, invoice_amount_uzs}` |
| `reasons[]` | top 3 money-trail reasons `{code, text_uz, weight, evidence_refs[]}` (`weight` = share of the loan's risk-raising push, 0..1) + policy reasons `{code, text_uz, weight: null, evidence_refs[], source: "fusion_policy"}`; plain Uzbek, never accusatory; every `evidence_refs` value is an `evidence[].ref`. `score_loan` reasons have no `evidence_refs` |
| `escalations[]` | `{code, min_color, text_uz, evidence_refs[], raises_color}` (see "Evidence fusion") |
| `evidence[]` | `{ref, kind, summary_uz, data}`: `txn` (`ref` = txn id; `data` = `{txn_id, ts, type, amount_uzs, counterparty_id}`), `invoice` (`invoice:<loan_id>`; `data` = invoice), `graph` (`graph:<supplier_id>`; `data` = `{supplier_name, in_degree, cash_share, exit_share, active_span_days, employee_link}`), `photo` (`photo:<photo_id>` or `photo:sandbox:<loan_id>:<n>`; `data` = `{photo, uploaded, duplicate_of, geo_ok, distance_km, time_ok, asset_match}`). `ref` is what `/draft_report` cites as `evidence_ref` |

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
| `PHOTO_ASSET_MISMATCH` | yellow | `asset_match.matches_purpose == false`: the photo does not show the asset the loan was for (CLIP + linear probe; v1.5: active) |

- `priority` = colour rank (green 0, yellow 1, red 2) + `risk`: the queue sort key.
- One escalation per code; if several photos trigger the same code, their texts are joined and all evidence refs are kept.
- Reason order (`get_loan().reasons`, and `top_reason` == `reasons[0].text_uz`, or "Sezilarli xavf belgisi aniqlanmadi"):
  policy reasons that **raised** the colour, then money-trail reasons, then policy reasons that did not raise it.
- Asset rule (one rule for `label`, `matches_purpose` and the reason text): `matches_purpose = (label == the loan's purpose) and
  probability >= tau` (tau chosen in cross-validation, `ml/artifacts/asset_probe_v1.json`). A different thing found → "Rasmda kredit
  maqsadiga mos aktiv aniqlanmadi (aniqlangan: <label_uz>)"; the right class with low confidence → "Rasmdagi aktiv ishonchli aniqlanmadi
  (eng ehtimoliy: <label_uz>, ishonch past)"; the text never says "no matching asset" when `label` equals the purpose.
- Sandbox view: uploads made with `check_photo(..., sandbox_id=...)` escalate only that sandbox's `get_queue` / `get_loan`;
  other sandboxes and the default view are unaffected; `reset_sandbox` removes the effect.

### Errors
Every error → `{detail: "<Uzbek message>"}`:
`400` bad file/input (incl. `InvalidImageError`, `InvalidRequestError`, `ScenarioError`) · `404` unknown `loan_id` · `413` file too large (> 8 MB) ·
`422` validation · `429` rate limit · `502` ML/LLM failure ·
`503` function not connected yet (`NotImplementedError`: "Bu funksiya hali ulanmagan") or LLM not configured.

## Python interface (backend imports `ml/` directly, only via `backend/app/ml_adapter.py`)

| Endpoint | Backend calls | Owner |
|---|---|---|
| `GET /api/queue`, `GET /api/queue/facets` | `ml.api.get_queue(sandbox_id)` (paging / filters / counts in the backend) | ml |
| `GET /api/loan/{id}` | `ml.api.get_loan(loan_id, sandbox_id)` | ml |
| `POST /api/score_loan` | `ml.api.score_loan(loan_id)` | ml |
| `POST /api/check_photo` | `ml.api.check_photo(loan_id, image_bytes, lat, lon, captured_at, sandbox_id)` → also returns internal `blurred_image: bytes` (stored by backend, never sent over HTTP) | ml |
| `POST /api/admin/reset` | `ml.api.reset_sandbox(None)` (`None` = every sandbox) | ml |
| service start-up (warm-up thread) | `ml.api.warm_up()` → `{portfolio_s, face_detector_s, clip_s, total_s, clip_loaded, clip_error}` (v1.5; replaces `get_queue()` alone) | ml |
| `GET /api/queue/legend` | `ml.api.get_color_thresholds()` | ml |
| `POST /api/score_scenario` | `ml.api.score_scenario(scenario)` | ml |
| `GET /api/scenario_presets` | `ml.api.get_scenario_presets()` | ml |
| `POST /api/draft_report` | backend (LLM) | backend |
| `POST /api/decision`, `GET /api/audit` | backend (audit log) | backend |

### Exceptions

| `ml` raises | Meaning | HTTP |
|---|---|---|
| `ml.api.LoanNotFoundError` (KeyError) | unknown or invalid `loan_id` (any type) | 404 |
| `ml.api.InvalidImageError` (ValueError) | empty / not an image / corrupted image | 400 |
| `ml.api.InvalidImageError` "image larger than …" | > 10 MB (the backend already rejects > 8 MB) | 413 |
| `ml.api.InvalidRequestError` (ValueError) | malformed `sandbox_id` (not a string, empty, > 64 characters, or characters outside `[A-Za-z0-9_.:-]`); message in Uzbek | 400 |
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
- `asset_match` = `{label, score, matches_purpose, label_uz}` from OpenCLIP ViT-B-32 (CPU) + a linear probe: `label` is the most likely of the 5
  loan purposes or `other`, `score` its probability, `matches_purpose` per the asset rule above, `label_uz` an Uzbek name of what was found (for
  non-assets one of a few everyday categories, e.g. "avtomobil"). It is `null` only if the CLIP stack / weights are unavailable (the failure is
  cached and logged once, nothing is downloaded at request time, no `PHOTO_ASSET_MISMATCH` is raised). The asset check is a flag for the officer,
  not proof (measured limits: `docs/metrics.md`). `geo_ok` / `time_ok` are `null` when the input is missing.
- `sandbox_id` is validated (`InvalidRequestError` → 400, see the exceptions table); the backend's sandbox ids are well-formed by construction.
- The photo result does **not** change `risk`; it can only raise colour / priority through the evidence-fusion policy.

### Scenario scoring (ml; HTTP endpoints since v1.6)

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

### ML deployment (CLIP weights and environment)

Details and the Dockerfile snippet: [`ml/README.md`](../ml/README.md). Short version: pre-download the CLIP weights **at build time**
(`python -m ml.photo.asset --download`); at runtime the service never downloads. Environment: `MAQSAD_CLIP_CACHE` (weights directory, same at
build and runtime), `HF_HUB_OFFLINE=1`, `MAQSAD_ASSET_CHECK=off` (switch the asset check off), `MAQSAD_TORCH_THREADS` / `OMP_NUM_THREADS` /
`MKL_NUM_THREADS` (thread pinning; `1` matches the cost model in `docs/cost.md`).

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

1. Demo portfel (`demo` split, seed 7), warm-up ~6-7 s (o'lchangan); backend uni fonda isitadi.
2. Rang qoidasi: navbatdagi o'rin bo'yicha 10 % qizil / 10 % sariq; `ml.api.get_color_thresholds()`.
3. Evidence fusion: queue / loan'da `base_color`, `priority`, `escalations[]`; navbat `priority` bo'yicha tartiblanadi;
   policy sabablari `weight: null`, `source: "fusion_policy"`; `evidence.kind` ga `photo` qo'shildi.
4. `get_queue` / `get_loan` ixtiyoriy `sandbox_id` qabul qiladi (backend cookie'dagi sandbox'ni uzatadi).
5. `reset_sandbox(None)` = hamma sandbox'lar (ml v2 da amalga oshirilgan); backend faqat shuni chaqiradi.
6. `score_scenario`, `get_scenario_presets`, `ScenarioError` (→ 400) — ml'da tayyor, backend HTTP endpointi keyingi qadam.
7. Istisnolar jadvali (yuqorida).

## v1.3 o'zgarishlar (xodim paneli navbati)

1. `GET /api/queue` endi sahifalangan obyekt qaytaradi: `{items, total, counts:{red, yellow, green}}` (avval ro'yxat edi).
   Ixtiyoriy query: `limit` (1..200, default 50), `offset` (≥ 0), `color`, `region`, `purpose`.
   Tartib — ml'dagi `priority` bo'yicha, backend o'zgartirmaydi. `counts` — filtrlardan **oldingi**, sandbox navbatining
   to'liq soni; `total` — filtrdan keyingi soni (sahifalash uchun).
2. `GET /api/queue/facets` → `{regions, purposes}` (navbatdagi qiymatlar, alifbo tartibida) — filtrlar uchun.
3. `GET /api/queue/legend` → `ml.api.get_color_thresholds()` 1:1 (rang legendasi). Yangi ml funksiyasi kerak emas.
4. `/api/health` `functions` ichiga `get_color_thresholds` qo'shildi.
5. ml tomonda o'zgarish yo'q: `get_queue(sandbox_id)` va `get_color_thresholds()` avvalgidek chaqiriladi.

## v1.4 o'zgarishlar (kredit sahifasi)

1. `GET /api/session` → `{officer_id}` qo'shildi (faqat backend, ml'da o'zgarish yo'q). Demo'da `/decision` uchun
   `officer_id` shu yerdan olinadi; sandbox cookie `httponly` bo'lgani uchun frontend uni o'qiy olmaydi.
2. `/api/audit` — chaqiruvchi sandbox'ining ko'rinishi (avval ham shunday ishlagan, endi yozib qo'yildi); hash-zanjir
   barcha sandbox'lar uchun bitta (global).
3. `/api/queue`, `/api/loan` shakli o'zgarmadi.

## v1.5 o'zgarishlar (CLIP aktiv tekshiruvi, ml)

1. `check_photo().asset_match` = `{label, score, matches_purpose, label_uz}` (avval `null` edi); `PHOTO_ASSET_MISMATCH` faol. Qoida:
   `matches_purpose = (label == kredit maqsadi) and p >= tau`; sabab matni shu qoidaga mos (yuqorida).
2. `ml.api.warm_up()` qo'shildi: portfel + yuz detektori + CLIP; `get_queue()` o'rniga shuni chaqirish tavsiya etiladi. CLIP yuklanmasa
   (`clip_loaded: false`) servis ishlayveradi: `asset_match: null`, eskalatsiya yo'q, so'rov vaqtida hech narsa yuklab olinmaydi.
3. `ml.api.InvalidRequestError` (noto'g'ri `sandbox_id`) → 400.
4. `evidence[kind=photo].data.asset_match` qo'shildi. `photos[]` shakli o'zgarmadi (portfel rasmlarining `asset_match`i in-sample bo'lgani uchun HTTP shaklida yo'q).
5. CLIP vaznlari build vaqtida oldindan yuklanadi, `MAQSAD_*` o'zgaruvchilari va Dockerfile: `ml/README.md`.
6. HTTP qatlami (v1.4) o'zgarishsiz: `/api` prefiksi, `{items, total, counts}`, `{detail}` xato shakli, health, admin, session.

## v1.6 o'zgarishlar (ssenariy sinash, backend + frontend)

1. `GET /api/scenario_presets` va `POST /api/score_scenario` backend'da ochildi (`ml.api.get_scenario_presets` / `score_scenario`),
   shakllar yuqoridagi "Scenario scoring" bo'limidagidek, ml tomonda o'zgarish yo'q. Ssenariy hech qayerga yozilmaydi (audit ham, portfel ham).
2. Backend ml bilan bir xil kirish qoidalarini Pydantic'da (`extra=forbid`, qat'iy `int`) tekshiradi: noma'lum kalit yo'q,
   `snapshot_day` 2..364, `invoice_amount_uzs` ixtiyoriy, bitta summa <= kredit x 10, kirimlar yig'indisi <= kredit x 20,
   `day` <= `snapshot_day`, <= 60 tranzaksiya. Noto'g'ri kirish **400** (422 emas) va o'zbekcha `{detail}`; `ScenarioError` / `InvalidRequestError`
   ham 400. Hudud nomi, mavjud `supplier_id` va balans tekshiruvi ml'da qoladi (400).
3. `/api/health` `functions` ichiga `score_scenario`, `get_scenario_presets` qo'shildi. Stub rejimda (`ml_stub` da ssenariy modeli yo'q)
   ikkalasi `"missing"` va 503 qaytaradi.
4. Frontend: `/scenario` sahifasi, navbatda "Ssenariy sinash" havolasi. `.github/workflows/ci.yml` smoke testi yangilandi.

## v1.7 o'zgarishlar (dalolatnoma keshi va narx, backend + frontend)

1. `POST /api/draft_report` so'rovi: `regenerate: bool = false` qo'shildi (`true` = keshni chetlab o'tish, "Qayta yaratish").
2. Javobga qo'shildi: `cached: bool`, `input_tokens`, `output_tokens` (`int|null`). Kesh: bir sandbox + kredit uchun bitta qoralama,
   kalit = dalillar + sabablar + model + prompt versiyasi (dalil o'zgarsa, masalan rasm yuklansa, kesh o'z-o'zidan yaroqsiz bo'ladi).
   Keshdan javobda `cached: true`, `cost_usd: 0.0`, tokenlar `null`, `latency_ms` = keshdan o'qish vaqti. Xato javob keshga yozilmaydi.
   Admin reset keshni ham tozalaydi.
3. Env: `LLM_REASONING_EFFORT` (default `minimal`, bo'sh = yuborilmaydi), narx `LLM_PRICE_IN_PER_1M` / `LLM_PRICE_OUT_PER_1M`
   (eski `..._PER_MTOK` nomlari ham o'qiladi). ml tomonda o'zgarish yo'q.

## v1.8 o'zgarishlar (dalolatnoma navbati, umumiy kesh, pre-warm; backend + frontend)

Maqsad: bir necha hakam bir vaqtda dalolatnoma so'rasa, hech kim xato ko'rmasin (bepul Gemini kaliti cheklangan). ml tomonda o'zgarish yo'q.

1. `POST /api/draft_report` endi ikki xil javob beradi:
   - **200** `DraftReport` — keshda bor (darhol; `cached: true`);
   - **202** `{job_id, status: "queued"|"running", position, eta_s}` — navbatga qo'yildi.
   `DraftReport` ga `generated_at` (UTC ISO) qo'shildi. Xatolar: 503 (`LLM ulanmagan` yoki "Dalolatnoma xizmati hozir band, birozdan keyin
   qayta urinib ko'ring." — navbatda > 30 ta kutayotgan ish, yoki provayder kvotasi tugagan), 429 (`regenerate` daqiqada 1 marta, sandbox bo'yicha).
2. `GET /api/draft_report/job/{job_id}` → `{job_id, status: queued|running|done|error, position, eta_s, result?, error?}`.
   `position`: 1 = keyingi; ishlayotganda va tugaganda 0 (ishlayotgan chaqiruv kutayotganlar uchun 1 o'rin hisoblanadi). `eta_s` — taxmin
   (oxirgi chaqiruvlarning o'rtacha davomiyligi va daqiqalik oyna bo'yicha), noma'lum bo'lsa `null`. Job faqat o'z sandbox'iga ko'rinadi, boshqasiga
   va noma'lum id'ga **404** (mijoz qayta yuboradi: kesh yoki yangi job javob beradi). `status: "error"` — `error` o'zbekcha matn.
   Job 120 s dan oshsa `error` = "band" xabari (task o'zi davom etib, keshni to'ldiradi). Joblar xotirada: restartdan keyin yo'qoladi.
3. Navbat: barcha LLM chaqiruvlari bitta jarayon ichidagi navbatdan o'tadi; bir vaqtda bitta chaqiruv, har 60 s oynada `LLM_RPM` tadan ko'p emas
   (default 4; sirpanuvchi oyna, token bucket'dan qat'iyroq). Bir xil kalit (kredit + dalillar + model + prompt) = bitta task, har so'rovchi o'z `job_id`'sini oladi.
   Provayderdan 429 kelsa task navbat boshida qoladi, navbat provayder aytgan vaqtgacha (kamida 5 s, ko'pi bilan 30 s gacha o'sib) to'xtaydi;
   provayder bir job umridan uzoqroq kutishni so'rasa (masalan kunlik kvota tugagan: "retry in 1h49m") kutayotgan joblar darhol "band" xabari bilan tugaydi va
   yangilari shu vaqtgacha 503 oladi.
4. Kesh `DATA_DIR` dagi SQLite'da (restart va redeploydan keyin saqlanadi). Kalit: dalillar + sabablar + model + prompt versiyasi. Kreditga shu sandbox'ning
   qo'shimcha dalili (yuklangan rasm / eskalatsiya) bo'lmasa — **umumiy** kesh (hamma ko'radi); bo'lsa — shu sandbox'niki. `regenerate: true` keshni
   chetlab o'tadi va natija keshni yangilaydi.
5. `POST /api/admin/reset` sandbox keshini tozalaydi, umumiy keshni faqat `?clear_shared_cache=true` bilan.
6. `POST /api/admin/prewarm?n=40` (header `X-Admin-Token`, 202) → `{total, done, skipped, failed, running}`: asosiy navbatning eng yuqori `n` ta krediti
   (`priority` bo'yicha: qizil + eskalatsiyali) uchun umumiy keshga qoralama yaratadi, o'sha navbat orqali; keshda borini o'tkazib yuboradi. Foydalanuvchi
   so'rovlari pre-warm'dan oldin turadi (ishlayotgan chaqiruv to'xtatilmaydi). `GET /api/admin/prewarm` — holat. Token'siz yoki noto'g'ri token → 403.
7. Env: `LLM_RPM` (default 4). `llm.py`: 429 endi `LLMRateLimited` (Retry-After yoki "retry in ..." matni bo'yicha kutish vaqti bilan).

## Open points
- Backend: call `ml.api.warm_up()` at start-up and show `clip_loaded` / `clip_error` in `/api/health`.
- CLIP weights licence for production needs a legal review (see "Known failure cases" in `docs/metrics.md`).
