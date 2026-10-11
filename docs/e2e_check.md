# End-to-end check of the live demo

Live site: https://maqsad-production-d704.up.railway.app. Run 2026-10-10 23:27:49 UTC - 2026-10-10 23:28:35 UTC, model_version `money_trail_v2`. Scripts: `qa/e2e_live.py` (HTTP checks, writes `qa/e2e_results.json`), `qa/e2e_screens.py` (Playwright / Chromium screenshots), `qa/e2e_render.py` (this file). No app / ML code was changed. No `/api/admin/*` call was made. Everything that wrote data used `officer_id = e2e-test` or a throw-away sandbox (list at the end).

**58 of 61 HTTP checks passed; the 3 failures are explained under the table (1 is the missing evidence_refs of an absence reason, 2 are English fragments in 422 messages).**

## 1. Results

| # | Check | Result | Evidence | Latency |
|---|---|---|---|---|
| 1 Health | GET /api/health: ml_mode real, warmup done, clip_loaded true, all functions real | PASS | {"ml_mode": "real", "model_version": "money_trail_v2", "warmup": "done", "functions": {"get_queue": "real", "get_loan": "real", "score_loan": "real", "check_photo": "real", "reset_sandbox": "real", "get_color_thresholds": "real", "score_scenario": "real", "get_scenario_presets": "real"}, "clip_loaded": true} | 319 ms |
| 1 Health | headers X-ML-Mode / X-Model-Version match body | PASS | X-ML-Mode=real X-Model-Version=money_trail_v2 |  |
| 2 Queue | page 1: 50 items, total 5000, counts sum to total | PASS | total 5000 counts {'red': 513, 'yellow': 501, 'green': 3986} | 191 ms |
| 2 Queue | page 2: no overlap with page 1, priority non-increasing across pages | PASS | p1 last 2.991 p2 first 2.991 | 175 ms |
| 2 Queue | top_reason present, Uzbek, no None/nan/{/English (100 items) | PASS | e.g. 'Kredit berilganidan beri tushum qayd etilmagan' |  |
| 2 Queue | colour >= base_color, colours valid | PASS | counts {'red': 513, 'yellow': 501, 'green': 3986} |  |
| 2 Queue | filter region=Andijon | PASS | total 370 | 180 ms |
| 2 Queue | filter purpose=tikuv_dastgohi | PASS | total 1022 | 186 ms |
| 2 Queue | unknown region -> 200 empty page | PASS | total 0 |  |
| 2 Queue | bad color -> 422 | PASS | HTTP 422 {'detail': "So'rov ma'lumotlari noto'g'ri — color: Input should be 'green', 'yellow' or 'red'"} |  |
| 2 Queue | legend: 500 red / 500 yellow budget + escalation counts | PASS | n_red 500 n_yellow 500 escalated_to_red 13 |  |
| 2 Loan | GET /api/loan/L1504044 (red): colour = queue, top_reason == reasons[0], Uzbek reasons | PASS | 3 reasons, 5 evidence, 8 tx, invoice=yes, photos=0; reason0='Kredit berilganidan beri tushum qayd etilmagan' | 110 ms |
| 2 Loan | red loan: every reason has evidence_refs that resolve to evidence[] | **FAIL** | refs ['T150016477', 'invoice:L1504044', 'T150015906', 'T150015921'] evidence keys ['ref', 'kind', 'summary_uz', 'data'] |  |
| 2 Loan | GET /api/loan/L1503806 (yellow): colour = queue, top_reason == reasons[0], Uzbek reasons | PASS | 3 reasons, 3 evidence, 8 tx, invoice=yes, photos=0; reason0="Kredit davrida oyiga o'rtacha 0.1 ta tushum qayd etilgan" | 108 ms |
| 2 Loan | GET /api/loan/L1504711 (green): colour = queue, top_reason == reasons[0], Uzbek reasons | PASS | 3 reasons, 1 evidence, 2 tx, invoice=yes, photos=0; reason0='Yirik kirim qayd etilmagan' | 110 ms |
| 2 Loan | photos[] of a portfolio loan (L1500951): asset route returns 200 image/jpeg | PASS | /api/asset/photos/tikuv_dastgohi_02.jpg -> 200 image/jpeg 27314 B | 154 ms |
| 2 Loan | unknown loan id -> 404 with Uzbek detail | PASS | HTTP 404 detail='Kredit topilmadi: L9999999' | 104 ms |
| 3 Photo | A: other image (phones) for green L1504130: asset_match not null, matches_purpose false, label_uz present | PASS | {"label": "other", "score": 0.8649, "matches_purpose": false, "label_uz": "telefon yoki noutbuk"} server 127.06 ms | 917 ms |
| 3 Photo | A: L1504130 colour in sandbox A >= yellow with PHOTO_ASSET_MISMATCH | PASS | colour yellow escalations ['PHOTO_ASSET_MISMATCH']; text: ['Rasmda kredit maqsadiga mos aktiv aniqlanmadi (aniqlangan: telefon yoki noutbuk)'] |  |
| 3 Photo | B (new cookie): L1504130 is still green, no escalation (isolation) | PASS | colour green escalations [] |  |
| 3 Photo | A queue counts differ from B (A has the escalation) | PASS | A {'red': 513, 'yellow': 502, 'green': 3985} B {'red': 513, 'yellow': 501, 'green': 3986} |  |
| 3 Photo | A: sewing machine NOT in the index (tikuv_dastgohi_16.jpg) for green L1503050: matches_purpose true, no escalation | PASS | {"label": "tikuv_dastgohi", "score": 0.77, "matches_purpose": true, "label_uz": "tikuv mashinasi"} duplicate_of [] colour green server 134.89 ms | 350 ms |
| 3 Photo | A: portfolio image of ANOTHER loan (tikuv_dastgohi_15.jpg = L1503448) for L1500270: duplicate_of lists it, colour red, PHOTO_DUPLICATE | PASS | duplicate_of [{'loan_id': 'L1503448', 'similarity': 1.0}] colour red codes ['PHOTO_DUPLICATE'] server 124.07 ms | 336 ms |
| 3 Photo | B: L1500270 still green | PASS | colour green |  |
| 3 Photo | uploaded (blurred) photo: owner gets 200 image, other sandbox does not | PASS | owner 200, other 404 |  |
| 3 Errors | empty file -> 4xx Uzbek detail | PASS | HTTP 400 detail="Rasm fayli bo'sh." |  |
| 3 Errors | .txt renamed to .jpg -> 4xx | PASS | HTTP 400 detail='Faqat JPEG, PNG yoki WebP rasm qabul qilinadi.' |  |
| 3 Errors | 9 MB file (limit 8 MB) -> 413 | PASS | HTTP 413 detail='Rasm hajmi 8 MB dan oshmasligi kerak.' |  |
| 3 Errors | unknown loan id -> 404 | PASS | HTTP 404 detail='Kredit topilmadi: L0000000' |  |
| 3 Errors | empty loan id -> 4xx | **FAIL** | HTTP 422 detail="So'rov ma'lumotlari noto'g'ri — loan_id: Field required" |  |
| 3 Errors | no image part -> 4xx | PASS | HTTP 400 detail="So'rov multipart/form-data (asosiy) yoki application/json bo'lishi kerak." |  |
| 3 Errors | invalid lat/lon -> 4xx | PASS | HTTP 422 detail="So'rov ma'lumotlari noto'g'ri — lat: Input should be less than or equal to 90; lon: Input should be a valid number, unable to parse string as a number" |  |
| 3 Errors | invalid captured_at -> 4xx | **FAIL** | HTTP 422 detail="So'rov ma'lumotlari noto'g'ri — captured_at: Input should be a valid datetime or date, invalid character in year" |  |
| 3 Errors | malformed sandbox cookie -> no 500 (server replaces it with a fresh sandbox) | PASS | HTTP 200 {"asset_match":{"label":"other","score":0.8649,"matches_purpose":false,"label_uz":"telefon yoki nout |  |
| 3 Photo | 5 parallel uploads from 5 sandboxes (other_00..04 for the same loan): all 200, each sees only its own escalation, no cross-duplicates | PASS | other_00.jpg: HTTP 200, server 131.05 ms, client 660 ms, esc ['PHOTO_ASSET_MISMATCH']; other_01.jpg: HTTP 200, server 214.36 ms, client 897 ms, esc ['PHOTO_ASSET_MISMATCH']; other_02.jpg: HTTP 200, server 202.88 ms, client 759 ms, esc ['PHOTO_ASSET_MISMATCH']; other_03.jpg: HTTP 200, server 329.09 ms, client 992 ms, esc ['PHOTO_ASSET_MISMATCH']; other_04.jpg: HTTP 200, server 206.5 ms, client 1290 ms, esc ['PHOTO_ASS | 1290 ms |
| 4 Scenario | presets load (4) | PASS | ['honest_purchase', 'fast_cash_out', 'supplier_ring', 'asset_resale'] | 644 ms |
| 4 Scenario | score preset honest_purchase | PASS | risk 0.014 colour green percentile 33.9 reason0 "Ta'minotchiga kredit summasining 98 foizi o'tkazilgan" | 251 ms |
| 4 Scenario | score preset fast_cash_out | PASS | risk 0.993 colour red percentile 99.3 reason0 "Kredit tushgandan keyingi 24 soat ichida mablag'ning 84 foizi naqd yechilgan" | 255 ms |
| 4 Scenario | score preset supplier_ring | PASS | risk 0.481 colour yellow percentile 86.2 reason0 "Ta'minotchi olgan mablag'ning 69 foizini naqd yechgan" | 274 ms |
| 4 Scenario | score preset asset_resale | PASS | risk 0.506 colour yellow percentile 86.8 reason0 'Kredit berilganidan beri tushum qayd etilmagan' | 250 ms |
| 4 Scenario | cash withdrawal day 30 -> lower risk than day 1 | PASS | day1: risk 0.993, colour red, percentile 99.3, first reason: Kredit tushgandan keyingi 24 soat ichida mablag'ning 84 foizi naqd yechilgan // day30: risk 0.598, colour yellow, percentile 88.8, first reason: Hisob-fakturadagi summaning 90 foizi ta'minotchiga to'lanmagan |  |
| 4 Scenario | all paid to supplier -> lower risk than cash-out | PASS | paid: risk 0.130, colour green, percentile 72.2, first reason: Yirik kirim qayd etilmagan |  |
| 4 Scenario | invalid: negative amount -> 400/422 with Uzbek message | PASS | HTTP 400 detail="Ssenariy noto'g'ri: amount_uzs: qiymat ruxsat etilgan oralig'dan tashqarida." |  |
| 4 Scenario | invalid: unknown tx type -> 400/422 with Uzbek message | PASS | HTTP 400 detail="Ssenariy noto'g'ri: transactions.0.type: ruxsat etilgan qiymatlardan biri bo'lishi kerak ('cash_withdrawal', 'supplier_payment', 'p2p_out', 'p2p_in', 'external_transfer' or 'income')." |  |
| 4 Scenario | invalid: snapshot_day 1000 -> 400/422 with Uzbek message | PASS | HTTP 400 detail="Ssenariy noto'g'ri: snapshot_day: qiymat ruxsat etilgan oralig'dan tashqarida." |  |
| 4 Scenario | invalid: snapshot_day 1 -> 400/422 with Uzbek message | PASS | HTTP 400 detail="Ssenariy noto'g'ri: snapshot_day: qiymat ruxsat etilgan oralig'dan tashqarida." |  |
| 4 Scenario | invalid: unknown extra field -> 400/422 with Uzbek message | PASS | HTTP 400 detail="Ssenariy noto'g'ri: hacker: noma'lum kalit." |  |
| 4 Scenario | invalid: unknown purpose -> 400/422 with Uzbek message | PASS | HTTP 400 detail="Ssenariy noto'g'ri: purpose: ruxsat etilgan qiymatlardan biri bo'lishi kerak ('tikuv_dastgohi', 'chorva', 'issiqxona', 'parranda' or 'asalarichilik')." |  |
| 4 Scenario | invalid: negative tx amount -> 400/422 with Uzbek message | PASS | HTTP 400 detail="Ssenariy noto'g'ri: transactions.0.amount_uzs: qiymat ruxsat etilgan oralig'dan tashqarida." |  |
| 4 Scenario | invalid: tx day beyond snapshot -> 400/422 with Uzbek message | PASS | HTTP 400 detail="Ssenariy noto'g'ri: transactions.0.day: qiymat ruxsat etilgan oralig'dan tashqarida." |  |
| 5 Decision | POST /api/decision on L1504044 (officer e2e-test, visit) | PASS | HTTP 200 {'audit_id': 73} | 1072 ms |
| 5 Decision | GET /api/audit shows the row with ai_recommendation and model_version | PASS | {"ts": "2026-10-10T23:28:29.150+00:00", "actor": "e2e-test", "action": "visit", "ai_recommendation": "red (risk=0.998)", "model_version": "money_trail_v2", "inputs_hash": "5bbb67cf6c77487116b4bf26ba2ad4f2839ae1f6b53051953e7b7b921012edd9"} | 221 ms |
| 5 Decision | invalid action -> 4xx with Uzbek message | PASS | HTTP 422 "So'rov ma'lumotlari noto'g'ri — action: Input should be 'approve', 'reject' or 'visit'" |  |
| 5 Report | POST /api/draft_report on L1504044: text_uz + facts each with evidence_ref (or flagged unverified) | PASS | HTTP 200 cached=True model=gemini-3.6-flash facts=5 unverified=0 latency_ms=10.0 cost_usd=0.0 | 231 ms |
| 5 Report | second call is cached: same text, faster, cost 0 | PASS | call1 231 ms cost 0.0 cached True; call2 225 ms cost 0.0 cached True same_text=True | 225 ms |
| 6 Frontend | GET /: 200 HTML referencing built JS (JS loads) | PASS | HTTP 200 js=['/assets/index-BzyvEJOF.js'] | 217 ms |
| 6 Frontend | GET /loan/L1504044: 200 HTML referencing built JS (JS loads) | PASS | HTTP 200 js=['/assets/index-BzyvEJOF.js'] | 216 ms |
| 6 Frontend | GET /capture?loan=L1504130: 200 HTML referencing built JS (JS loads) | PASS | HTTP 200 js=['/assets/index-BzyvEJOF.js'] | 215 ms |
| 6 Frontend | GET /scenario: 200 HTML referencing built JS (JS loads) | PASS | HTTP 200 js=['/assets/index-BzyvEJOF.js'] | 215 ms |
| 6 Frontend | GET /capture: 200 HTML referencing built JS (JS loads) | PASS | HTTP 200 js=['/assets/index-BzyvEJOF.js'] | 222 ms |
| 6 Frontend | GET /nonexistent-page: 200 HTML referencing built JS (JS loads) | PASS | HTTP 200 js=['/assets/index-BzyvEJOF.js'] | 217 ms |

Notes on the failures: `red loan: every reason has evidence_refs ...` failed because the **first reason of L1504044 has an empty `evidence_refs`** (every ref that exists does resolve: `T...` ids are transactions in `transactions[]`, `invoice:` is the invoice; this is bug 3). `empty loan id` and `invalid captured_at` returned 422 as required, but with English pydantic fragments in `detail` (bug 2).

## 2. Photo latency (three sequential uploads, fresh sandbox)

| Image | server `latency_ms` | client round trip |
|---|---|---|
| other_06.jpg | 118.61 ms | 534 ms |
| tikuv_dastgohi_16.jpg | 124.08 ms | 238 ms |
| chorva_09.jpg | 122.77 ms | 235 ms |

Other uploads above: server 124-214 ms, client 570-940 ms (the first request after idle is the slowest).

## 3. Scenario comparison (fast cash-out preset)

- **cash-out day 1**: risk 0.993, colour red, percentile 99.3, first reason: Kredit tushgandan keyingi 24 soat ichida mablag'ning 84 foizi naqd yechilgan
- **cash-out day 30**: risk 0.598, colour yellow, percentile 88.8, first reason: Hisob-fakturadagi summaning 90 foizi ta'minotchiga to'lanmagan
- **all paid to supplier**: risk 0.130, colour green, percentile 72.2, first reason: Yirik kirim qayd etilmagan

## 4. Frontend (headless Chromium, screenshots in `docs/e2e_screens/`)

| Page | Viewport | Console errors | HTTP >= 400 | Horizontal scroll | 'None/nan/undefined' in text |
|---|---|---|---|---|---|
| / | mobile | 0 | 0 | no | 0 |
| /loan/L1504044 | mobile | 0 | 0 | no | 0 |
| /capture?loan=L1504130 | mobile | 0 | 0 | no | 0 |
| /scenario | mobile | 0 | 0 | no | 0 |
| / | desktop | 0 | 0 | no | 0 |
| /loan/L1504044 | desktop | 0 | 0 | no | 0 |
| /capture?loan=L1504130 | desktop | 0 | 0 | no | 0 |
| /scenario | desktop | 0 | 0 | no | 0 |

Screens: queue, loan_red, capture, scenario x mobile (390x844) / desktop (1366x850); e.g. `docs/e2e_screens/loan_red_mobile.png`. Unknown routes (`/nonexistent-page`) return the SPA shell (200).

## 5. Bugs

| Severity | Owner | Bug | Suggested fix |
|---|---|---|---|
| MAJOR | frontend | No demo / QR page exists: the frontend routes are only `/`, `/loan/{id}`, `/capture?loan={id}`, `/scenario` (see `frontend/src/App.tsx`). If the plan is a QR code for judges, nothing generates it; judges would have to type `/capture?loan=ID`. | Add a `/demo` page that lists the demo loan ids with a QR code each pointing to `/capture?loan=ID` (or print the QR codes into the deck). |
| MINOR | backend | Request-validation errors are Uzbek only in the prefix: `So'rov ma'lumotlari noto'g'ri - loan_id: Field required`, `captured_at: Input should be a valid datetime ...` (English pydantic text reaches the user). Same for the scenario enum message (`... 'parranda' or 'asalarichilik')`) and technical field names (`snapshot_day`, `amount_uzs`). | Map pydantic error types (`missing`, `datetime_parsing`, `literal_error`) to Uzbek sentences and field names to Uzbek labels in the 400/422 handler. |
| MINOR | ml | The top reason of the #1 red loan (`Kredit berilganidan beri tushum qayd etilmagan`, L1504044) has `evidence_refs: []` (an absence-of-income reason has no event to cite); the UI says 'Alohida dalil havolasi yo'q'. A judge may read this as an unsupported claim. Also green loans show risk-raising wording as `top_reason` (e.g. 'Yirik kirim qayd etilmagan' on a green loan). | Cite the loan window as evidence (e.g. a `window:` ref) for absence reasons; in the UI label green-loan reasons as the factors that raised the score most rather than as findings. |
| MINOR | ml/backend | The draft report could only be checked from the shared cache (both calls `cached: true`, cost 0, ~230 ms client time). A fresh generation (latency, cost_usd, rate-limit behaviour) was NOT tested to avoid LLM cost / quota. | Before the demo, run one `regenerate: true` on a loan that is not pre-warmed and record latency and cost. |

No BLOCKER was found: queue, loan pages, photo flow (asset mismatch / match / duplicate / isolation / errors / parallel), scenario, decision + audit, draft report (cached) and all frontend pages worked on the live site.

## 6. Demo helpers

Green loans without photo escalation **in a fresh sandbox**, not in the portfolio photo index (a first upload of a new sewing-machine photo does not hit a duplicate).

- `tikuv_dastgohi` green: `L1504130`, `L1503050`, `L1500270`
- `chorva` green: `L1503497`, `L1500884`
- red loan for the live walkthrough: `L1504044` (rank 1 of the queue, 3 reasons with evidence, invoice block, transactions; reasons: Kredit berilganidan beri tushum qayd etilmagan; Hisob-fakturadagi summaning 94 foizi ta'minotchiga to'lanmagan; Kredit tushgandan keyin 7 kun ichida mablag'ning 67 foizi naqd yechilgan)

Photo test images: `ml/assets/photos/tikuv_dastgohi_16.jpg` (sewing machine, not in the index -> matches, no escalation), `other_05.jpg` (phones -> `PHOTO_ASSET_MISMATCH`, yellow), `tikuv_dastgohi_15.jpg` (in the index -> `PHOTO_DUPLICATE`, red).

## 7. What this run created on the live service

- Audit rows: id 73 (loan `L1504044`, actor `e2e-test`, action `visit`, comment `e2e test`; visible only in that sandbox's view). The audit log is a hash chain, so the row cannot be removed.
- 10 throw-away sandboxes (random UUID cookies; sandbox photos live in server memory only); ids in `qa/e2e_results.json`. Not reset (no admin call).
- Draft report: no new LLM generation (both calls were served from the cache).
