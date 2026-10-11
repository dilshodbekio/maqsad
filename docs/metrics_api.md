# Application (backend) latency and LLM report cost

Measured numbers only. Reproduce the latency table with `python backend/scripts/measure_api.py` (optional `--base`, `--n`, `--only`).

## Method
- Service: https://maqsad-production-d704.up.railway.app (model `money_trail_v2`, `ml_mode` real).
- Measured on 2026-10-10 21:12 UTC (`measured_at_utc` in the script's JSON).
- Client: Python `httpx`, sequential requests, from a developer laptop. **Network time is included**: these are not pure server processing times.
- n = 20 requests per endpoint. Percentiles use nearest-rank on the sorted sample; with n = 20, p95 is the 19th-slowest request.
- Between `check_photo` requests there is a 3.2 s pause (the app's limit is 20 requests per minute).

## Results (n = 20, errors = 0)

| Endpoint | p50, ms | p95, ms | min–max, ms |
|---|---|---|---|
| `GET /api/queue?limit=50` | 292.2 | 550.8 | 283.3–569.6 |
| `GET /api/loan/{id}` (cycling over 50 loans) | 232.1 | 241.4 | 223.1–241.6 |
| `POST /api/check_photo` (`ml/assets/photos/asalarichilik_00.jpg`, 38 KB) | 362.9 | 375.3 | 352.7–576.9 |

The server-side `latency_ms` in the `check_photo` response: p50 124.1 ms, p95 136.1 ms (model processing time; the rest is network and upload).

## Draft report (`/api/draft_report`)

Two sources, both from the production service with the Gemini model `gemini-3.6-flash` and `reasoning_effort` = `minimal`. Both samples are small (n = 3 each), so they are listed as they are; **no p50 / p95 is claimed**.

### (a) Production Railway logs, 2026-10-11, n = 3 LLM calls

One line per call in the service log (`LLM call loan=... model=... latency_ms=... in=... out=... cost_usd=...`, written by `backend/app/llm.py`).

| Call | latency, ms | input tokens | output tokens | cost_usd |
|---|---|---|---|---|
| 1 | 5950 | 835 | 498 | 0.002494 |
| 2 | 3859 | 1219 | 722 | 0.003622 |
| 3 | 4375 | 1635 | 789 | 0.004185 |

`cost_usd` is computed from the tokens and the **official published price** of the model (`LLM_PRICE_IN_PER_1M`, `LLM_PRICE_OUT_PER_1M`). The API key is on the free tier, so the **actual charge was $0**; the figures show what the same calls would cost on a paid plan. Output tokens include thinking tokens (Gemini).

### (b) `qa/e2e_draft_live.json`: live three-language check (run committed 2026-10-11 04:31 UTC)

Produced by `qa/e2e_draft_live.py`: for one red loan (`L1504044`) and each language, a new sandbox asks for a fresh draft (`regenerate: true`), then asks again (cache). `latency_ms` is the value in the API response; `client_ms` is measured by the script and includes the queue wait, the polling interval (3 s) and the network.

| Language | fresh `latency_ms` | fresh `client_ms` | input / output tokens | `cost_usd` | facts (unverified) | cached `latency_ms` | cached `client_ms` |
|---|---|---|---|---|---|---|---|
| uz | 3497.2 | 6979 | 1351 / 682 | 0.003571 | 6 (0) | 14.7 | 200 |
| en | 8499.8 | 10436 | 1187 / 737 | 0.003654 | 6 (0) | 25.5 | 234 |
| ru | 3599.0 | 6596 | 1239 / 740 | 0.003704 | 6 (0) | 10.6 | 121 |

Cache hits report `cached: true`, `cost_usd: 0.0` and no tokens (no LLM call was made). The same free-tier note as in (a) applies to `cost_usd`.

### Local trial of the reasoning setting (not production; 1 draft per cell, 3 loans: red / yellow / green; not p50 / p95)

| `reasoning_effort` | latency, s (red / yellow / green) | unverified facts |
|---|---|---|
| not sent (the model's own "thinking") | 11.5 / 7.8 / 17.0 | 0 / 0 / 1 |
| `none` | 3.7 / 5.8 / 2.5 | 0 / 0 / 1 |
| `minimal` | 3.6 / (quota error) / (quota error) | 0 / – / – |

With `minimal`, the yellow and green loans could not be measured because of the Gemini quota (429). Quality: the red loan's text matched the evidence numbers (5 of 5 facts passed the fact check). The conclusion rests on 3 samples and is not statistical.

## Limit: Gemini quota
The key in the local `.env` is on the free tier: Google's 429 answer named `generate_content_free_tier_requests` with a limit of 5 requests per minute (`gemini-3.6-flash`). This was not checked for the production key. Uncached measurements need a pause of about 13 s between calls. The application queues LLM calls to stay under the limit (`LLM_RPM`, `backend/app/draft_queue.py`); see `docs/known_issues_app.md`.
