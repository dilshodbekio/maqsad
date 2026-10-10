# Cost per check

All CPU times were **measured** by `python -m ml.eval.measure_cost` on the machine below (process CPU seconds, warm, 100 calls per component); prices are public prices with their source and date. Numbers are for the prototype code on synthetic data. The cost basis is the **single-thread** measurement (OpenCV, torch and OpenMP pinned to one thread = one vCPU): with all cores the process CPU time is inflated by spinning worker threads (see section 1b) and does not mean work done.

Machine: Intel64 Family 6 Model 154 Stepping 3, GenuineIntel, 12 logical cores, Python 3.13.15. Resident memory after warm-up: 1179 MB.

## 1. Measured CPU time (single thread = one vCPU; the cost basis)

| Component | CPU time per call | Wall time per call |
|---|---|---|
| Money-trail score (`score_loan`: model inference + reasons) | 3 ms | 3 ms |
| Scenario score (`score_scenario`) | 96 ms | 98 ms |
| `check_photo` total (decode, hash, duplicates, face detect + blur, CLIP asset check, re-encode; 1600 px) | 502 ms | 520 ms |
| &nbsp;&nbsp;of which decode + perceptual hashes | 31 ms | 32 ms |
| &nbsp;&nbsp;of which face detection + blur | 243 ms | 246 ms |
| &nbsp;&nbsp;of which CLIP asset check (embedding + probe) | 200 ms | 201 ms |
| Warm-up once at start (`ml.api.warm_up()`: demo portfolio + features + scores + queue, face detector, CLIP) | 15 s | 16 s |

### 1b. The same on all cores of this machine (latency, not cost)

| Component | CPU time (all threads) | Wall time |
|---|---|---|
| Money-trail score (`score_loan`: model inference + reasons) | 62 ms | 7 ms |
| Scenario score (`score_scenario`) | 987 ms | 152 ms |
| `check_photo` total (decode, hash, duplicates, face detect + blur, CLIP asset check, re-encode; 1600 px) | 3017 ms | 403 ms |
| Warm-up once | 44 s | 13 s |

Warm-up breakdown (wall): portfolio 6.02 s, face detector 0.03 s, CLIP 9.68 s. Warm-up is paid once per process, not per check.

## 2. Server prices used (public, read 2026-10-10)

- **Railway** (usage-based): CPU $20 per vCPU per month ($0.000463 per vCPU-minute), memory $10 per GB per month ($0.000231 per GB-minute). Source: https://docs.railway.com/reference/pricing/plans (read 2026-10-10). usage-based; the plan fee (Hobby $5 / Pro $20 per month) includes usage up to the same amount (stated on the same page).
- **Dedicated CPU server, on-prem-like option**: Hetzner AX42 (AMD Ryzen 7 PRO 8700GE, 8 cores / 16 threads, 64 GB DDR5 ECC). Hardware: https://www.hetzner.com/dedicated-rootserver/ax42/. Price: **68.19 EUR per month** (source: https://www.deskmodder.de/blog/?p=213645, read 2026-10-10) - **not confirmed on the provider's page**: the hardware is stated on Hetzner's page, but the price fields were empty when the page was read; 68.19 EUR/month (excl. VAT, effective 2026-04-01) comes from a third-party news report and must be checked on hetzner.com.

## 3. Cost per 1 000 checks (variable CPU cost)

| Check | Railway (CPU only), USD per 1 000 | Dedicated server at 100% core utilisation, EUR per 1 000 |
|---|---|---|
| Money-trail score | 0.00002 | 0.00001 |
| Scenario score | 0.00074 | 0.00031 |
| Photo check (incl. CLIP) | 0.00387 | 0.00165 |

Formulas: Railway = CPU seconds x 1 000 x $0.000463 / 60. Dedicated = CPU seconds x 1 000 / 3 600 x (68.19 EUR / 720 h / 8 cores) = CPU seconds x 1 000 x 3.288e-06 EUR per core-second.

## 4. Always-on cost (not per check)

- Railway memory for the resident model + portfolio: 1.15 GB x $10 = **$11.51 per month** (memory is billed while the service runs, whatever the number of checks).
- Dedicated server: **68.19 EUR per month** flat (see the caveat above).

## 5. LLM report (`/draft_report`)

- Prototype: an external LLM is called only in `/draft_report`; the cost per report is **`LLM_COST_USD_PER_REPORT` = (parameter)** - take it from the backend's logged `cost_usd` (Oybek logs it); it is not estimated here.
- Production design: a local LLM (vLLM / Ollama) on the bank's own server, so cost = server time: **`LLM_SECONDS_PER_REPORT` = (parameter, to be measured with the chosen model)** x the core-second price above (Railway $7.717e-06 per vCPU-second; dedicated 3.288e-06 EUR per core-second; a GPU server needs its own price).

## 6. Assumptions (for the ROI model)

- Taxmin: one check = one call of the component in the table of section 1 on a warm process; the measured CPU time on this machine is representative of the server (a cloud vCPU may be slower or faster).
- Taxmin: the photo check uses a 1 600 px JPEG (the largest size the service accepts after downscaling); smaller photos cost less.
- Taxmin: Railway bills CPU for the time it is used; we count only the measured CPU seconds per check and the always-on memory separately, we do not count idle CPU, egress or storage.
- Taxmin: the dedicated-server column assumes 100% core utilisation; at utilisation u the cost per check is divided by u (for example at 30% it is 3.3 times higher).
- Taxmin: one month = 30 days = 720 hours.
- Taxmin: exchange rates and VAT are not applied; prices are quoted in their original currency.
- Taxmin: the money-trail features are computed once for the whole portfolio in a batch (warm-up); the per-check cost excludes it. A new loan needs only its own feature row.
- Taxmin: no cost for the bank officer's time, development, hosting of the dashboard or the audit log is included.

