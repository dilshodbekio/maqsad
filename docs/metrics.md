# Metrics

**All numbers are on synthetic data; real accuracy is measured in a pilot on historical loans.** Every number is produced by a script (named next to it) and stored in `ml/artifacts/eval_*.json`; `python -m ml.eval.build_metrics` only formats them. Fraud rates, typology mix and all generator parameters are assumptions (`data/README.md`).

## 1. Data & splits

| Split | Role | Seed | Snapshot | Loans | Fraud rate (assumed) |
|---|---|---|---|---|---|
| train | training and cross-validation only | 42 | 2026-03-31 | 5000 | 11% |
| dev | development: looked at while building features, reasons and the v1 -> v2 fixes, so NOT an unbiased test set; kept for comparison with v1 | 1337 | 2026-07-31 | 5000 | 13% |
| holdout | evaluated **exactly once**, after the model was frozen; the headline numbers | 2026 | 2026-07-31 | 5000 | 13% |
| demo | served by `ml.api` to the dashboard and used for the photo index and latency; never used for metrics | 7 | 2026-07-31 | 5000 | 13% |

All splits have their own ID space, hence disjoint loans, borrowers, suppliers, employees, photos and transactions (tested in `ml/tests/test_generator.py`). Dev, holdout and demo share the same shifted generator parameters (more fraud, more stealth cases, slower cash-out, ... see `ml/generator/config.py`); only the seed differs.

| Artifact | sha256 (LF-normalised file bytes) |
|---|---|
| `ml/artifacts/money_trail_v2.txt` | `8b4767b0ff1e1a4284cd792061eab0bfb1272c7e4f5bc336243bcbadf3c4f174` |
| `ml/artifacts/money_trail_v1.txt` (kept for comparison) | `808def7894bb36a922c2073c736aa097df37cdaa8d86cd112d805eb1aefbae38` |

Holdout run: commit `b6afbb7c38dc67712f5f705a45cb20f2fd096474`, 2026-10-10T12:59:14+00:00, model sha256 `8b4767b0ff1e1a4284cd792061eab0bfb1272c7e4f5bc336243bcbadf3c4f174`; evaluated once.

## 2. Money-trail v2 (holdout = headline)

- Model `money_trail_v2`: LightGBM, 29 money-trail features (`ml/money_trail/features.py`; no photo / geo, no labels), trained on the train split only (fixed seed and hyper-parameters, no search). Positives = all fraud loans.
- Cross-validation on train (5-fold, grouped by declared supplier, out-of-fold; `python -m ml.money_trail.train`): ROC-AUC 0.910, PR-AUC 0.726 (random 0.110), recall@20% 0.782 (random 0.200).

### Recall@20% inspection budget

holdout split: 5000 loans, 650 fraud (13.0%), 1070 declared suppliers. 95% CIs: *plain* = bootstrap over loans, *clustered* = bootstrap over suppliers (loans of one supplier resampled together); 1000 resamples each.

| Method | recall@20% | 95% CI plain | 95% CI clustered by supplier |
|---|---|---|---|
| Money-trail model (LightGBM) | 0.782 | [0.750, 0.813] | [0.728, 0.832] |
| Graded rule: cash_share_d1 | 0.274 | [0.249, 0.299] | [0.249, 0.304] |
| Best single feature (chosen on train): income_gap_days | 0.463 | [0.427, 0.499] | [0.419, 0.507] |
| 0/1 rule: cash_share_d1 >= 0.5 | 0.264 | [0.248, 0.280] | [0.247, 0.285] |
| Random selection | 0.200 (expected value) | | |

The 0/1 rule flags only 1.7% of loans (precision 72.1%, recall 9.5% counting flagged loans only); the rest of its 20% budget is filled at random, so its recall@20% is not 'rule only'. The graded rule ranks all loans by the same `cash_share_d1` and is the fairer baseline. The best single feature (`income_gap_days`) was chosen on the train split.

### Ranking quality of the model

| Metric | value | 95% CI plain | 95% CI clustered by supplier | Random |
|---|---|---|---|---|
| ROC-AUC | 0.910 | [0.896, 0.923] | [0.882, 0.933] | 0.500 |
| PR-AUC | 0.745 | [0.713, 0.773] | [0.692, 0.795] | 0.130 |

### Share of each group inside the 20% budget

Fraud groups: recall@20% of the group. Honest groups: false-positive rate (share of the group inside the budget). CI (clustered by supplier) only for groups with n >= 50; smaller groups show counts only. Columns after the model: baselines (point estimates).

| Group | n | model [95% CI] | Graded rule | Best single feature (chosen on train) | 0/1 rule |
|---|---|---|---|---|---|
| T1 | 195 | 99.0% [97.5, 100.0] | 54.2% | 55.9% | 44.5% |
| T2 | 195 | 93.3% [85.1, 99.3] | 17.2% | 44.6% | 18.6% |
| T3 | 130 | 56.9% [47.0, 65.7] | 14.2% | 65.4% | 18.6% |
| T4 | 130 | 45.4% [29.1, 61.7] | 15.6% | 15.4% | 18.6% |
| T1 plain | 133 | 98.5% [96.4, 100.0] | 68.5% | 54.9% | 55.9% |
| T1 stealth | 62 | 100.0% [100.0, 100.0] | 23.7% | 58.1% | 19.9% |
| T2 plain | 161 | 91.9% [82.1, 99.0] | 18.0% | 44.1% | 18.6% |
| T2 stealth | 34 | 100.0% | 13.4% | 47.1% | 18.6% |
| T4 plain | 96 | 58.3% [38.7, 75.3] | 15.4% | 13.5% | 18.6% |
| T4 stealth | 34 | 8.8% | 16.0% | 20.6% | 18.6% |
| T3 resale visible at snapshot | 109 | 67.0% [55.8, 76.5] | 14.0% | 78.0% | 18.6% |
| T3 resale not yet happened | 21 | 4.8% | 15.0% | 0.0% | 18.6% |
| honest/normal | 3410 | 7.8% [6.6, 9.2] | 16.5% | 16.1% | 19.0% |
| honest/cash_heavy | 287 | 7.7% [4.4, 11.5] | 54.0% | 7.7% | 20.0% |
| honest/livestock_seller | 271 | 22.5% [17.1, 27.6] | 14.7% | 27.7% | 18.9% |
| honest/oked_mismatch | 382 | 37.2% [29.0, 45.0] | 16.6% | 13.6% | 19.0% |
| honest/all | 4350 | 11.3% [9.9, 12.6] | 18.9% | 16.1% | 19.0% |

Fraud positives not yet visible at the snapshot (counted as positives, they look honest): T3 21 (total 21).

### Colours by queue rank

Rule: red = top 10% of the portfolio by risk, yellow = next 10%, green = rest (red + yellow = 20% inspection budget). On this portfolio: red = risk >= 0.6939 (500 loans), yellow = risk >= 0.2749 (500 loans).

| Colour | loans | fraud rate inside |
|---|---|---|
| red | 500 | 73.6% |
| yellow | 500 | 28.0% |
| green | 4000 | 3.5% |

Red + yellow (the 20% budget) contain 78.2% of all fraud loans.

### Latency and size

Measured by `python -m ml.eval.eval_money_trail` `--latency` on the demo portfolio (not used for metrics), 3 runs x 500 warm single `score_loan` calls; range across the runs. Intel64 Family 6 Model 154 Stepping 3, GenuineIntel, Python 3.13.15, single thread.

| Item | Value (range over runs) |
|---|---|
| `score_loan` p50 (model inference + reasons) | 6.6 - 7.2 ms |
| `score_loan` p95 | 7.5 - 8.5 ms |
| same, wall clock p50 / p95 | 6.7 - 7.3 ms / 7.6 - 8.6 ms |
| Cold start: imports + demo portfolio generated, featured and scored once (3 fresh processes) | 6.3 - 7.3 s (of which portfolio build 3.7 - 4.3 s) |
| Model artifact `money_trail_v2.txt` | 513 KB |
| Cost per check | 0 (local model, no external API) |

### Development split (dev) - same tables, for comparison with v1

#### Recall@20% inspection budget

dev split: 5000 loans, 650 fraud (13.0%), 1053 declared suppliers. 95% CIs: *plain* = bootstrap over loans, *clustered* = bootstrap over suppliers (loans of one supplier resampled together); 1000 resamples each.

| Method | recall@20% | 95% CI plain | 95% CI clustered by supplier |
|---|---|---|---|
| Money-trail model (LightGBM) | 0.795 | [0.767, 0.824] | [0.747, 0.842] |
| Graded rule: cash_share_d1 | 0.288 | [0.265, 0.313] | [0.260, 0.318] |
| Best single feature (chosen on train): income_gap_days | 0.425 | [0.389, 0.457] | [0.378, 0.472] |
| 0/1 rule: cash_share_d1 >= 0.5 | 0.250 | [0.236, 0.264] | [0.235, 0.266] |
| Random selection | 0.200 (expected value) | | |

The 0/1 rule flags only 1.4% of loans (precision 70.0%, recall 7.5% counting flagged loans only); the rest of its 20% budget is filled at random, so its recall@20% is not 'rule only'. The graded rule ranks all loans by the same `cash_share_d1` and is the fairer baseline. The best single feature (`income_gap_days`) was chosen on the train split.

#### Ranking quality of the model

| Metric | value | 95% CI plain | 95% CI clustered by supplier | Random |
|---|---|---|---|---|
| ROC-AUC | 0.915 | [0.901, 0.928] | [0.888, 0.937] | 0.500 |
| PR-AUC | 0.769 | [0.743, 0.797] | [0.725, 0.815] | 0.130 |

#### Share of each group inside the 20% budget

Fraud groups: recall@20% of the group. Honest groups: false-positive rate (share of the group inside the budget). CI (clustered by supplier) only for groups with n >= 50; smaller groups show counts only. Columns after the model: baselines (point estimates).

| Group | n | model [95% CI] | Graded rule | Best single feature (chosen on train) | 0/1 rule |
|---|---|---|---|---|---|
| T1 | 195 | 97.9% [95.9, 99.5] | 54.4% | 50.8% | 39.3% |
| T2 | 195 | 89.7% [83.0, 95.4] | 19.3% | 39.0% | 18.9% |
| T3 | 130 | 59.2% [50.9, 68.8] | 15.9% | 61.5% | 18.9% |
| T4 | 130 | 56.9% [39.1, 73.1] | 17.3% | 16.2% | 18.9% |
| T1 plain | 129 | 98.4% [96.2, 100.0] | 69.0% | 43.4% | 49.7% |
| T1 stealth | 66 | 97.0% [92.3, 100.0] | 25.9% | 65.2% | 18.9% |
| T2 plain | 160 | 91.9% [84.9, 98.0] | 18.3% | 38.1% | 18.9% |
| T2 stealth | 35 | 80.0% | 23.8% | 42.9% | 18.9% |
| T4 plain | 102 | 64.7% [44.4, 83.1] | 19.0% | 15.7% | 18.9% |
| T4 stealth | 28 | 28.6% | 11.1% | 17.9% | 18.9% |
| T3 resale visible at snapshot | 103 | 69.9% [61.1, 79.4] | 15.4% | 76.7% | 18.9% |
| T3 resale not yet happened | 27 | 18.5% | 17.7% | 3.7% | 18.9% |
| honest/normal | 3446 | 7.7% [6.3, 9.1] | 16.3% | 16.7% | 19.1% |
| honest/cash_heavy | 259 | 6.9% [4.2, 10.1] | 56.4% | 7.7% | 20.4% |
| honest/livestock_seller | 259 | 17.4% [12.3, 22.6] | 15.2% | 26.3% | 19.8% |
| honest/oked_mismatch | 386 | 40.4% [31.8, 48.4] | 17.1% | 15.5% | 19.5% |
| honest/all | 4350 | 11.1% [9.8, 12.4] | 18.7% | 16.6% | 19.3% |

Fraud positives not yet visible at the snapshot (counted as positives, they look honest): T1 1, T3 27 (total 28).

#### Colours by queue rank

Rule: red = top 10% of the portfolio by risk, yellow = next 10%, green = rest (red + yellow = 20% inspection budget). On this portfolio: red = risk >= 0.6249 (500 loans), yellow = risk >= 0.2521 (500 loans).

| Colour | loans | fraud rate inside |
|---|---|---|
| red | 500 | 78.4% |
| yellow | 500 | 25.0% |
| green | 4000 | 3.3% |

Red + yellow (the 20% budget) contain 79.5% of all fraud loans.

## 3. v1 -> v2 change log

An independent review of v1 found the problems below; each row says what changed. The dev numbers before / after come from `python -m ml.eval.eval_money_trail` `--split dev` (v1 = frozen `features_v1.py` + `money_trail_v1`; v2 = current).

| Review finding | Change in v2 |
|---|---|
| Demo portfolio was the test split: looking at the dashboard meant looking at test loans; 'run once' not checkable | Splits renamed/added: `dev` (old test, seed 1337), `holdout` (seed 2026, evaluated once, run recorded with model sha256 + commit), `demo` (seed 7, served by `ml.api`, photo index). |
| 'No regular income' was the biggest driver of the queue, but it is normal for young loans | Income features are age-aware: `income_rate_per_month`, `income_missing_flag` (only for loans older than 90 days; ASSUMPTION from the generator's income timing, U(20, 90) days), `income_gap_days`, `income_stopped_flag`. |
| Reason texts: wrong facts, 'atigi' with large values, wrong sentinel meaning, duplicate facts, '78% i' | Texts rewritten (`ml/money_trail/reasons.py`); 'atigi' only below low thresholds; explicit sentinels ('no payment received yet', 'no activity yet', 'no invoice'); at most one reason per group of equivalent facts (income / cash share / supplier payment); '78 foizi' forms, values above 100% shown; `sup_in_degree` states what it counts (loans naming the supplier on the invoice). |
| Colour explanation contradicted the measurement (fixed train thresholds drift on a shifted portfolio) | Colours by queue rank on the served portfolio (top 10% red, next 10% yellow); thresholds returned by `ml.api.get_color_thresholds()` and listed above; no causal claim is made. |
| Not-visible positives were mis-counted (28 T3)  | Reported by typology (T3 and T1 differ). |
| Bootstrap ignored supplier clustering; CI too narrow | Supplier-clustered bootstrap next to the plain one; groups with n < 50 get counts only. |
| Latency varied between runs | Reported as a range over 3 runs. |
| Unhashable `loan_id` raised TypeError; `invoice_gap` hid 'no invoice' | `LoanNotFoundError` for any invalid id; `invoice_gap = -1` means no invoice. |
| API contract incomplete | `docs/api_contract.md` updated (sandbox_id, reset_sandbox, faces_count, blurred_image, photos[], evidence[], evidence_refs, colour rule). |

Not changed: reasons do not yet say which direction is risky relative to a typical honest value (review minor 14); `income` rows still include one-off sales to businesses (generator), so the texts say 'tushum', not 'muntazam tushum'.

### Dev: v1 vs v2

| Metric (dev) | v1 | v2 |
|---|---|---|
| recall@20% (model) | 0.783 | 0.795 |
| ROC-AUC | 0.914 | 0.915 |
| PR-AUC | 0.756 | 0.769 |
| recall@20% T1 | 98.5% | 97.9% |
| recall@20% T2 | 85.6% | 89.7% |
| recall@20% T3 | 60.8% | 59.2% |
| recall@20% T4 | 54.6% | 56.9% |
| honest false-positive rate in the budget | 11.3% | 11.1% |

### Ablation by loan age at the snapshot (dev)

Recall = share of the age group's fraud loans inside the 20% budget; FPR = share of its honest loans inside the budget (the budget is the top 20% of the whole portfolio by each model's own score).

| Loan age | loans (fraud / honest) | v1 recall | v2 recall | v1 honest FPR | v2 honest FPR |
|---|---|---|---|---|---|
| < 60 days | 759 (87 / 672) | 67.8% | 69.0% | 21.1% | 20.2% |
| 60-180 days | 1796 (264 / 1532) | 79.9% | 79.5% | 12.1% | 11.6% |
| > 180 days | 2445 (299 / 2146) | 79.9% | 82.6% | 7.6% | 7.9% |

Reading the ablation: the age-aware encoding changes the honest false-positive rate of loans younger than 60 days only from 21.1% to 20.2%; young loans stay clearly harder than old ones. The reason is in the data, not in a missed fix: among loans younger than 60 days 79.9% of the honest and 89.7% of the fraud loans have no income yet (our generator gives fraud typologies a lower income probability and a later start), so 'no income yet' keeps a small genuine likelihood ratio; the v2 encoding (`income_rate_per_month = -1` = income not due yet, never shown as a reason) only removes the age artefact. The generator's income timing is an assumption (`data/README.md`); real borrowers must be re-measured. The first v2 draft encoded 'not due yet' as 0; a dev diagnostic showed that this feature was still the top driver for most flagged young honest loans, so it was changed to -1 **before** the holdout was run (this did not change the dev numbers materially). The honest false-positive rate of young loans is therefore a known limitation.

## 4. Photo module

**All numbers are on a small library of openly licensed photos (CC / public domain) and are produced by `python -m ml.eval.eval_photo`. They say nothing about real borrower photos; real accuracy is measured in the pilot.**

- Library: 100 asset photos (Wikimedia Commons; `ml/assets/photos/manifest.csv`) + 11 photos with people for the face-blur test; 4726 KB in the repo.
- Portfolio photo index (`ml/artifacts/photo_index_v1.json`): 90 images on 100 loans of the demo portfolio; 10 images are shared by 2-3 loans (the generator's photo-reuse groups). Photos are matched to the loan's purpose.

| Purpose | library images | in index | loans with a photo |
|---|---|---|---|
| tikuv_dastgohi | 20 | 18 | 20 |
| chorva | 20 | 18 | 20 |
| issiqxona | 20 | 18 | 20 |
| parranda | 20 | 18 | 20 |
| asalarichilik | 20 | 18 | 20 |

### Duplicate detection under transformations

Hashes: pHash + dHash (64 bit each) of the whole image and of its central 80% crop; distance = hamming/128, for the crop-tolerant variant the minimum over whole/central combinations. Positives = a transformed copy against its own original. Hard negatives = the same transformed copy against a *different* image of the same purpose (e.g. two different sewing machines). The threshold is the largest distance with pooled precision >= 0.999 on the tuning half; numbers below are on the other half (50 images; 50 for tuning).

| Variant | max distance (tuned on half A) | recall (held-out) | precision (held-out) | false positives |
|---|---|---|---|---|
| p | 0.250 | 82.5% | 100.0% | 0 / 3840 |
| pd | 0.297 | 87.8% | 98.3% | 6 / 3840 |
| pd_crop | 0.211 | 84.5% | 100.0% | 0 / 3840 |

**Deployed: `pd_crop`, max distance 0.2109 (similarity >= 0.7891).** The deployed variant and threshold were re-tuned on all 100 images (stricter threshold, more hard negatives) after the held-out numbers were fixed; the table above shows how the tune-on-one-half procedure generalises. With the deployed setting no two single-loan images of the portfolio index match each other. Held-out per-transform numbers for the pd_crop variant (threshold tuned on half A):

| Transform | positives | recall | negatives | false positives | precision |
|---|---|---|---|---|---|
| original | 50 | 100.0% | 480 | 0 | 100.0% |
| crop 10% | 50 | 64.0% | 480 | 0 | 100.0% |
| crop 20% | 50 | 20.0% | 480 | 0 | 100.0% |
| JPEG q=30 | 50 | 100.0% | 480 | 0 | 100.0% |
| resize 50% | 50 | 100.0% | 480 | 0 | 100.0% |
| rotate 5 deg | 50 | 96.0% | 480 | 0 | 100.0% |
| brightness +30% | 50 | 100.0% | 480 | 0 | 100.0% |
| brightness -30% | 50 | 100.0% | 480 | 0 | 100.0% |
| screenshot | 50 | 96.0% | 480 | 0 | 100.0% |

### Face blur (YuNet, OpenCV zoo, MIT; Haar cascade as fallback)

Counted by eye on the 11 `faces_test` images (`ml/assets/photos/faces_test_counts.csv`; faces that are very small, dark or out of focus are noted there and not counted).

| Item | Value |
|---|---|
| Faces present (hand count) | 14 |
| Faces detected by `yunet` | 15 (true 14, false 1, missed 0) |
| Face recall / precision (count based) | 100.0% / 93.3% |
| Detectable faces left after blur (YuNet re-run) | 0 |
| Detectable faces left after blur (independent Haar cascade; it found 6 before blur) | 1 |

Detection is count based (no box annotations), so a miss and a false alarm on the same image can cancel out.

### Latency

| Item | Value |
|---|---|
| `check_photo` p50 / p95 (500 calls, warm, 1600x1119 px, 164 KB JPEG; decode, hashing, duplicate search, face detection + blur, re-encode) | 296 ms / 316 ms |
| same, wall clock | 297 ms / 317 ms |
| Cost per check | 0 (local, no external API) |

### Known failure cases

- **Heavy crops:** crop 20% is found only 20.0% of the time, crop 10% 64.0% (random crop position). Hashing is not crop-proof.
- **Different photo of the same thing** (the same cow from another angle, the same machine in another room) is not a duplicate by hash; recognising an individual animal or machine is out of scope.
- **Photo of a screen** is only covered by the simulated 'screenshot' transform (border + colour shift); moire from a real screen photo and perspective distortion were not tested.
- **Larger rotation / mirroring** were not tested (only 5 degrees).
- **Faces:** a decorated bull head was detected as a face once (`faces_test_04`); a dark, out-of-focus face in the background (`faces_test_08`) is not counted and was not detected, so it is not blurred. Side and tiny faces are the weak spot of any detector; the 'faces left' numbers above only show that *detectable* faces are gone.
- **Small library:** hard negatives come from ~10 images per purpose and half; with a real portfolio of thousands of photos the false-positive rate at this threshold must be re-measured.
- The photo result does not change the money-trail risk; combining them is a later step.

## 5. Known failure cases

- **Weakest fraud groups (holdout):** T3 resale not yet happened 4.8% in budget (n=21); T4 stealth 8.8% in budget (n=34); T4 45.4% in budget (n=130); T3 56.9% in budget (n=130).
- **Honest look-alikes that get flagged most:** honest/oked_mismatch 37.2% (n=382); honest/livestock_seller 22.5% (n=271).
- **Fraud that has not happened yet cannot be seen:** 21 positives (T3 21) look honest at the snapshot by construction; they count against recall.
- **'No income' and young loans:** v2 no longer penalises loans younger than 90 days, but the 90-day limit is an assumption about how our synthetic generator times income; real borrowers (seasonal income, grace periods) must be re-measured.
- **Everything is synthetic:** fraud typologies, rates and the honest-but-suspicious profiles are assumptions; the generator's own structure (e.g. ring bursts, photo-reuse rates the same for all typologies) is what the model learns. See `ml/generator/README.md` (Known limitations: counterparty ID collisions, ring size up to ~12 vs. 71 in a real case).
- **Photo module:** see the failure cases in section 4 (hash crop robustness, screen photos, side faces).
- The photo result does not change the money-trail risk; combining them is a later step.

## 6. Reproduce

```
.venv/Scripts/python -m pip install -r ml/requirements.txt
.venv/Scripts/python -m ml.money_trail.train                      # model v2 (train split only)
.venv/Scripts/python -m ml.eval.eval_money_trail --split dev     # dev tables + v1/v2 comparison + age ablation
.venv/Scripts/python -m ml.eval.eval_money_trail --split holdout # EXACTLY ONCE (refuses to run twice; --force-holdout-rerun only reproduces)
.venv/Scripts/python -m ml.eval.eval_money_trail --latency
.venv/Scripts/python -m ml.eval.eval_photo --write-config        # photo duplicate variant + threshold
.venv/Scripts/python -m ml.photo.index                           # photo index on the demo split
.venv/Scripts/python -m ml.eval.eval_photo
.venv/Scripts/python -m ml.eval.build_metrics                    # regenerate this file from the stored results
.venv/Scripts/python -m pytest ml/tests -q
```
