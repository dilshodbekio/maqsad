# Metrics

Every number below is produced by a script in `ml/eval/` or `ml/money_trail/` (named next to it). All numbers are on synthetic data.

<!-- money-trail:start -->
## Money-trail model (synthetic data)

**Numbers are on synthetic data; real accuracy is measured in the pilot on historical loans.** Positives = all fraud loans of the generator (11-13% of loans, an assumption), including 28 T3 loans whose resale has not happened yet at the snapshot (they look honest by construction). Photo / geo signals are not used by this model and are evaluated separately.

- Model `money_trail_v1`: LightGBM, 28 money-trail features, trained on the train split (seed 42, 5000 loans, 550 fraud). Test split: seed 1337, 5000 loans, 650 fraud (13.0%).
- Test numbers: produced by `python -m ml.eval.eval_money_trail`; CV numbers and thresholds: `python -m ml.money_trail.train` (stored in `ml/artifacts/money_trail_v1.json`). 95% CIs: bootstrap over test loans (1000 resamples, seed 0).

### Cross-validation on the train split (5-fold, grouped by supplier)

| Metric | Out-of-fold value | Random |
|---|---|---|
| ROC-AUC | 0.902 | 0.500 |
| PR-AUC | 0.718 | 0.110 |
| recall@20% | 0.780 | 0.200 |

### Test split: recall@20% inspection budget

| Method | recall@20% [95% CI] |
|---|---|
| Money-trail model (LightGBM) | 0.783 [0.754, 0.816] |
| Graded rule: cash_share_d1 | 0.288 [0.265, 0.313] |
| Best single feature (chosen on train): income_count | 0.411 [0.385, 0.440] |
| 0/1 rule: cash_share_d1 >= 0.5 | 0.250 [0.236, 0.264] |
| Random selection | 0.200 (expected value) |

The 0/1 rule flags only 1.4% of loans (precision 70.0%, recall 7.5% when flagged loans only are counted); the rest of its 20% budget is filled at random, which is why its recall@20% is not 'rule only'. The graded rule ranks all loans by the same cash_share_d1 and is the fairer baseline.

### Test split: ranking quality of the model

| Metric | Value [95% CI] | Random |
|---|---|---|
| ROC-AUC | 0.914 [0.901, 0.927] | 0.500 |
| PR-AUC | 0.756 [0.728, 0.786] | 0.130 |

### Test split: recall@20% per typology

| Typology | n | Money-trail model (LightGBM) | Graded rule: cash_share_d1 | Best single feature (chosen on train): income_count |
|---|---|---|---|---|
| T1 | 195 | 98.5% | 54.4% | 59.8% |
| T2 | 195 | 85.6% | 19.3% | 41.0% |
| T3 | 130 | 60.8% | 15.9% | 33.5% |
| T4 | 130 | 54.6% | 17.3% | 20.9% |

Model, harder subgroups (share of the group inside the 20% budget):

| Group | n | in budget |
|---|---|---|
| T1 plain | 129 | 99.2% |
| T1 stealth | 66 | 97.0% |
| T2 plain | 160 | 88.1% |
| T2 stealth | 35 | 74.3% |
| T4 plain | 102 | 61.8% |
| T4 stealth | 28 | 28.6% |
| T3 resale visible at snapshot | 103 | 75.7% |
| T3 resale not yet happened | 27 | 3.7% |

### Test split: honest loans inside the 20% budget (false-positive rate)

| Honest profile | n | Money-trail model (LightGBM) | Graded rule: cash_share_d1 | Best single feature (chosen on train): income_count |
|---|---|---|---|---|
| normal | 3446 | 7.7% | 16.3% | 17.9% |
| cash_heavy | 259 | 6.6% | 56.4% | 4.2% |
| livestock_seller | 259 | 19.7% | 15.2% | 11.4% |
| oked_mismatch | 386 | 40.7% | 17.1% | 19.3% |
| all honest | 4350 | 11.3% | 18.7% | 16.8% |

### Colours (budget logic)

Thresholds come from the train OOF score quantiles: red = score >= 0.5030 (top 10%), yellow = score >= 0.1674 (next 10%), green = rest. On the test split the model is applied with these fixed thresholds:

| Colour | loans | share | fraud rate inside |
|---|---|---|---|
| red | 589 | 11.8% | 71.0% |
| yellow | 723 | 14.5% | 19.4% |
| green | 3688 | 73.8% | 2.5% |

Red + yellow together are 26.2% of the test loans and contain 85.8% of all fraud loans. That is above the nominal 20% budget because the thresholds are fixed on train while the test split has more fraud (13% vs 11%) and more stealth cases; the exact share of each cause was not measured.

### Latency and size

| Item | Value |
|---|---|
| `score_loan` latency (model inference + reasons, warm, 500 single calls), p50 / p95 | 6.64 ms / 7.76 ms |
| same, wall clock including call overhead, p50 / p95 | 6.68 ms / 7.80 ms |
| First call: demo portfolio generated + features + scores (once, in memory) | 3.3 s |
| Model artifact `money_trail_v1.txt` | 512 KB |
| Cost per check | 0 (local model, no external API) |

Measured on: Intel64 Family 6 Model 154 Stepping 3, GenuineIntel, Python 3.13.15, single thread. `score_loan` uses the cached feature row of the loan; building features for the whole portfolio is a one-time batch step.
<!-- money-trail:end -->

<!-- photo:start -->
## Photo module (openly licensed images)

**All numbers are on a small library of openly licensed photos (CC / public domain) and are produced by `python -m ml.eval.eval_photo`. They say nothing about real borrower photos; real accuracy is measured in the pilot.**

- Library: 100 asset photos (Wikimedia Commons; `ml/assets/photos/manifest.csv`) + 11 photos with people for the face-blur test; 4726 KB in the repo.
- Portfolio photo index (`ml/artifacts/photo_index_v1.json`): 90 images on 101 loans of the demo portfolio; 10 images are shared by 2-3 loans (the generator's photo-reuse groups). Photos are matched to the loan's purpose.

| Purpose | library images | in index | loans with a photo |
|---|---|---|---|
| tikuv_dastgohi | 20 | 18 | 20 |
| chorva | 20 | 18 | 20 |
| issiqxona | 20 | 18 | 20 |
| parranda | 20 | 18 | 21 |
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
| `check_photo` p50 / p95 (500 calls, warm, 1600x1119 px, 164 KB JPEG; decode, hashing, duplicate search, face detection + blur, re-encode) | 312 ms / 379 ms |
| same, wall clock | 313 ms / 380 ms |
| Cost per check | 0 (local, no external API) |

### Known failure cases

- **Heavy crops:** crop 20% is found only 20.0% of the time, crop 10% 64.0% (random crop position). Hashing is not crop-proof.
- **Different photo of the same thing** (the same cow from another angle, the same machine in another room) is not a duplicate by hash; recognising an individual animal or machine is out of scope.
- **Photo of a screen** is only covered by the simulated 'screenshot' transform (border + colour shift); moire from a real screen photo and perspective distortion were not tested.
- **Larger rotation / mirroring** were not tested (only 5 degrees).
- **Faces:** a decorated bull head was detected as a face once (`faces_test_04`); a dark, out-of-focus face in the background (`faces_test_08`) is not counted and was not detected, so it is not blurred. Side and tiny faces are the weak spot of any detector; the 'faces left' numbers above only show that *detectable* faces are gone.
- **Small library:** hard negatives come from ~10 images per purpose and half; with a real portfolio of thousands of photos the false-positive rate at this threshold must be re-measured.
- The photo result does not change the money-trail risk; combining them is a later step.
<!-- photo:end -->
