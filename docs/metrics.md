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
