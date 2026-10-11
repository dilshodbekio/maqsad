# ml/generator - synthetic portfolio generator

    python -m ml.generator.build --split all        # train (42), dev (1337), holdout (2026), demo (7); 5000 loans each
    python -m ml.generator.check_leakage --split all
    python -m ml.generator.sample                   # data/sample/sample_loan.json for the backend

Output: `data/generated/{train,dev,holdout,demo}/*.parquet` + `meta.json` (gitignored).
Tables: `loans`, `transactions` (borrower and supplier accounts; supplier rows have `loan_id = null`),
`suppliers`, `employees`, `photo_manifest`. `labels.parquet` is evaluation-only
(never a model feature). Parameter shifts between train and the other splits: `config.py` (TRAIN vs DEV; HOLDOUT and DEMO only change seed and ID space).

All numbers here are ASSUMPTIONS about a synthetic world (fraud rate 11% train / 13% dev, holdout, demo,
typology mix 30/30/20/20, OKED codes illustrative); none come from real bank data.
Photos: only `photo_id` + manifest for now (`image_file` / `license` filled when real images are assigned).

## Conventions
- **Evaluation positives:** primary metrics (recall@20% etc.) count ALL fraud as positive, including T3 loans whose
  resale has not happened yet at the snapshot (`labels.fraud_visible == False`). Always additionally report how many
  positives were not visible at the snapshot (`meta.json: fraud_not_yet_visible`; the leakage report prints it too).
- **Photo features out of the money-trail model:** `photo_far_km`, `photo_dup_count` (and any `photo_*` / geo column)
  are inputs of `ml/photo` and are combined with the money-trail score later, in the final risk score. The
  money-trail model must not use them. `check_leakage.py` includes them only to check they are not trivially separable.
- **Labels:** `labels.parquet` is read only by `ml/eval/` and `check_leakage.py`; never a feature.
- **T4 stealth:** ~25% of T4 loans use a supplier whose OKED matches the purpose; for them only weak secondary
  signals (recent supplier, invoice-payment gap, higher supplier cash-out) and the photo remain.
- **Headline metric = money-trail score only.** Photo / geo signals (`photo_*`, coordinates) are evaluated separately and
  combined with the money-trail score later; any combined number depends on our assumed photo rates (`config.py`:
  `photo_reuse_*`, `photo_far_*`) and must say so. `check_leakage.py` reports them in their own section
  ("photo module inputs - not used by money-trail model").
- **Photo reuse:** only `photo_id` is reused. `photo_captured_at` and `photo_lat/lon` are always the loan's own
  (capture after disbursement; near the borrower, or far with probability `photo_far_*`).
- **Ledgers never go negative:** `flows.cap_outflows` replays each account in time order and caps every outflow at the
  balance (borrower accounts first; supplier accounts are then built from the settled supplier payments, shown as
  `income` rows, plus their own outflows). Tested in `ml/tests/test_generator.py`.
- **Short-window honest suppliers:** `honest_short_window_share` (0.18) of honest suppliers with >= 2 borrowers are only
  active for 20-120 days (new suppliers, or seasonal for parranda / issiqxona), so the T2 ring burst is not a free signal
  (`sup_tx_ts_span_days` T2 AUC ~0.84, intentionally below the 0.90 flag).
- **Leakage check:** `check_leakage.py` covers (A) engineered money-trail features, (B) all raw columns of the four tables
  plus derived ones, (C) a depth-2 tree on B (supplier-grouped CV), (D) photo module inputs. It flags AUC >= 0.90
  except the intended T1 features (fast cash-out, "no supplier payment" = `sup_pay_share` / `invoice_gap` /
  `share_supplier_payment`, first-outflow timing). Re-run it after every generator change.
- Project-level documentation (method, assumptions, licenses): `data/README.md`.

## Known limitations
- **Counterparty ID collisions:** person / business / external counterparty IDs are random numbers drawn independently,
  so the same ID can appear on unrelated loans (and a few across splits). Any counterparty-graph feature would
  pick up these false shared links; do not build one without first switching to counter-based unique IDs.
- **Ring size:** rings have ~4 + Poisson(3) borrowers (max ~12); the real case in `docs/brief.md` §2 had 71 borrowers.
  Large rings are not modelled.
