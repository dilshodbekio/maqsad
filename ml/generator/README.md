# ml/generator - synthetic portfolio generator

    python -m ml.generator.build --split all        # train (seed 42) + test (seed 1337), 5000 loans each
    python -m ml.generator.check_leakage --split all
    python -m ml.generator.sample                   # data/sample/sample_loan.json for the backend

Output: `data/generated/{train,test}/*.parquet` + `meta.json` (gitignored).
Tables: `loans`, `transactions` (borrower and supplier accounts; supplier rows have `loan_id = null`),
`suppliers`, `employees`, `photo_manifest`. `labels.parquet` is evaluation-only
(never a model feature). Parameter shifts between train and test: `config.py` (TRAIN vs TEST).

All numbers here are ASSUMPTIONS about a synthetic world (fraud rate 11% train / 13% test,
typology mix 30/30/20/20, OKED codes illustrative); none come from real bank data.
Photos: only `photo_id` + manifest for now (`image_file` / `license` filled when real images are assigned).
Documentation for `data/README.md` is pending team agreement (shared file).

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
- Project-level documentation (method, assumptions, licenses): `data/README.MD`.
