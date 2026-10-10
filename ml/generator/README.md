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
