# Data, assumptions and licenses

All data in this project is **synthetic** or openly licensed. No real personal or bank data is used.
Code-level detail: [`ml/generator/README.md`](../ml/generator/README.md).

## 1. Synthetic portfolio (`data/generated/`, not committed)

**Generation method.** `python -m ml.generator.build --split all` creates two independent portfolios of 5000 loans
(numpy RNG; `ml/generator/`). Tables: `loans`, `transactions` (borrower and supplier accounts), `suppliers`,
`employees`, `photo_manifest`, plus `labels` (evaluation only, never a feature). Five loan purposes:
tikuv_dastgohi, chorva, issiqxona, parranda, asalarichilik. Every behavioural parameter is sampled from a
distribution; fraud variants deliberately overlap with honest ones. `python -m ml.generator.check_leakage`
measures single-feature AUC per typology and a rule baseline.

**Fraud typologies** (modelled on prosecuted cases, `docs/brief.md` §2):
| Code | Pattern | Real-case basis |
|---|---|---|
| T1 fast cash-out | large share withdrawn as cash on day 0-1, little or no supplier payment (a "stealth" share drips out over 5-10 days) | loan money cashed out instead of spent: `docs/brief.md` §2 case 3 (2.18 bn UZS cashed out for 71 borrowers, [yuz.uz](https://yuz.uz/uz/news/bank-xodimining-kirdikorlari-fosh-etildi)) and case 4 (386.7 mln UZS taken with fake salary data, [yuz.uz](https://yuz.uz/uz/news/mikroqarz-va-imtiyozli-kredit-mablaglarini-ozlashtirgan-shaxslar-aniqlandi)) |
| T2 supplier ring | many borrowers pay one supplier, which cashes out fast; supplier may be linked to a bank employee | employee's relative as supplier for 71 citizens' loans ([yuz.uz](https://yuz.uz/uz/news/bank-xodimining-kirdikorlari-fosh-etildi)) |
| T3 asset resale | supplier paid, later a one-off inflow resembling an asset sale, then cash-out; regular income often stops | 160 head of livestock sold for cash ([spot.uz](https://www.spot.uz/oz/2024/04/29/credit-money/)); sewing machines sold for cash ([yuz.uz](https://yuz.uz/uz/news/mikroqarz-va-imtiyozli-kredit-mablaglarini-ozlashtirgan-shaxslar-aniqlandi)) |
| T4 fake invoice | supplier OKED does not match the purpose (~25% of T4 use a matching-OKED firm; only weak secondary signals remain) | `docs/brief.md` §2 case 1: equipment never delivered, "production started" on fake documents, 8 bn UZS ([yuz.uz](https://yuz.uz/uz/news/10-mlrddan-ziyod-imtiyozli-kredit-mablaglari-talon-toroj-qilindi)) |

**Honest-but-suspicious profiles** (so models cannot simply flag "cash" or "mismatch"): livestock/poultry seller who
legally sells stock; cash-heavy rural business; honest loans with generic-trade or unrelated OKED supplier (6-8% of honest loans);
honest buyers who pay the asset in cash; honest one-off large inflows; cash-heavy honest suppliers.

**Splits and shift.** Train: seed 42, snapshot 2026-03-31. Dev (seed 1337, looked at during development), holdout (seed 2026, evaluated once) and demo (seed 7, served by the app, never used for metrics) share the shifted parameters below: snapshot 2026-07-31, disjoint ID spaces,
different regional mix, and shifted parameters (fraud rate 11% -> 13%, stealth share 25% -> 35%, slower T1 cash-out,
smaller rings, fewer employee links, more honest-suspicious borrowers, loan sizes +10%). See `ml/generator/config.py`.

**Snapshot convention.** Disbursements are spread over ~12 months; each split has one `snapshot_date` and all
transactions after it are dropped. Features may only use data up to the snapshot. Some fraud events (T3 resale) have
not happened yet for recent loans; they are still positives (`labels.fraud_visible` marks them).

**Ledger consistency.** Every account is replayed in time order and each outflow (cash withdrawal, supplier payment,
P2P out, external transfer) is capped at the available balance, so no account (borrower or supplier) ever goes negative.
Supplier accounts show the settled supplier payments as `income` rows and their own cash-outs / transfers.

**Photos.** Currently only `photo_id`s and a manifest (`photo_manifest`); some photos are reused across loans. Only the
image (`photo_id`) is reused: capture time and location always belong to the loan itself (capture is after disbursement,
location is near or - with the configured far-photo probability - far from the borrower). Real images are assigned later
(see Licenses).

**Evaluation convention.** The headline metric is the **money-trail score only** (recall@20% etc.). Photo / geo signals
(`photo_*`, coordinates) are evaluated separately and combined with the money-trail score later in the final risk score;
the money-trail model never uses them. Any number that includes photo signals depends on our assumed photo rates (below)
and must be labelled as such. `ml/generator/check_leakage.py` reports photo/geo features in a separate section.

## 2. Assumptions
Assumption: fraud rate is 11% (train) and 13% (dev, holdout, demo); real rates are unknown and are measured in the pilot.
Assumption: typology mix among fraud is T1 30% / T2 30% / T3 20% / T4 20%.
Assumption: loan amounts are log-normal around illustrative medians per purpose (30-80 mln UZS), sigma 0.35.
Assumption: region centroids are rounded approximations; borrower coordinates = centroid + noise (sd 0.25 deg).
Assumption: OKED codes are illustrative and not checked against the official classifier.
Assumption: honest livestock/poultry sellers are ~5% of loans, cash-heavy rural businesses ~5%, honest OKED mismatch 6-8% of honest loans.
Assumption: 25-35% of T1/T2 are "stealth" (look clean); 25% of T4 use a matching-OKED supplier.
Assumption: ring size is 4 + Poisson(3) (train) / Poisson(2.5) (dev, holdout, demo), stealth rings have 2-3 borrowers; 40% / 33% of ring suppliers link to a bank employee, 2.5% of honest suppliers do. A linked employee works in the branch of one of the supplier's borrowers with probability 80% (rings) / 75% train, 70% dev, holdout, demo (honest suppliers).
Assumption: 35% of fraud loans (both splits) and 2% of honest loans reuse a photo; 25% of fraud and 4% of honest photos are taken far from the borrower. These rates are the same for all typologies and not shifted between the splits.
Assumption: 18% (both splits) of honest suppliers with >= 2 borrowers are active in a short window only (20-120 days): newly registered suppliers (registered 3-45 days before their first payment) and, for parranda / issiqxona, seasonal suppliers (March-August). Ring suppliers burst within 30-150 days (brief §2 case 3), so supplier activity span alone is a strong but not decisive T2 signal (single-feature AUC ~0.84).
Assumption: all supplier-account flows are derived from loan payments only (no other business activity is simulated).
Assumption: all numbers measured on this synthetic data say nothing about real accuracy; real accuracy is measured in the pilot.

## 2b. Evidence fusion and scenario scoring (not data, but assumptions)
Assumption: there is no labelled data for photo signals, so photo evidence enters the queue through a **documented policy** (`ml/fusion.py`), not a learned model:
a photo duplicate escalates a loan to at least red, a photo taken more than 5 km from the borrower (assumed limit) or with an implausible capture time to at least yellow;
escalation never lowers the money-trail colour and never changes the money-trail `risk`. No accuracy is claimed for the policy.
Assumption: scenario scoring (`ml.api.score_scenario`) gives a new supplier no outflows of its own; without an `invoice_amount_uzs` there is no invoice (the training convention for "no invoice" is used); scenarios are limited to loans 2-364 days old (the model's training range).

## 2c. Datasets considered (and why they do not fit)
Maqsad needs, per loan: the **loan purpose**, the **declared supplier / invoice**, and the **money flows after disbursement** (cash-out timing, supplier
payment, what the supplier does next). We looked at the public datasets below; none of them has that combination, so the project uses its own synthetic
generator (section 1). Facts come from the papers / pages linked next to each line (the Kaggle pages themselves are not machine-readable, so we cite the
papers and mirrors that describe them); where a detail could not be verified it is not stated.

| Dataset | What it is (verified) | Why it does not fit |
|---|---|---|
| PaySim ([Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1), [EMSS 2016 paper](https://msc-les.org/proceedings/emss/emss2016/emss2016_249.html)) | A **simulator** of mobile-money transactions built from a sample of one month of real financial logs of a mobile-money service in an African country; the Kaggle copy is a reduced (1/4) version of 30 simulated days with five transaction types (CASH-IN, CASH-OUT, DEBIT, PAYMENT, TRANSFER) ([mirror description](https://baselight.app/u/kaggle/dataset/ealaxi_paysim1)). | Synthetic; mobile-money wallets, not loans: no loan purpose, supplier or invoice. Its description warns that the balance columns must not be used for fraud detection. |
| IBM Transactions for AML ([Kaggle](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml), [paper](https://arxiv.org/abs/2306.16424)) | Synthetic AML datasets made with an agent-based generator calibrated to match real transactions; labels are complete (unlike real data, where much laundering is never detected). | Synthetic; money laundering between accounts, not misuse of preferential loans: no loan purpose, supplier or invoice. |
| SAML-D ([Kaggle](https://www.kaggle.com/datasets/berkanoztas/synthetic-transaction-monitoring-dataset-aml), [Bournemouth / IEEE ICEBE 2023](https://eprints.bournemouth.ac.uk/40982/)) | Synthetic AML transaction-monitoring dataset with 12 features and 28 typologies; a 2024 survey reports 9,504,852 transactions with about 0.1039% suspicious ([arXiv:2404.14746](https://arxiv.org/pdf/2404.14746)). | Synthetic; focused on geographic locations, high-risk countries and payment types; no loans, suppliers or invoices. |
| Credit Card Fraud ([Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), [OpenML 1597](https://api.openml.org/d/1597)) | Card transactions of European cardholders over two days in September 2013: 492 frauds among 284,807 transactions (0.172%); V1-V28 are PCA components, only Time and Amount are untransformed (research collaboration of Worldline and the ULB Machine Learning Group). | Real but anonymised card payments: features are PCA components, so no loan purpose, supplier, invoice or interpretable flow; it is not loan data. |
| Home Credit Default Risk ([Kaggle](https://www.kaggle.com/competitions/home-credit-default-risk/data); tables as described in [community write-ups](https://github.com/NoxMoon/home-credit-default-risk)) | Consumer-credit data: application table with a default TARGET, credit-bureau history, previous applications, instalment repayment history, POS / cash and credit-card monthly balances. | Predicts default at application time from application and bureau data; no supplier / invoice and no account-level flows after disbursement (to our reading of the table list; column definitions are in the competition's `HomeCredit_columns_description.csv`). |
| Lending Club loan data ([example description of the loan schema](https://rdrr.io/github/tessington/qsci381/man/loans_full_schema.html), Kaggle release `accepted_2007_to_2018Q2.csv`) | Loan-level data of loans actually made on the Lending Club platform, with a loan-purpose category and a loan status. | Loan-level only: no transactions after disbursement, no supplier or invoice; peer-to-peer consumer loans, a different product from preferential microloans for assets. |

Most of these are synthetic themselves; the real ones are anonymised or application-time only. **None combines loan purpose + supplier + invoice + post-disbursement flows**,
which is why the generator reproduces the schemes from the prosecuted cases in `docs/brief.md` §2 (assumptions in section 2). Real accuracy is measured in a pilot on historical loans.

## 3. Licenses
| Item | Use | License |
|---|---|---|
| numpy 2.5.3 | generator | BSD-3-Clause |
| pandas 3.0.6 | generator, features | BSD-3-Clause |
| pyarrow 26.0.0 | parquet I/O | Apache-2.0 |
| scikit-learn 1.9.1 | leakage check, grouped CV, metrics | BSD-3-Clause |
| Images: 154 photos from Wikimedia Commons (tikuv_dastgohi 20, chorva 20, issiqxona 20, parranda 20, asalarichilik 20, faces_test 11, other 43), saved as <= 512 px JPEG in `ml/assets/photos/` | demo photo index, duplicate-detection, face-blur and asset-check (CLIP) tests; `other` = random everyday photos used as the negative class | CC BY-SA 4.0 x84, CC BY 4.0 x17, CC0 x17, CC BY-SA 3.0 x12, Public domain x11, CC BY 2.0 x6, CC BY-SA 2.0 x5, CC BY 3.0 x2. Per-file title, author, license and source URL: [`ml/assets/photos/manifest.csv`](../ml/assets/photos/manifest.csv). Attribution (author + license + source link) is given there as required by CC BY / CC BY-SA; images were downscaled and re-encoded only. `faces_test` images show real people (openly licensed) and are used only to test face blurring. |
| LightGBM 4.7.0 | money-trail model (`ml/money_trail/`) | MIT |
| pytest 9.1.1 | tests | MIT |
| opencv-python-headless 4.11.0.86 | image decode helpers, face detection (YuNet, Haar), blur | Apache-2.0 |
| YuNet face detector `face_detection_yunet_2023mar.onnx` (OpenCV Zoo, 233 KB, `ml/photo/models/`) | face detection before blurring | MIT (Shiqi Yu) |
| OpenCV Haar cascade `haarcascade_frontalface_default.xml` (shipped inside opencv) | fallback / second detector in tests | BSD-style (OpenCV / Intel license) |
| Pillow 12.3.0 | image decode / re-encode | MIT-CMU (HPND) |
| OpenCLIP `open_clip_torch` 3.3.0 | asset check (`ml/photo/asset.py`) | MIT ([mlfoundations/open_clip](https://github.com/mlfoundations/open_clip/blob/main/LICENSE)) |
| CLIP ViT-B-32 weights (OpenAI), **not committed** | image embeddings for the asset check; downloaded once (354 MB, sha256 checked by open_clip) from OpenAI's public URL into the cache directory `MAQSAD_CLIP_CACHE` (default `ml/.cache/clip`, git-ignored). A Dockerfile can pre-download them at build time with `python -m ml.photo.asset --download`. | MIT ([openai/CLIP](https://github.com/openai/CLIP/blob/main/LICENSE)); **needs legal review for production**: OpenAI's model card calls any deployed use of the model out of scope; alternative: a LAION-trained OpenCLIP checkpoint with its own weight licence (`docs/metrics.md`, Known failure cases) |
| torch 2.14.1 (CPU wheels), torchvision 0.29.1 | CLIP inference (`--extra-index-url https://download.pytorch.org/whl/cpu` in `ml/requirements.txt`) | BSD-3-Clause |
| timm 1.0.30 | dependency of open_clip | Apache-2.0 |
| psutil 7.2.2, openpyxl 3.1.5 | memory measurement (`ml/eval`), `docs/sample_loans.xlsx` | BSD-3-Clause, MIT |
| Asset probe `ml/artifacts/asset_probe_v1.npz` | logistic regression on CLIP embeddings of our library (11 KB, committed) | project artifact (derived from the openly licensed library above) |
