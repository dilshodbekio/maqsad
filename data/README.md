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

**Train/test shift.** Train: seed 42, snapshot 2026-03-31. Test: seed 1337, snapshot 2026-07-31, disjoint ID space,
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
Assumption: fraud rate is 11% (train) and 13% (test); real rates are unknown and are measured in the pilot.
Assumption: typology mix among fraud is T1 30% / T2 30% / T3 20% / T4 20%.
Assumption: loan amounts are log-normal around illustrative medians per purpose (30-80 mln UZS), sigma 0.35.
Assumption: region centroids are rounded approximations; borrower coordinates = centroid + noise (sd 0.25 deg).
Assumption: OKED codes are illustrative and not checked against the official classifier.
Assumption: honest livestock/poultry sellers are ~5% of loans, cash-heavy rural businesses ~5%, honest OKED mismatch 6-8% of honest loans.
Assumption: 25-35% of T1/T2 are "stealth" (look clean); 25% of T4 use a matching-OKED supplier.
Assumption: ring size is 4 + Poisson(3) (train) / Poisson(2.5) (test), stealth rings have 2-3 borrowers; 40% / 33% of ring suppliers link to a bank employee, 2.5% of honest suppliers do. A linked employee works in the branch of one of the supplier's borrowers with probability 80% (rings) / 75% train, 70% test (honest suppliers).
Assumption: 35% of fraud loans (both splits) and 2% of honest loans reuse a photo; 25% of fraud and 4% of honest photos are taken far from the borrower. These rates are the same for all typologies and not shifted between train and test.
Assumption: 18% (both splits) of honest suppliers with >= 2 borrowers are active in a short window only (20-120 days): newly registered suppliers (registered 3-45 days before their first payment) and, for parranda / issiqxona, seasonal suppliers (March-August). Ring suppliers burst within 30-150 days (brief §2 case 3), so supplier activity span alone is a strong but not decisive T2 signal (single-feature AUC ~0.84).
Assumption: all supplier-account flows are derived from loan payments only (no other business activity is simulated).
Assumption: all numbers measured on this synthetic data say nothing about real accuracy; real accuracy is measured in the pilot.

## 3. Licenses
| Item | Use | License |
|---|---|---|
| numpy 2.5.3 | generator | BSD-3-Clause |
| pandas 3.0.6 | generator, features | BSD-3-Clause |
| pyarrow 26.0.0 | parquet I/O | Apache-2.0 |
| scikit-learn 1.9.1 | leakage check (ROC-AUC) | BSD-3-Clause |
| Images (photo set, asset-type probe) | TBD | TBD (openly licensed only; each file to be listed here) |
| Models (LightGBM, CLIP, ...) | TBD | TBD |
