# ml/ — AI/ML layer (owner: Dilshodbek)

Read the root `CLAUDE.md` and `docs/brief.md` (section 6.1) first.
This layer carries most of the Technical (25%) and Innovation (25%) score. Judges will ask:
"Is this a system or just a model call? What quality did YOU measure on YOUR test set? Latency? Cost? Known failure cases?"

## Layout
```
ml/
├── api.py              # public interface used by backend (see root CLAUDE.md)
├── generator/          # synthetic portfolio generator
├── money_trail/        # features, LightGBM, SHAP → reasons
├── photo/              # duplicate detection (pHash/embeddings), face blur, asset type (CLIP + probe)
├── eval/               # evaluation scripts → docs/metrics.md
├── artifacts/          # trained models, photo index (small only)
└── requirements.txt
```

## Priority order (MVP first; stop adding scope if behind)
1. **Generator** (~2h) — FIRST, everything depends on it
2. **Money-trail model** (~3h)
3. **Photo duplicate detection + face blur** (~1.5h)
4. Asset type (CLIP zero-shot → linear probe) — only after 1–3 are wired into the dashboard
5. Metrics + known failure cases → `docs/metrics.md`

Deadline: by end of day 1, items 1–3 are connected to Oybek's dashboard and deployed.

## 1. Generator (`ml/generator/`)
- 2–5k loans. Purposes: `tikuv_dastgohi, chorva, issiqxona, parranda, asalarichilik`.
- Tables: `loans`, `transactions`, `suppliers`, `employees`. Fields per `docs/brief.md` §6.1.
- Transaction types: `disbursement, cash_withdrawal, supplier_payment, p2p_out, p2p_in, external_transfer, income`.
- Fraud typologies (based on real prosecuted cases):
  - **T1 fast cash-out**: large share withdrawn as cash on day 0–1, no supplier payment
  - **T2 supplier ring**: many borrowers → one supplier → supplier cashes out fast; supplier may be linked to a bank employee
  - **T3 asset resale**: supplier payment exists, later a one-off income resembling an asset sale + cash-out
  - **T4 fake invoice**: supplier OKED does not match loan purpose
- **Honest-but-suspicious borrowers** (must exist): farmer who legally sold livestock, cash-heavy rural business, generic-trade OKED supplier used legitimately.
- Fraud rate: ~8–15% — document as an ASSUMPTION in `data/README.md`.
- Test set: generated separately with a **different seed and slightly different parameters**.
- Photos: each loan linked to an openly licensed image; a portion deliberately reused across loans (photo-duplicate scheme). Licenses → `data/README.md`.

### Anti-leakage rules (critical)
- Parameters are **sampled from distributions**, not fixed thresholds. Some fraudsters look "clean" (cash out over 5–10 days, small supplier payment).
- Some honest loans also have OKED mismatch (generic trade) — otherwise T4 recall is trivially 100%.
- T2/T3 must be hard for a single rule: signal should come from feature combinations (supplier in-degree + supplier's downstream cash flow, etc.).
- Never use the typology label or generator-internal fields as features.

## 2. Money-trail model (`ml/money_trail/`)
- Features: time from disbursement to first cash withdrawal; cash share within 7/30 days; supplier payment present + share; supplier in-degree; supplier downstream cash share; share leaving the bank; P2P count; supplier–employee link; OKED ↔ purpose match.
- LightGBM + SHAP. Map each top SHAP contribution to `{code, text_uz, weight}` in plain Uzbek.
- **Split grouped by supplier** (T2 suppliers must not appear in both train and validation).

## Evaluation (`ml/eval/` → `docs/metrics.md`)
- **Primary metric: recall@20% inspection budget** — model vs. random selection vs. simple rule baseline
  (e.g. "≥50% cash on day 0–1 → red").
- Secondary: ROC-AUC, PR-AUC, recall per typology, false positive rate on honest-suspicious borrowers.
- Photo duplicates: precision/recall under crop, JPEG compression, resize, screenshot.
- Per component: p50/p95 latency, cost per check.
- "Known failure cases" list — honest.
- Every number in `docs/metrics.md` must be reproducible by a script in `ml/eval/`. State that numbers are on synthetic data; real accuracy is measured in the pilot.
