"""Assemble docs/metrics.md from the stored evaluation results (ml/artifacts/eval_*.json).

    .venv/Scripts/python -m ml.eval.build_metrics

Every number in the document comes from a JSON written by a script in ml/eval/ or ml/money_trail/
(named next to the number); this script only formats them.
"""
from __future__ import annotations

import json
from pathlib import Path

from ml.generator.config import SPLITS
from ml.money_trail.model import ARTIFACTS, artifact_paths

DOCS = Path(__file__).resolve().parents[2] / "docs" / "metrics.md"
RES_DEV = ARTIFACTS / "eval_money_trail_dev.json"
RES_HOLDOUT = ARTIFACTS / "eval_money_trail_holdout.json"
RES_LATENCY = ARTIFACTS / "eval_latency.json"
RES_PHOTO = ARTIFACTS / "eval_photo.json"
RES_SCENARIO = ARTIFACTS / "eval_scenario.json"
M = "Money-trail model (LightGBM)"
EVAL = "`python -m ml.eval.eval_money_trail`"


def _load(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def pct(x, d=1):
    return "-" if x is None else f"{100 * x:.{d}f}%"


def fci(point, c, d=3):
    return f"{point:.{d}f}" + ("" if not c else f" [{c[0]:.{d}f}, {c[1]:.{d}f}]")


def cis(c, d=3):
    return "-" if not c else f"[{c[0]:.{d}f}, {c[1]:.{d}f}]"


def sha(version):
    import hashlib
    p = artifact_paths(version)[0]
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest() if p.exists() else "-"


# ------------------------------------------------------------------ money-trail result tables
def render_result(r: dict, h: str) -> list[str]:
    L = []
    w = L.append
    ci = r["ci"]
    w(f"{h} Recall@20% inspection budget")
    w("")
    w(f"{r['split']} split: {r['n']} loans, {r['n_pos']} fraud ({pct(r['fraud_rate'])}), {r['n_suppliers']} declared suppliers. "
      "95% CIs: *plain* = bootstrap over loans, *clustered* = bootstrap over suppliers (loans of one supplier resampled together); "
      f"1000 resamples each.")
    w("")
    w("| Method | recall@20% | 95% CI plain | 95% CI clustered by supplier |")
    w("|---|---|---|---|")
    for k, v in r["recall"].items():
        w(f"| {k} | {v:.3f} | {cis(ci['recall'][k]['plain'])} | {cis(ci['recall'][k]['clustered'])} |")
    w(f"| Random selection | {r_random():.3f} (expected value) | | |")
    w("")
    r01 = r["rule01"]
    w(f"The 0/1 rule flags only {pct(r01['flagged'])} of loans (precision {pct(r01['precision'])}, recall {pct(r01['flag_recall'])} "
      "counting flagged loans only); the rest of its 20% budget is filled at random, so its recall@20% is not 'rule only'. "
      "The graded rule ranks all loans by the same `cash_share_d1` and is the fairer baseline. "
      f"The best single feature (`{r['baseline_feature'][0]}`) was chosen on the train split.")
    w("")
    w(f"{h} Ranking quality of the model")
    w("")
    w("| Metric | value | 95% CI plain | 95% CI clustered by supplier | Random |")
    w("|---|---|---|---|---|")
    for name, key, rnd in (("ROC-AUC", "roc", "0.500"), ("PR-AUC", "pr", f"{r['fraud_rate']:.3f}")):
        v = r[key]
        w(f"| {name} | {v:.3f} | {cis(ci[key]['plain'])} | {cis(ci[key]['clustered'])} | {rnd} |")
    w("")
    w(f"{h} Share of each group inside the 20% budget")
    w("")
    w("Fraud groups: recall@20% of the group. Honest groups: false-positive rate (share of the group inside the budget). "
      f"CI (clustered by supplier) only for groups with n >= 50; smaller groups show counts only. Columns after the model: "
      "baselines (point estimates).")
    w("")
    names = list(r["recall"])
    w("| Group | n | " + " | ".join(["model [95% CI]"] + [n.split(":")[0] for n in names[1:]]) + " |")
    w("|---|---|" + "---|" * len(names))
    for k, shares in r["groups"].items():
        g = ci["groups"].get(k)
        first = pct(shares[M]) + (f" [{100 * g[0]:.1f}, {100 * g[1]:.1f}]" if g else "")
        w(f"| {k} | {r['group_n'][k]} | {first} | " + " | ".join(pct(shares[n]) for n in names[1:]) + " |")
    w("")
    nv = r["not_visible_by_typology"]
    w(f"Fraud positives not yet visible at the snapshot (counted as positives, they look honest): "
      + ", ".join(f"{t} {c}" for t, c in nv.items() if c) + f" (total {sum(nv.values())}).")
    w("")
    w(f"{h} Colours by queue rank")
    w("")
    t = r["thresholds"]
    w(f"Rule: {t['rule']}. On this portfolio: red = risk >= {t['red']:.4f} ({t['n_red']} loans), yellow = risk >= {t['yellow']:.4f} "
      f"({t['n_yellow']} loans).")
    w("")
    w("| Colour | loans | fraud rate inside |")
    w("|---|---|---|")
    for c, v in r["colors"].items():
        w(f"| {c} | {v['n']} | {pct(v['fraud_rate'])} |")
    w("")
    w(f"Red + yellow (the 20% budget) contain {pct(r['red_yellow_recall'])} of all fraud loans.")
    w("")
    return L


def r_random():
    return 0.20


def render_latency() -> list[str]:
    lat = _load(RES_LATENCY)
    if not lat:
        return ["_Latency not measured yet: `python -m ml.eval.eval_money_trail --latency`._", ""]
    rng = lambda key, f="{:.1f}": f"{f.format(min(r[key] for r in lat['runs']))} - {f.format(max(r[key] for r in lat['runs']))}"
    L = ["### Latency and size", "",
         f"Measured by {EVAL} `--latency` on the demo portfolio (not used for metrics), {len(lat['runs'])} runs x {lat['n']} "
         f"warm single `score_loan` calls; range across the runs. {lat['cpu']}, Python {lat['python']}, single thread.", "",
         "| Item | Value (range over runs) |", "|---|---|",
         f"| `score_loan` p50 (model inference + reasons) | {rng('p50')} ms |",
         f"| `score_loan` p95 | {rng('p95')} ms |",
         f"| same, wall clock p50 / p95 | {rng('wall_p50')} ms / {rng('wall_p95')} ms |",
         f"| Cold start: imports + demo portfolio generated, featured and scored once ({len(lat['cold'])} fresh processes) | "
         f"{min(c['total_s'] for c in lat['cold']):.1f} - {max(c['total_s'] for c in lat['cold']):.1f} s "
         f"(of which portfolio build {min(c['init_s'] for c in lat['cold']):.1f} - {max(c['init_s'] for c in lat['cold']):.1f} s) |"]
    mp = artifact_paths("money_trail_v2")[0]
    L += [f"| Model artifact `money_trail_v2.txt` | {mp.stat().st_size / 1024:.0f} KB |", "| Cost per check | 0 (local model, no external API) |", ""]
    return L


def render_scenario_fusion() -> list[str]:
    from ml import fusion
    L = []
    w = L.append
    w("## 5. Scenario scoring and evidence fusion")
    w("")
    w("### Scenario scoring (judges build their own loan)")
    w("")
    w("`ml.api.score_scenario` scores a loan + transactions typed in by the user with the same frozen model v2, the same features "
      "and the same reason templates as a portfolio loan. The scenario is added as one extra loan to a *copy* of the demo "
      "portfolio's context (the cached portfolio is never modified); `percentile` is its rank among the demo portfolio's risks. "
      "Every score is computed live: changing one transaction changes the score.")
    w("")
    sc = _load(RES_SCENARIO)
    if sc:
        ex = sc["example"]
        w("Example (`fast_cash_out` preset, only the day of the 42 000 000 so'm cash withdrawal changed):")
        w("")
        w("| Cash withdrawal on day | risk | colour | percentile in the demo portfolio | cash_share_d1 | first top reason |")
        w("|---|---|---|---|---|---|")
        for e in ex:
            w(f"| {e['cash_day']} | {e['risk']:.4f} | {e['color']} | {e['percentile']:.1f} | {e['cash_share_d1']:.2f} | {e['top_reason']} |")
        w("")
        w(f"Latency of `score_scenario` ({len(sc['runs'])} runs x {sc['runs'][0]['n']} calls over the 4 presets, inputs varied; {sc['cpu']}, "
          f"Python {sc['python']}), range across runs: p50 {min(r['p50'] for r in sc['runs']):.0f} - {max(r['p50'] for r in sc['runs']):.0f} ms, "
          f"p95 {min(r['p95'] for r in sc['runs']):.0f} - {max(r['p95'] for r in sc['runs']):.0f} ms, "
          f"slowest call {max(r['max'] for r in sc['runs']):.0f} ms (limit in the tests: 500 ms). Produced by `python -m ml.eval.eval_scenario`.")
    else:
        w("_Not measured yet: `python -m ml.eval.eval_scenario`._")
    w("")
    w("Assumptions: the invoice amount equals the loan amount; a new supplier has no outflows of its own; for an existing supplier the "
      "real demo ledger gives the supplier-graph features (in-degree, downstream cash share); the context is restricted to the "
      "supplier's loans, which gives exactly the same features as a full copy (tested).")
    w("")
    w("### Fusion policy (photo -> queue)")
    w("")
    w("**There is no labelled data for photo signals, so fusion is a documented, configurable policy (`ml/fusion.py`, one dict), "
      "not a learned model; no accuracy is claimed for it.** Each triggered signal escalates a loan to a minimum colour. "
      "Escalation only raises priority: it never lowers the money-trail colour, and the money-trail `risk` is never changed. "
      "`priority` = colour rank + risk (the queue sort key); `escalations[]` and policy reasons (Uzbek, factual, with evidence refs "
      "to the photo) are added to the queue items and to `get_loan`.")
    w("")
    w("| Code | Minimum colour | Active | Triggered when |")
    w("|---|---|---|---|")
    for row in fusion.policy_table():
        w(f"| `{row['code']}` | {row['min_color']} | {'yes' if row['active'] else 'no (later)'} | {row['when']} |")
    w("")
    w("The demo portfolio's own photos are evaluated with the same policy (duplicates in the index, distance, capture time); a sandbox's "
      "uploaded photos escalate loans only in that sandbox's view (`get_queue(sandbox_id)`, `get_loan(loan_id, sandbox_id)`) and "
      "`reset_sandbox` removes them. The 5 km distance limit and the photo rates of the generator are assumptions.")
    w("")
    return L


def render() -> str:
    dev, hold, photo = _load(RES_DEV), _load(RES_HOLDOUT), _load(RES_PHOTO)
    L = []
    w = L.append
    w("# Metrics")
    w("")
    w("**All numbers are on synthetic data; real accuracy is measured in a pilot on historical loans.** "
      "Every number is produced by a script (named next to it) and stored in `ml/artifacts/eval_*.json`; "
      "`python -m ml.eval.build_metrics` only formats them. Fraud rates, typology mix and all generator parameters are "
      "assumptions (`data/README.md`).")
    w("")
    w("## 1. Data & splits")
    w("")
    w("| Split | Role | Seed | Snapshot | Loans | Fraud rate (assumed) |")
    w("|---|---|---|---|---|---|")
    role = {"train": "training and cross-validation only", "dev": "development: looked at while building features, reasons and "
            "the v1 -> v2 fixes, so NOT an unbiased test set; kept for comparison with v1",
            "holdout": "evaluated **exactly once**, after the model was frozen; the headline numbers",
            "demo": "served by `ml.api` to the dashboard and used for the photo index and latency; never used for metrics"}
    for name, p in SPLITS.items():
        w(f"| {name} | {role[name]} | {p.seed} | {p.snapshot_date} | {p.n_loans} | {pct(p.fraud_rate, 0)} |")
    w("")
    w("All splits have their own ID space, hence disjoint loans, borrowers, suppliers, employees, photos and transactions "
      "(tested in `ml/tests/test_generator.py`). Dev, holdout and demo share the same shifted generator parameters "
      "(more fraud, more stealth cases, slower cash-out, ... see `ml/generator/config.py`); only the seed differs.")
    w("")
    w("| Artifact | sha256 (LF-normalised file bytes) |")
    w("|---|---|")
    w(f"| `ml/artifacts/money_trail_v2.txt` | `{sha('money_trail_v2')}` |")
    w(f"| `ml/artifacts/money_trail_v1.txt` (kept for comparison) | `{sha('money_trail_v1')}` |")
    w("")
    if hold:
        run = hold["run"]
        w(f"Holdout run: commit `{run['commit']}`{' (working tree was dirty)' if run['dirty'] else ''}, {run['run_at']}, "
          f"model sha256 `{run['model_sha256']}`" + (f"; **re-run {run['reruns']} time(s) to reproduce** "
          f"(first run at commit `{run['first_run']['commit'][:10]}`; numbers identical: {hold.get('rerun_identical_numbers')})."
           if run["reruns"] else "; evaluated once."))
        w("")

    w("## 2. Money-trail v2 (holdout = headline)")
    w("")
    cv = (_load(artifact_paths("money_trail_v2")[1]) or {}).get("cv")
    w("- Model `money_trail_v2`: LightGBM, 29 money-trail features (`ml/money_trail/features.py`; no photo / geo, no labels), "
      "trained on the train split only (fixed seed and hyper-parameters, no search). Positives = all fraud loans.")
    if cv:
        w(f"- Cross-validation on train (5-fold, grouped by declared supplier, out-of-fold; `python -m ml.money_trail.train`): "
          f"ROC-AUC {cv['roc_auc']:.3f}, PR-AUC {cv['pr_auc']:.3f} (random {cv['pr_auc_random']:.3f}), recall@20% {cv['recall_at_20']:.3f} (random 0.200).")
    w("")
    if hold:
        L += render_result(hold, "###")
    else:
        w("_The holdout has not been evaluated yet: `python -m ml.eval.eval_money_trail --split holdout` (once)._")
        w("")
    L += render_latency()
    if dev:
        w("### Development split (dev) - same tables, for comparison with v1")
        w("")
        L += render_result(dev["v2"], "####")

    w("## 3. v1 -> v2 change log")
    w("")
    w("An independent review of v1 found the problems below; each row says what changed. The dev numbers before / after "
      f"come from {EVAL} `--split dev` (v1 = frozen `features_v1.py` + `money_trail_v1`; v2 = current).")
    w("")
    w("| Review finding | Change in v2 |")
    w("|---|---|")
    for a, b in (
        ("Demo portfolio was the test split: looking at the dashboard meant looking at test loans; 'run once' not checkable",
         "Splits renamed/added: `dev` (old test, seed 1337), `holdout` (seed 2026, evaluated once, run recorded with model sha256 + commit), `demo` (seed 7, served by `ml.api`, photo index)."),
        ("'No regular income' was the biggest driver of the queue, but it is normal for young loans",
         "Income features are age-aware: `income_rate_per_month`, `income_missing_flag` (only for loans older than 90 days; ASSUMPTION from the generator's income timing, U(20, 90) days), `income_gap_days`, `income_stopped_flag`."),
        ("Reason texts: wrong facts, 'atigi' with large values, wrong sentinel meaning, duplicate facts, '78% i'",
         "Texts rewritten (`ml/money_trail/reasons.py`); 'atigi' only below low thresholds; explicit sentinels ('no payment received yet', 'no activity yet', 'no invoice'); at most one reason per group of equivalent facts (income / cash share / supplier payment); '78 foizi' forms, values above 100% shown; `sup_in_degree` states what it counts (loans naming the supplier on the invoice)."),
        ("Colour explanation contradicted the measurement (fixed train thresholds drift on a shifted portfolio)",
         "Colours by queue rank on the served portfolio (top 10% red, next 10% yellow); thresholds returned by `ml.api.get_color_thresholds()` and listed above; no causal claim is made."),
        ("Not-visible positives were mis-counted (28 T3) ", "Reported by typology (T3 and T1 differ)."),
        ("Bootstrap ignored supplier clustering; CI too narrow", "Supplier-clustered bootstrap next to the plain one; groups with n < 50 get counts only."),
        ("Latency varied between runs", "Reported as a range over 3 runs."),
        ("Unhashable `loan_id` raised TypeError; `invoice_gap` hid 'no invoice'", "`LoanNotFoundError` for any invalid id; `invoice_gap = -1` means no invoice."),
        ("API contract incomplete", "`docs/api_contract.md` updated (sandbox_id, reset_sandbox, faces_count, blurred_image, photos[], evidence[], evidence_refs, colour rule)."),
    ):
        w(f"| {a} | {b} |")
    w("")
    w("Not changed: reasons do not yet say which direction is risky relative to a typical honest value (review minor 14); "
      "`income` rows still include one-off sales to businesses (generator), so the texts say 'tushum', not 'muntazam tushum'.")
    w("")
    if dev:
        v1, v2 = dev["v1"], dev["v2"]
        w("### Dev: v1 vs v2")
        w("")
        w("| Metric (dev) | v1 | v2 |")
        w("|---|---|---|")
        w(f"| recall@20% (model) | {v1['recall'][M]:.3f} | {v2['recall'][M]:.3f} |")
        w(f"| ROC-AUC | {v1['roc']:.3f} | {v2['roc']:.3f} |")
        w(f"| PR-AUC | {v1['pr']:.3f} | {v2['pr']:.3f} |")
        for t in ("T1", "T2", "T3", "T4"):
            w(f"| recall@20% {t} | {pct(v1['groups'][t][M])} | {pct(v2['groups'][t][M])} |")
        w(f"| honest false-positive rate in the budget | {pct(v1['groups']['honest/all'][M])} | {pct(v2['groups']['honest/all'][M])} |")
        w("")
        w("### Ablation by loan age at the snapshot (dev)")
        w("")
        w("Recall = share of the age group's fraud loans inside the 20% budget; FPR = share of its honest loans inside the budget "
          "(the budget is the top 20% of the whole portfolio by each model's own score).")
        w("")
        w("| Loan age | loans (fraud / honest) | v1 recall | v2 recall | v1 honest FPR | v2 honest FPR |")
        w("|---|---|---|---|---|---|")
        for a1, a2 in zip(v1["age"], v2["age"]):
            w(f"| {a1['bucket']} | {a1['n']} ({a1['n_fraud']} / {a1['n_honest']}) | {pct(a1['recall'])} | {pct(a2['recall'])} | "
              f"{pct(a1['honest_fpr'])} | {pct(a2['honest_fpr'])} |")
        w("")
        y1, y2 = v1["young_no_income"], v2["young_no_income"]
        w("Reading the ablation: the age-aware encoding changes the honest false-positive rate of loans younger than 60 days only "
          f"from {pct(v1['age'][0]['honest_fpr'])} to {pct(v2['age'][0]['honest_fpr'])}; young loans stay clearly harder than old ones. "
          f"The reason is in the data, not in a missed fix: among loans younger than 60 days {pct(y2['honest'])} of the honest and "
          f"{pct(y2['fraud'])} of the fraud loans have no income yet (our generator gives fraud typologies a lower income "
          "probability and a later start), so 'no income yet' keeps a small genuine likelihood ratio; "
          "the v2 encoding (`income_rate_per_month = -1` = income not due yet, never shown as a reason) only removes the "
          "age artefact. The generator's income timing is an assumption (`data/README.md`); "
          "real borrowers must be re-measured. The first v2 draft encoded 'not due yet' as 0; a dev diagnostic showed that this feature "
          "was still the top driver for most flagged young honest loans, so it was changed to -1 **before** the holdout was run "
          "(this did not change the dev numbers materially). The honest false-positive rate of young loans is therefore a known limitation.")
        w("")

    w("## 4. Photo module")
    w("")
    if photo:
        from ml.eval.eval_photo import render as render_photo
        sec = render_photo(photo["dup"], photo["faces"], photo["lat"], photo["idx"])
        sec = "\n".join(l for l in sec.splitlines() if not l.startswith("<!--"))
        sec = sec.replace("## Photo module (openly licensed images)\n", "").replace("\n### ", "\n### ")
        w(sec.strip())
        w("")
    else:
        w("_Not evaluated yet: `python -m ml.eval.eval_photo`._")
        w("")

    L += render_scenario_fusion()
    w("## 6. Known failure cases")
    w("")
    ref = hold or (dev["v2"] if dev else None)
    if ref:
        weak = sorted(((v[M], k) for k, v in ref["groups"].items() if v[M] is not None and not k.startswith("honest")), key=lambda x: x[0])[:4]
        w(f"- **Weakest fraud groups ({ref['split']}):** " + "; ".join(f"{k} {pct(s)} in budget (n={ref['group_n'][k]})" for s, k in weak) + ".")
        hon = sorted(((v[M], k) for k, v in ref["groups"].items() if k.startswith("honest/") and k != "honest/all"), reverse=True)[:2]
        w("- **Honest look-alikes that get flagged most:** " + "; ".join(f"{k} {pct(s)} (n={ref['group_n'][k]})" for s, k in hon) + ".")
        nv = ref["not_visible_by_typology"]
        w(f"- **Fraud that has not happened yet cannot be seen:** {sum(nv.values())} positives ({', '.join(f'{t} {c}' for t, c in nv.items() if c)}) "
          "look honest at the snapshot by construction; they count against recall.")
    w("- **'No income' and young loans:** v2 no longer penalises loans younger than 90 days, but the 90-day limit is an assumption "
      "about how our synthetic generator times income; real borrowers (seasonal income, grace periods) must be re-measured.")
    w("- **Everything is synthetic:** fraud typologies, rates and the honest-but-suspicious profiles are assumptions; the generator's "
      "own structure (e.g. ring bursts, photo-reuse rates the same for all typologies) is what the model learns. See `ml/generator/README.md` "
      "(Known limitations: counterparty ID collisions, ring size up to ~12 vs. 71 in a real case).")
    w("- **Photo module:** see the failure cases in section 4 (hash crop robustness, screen photos, side faces).")
    w("- **Fusion is a policy, not a measured model:** its thresholds (e.g. 5 km) and escalation colours are choices, not fitted values; "
      "its effect on recall / precision cannot be measured without labelled photo data.")
    w("- **Scenarios:** a hand-built scenario can describe behaviour the generator never produces; the score then shows what the "
      "model does outside the synthetic world, not that the behaviour is fraud. Assumptions are listed in section 5.")
    w("- The photo result does not change the money-trail `risk` value; it only raises the colour / priority (policy in section 5).")
    w("")
    w("## 7. Reproduce")
    w("")
    w("```")
    w(".venv/Scripts/python -m pip install -r ml/requirements.txt")
    w(".venv/Scripts/python -m ml.money_trail.train                      # model v2 (train split only)")
    w(".venv/Scripts/python -m ml.eval.eval_money_trail --split dev     # dev tables + v1/v2 comparison + age ablation")
    w(".venv/Scripts/python -m ml.eval.eval_money_trail --split holdout # EXACTLY ONCE (refuses to run twice; --force-holdout-rerun only reproduces)")
    w(".venv/Scripts/python -m ml.eval.eval_money_trail --latency")
    w(".venv/Scripts/python -m ml.eval.eval_photo --write-config        # photo duplicate variant + threshold")
    w(".venv/Scripts/python -m ml.photo.index                           # photo index on the demo split")
    w(".venv/Scripts/python -m ml.eval.eval_photo")
    w(".venv/Scripts/python -m ml.eval.eval_scenario                    # scenario latency + example")
    w(".venv/Scripts/python -m ml.eval.build_metrics                    # regenerate this file from the stored results")
    w(".venv/Scripts/python -m pytest ml/tests -q")
    w("```")
    return "\n".join(L) + "\n"


def main():
    DOCS.write_text(render(), encoding="utf-8", newline="\n")
    print("wrote", DOCS)


if __name__ == "__main__":
    main()
