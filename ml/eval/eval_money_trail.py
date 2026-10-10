"""Evaluate the money-trail model on the TEST split and write docs/metrics.md.

    .venv/Scripts/python -m ml.eval.eval_money_trail

Run ONCE after training is final (the test split is never used for tuning; the "best single feature"
baseline is chosen on the TRAIN split). Positives = ALL fraud loans (generator convention), including T3
loans whose resale has not happened yet at the snapshot; their number is reported. Everything printed here
is computed by this script; docs/metrics.md is regenerated between the money-trail markers.
"""
from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from ml.money_trail.features import FEATURE_COLUMNS
from ml.money_trail.metrics import BUDGET, recall_at_budget
from ml.money_trail.model import META_PATH, MODEL_PATH, load_model
from ml.money_trail.train import load_split

DOCS = Path(__file__).resolve().parents[2] / "docs" / "metrics.md"
START, END = "<!-- money-trail:start -->", "<!-- money-trail:end -->"
SCRIPT = "`python -m ml.eval.eval_money_trail`"
N_BOOT, BOOT_SEED, N_LAT = 1000, 0, 500
TYPOLOGIES = ("T1", "T2", "T3", "T4")
PROFILES = ("normal", "cash_heavy", "livestock_seller", "oked_mismatch")


def inclusion(score: np.ndarray, budget: float = BUDGET) -> np.ndarray:
    """Probability that each loan is inside the top `budget` share (ties at the cut-off filled at random)."""
    n = len(score)
    B = max(1, int(round(budget * n)))
    thr = np.partition(score, -B)[-B]
    above, tied = score > thr, score == thr
    w = np.zeros(n)
    w[above] = 1.0
    w[tied] = (B - above.sum()) / tied.sum()
    return w


def ci(values) -> tuple[float, float]:
    return float(np.nanpercentile(values, 2.5)), float(np.nanpercentile(values, 97.5))


def pct(x, d=1) -> str:
    return f"{100 * x:.{d}f}%"


def fmt_ci(point, lohi, d=3) -> str:
    return f"{point:.{d}f} [{lohi[0]:.{d}f}, {lohi[1]:.{d}f}]"


def best_single_feature(Xtr: pd.DataFrame, ytr: np.ndarray) -> tuple[str, int]:
    """Single money-trail feature (and direction) with the best recall@20% on the TRAIN split."""
    best = max(((recall_at_budget(sign * Xtr[c].to_numpy(float), ytr), c, sign)
                for c in FEATURE_COLUMNS for sign in (1, -1) if Xtr[c].nunique() > 1))
    return best[1], best[2]


def latency(n=N_LAT) -> dict:
    from ml import api
    api.get_queue()                                           # warm: portfolio generated + scored
    from ml.money_trail.service import get_portfolio
    P = get_portfolio()
    ids = np.random.default_rng(1).choice(list(P.loans.index), n, replace=False)
    for lid in ids[:20]:
        api.score_loan(lid)                                   # warm-up calls, not counted
    inner, wall = [], []
    for lid in ids:
        t0 = time.perf_counter()
        r = api.score_loan(lid)
        wall.append((time.perf_counter() - t0) * 1000)
        inner.append(r["latency_ms"])
    q = lambda a, p: float(np.percentile(a, p))
    return dict(n=n, p50=q(inner, 50), p95=q(inner, 95), wall_p50=q(wall, 50), wall_p95=q(wall, 95),
                init_s=P.init_s)


def evaluate() -> dict:
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    model = load_model()
    Xtr, ytr, _, _ = load_split("train")
    Xte, y, _, tables = load_split("test")
    lab = tables["labels"].set_index("loan_id").reindex(Xte.index)
    typ = lab["typology"].fillna("honest").to_numpy()
    n, n_pos = len(y), int(y.sum())

    score = model.score(Xte)
    bf_name, bf_sign = best_single_feature(Xtr, ytr)
    scorers = {
        "Money-trail model (LightGBM)": score,
        "Graded rule: cash_share_d1": Xte["cash_share_d1"].to_numpy(float),
        f"Best single feature (chosen on train): {bf_name}": bf_sign * Xte[bf_name].to_numpy(float),
    }
    rule01 = (Xte["cash_share_d1"].to_numpy(float) >= 0.5).astype(float)
    res = dict(meta=meta, n=n, n_pos=n_pos, fraud_rate=float(y.mean()),
               not_visible=int((lab["is_fraud"] & ~lab["fraud_visible"]).sum()), bf=(bf_name, bf_sign))

    # ---- headline metrics + bootstrap CIs
    rng = np.random.default_rng(BOOT_SEED)
    idx = [rng.integers(0, n, n) for _ in range(N_BOOT)]
    rows = {}
    for name, s in scorers.items():
        boots = [recall_at_budget(s[i], y[i]) for i in idx]
        rows[name] = dict(recall=recall_at_budget(s, y), recall_ci=ci(boots))
    ms = scorers["Money-trail model (LightGBM)"]
    rows["Money-trail model (LightGBM)"].update(
        roc=float(roc_auc_score(y, ms)), roc_ci=ci([roc_auc_score(y[i], ms[i]) for i in idx]),
        pr=float(average_precision_score(y, ms)), pr_ci=ci([average_precision_score(y[i], ms[i]) for i in idx]))
    res["rows"] = rows
    boots01 = [recall_at_budget(rule01[i], y[i]) for i in idx]
    res["rule01"] = dict(flagged=float(rule01.mean()), precision=float(y[rule01 > 0].mean()),
                         flag_recall=float(y[rule01 > 0].sum() / n_pos), recall20=recall_at_budget(rule01, y),
                         recall20_ci=ci(boots01))
    res["random_recall20"] = BUDGET

    # ---- per typology (recall inside the 20% budget), stealth split
    per_t = {}
    for name, s in scorers.items():
        w = inclusion(s)
        per_t[name] = {t: float(w[typ == t].sum() / (typ == t).sum()) for t in TYPOLOGIES}
    res["per_typology"] = per_t
    res["n_typology"] = {t: int((typ == t).sum()) for t in TYPOLOGIES}
    w = inclusion(ms)
    st = lab["stealth"].to_numpy()
    res["stealth"] = {f"{t} {'stealth' if s else 'plain'}": (int(((typ == t) & (st == s)).sum()),
                                                            float(w[(typ == t) & (st == s)].mean()))
                      for t in ("T1", "T2", "T4") for s in (False, True) if ((typ == t) & (st == s)).any()}
    vis = lab["fraud_visible"].to_numpy()
    res["t3_visibility"] = {k: (int(m.sum()), float(w[m].mean())) for k, m in
                            (("T3 resale visible at snapshot", (typ == "T3") & vis),
                             ("T3 resale not yet happened", (typ == "T3") & ~vis))}

    # ---- honest-but-suspicious profiles inside the budget
    prof = lab["profile"].to_numpy()
    honest = ~lab["is_fraud"].to_numpy()
    res["profiles"] = {name: {p: (int((honest & (prof == p)).sum()), float(inclusion(s)[honest & (prof == p)].mean()))
                              for p in PROFILES} for name, s in scorers.items()}
    res["honest_fpr"] = {name: float(inclusion(s)[honest].mean()) for name, s in scorers.items()}

    # ---- colours
    colors = np.array([model.color(v) for v in ms])
    res["colors"] = {c: (int((colors == c).sum()), float(y[colors == c].mean())) for c in ("red", "yellow", "green")}
    res["red_yellow_recall"] = float(y[colors != "green"].sum() / n_pos)
    res["thresholds"] = (model.red, model.yellow)

    res["latency"] = latency()
    res["sizes"] = dict(model=MODEL_PATH.stat().st_size, meta=META_PATH.stat().st_size)
    return res


def render(r: dict) -> str:
    m, rows = r["meta"], r["rows"]
    cv = m["cv"]
    mt = rows["Money-trail model (LightGBM)"]
    L = []
    w = L.append
    w(START)
    w("## Money-trail model (synthetic data)")
    w("")
    w("**Numbers are on synthetic data; real accuracy is measured in the pilot on historical loans.** "
      "Positives = all fraud loans of the generator (11-13% of loans, an assumption), including "
      f"{r['not_visible']} T3 loans whose resale has not happened yet at the snapshot (they look honest by construction). "
      "Photo / geo signals are not used by this model and are evaluated separately.")
    w("")
    w(f"- Model `{m['model_version']}`: LightGBM, {len(m['features'])} money-trail features, trained on the train split "
      f"(seed {m['train']['seed']}, {m['train']['rows']} loans, {m['train']['fraud']} fraud). "
      f"Test split: seed 1337, {r['n']} loans, {r['n_pos']} fraud ({pct(r['fraud_rate'])}).")
    w("- Test numbers: produced by " + SCRIPT + f"; CV numbers and thresholds: `python -m ml.money_trail.train` "
      "(stored in `ml/artifacts/money_trail_v1.json`). 95% CIs: bootstrap over test loans "
      f"({N_BOOT} resamples, seed {BOOT_SEED}).")
    w("")
    w("### Cross-validation on the train split (5-fold, grouped by supplier)")
    w("")
    w("| Metric | Out-of-fold value | Random |")
    w("|---|---|---|")
    w(f"| ROC-AUC | {cv['roc_auc']:.3f} | 0.500 |")
    w(f"| PR-AUC | {cv['pr_auc']:.3f} | {cv['pr_auc_random']:.3f} |")
    w(f"| recall@20% | {cv['recall_at_20']:.3f} | {cv['random_recall_at_20']:.3f} |")
    w("")
    w("### Test split: recall@20% inspection budget")
    w("")
    w("| Method | recall@20% [95% CI] |")
    w("|---|---|")
    for name, v in rows.items():
        w(f"| {name} | {fmt_ci(v['recall'], v['recall_ci'])} |")
    r01 = r["rule01"]
    w(f"| 0/1 rule: cash_share_d1 >= 0.5 | {fmt_ci(r01['recall20'], r01['recall20_ci'])} |")
    w(f"| Random selection | {r['random_recall20']:.3f} (expected value) |")
    w("")
    w(f"The 0/1 rule flags only {pct(r01['flagged'])} of loans (precision {pct(r01['precision'])}, recall {pct(r01['flag_recall'])} "
      "when flagged loans only are counted); the rest of its 20% budget is filled at random, which is why its recall@20% "
      "is not 'rule only'. The graded rule ranks all loans by the same cash_share_d1 and is the fairer baseline.")
    w("")
    w("### Test split: ranking quality of the model")
    w("")
    w("| Metric | Value [95% CI] | Random |")
    w("|---|---|---|")
    w(f"| ROC-AUC | {fmt_ci(mt['roc'], mt['roc_ci'])} | 0.500 |")
    w(f"| PR-AUC | {fmt_ci(mt['pr'], mt['pr_ci'])} | {r['fraud_rate']:.3f} |")
    w("")
    w("### Test split: recall@20% per typology")
    w("")
    names = list(r["per_typology"])
    w("| Typology | n | " + " | ".join(names) + " |")
    w("|---|---|" + "---|" * len(names))
    for t in TYPOLOGIES:
        w(f"| {t} | {r['n_typology'][t]} | " + " | ".join(pct(r["per_typology"][nm][t]) for nm in names) + " |")
    w("")
    w("Model, harder subgroups (share of the group inside the 20% budget):")
    w("")
    w("| Group | n | in budget |")
    w("|---|---|---|")
    for k, (cnt, share) in {**r["stealth"], **r["t3_visibility"]}.items():
        w(f"| {k} | {cnt} | {pct(share)} |")
    w("")
    w("### Test split: honest loans inside the 20% budget (false-positive rate)")
    w("")
    w("| Honest profile | n | " + " | ".join(names) + " |")
    w("|---|---|" + "---|" * len(names))
    for p in PROFILES:
        cnt = r["profiles"][names[0]][p][0]
        w(f"| {p} | {cnt} | " + " | ".join(pct(r["profiles"][nm][p][1]) for nm in names) + " |")
    w("| all honest | " + str(r["n"] - r["n_pos"]) + " | " + " | ".join(pct(r["honest_fpr"][nm]) for nm in names) + " |")
    w("")
    w("### Colours (budget logic)")
    w("")
    red, yellow = r["thresholds"]
    w(f"Thresholds come from the train OOF score quantiles: red = score >= {red:.4f} (top 10%), yellow = score >= "
      f"{yellow:.4f} (next 10%), green = rest. On the test split the model is applied with these fixed thresholds:")
    w("")
    w("| Colour | loans | share | fraud rate inside |")
    w("|---|---|---|---|")
    for c, (cnt, fr) in r["colors"].items():
        w(f"| {c} | {cnt} | {pct(cnt / r['n'])} | {pct(fr)} |")
    w("")
    ry = (r["colors"]["red"][0] + r["colors"]["yellow"][0]) / r["n"]
    w(f"Red + yellow together are {pct(ry)} of the test loans and contain {pct(r['red_yellow_recall'])} of all fraud loans. "
      "That is above the nominal 20% budget because the thresholds are fixed on train while the test split has more fraud "
      "(13% vs 11%) and more stealth cases; the exact share of each cause was not measured.")
    w("")
    lt = r["latency"]
    w("### Latency and size")
    w("")
    w(f"| Item | Value |")
    w("|---|---|")
    w(f"| `score_loan` latency (model inference + reasons, warm, {lt['n']} single calls), p50 / p95 | "
      f"{lt['p50']:.2f} ms / {lt['p95']:.2f} ms |")
    w(f"| same, wall clock including call overhead, p50 / p95 | {lt['wall_p50']:.2f} ms / {lt['wall_p95']:.2f} ms |")
    w(f"| First call: demo portfolio generated + features + scores (once, in memory) | {lt['init_s']:.1f} s |")
    w(f"| Model artifact `money_trail_v1.txt` | {r['sizes']['model'] / 1024:.0f} KB |")
    w(f"| Cost per check | 0 (local model, no external API) |")
    w("")
    w(f"Measured on: {platform.processor() or platform.machine()}, Python {platform.python_version()}, single thread. "
      "`score_loan` uses the cached feature row of the loan; building features for the whole portfolio is a one-time batch step.")
    w(END)
    return "\n".join(L) + "\n"


def write_docs(section: str):
    head = ("# Metrics\n\nEvery number below is produced by a script in `ml/eval/` or `ml/money_trail/` "
            "(named next to it). All numbers are on synthetic data.\n\n")
    text = DOCS.read_text(encoding="utf-8") if DOCS.exists() else head
    if START in text and END in text:
        a, b = text.index(START), text.index(END) + len(END)
        text = text[:a] + section.rstrip("\n") + text[b:]
    else:
        text = text.rstrip("\n") + "\n\n" + section
    DOCS.write_text(text, encoding="utf-8", newline="\n")


def main():
    r = evaluate()
    section = render(r)
    write_docs(section)
    print(section)


if __name__ == "__main__":
    main()
