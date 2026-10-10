"""Evaluate the money-trail model; results go to ml/artifacts/eval_*.json and docs/metrics.md (build_metrics.py).

    .venv/Scripts/python -m ml.eval.eval_money_trail --split dev        # development split (may be run any time)
    .venv/Scripts/python -m ml.eval.eval_money_trail --split holdout    # EXACTLY ONCE, after the model is frozen
    .venv/Scripts/python -m ml.eval.eval_money_trail --latency          # 3 x 500 score_loan calls on the demo portfolio

Positives = ALL fraud loans (generator convention), including T3 loans whose resale has not happened yet at the
snapshot; their number is reported by typology. The best-single-feature baseline is chosen on the TRAIN split.
Colours are assigned by queue rank on the evaluated portfolio (top 10% red, next 10% yellow). Confidence intervals:
bootstrap over loans ("plain") and bootstrap over declared suppliers ("clustered": loans of one supplier are
resampled together), 1000 resamples each; groups with n < 50 get counts only.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from ml.generator.build import build
from ml.generator.config import SPLITS
from ml.money_trail import features as feat_v2
from ml.money_trail import features_v1 as feat_v1
from ml.money_trail.colors import color_of, rank_thresholds
from ml.money_trail.metrics import BUDGET, recall_at_budget
from ml.money_trail.model import ARTIFACTS, artifact_paths, load_model

RES_DEV = ARTIFACTS / "eval_money_trail_dev.json"
RES_HOLDOUT = ARTIFACTS / "eval_money_trail_holdout.json"
RES_LATENCY = ARTIFACTS / "eval_latency.json"
N_BOOT, BOOT_SEED, MIN_CI_N = 1000, 0, 50
TYPOLOGIES = ("T1", "T2", "T3", "T4")
PROFILES = ("normal", "cash_heavy", "livestock_seller", "oked_mismatch")
AGE_BUCKETS = (("< 60 days", 0, 60), ("60-180 days", 60, 180), ("> 180 days", 180, 10_000))
FEATURES = {"v1": feat_v1, "v2": feat_v2}
MODEL_NAME = {"v1": "money_trail_v1", "v2": "money_trail_v2"}


# ------------------------------------------------------------------ helpers
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


def ci(values) -> list[float] | None:
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    return None if len(v) == 0 else [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def sha256_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def git(*args) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, cwd=ARTIFACTS.parents[1]).stdout.strip()


def load_tables(split: str) -> dict:
    return build(SPLITS[split])


def features_of(version: str, t: dict) -> pd.DataFrame:
    return FEATURES[version].build_features(t["loans"], t["transactions"], t["suppliers"], t["employees"])


def best_single_feature(version: str) -> tuple[str, int]:
    """Single money-trail feature (and sign) with the best recall@20% on the TRAIN split."""
    t = load_tables("train")
    X = features_of(version, t)
    y = t["labels"].set_index("loan_id").reindex(X.index)["is_fraud"].to_numpy().astype(int)
    best = max(((recall_at_budget(sg * X[c].to_numpy(float), y), c, sg)
                for c in FEATURES[version].FEATURE_COLUMNS for sg in (1, -1) if X[c].nunique() > 1))
    return best[1], best[2]


def group_masks(lab: pd.DataFrame) -> dict[str, np.ndarray]:
    typ = lab["typology"].fillna("honest").to_numpy()
    st, vis = lab["stealth"].to_numpy(), lab["fraud_visible"].to_numpy()
    prof, honest = lab["profile"].to_numpy(), ~lab["is_fraud"].to_numpy()
    m = {t: typ == t for t in TYPOLOGIES}
    for t in ("T1", "T2", "T4"):
        m[f"{t} plain"], m[f"{t} stealth"] = (typ == t) & ~st, (typ == t) & st
    m["T3 resale visible at snapshot"] = (typ == "T3") & vis
    m["T3 resale not yet happened"] = (typ == "T3") & ~vis
    for p in PROFILES:
        m[f"honest/{p}"] = honest & (prof == p)
    m["honest/all"] = honest
    return m


def cluster_indices(groups: np.ndarray, n_boot: int, rng):
    _, inv = np.unique(groups, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    cuts = np.flatnonzero(np.diff(inv[order])) + 1
    members = np.split(order, cuts)
    for _ in range(n_boot):
        picks = rng.integers(0, len(members), len(members))
        yield np.concatenate([members[k] for k in picks])


def resample_metrics(scorers: dict, y, masks, idx_iter, model_name) -> dict:
    """Bootstrap distributions: recall@20% of every scorer; ROC / PR of the model; group shares of the model."""
    out = {"recall": {k: [] for k in scorers}, "roc": [], "pr": [], "groups": {k: [] for k in masks}}
    for i in idx_iter:
        yi = y[i]
        if yi.sum() == 0 or yi.sum() == len(yi):
            continue
        for k, s in scorers.items():
            out["recall"][k].append(recall_at_budget(s[i], yi))
        ms = scorers[model_name][i]
        out["roc"].append(roc_auc_score(yi, ms))
        out["pr"].append(average_precision_score(yi, ms))
        w = inclusion(ms)
        for k, m in masks.items():
            mi = m[i]
            out["groups"][k].append(float(w[mi].mean()) if mi.any() else np.nan)
    return out


# ------------------------------------------------------------------ main evaluation of one split / version
def evaluate_split(split: str, version: str = "v2", boot: bool = True) -> dict:
    t = load_tables(split)
    X = features_of(version, t)
    lab = t["labels"].set_index("loan_id").reindex(X.index)
    loans = t["loans"].set_index("loan_id").reindex(X.index)
    y = lab["is_fraud"].to_numpy().astype(int)
    n, n_pos = len(y), int(y.sum())
    model = load_model(MODEL_NAME[version])
    score = model.score(X)
    bf_name, bf_sign = best_single_feature(version)
    M = "Money-trail model (LightGBM)"
    scorers = {
        M: score,
        "Graded rule: cash_share_d1": X["cash_share_d1"].to_numpy(float),
        f"Best single feature (chosen on train): {bf_name}": bf_sign * X[bf_name].to_numpy(float),
        "0/1 rule: cash_share_d1 >= 0.5": (X["cash_share_d1"].to_numpy(float) >= 0.5).astype(float),
    }
    masks = group_masks(lab)
    groups = loans["supplier_id"].to_numpy()
    res = dict(split=split, version=version, n=n, n_pos=n_pos, fraud_rate=float(y.mean()),
               n_suppliers=int(len(np.unique(groups))), baseline_feature=[bf_name, bf_sign],
               not_visible_by_typology={t_: int(((lab["typology"] == t_) & ~lab["fraud_visible"]).sum()) for t_ in TYPOLOGIES},
               group_n={k: int(m.sum()) for k, m in masks.items()})
    # point estimates
    res["recall"] = {k: recall_at_budget(s, y) for k, s in scorers.items()}
    res["roc"], res["pr"] = float(roc_auc_score(y, score)), float(average_precision_score(y, score))
    r01 = scorers["0/1 rule: cash_share_d1 >= 0.5"] > 0
    res["rule01"] = dict(flagged=float(r01.mean()), precision=float(y[r01].mean()), flag_recall=float(y[r01].sum() / n_pos))
    res["groups"] = {k: {nm: float(inclusion(s)[m].mean()) if m.any() else None for nm, s in scorers.items()}
                     for k, m in masks.items()}
    # colours by queue rank on this portfolio
    thr = rank_thresholds(score)
    colors = np.array([color_of(v, thr) for v in score])
    res["thresholds"] = thr
    res["colors"] = {c: dict(n=int((colors == c).sum()), fraud_rate=float(y[colors == c].mean())) for c in ("red", "yellow", "green")}
    res["red_yellow_recall"] = float(y[colors != "green"].sum() / n_pos)
    # age ablation material
    age = ((loans["snapshot_date"] - loans["disbursed_at"]) / pd.Timedelta(days=1)).to_numpy()
    w = inclusion(score)
    honest = y == 0
    res["age"] = [dict(bucket=name, n=int(((age >= lo) & (age < hi)).sum()),
                       n_fraud=int((((age >= lo) & (age < hi)) & ~honest).sum()),
                       recall=float(w[((age >= lo) & (age < hi)) & ~honest].mean()),
                       honest_fpr=float(w[((age >= lo) & (age < hi)) & honest].mean()),
                       n_honest=int((((age >= lo) & (age < hi)) & honest).sum()))
                  for name, lo, hi in AGE_BUCKETS]
    # how many young loans (< 60 days) have no income yet: honest vs fraud (explains what the age ablation can and cannot fix)
    no_income = (X["income_rate_per_month"] < 0) if "income_rate_per_month" in X else (X["income_count"] == 0)
    young = age < 60
    res["young_no_income"] = dict(honest=float(no_income.to_numpy()[young & honest].mean()),
                                  fraud=float(no_income.to_numpy()[young & ~honest].mean()))
    if boot:
        rng = np.random.default_rng(BOOT_SEED)
        plain = resample_metrics(scorers, y, masks, (rng.integers(0, n, n) for _ in range(N_BOOT)), M)
        rng = np.random.default_rng(BOOT_SEED)
        clus = resample_metrics(scorers, y, masks, cluster_indices(groups, N_BOOT, rng), M)
        res["ci"] = dict(
            recall={k: dict(plain=ci(plain["recall"][k]), clustered=ci(clus["recall"][k])) for k in scorers},
            roc=dict(plain=ci(plain["roc"]), clustered=ci(clus["roc"])),
            pr=dict(plain=ci(plain["pr"]), clustered=ci(clus["pr"])),
            groups={k: (ci(clus["groups"][k]) if res["group_n"][k] >= MIN_CI_N else None) for k in masks})
    return res


# ------------------------------------------------------------------ runs
def run_dev():
    v2, v1 = evaluate_split("dev", "v2"), evaluate_split("dev", "v1")
    RES_DEV.write_text(json.dumps(dict(v2=v2, v1=v1), indent=1), encoding="utf-8", newline="\n")
    return v2, v1


def run_holdout(force: bool = False, allow_dirty: bool = False):
    prev = json.loads(RES_HOLDOUT.read_text(encoding="utf-8")) if RES_HOLDOUT.exists() else None
    if prev and not force:
        sys.exit(f"The holdout was already evaluated ({RES_HOLDOUT.name}, commit {prev['run']['commit'][:10]}). "
                 "It is evaluated exactly once; use --force-holdout-rerun only to REPRODUCE it (a re-run is recorded).")
    dirty = git("status", "--porcelain", "--", "ml")
    if dirty and not allow_dirty:
        sys.exit("Refusing to run the holdout with uncommitted changes under ml/ (commit the frozen model first):\n" + dirty)
    model_path, meta_path = artifact_paths("money_trail_v2")
    res = evaluate_split("holdout", "v2")
    res["run"] = dict(commit=git("rev-parse", "HEAD"), dirty=bool(dirty), run_at=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                      model_version="money_trail_v2", model_sha256=sha256_lf(model_path), meta_sha256=sha256_lf(meta_path),
                      reruns=(prev["run"].get("reruns", 0) + 1) if prev else 0,
                      first_run=(prev["run"] if prev else None))
    if prev:
        res["rerun_identical_numbers"] = prev["recall"] == res["recall"] and prev["roc"] == res["roc"]
    RES_HOLDOUT.write_text(json.dumps(res, indent=1), encoding="utf-8", newline="\n")
    return res


def run_latency(n=500, runs=3):
    from ml import api
    from ml.money_trail.service import get_portfolio
    api.get_queue()
    P = get_portfolio()
    all_runs = []
    for r in range(runs):
        ids = np.random.default_rng(100 + r).choice(list(P.loans.index), n, replace=False)
        for lid in ids[:20]:
            api.score_loan(lid)                                  # warm-up, not counted
        inner, wall = [], []
        for lid in ids:
            t0 = time.perf_counter()
            out = api.score_loan(lid)
            wall.append((time.perf_counter() - t0) * 1000)
            inner.append(out["latency_ms"])
        all_runs.append(dict(p50=float(np.percentile(inner, 50)), p95=float(np.percentile(inner, 95)),
                             wall_p50=float(np.percentile(wall, 50)), wall_p95=float(np.percentile(wall, 95))))
    cold = []
    code = ("import time;t=time.perf_counter();from ml import api;api.get_queue();"
            "from ml.money_trail.service import get_portfolio as g;print(time.perf_counter()-t, g().init_s)")
    for _ in range(runs):
        o = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ARTIFACTS.parents[1]).stdout.split()
        cold.append(dict(total_s=float(o[0]), init_s=float(o[1])))
    out = dict(n=n, runs=all_runs, cold=cold, portfolio="demo (seed 7)", python=sys.version.split()[0],
               cpu=__import__("platform").processor())
    RES_LATENCY.write_text(json.dumps(out, indent=1), encoding="utf-8", newline="\n")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "holdout"])
    ap.add_argument("--latency", action="store_true")
    ap.add_argument("--force-holdout-rerun", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args()
    if a.split == "dev":
        run_dev()
    elif a.split == "holdout":
        run_holdout(a.force_holdout_rerun, a.allow_dirty)
    elif a.latency:
        run_latency()
    else:
        ap.error("choose --split dev | --split holdout | --latency")
    from ml.eval.build_metrics import main as build_metrics
    build_metrics()


if __name__ == "__main__":
    main()
