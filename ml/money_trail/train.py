"""Train the money-trail model on the TRAIN split only.

    .venv/Scripts/python -m ml.money_trail.train

The portfolio is generated in memory (deterministic, seed 42). Writes
ml/artifacts/money_trail_v2.txt (LightGBM booster) and money_trail_v2.json (feature list, CV metrics).
Colours are NOT fixed here: they are assigned by queue rank on the served portfolio (ml/money_trail/colors.py). Positives = ALL fraud loans, including T3 loans whose resale has not happened
yet at the snapshot (generator convention). Only the train split is used here (dev / holdout are never touched).
"""
from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from ml.generator.build import build
from ml.generator.config import SPLITS

from .colors import rank_thresholds
from .features import FEATURE_COLUMNS, build_features
from .metrics import BUDGET, recall_at_budget
from .model import LGB_PARAMS, META_PATH, MODEL_PATH, MODEL_VERSION, SEED

N_FOLDS = 5


def load_split(name: str):
    """(features, y, supplier_id, tables) of a split, generated in memory."""
    t = build(SPLITS[name])
    X = build_features(t["loans"], t["transactions"], t["suppliers"], t["employees"])
    lab = t["labels"].set_index("loan_id").reindex(X.index)
    sup = t["loans"].set_index("loan_id")["supplier_id"].reindex(X.index)
    return X, lab["is_fraud"].to_numpy().astype(int), sup.to_numpy(), t


def fit(X: pd.DataFrame, y: np.ndarray) -> lgb.LGBMClassifier:
    spw = float((y == 0).sum() / max(1, (y == 1).sum()))
    clf = lgb.LGBMClassifier(**LGB_PARAMS, scale_pos_weight=spw)
    return clf.fit(X[FEATURE_COLUMNS].to_numpy(dtype=float), y)


def main():
    X, y, groups, _ = load_split("train")
    cv = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    for tr, va in cv.split(X, y, groups):
        assert not set(groups[tr]) & set(groups[va])          # grouped by declared supplier
        oof[va] = fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[va][FEATURE_COLUMNS].to_numpy(dtype=float))[:, 1]
    cv_metrics = dict(roc_auc=float(roc_auc_score(y, oof)),
                      pr_auc=float(average_precision_score(y, oof)),
                      recall_at_20=recall_at_budget(oof, y),
                      random_recall_at_20=BUDGET, pr_auc_random=float(y.mean()))
    oof_rank = rank_thresholds(oof)          # informational only: what the rank rule gives on train OOF scores

    final = fit(X, y)
    final.booster_.save_model(str(MODEL_PATH))
    meta = dict(
        model_version=MODEL_VERSION, features=FEATURE_COLUMNS,
        train_oof_rank_thresholds=oof_rank,
        cv=dict(folds=N_FOLDS, grouped_by="supplier_id", split="train", **cv_metrics),
        train=dict(rows=int(len(y)), fraud=int(y.sum()), seed=SPLITS["train"].seed),
        lgb_params={k: v for k, v in LGB_PARAMS.items()}, scale_pos_weight=float((y == 0).sum() / (y == 1).sum()),
        lightgbm_version=lgb.__version__,
    )
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    for path in (MODEL_PATH, META_PATH):          # LF line endings: same bytes (and sha256) on Windows and Linux
        path.write_bytes(path.read_bytes().replace(bytes([13, 10]), bytes([10])))
    print(f"CV ({N_FOLDS}-fold, grouped by supplier, train OOF): "
          + ", ".join(f"{k}={v:.3f}" for k, v in cv_metrics.items()))
    print(f"train OOF rank thresholds (informational): red>={oof_rank['red']:.4f} yellow>={oof_rank['yellow']:.4f}")
    print(f"saved {MODEL_PATH.name} ({MODEL_PATH.stat().st_size / 1024:.0f} KB) and {META_PATH.name}")


if __name__ == "__main__":
    main()
