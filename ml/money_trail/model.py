"""Loading and applying the trained money-trail model (artifacts in ml/artifacts/)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

MODEL_VERSION = "money_trail_v1"
ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
MODEL_PATH = ARTIFACTS / f"{MODEL_VERSION}.txt"
META_PATH = ARTIFACTS / f"{MODEL_VERSION}.json"
SEED = 42
# Modest, fixed hyper-parameters (no search). scale_pos_weight is set from the class ratio at fit time.
LGB_PARAMS = dict(
    objective="binary", n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=20,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
    random_state=SEED, n_jobs=1, deterministic=True, force_row_wise=True, verbose=-1,
)


@dataclass(frozen=True)
class MoneyTrailModel:
    booster: lgb.Booster
    features: list[str]
    red: float          # score >= red    -> "red"     (top 10% of train OOF scores)
    yellow: float       # score >= yellow -> "yellow"  (next 10%)
    meta: dict

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Risk score in 0..1 (model output; NOT a calibrated probability, classes are re-weighted)."""
        return self.booster.predict(X[self.features].to_numpy(dtype=float))

    def contributions(self, X: pd.DataFrame) -> np.ndarray:
        """Per-feature log-odds contributions (LightGBM pred_contrib), shape (n, n_features); bias dropped."""
        return self.booster.predict(X[self.features].to_numpy(dtype=float), pred_contrib=True)[:, :-1]

    def color(self, score: float) -> str:
        return "red" if score >= self.red else "yellow" if score >= self.yellow else "green"


def load_model(model_path: Path = MODEL_PATH, meta_path: Path = META_PATH) -> MoneyTrailModel:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return MoneyTrailModel(lgb.Booster(model_file=str(model_path)), meta["features"],
                           meta["thresholds"]["red"], meta["thresholds"]["yellow"], meta)
