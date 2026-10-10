"""Ranking metrics shared by the leakage check, training and evaluation."""
from __future__ import annotations

import numpy as np

BUDGET = 0.20


def recall_at_budget(score: np.ndarray, y: np.ndarray, budget: float = BUDGET) -> float:
    """Recall inside the top `budget` share of loans; ties at the cut-off are filled at random
    (expected value). NaN when there are no positives."""
    y = np.asarray(y)
    if y.sum() == 0:
        return float("nan")
    n = len(y)
    B = max(1, int(round(budget * n)))
    order = np.argsort(-score, kind="stable")
    s, yy = score[order], y[order]
    thr = s[B - 1]
    above = s > thr
    tied = s == thr
    take = min(B - int(above.sum()), int(tied.sum()))
    tp = yy[above].sum() + take * yy[tied].mean()
    return float(tp / y.sum())
