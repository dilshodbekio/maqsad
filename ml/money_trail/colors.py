"""Colour by queue rank: red = top 10% of the portfolio by risk, yellow = next 10%, green = the rest.

So red + yellow is exactly the 20% inspection budget of the portfolio being served. The thresholds are the risk
scores at those ranks, computed on that portfolio (and stored with the results), not fixed numbers from training.
"""
from __future__ import annotations

import math

import numpy as np

RED_SHARE, YELLOW_SHARE = 0.10, 0.10
RULE = "red = top 10% of the portfolio by risk, yellow = next 10%, green = rest (red + yellow = 20% inspection budget)"


def rank_thresholds(scores) -> dict:
    s = np.sort(np.asarray(scores, dtype=float))[::-1]
    n_red = max(1, math.ceil(RED_SHARE * len(s)))
    n_yellow = max(1, math.ceil(YELLOW_SHARE * len(s)))
    return dict(red=float(s[n_red - 1]), yellow=float(s[n_red + n_yellow - 1]), n_red=n_red, n_yellow=n_yellow,
                n_loans=len(s), rule=RULE)


def color_of(score: float, thr: dict) -> str:
    return "red" if score >= thr["red"] else "yellow" if score >= thr["yellow"] else "green"
