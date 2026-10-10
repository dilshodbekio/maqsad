"""Latency of scenario scoring (3 runs) and the fast-cash-out example (day 1 vs day 30) -> ml/artifacts/eval_scenario.json.

    .venv/Scripts/python -m ml.eval.eval_scenario

Uses the demo portfolio only (never a metrics split). Scores are always computed live by `ml.api.score_scenario`.
"""
from __future__ import annotations

import copy
import json
import platform
import sys

import numpy as np

from ml import api
from ml.money_trail.model import ARTIFACTS

RES = ARTIFACTS / "eval_scenario.json"
N_PER_PRESET, RUNS = 15, 3


def main():
    presets = api.get_scenario_presets()
    for p in presets:
        api.score_scenario(p["scenario"])                                  # warm
    runs = []
    for r in range(RUNS):
        ms = []
        for p in presets:
            scn = copy.deepcopy(p["scenario"])
            for k in range(N_PER_PRESET):
                if scn["transactions"]:                                    # vary an input so no call repeats the previous one
                    scn["transactions"][0]["day"] = int(min(scn["snapshot_day"], 1 + (k + r) % 5))
                ms.append(api.score_scenario(scn)["latency_ms"])
        runs.append(dict(p50=float(np.percentile(ms, 50)), p95=float(np.percentile(ms, 95)), max=float(max(ms)), n=len(ms)))
    fast = next(p for p in presets if p["id"] == "fast_cash_out")["scenario"]
    example = []
    for day in (1, 30):
        s = copy.deepcopy(fast)
        next(t for t in s["transactions"] if t["type"] == "cash_withdrawal")["day"] = day
        r = api.score_scenario(s)
        example.append(dict(cash_day=day, risk=r["risk"], color=r["color"], percentile=r["percentile"],
                            top_reason=r["reasons"][0]["text_uz"] if r["reasons"] else None,
                            cash_share_d1=r["features"]["cash_share_d1"], first_cash_days=r["features"]["first_cash_days"]))
    out = dict(runs=runs, example=example, presets=[p["id"] for p in presets], python=sys.version.split()[0],
               cpu=platform.processor())
    RES.write_text(json.dumps(out, indent=1), encoding="utf-8", newline="\n")
    from ml.eval.build_metrics import main as build_metrics
    build_metrics()
    for e in example:
        print(e)


if __name__ == "__main__":
    main()
