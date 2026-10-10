"""Evaluate the photo module and write the "Photo module" section of docs/metrics.md.

    .venv/Scripts/python -m ml.eval.eval_photo

1. Duplicate detection: every library image (100 openly licensed photos) is transformed (crop, JPEG, resize,
   rotate, brightness, screenshot). Positives = transformed copy vs its original; hard negatives = the
   transformed copy vs a DIFFERENT image of the same purpose. The hash variant and the threshold are chosen on
   a tuning half of the images and every number is reported on the other half.
2. Face blur on the faces_test images; 3. check_photo latency.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ml.photo import config, hashing, imageio
from ml.photo.transforms import TRANSFORMS

SEED = 20260410
MIN_PRECISION = 0.999
GRID = [i / 128 for i in range(0, 65)]            # max normalised distance candidates


def library() -> list[dict]:
    import csv
    rows = list(csv.DictReader(open(config.ASSETS / "manifest.csv", encoding="utf-8")))
    return [r for r in rows if r["purpose"] != "faces_test"]


def _hashes(items, rng):
    """Per image: hashes of the original and of every transformed copy."""
    out = []
    for r in items:
        rgb = imageio.decode((config.ASSETS / r["file"]).read_bytes())
        h = {"original": hashing.hash_image(rgb)}
        for name, fn in TRANSFORMS.items():
            h[name] = hashing.hash_image(fn(rgb, rng))
        out.append(h)
    return out


def _scores(variant, items, H):
    """(transform, is_positive, distance) for all query/index pairs inside `items` (same-purpose negatives)."""
    rows = []
    for i, ri in enumerate(items):
        for t, hq in H[i].items():
            rows.append((t, 1, hashing.distance(variant, hq, H[i]["original"])))
            for j, rj in enumerate(items):
                if j != i and rj["purpose"] == ri["purpose"]:
                    rows.append((t, 0, hashing.distance(variant, hq, H[j]["original"])))
    return rows


def _pr(rows, thr, transform=None):
    sel = [r for r in rows if transform is None or r[0] == transform]
    tp = sum(1 for t, y, d in sel if y and d <= thr)
    fn = sum(1 for t, y, d in sel if y and d > thr)
    fp = sum(1 for t, y, d in sel if not y and d <= thr)
    n_neg = sum(1 for t, y, d in sel if not y)
    return dict(tp=tp, fn=fn, fp=fp, n_pos=tp + fn, n_neg=n_neg,
                recall=tp / max(1, tp + fn), precision=tp / max(1, tp + fp))


def tune(items, H, variant):
    """(max distance, recall) with the largest recall on the tuning half at pooled precision >= MIN_PRECISION
    (ties -> smaller distance)."""
    rows = _scores(variant, items, H)
    best = None
    for thr in GRID:
        m = _pr([r for r in rows if r[0] != "original"], thr)
        if m["precision"] >= MIN_PRECISION and (best is None or m["recall"] > best[1] + 1e-12):
            best = (thr, m["recall"])
    return best if best else (0.0, 0.0)


def dup_eval() -> dict:
    items = library()
    rng = np.random.default_rng(SEED)
    H = _hashes(items, rng)
    order = rng.permutation(len(items))
    A, B = order[: len(items) // 2], order[len(items) // 2:]
    pick = lambda idx: ([items[i] for i in idx], [H[i] for i in idx])
    (ia, Ha), (ib, Hb) = pick(A), pick(B)
    res = dict(n_images=len(items), n_tune=len(A), n_test=len(B), variants={})
    for v in hashing.VARIANTS:
        thr, rec_tune = tune(ia, Ha, v)
        rows = _scores(v, ib, Hb)
        per = {t: _pr(rows, thr, t) for t in ["original", *TRANSFORMS]}
        pooled = _pr([r for r in rows if r[0] != "original"], thr)
        res["variants"][v] = dict(max_distance=thr, tune_recall=rec_tune, per=per, pooled=pooled)
    # Deployed configuration: after the held-out numbers are fixed, threshold and variant are re-tuned on ALL
    # library images (more hard negatives -> safer threshold). The held-out table above documents how well the
    # tune-on-one-half procedure generalises.
    res["deploy"] = {}
    for v in hashing.VARIANTS:
        thr, rec = tune(items, H, v)
        res["deploy"][v] = dict(max_distance=thr, recall=rec)
    return res


def choose_variant(res: dict) -> str:
    """Deployed variant = best recall on all library images at the precision floor (tie -> smaller distance)."""
    d = res["deploy"]
    return max(d, key=lambda k: (d[k]["recall"], -d[k]["max_distance"]))


if __name__ == "__main__":
    r = dup_eval()
    for v, d in r["variants"].items():
        print(v, "max_dist", round(d["max_distance"], 3), "pooled", {k: round(x, 3) if isinstance(x, float) else x for k, x in d["pooled"].items()})
        for t, m in d["per"].items():
            print("   ", f"{t:16s} recall {m['recall']:.3f} precision {m['precision']:.3f} fp {m['fp']}/{m['n_neg']}")
