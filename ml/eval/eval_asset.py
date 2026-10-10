"""Evaluate and train the asset check (CLIP zero-shot vs linear probe) -> ml/artifacts/eval_asset.json + the probe artifact.

    .venv/Scripts/python -m ml.eval.eval_asset

Data: the committed library (5 purposes x 20 + the "other" negatives; faces_test images are not used - they show people with
equipment, so their class is ambiguous). Stratified 5-fold cross-validation; the "matches the loan purpose" threshold tau is
chosen INSIDE every training fold (inner 5-fold out-of-fold probabilities, no peeking at the test fold). The binary task:
every image is paired with each of the 5 loan purposes; the pair is positive if the image shows that purpose.
The deployed probe is fitted on all images with tau from the all-data out-of-fold probabilities.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from ml.photo import asset, config, imageio
from ml.photo.transforms import TRANSFORMS

RES = config.ARTIFACTS / "eval_asset.json"
SEED, C, FOLDS = 0, 10.0, 5
CLS = list(asset.CLASSES)                         # 5 purposes + other
GRID = np.round(np.linspace(0.05, 0.95, 91), 2)
ROBUST = ("JPEG q=30", "resize 50%", "screenshot", "rotate 5 deg")


def library():
    rows = list(csv.DictReader(open(config.ASSETS / "manifest.csv", encoding="utf-8")))
    return [r for r in rows if r["purpose"] in CLS]


def fit(X, y):
    return LogisticRegression(C=C, max_iter=5000, class_weight="balanced").fit(X, y)


def probs_of(clf, X):
    out = np.zeros((len(X), len(CLS)))
    out[:, clf.classes_] = clf.predict_proba(X)
    return out


def expanded(P, y):
    """(scores, labels) of the binary task: image x each of the 5 purposes -> p(purpose), is it that purpose."""
    return np.concatenate([P[:, k] for k in range(5)]), np.concatenate([(y == k) for k in range(5)])


def prf(scores, labels, tau):
    pred = scores >= tau
    tp, fp, fn = int((pred & labels).sum()), int((pred & ~labels).sum()), int((~pred & labels).sum())
    return dict(tp=tp, fp=fp, fn=fn, precision=tp / max(1, tp + fp), recall=tp / max(1, tp + fn), n_pos=int(labels.sum()),
                n_neg=int((~labels).sum()), f1=2 * tp / max(1, 2 * tp + fp + fn))


def best_tau(scores, labels) -> float:
    """Threshold with the best F1 (ties -> the higher, more precise one)."""
    f = [prf(scores, labels, t)["f1"] for t in GRID]
    return float(GRID[max(range(len(GRID)), key=lambda i: (f[i], GRID[i]))])


def oof_probe_probs(X, y, seed=SEED):
    """Out-of-fold probe probabilities inside a training set (inner 5-fold)."""
    P = np.zeros((len(y), len(CLS)))
    for tr, va in StratifiedKFold(FOLDS, shuffle=True, random_state=seed).split(X, y):
        P[va] = probs_of(fit(X[tr], y[tr]), X[va])
    return P


def tau_probe(X, y):
    return best_tau(*expanded(oof_probe_probs(X, y), y))


def evaluate(am):
    rows = library()
    items = [(r["file"], r["purpose"]) for r in rows]
    y = np.array([CLS.index(p) for _, p in items])
    imgs = [imageio.decode((config.ASSETS / f).read_bytes()) for f, _ in items]
    X = np.stack([am.embed(im) for im in imgs])
    n = len(y)
    reserved = json.loads(config.INDEX_PATH.read_text(encoding="utf-8"))["reserved"]
    res = dict(n=n, counts={c: int((y == i).sum()) for i, c in enumerate(CLS)}, c=C, folds=FOLDS, seed=SEED)

    # ---------------- zero-shot on every image
    Pz = am.zero_shot_probs(X)
    zs_pred = Pz.argmax(1)
    res["zero_shot"] = dict(acc=float((zs_pred == y).mean()),
                            per_class={c: float((zs_pred[y == i] == i).mean()) for i, c in enumerate(CLS)})

    # ---------------- outer CV: probe, zero-shot with the same threshold protocol, robustness
    trans_emb = {}
    for t in ROBUST:
        rng = np.random.default_rng(1)
        trans_emb[t] = np.stack([am.embed(TRANSFORMS[t](im, rng)) for im in imgs])
    pred_probe = np.zeros(n, int)
    P_probe = np.zeros((n, len(CLS)))
    fold_tau_probe, fold_tau_zs = [], []
    bin_probe, bin_zs = [[], []], [[], []]
    rob = {t: dict(probe_correct=0, probe_same=0, zs_correct=0, zs_same=0) for t in ROBUST}
    for tr, te in StratifiedKFold(FOLDS, shuffle=True, random_state=SEED).split(X, y):
        clf = fit(X[tr], y[tr])
        tp_ = tau_probe(X[tr], y[tr])                                  # nested: only the training fold
        tz_ = best_tau(*expanded(Pz[tr], y[tr]))
        fold_tau_probe.append(tp_)
        fold_tau_zs.append(tz_)
        P = probs_of(clf, X[te])
        P_probe[te], pred_probe[te] = P, P.argmax(1)
        for store, probs, tau in ((bin_probe, P, tp_), (bin_zs, Pz[te], tz_)):
            s, l = expanded(probs, y[te])
            store[0].append(s >= tau)
            store[1].append(l)
        for t in ROBUST:
            Xt = trans_emb[t][te]
            pp, pz = probs_of(clf, Xt).argmax(1), am.zero_shot_probs(Xt).argmax(1)
            rob[t]["probe_correct"] += int((pp == y[te]).sum())
            rob[t]["probe_same"] += int((pp == P.argmax(1)).sum())
            rob[t]["zs_correct"] += int((pz == y[te]).sum())
            rob[t]["zs_same"] += int((pz == Pz[te].argmax(1)).sum())

    def binary(store):
        pred, lab = np.concatenate(store[0]), np.concatenate(store[1])
        out = prf(pred.astype(float), lab, 0.5)
        return out

    res["probe_cv"] = dict(acc=float((pred_probe == y).mean()),
                           per_class={c: float((pred_probe[y == i] == i).mean()) for i, c in enumerate(CLS)},
                           taus=fold_tau_probe, binary=binary(bin_probe))
    res["zero_shot"]["taus"] = fold_tau_zs
    res["zero_shot"]["binary"] = binary(bin_zs)
    res["robustness"] = {t: dict(n=n, **{k: v / n for k, v in d.items()}) for t, d in rob.items()}

    # ---------------- the 10 images left out of the portfolio index
    held = np.array([f in set(reserved) for f, _ in items])
    tr, te = np.where(~held)[0], np.where(held)[0]
    clf = fit(X[tr], y[tr])
    tau = tau_probe(X[tr], y[tr])
    tauz = best_tau(*expanded(Pz[tr], y[tr]))
    Ph = probs_of(clf, X[te])
    sp, lp = expanded(Ph, y[te])
    sz, lz = expanded(Pz[te], y[te])
    res["held_out"] = dict(n=int(len(te)), probe_acc=float((Ph.argmax(1) == y[te]).mean()),
                           zero_shot_acc=float((Pz[te].argmax(1) == y[te]).mean()), tau_probe=tau, tau_zero_shot=tauz,
                           probe_binary=prf(sp, lp, tau), zero_shot_binary=prf(sz, lz, tauz),
                           wrong=[dict(file=items[i][0], true=CLS[y[i]], probe=CLS[Ph[k].argmax()], zero_shot=CLS[Pz[i].argmax()])
                                  for k, i in enumerate(te) if Ph[k].argmax() != y[i] or Pz[i].argmax() != y[i]])

    # ---------------- false-match rate on random things (the "other" images) and on wrong-purpose assets
    # recompute with fold structure for the two groups
    fm = {"probe": [0, 0, 0, 0], "zero_shot": [0, 0, 0, 0]}                # [other fp, other n, wrong-asset fp, wrong-asset n]
    for tr, te in StratifiedKFold(FOLDS, shuffle=True, random_state=SEED).split(X, y):
        clf = fit(X[tr], y[tr])
        tp_, tz_ = tau_probe(X[tr], y[tr]), best_tau(*expanded(Pz[tr], y[tr]))
        for name, P, tau in (("probe", probs_of(clf, X[te]), tp_), ("zero_shot", Pz[te], tz_)):
            for local, i in enumerate(te):
                for k in range(5):
                    if y[i] == k:
                        continue
                    hit = P[local, k] >= tau
                    if y[i] == len(CLS) - 1:
                        fm[name][0] += hit
                        fm[name][1] += 1
                    else:
                        fm[name][2] += hit
                        fm[name][3] += 1
    res["false_match"] = {k: dict(other_rate=v[0] / v[1], other_pairs=v[1], wrong_asset_rate=v[2] / v[3], wrong_asset_pairs=v[3])
                          for k, v in fm.items()}
    res["misclassified"] = [dict(file=items[i][0], true=CLS[y[i]], probe=CLS[pred_probe[i]], score=round(float(P_probe[i].max()), 3),
                                 zero_shot=CLS[zs_pred[i]]) for i in range(n) if pred_probe[i] != y[i] or zs_pred[i] != y[i]]

    # ---------------- deployed probe: all images, tau from all-data out-of-fold probabilities
    tau_final = tau_probe(X, y)
    clf = fit(X, y)
    W = np.zeros((len(CLS), X.shape[1]))
    b = np.full(len(CLS), -50.0)
    W[clf.classes_], b[clf.classes_] = clf.coef_, clf.intercept_
    np.savez_compressed(asset.PROBE_NPZ, W=W.astype(np.float32), b=b.astype(np.float32))
    meta = dict(model=asset.MODEL_NAME, pretrained=asset.PRETRAINED, classes=CLS, tau=tau_final, C=C, n_train=n,
                counts=res["counts"], cv=dict(acc=res["probe_cv"]["acc"], binary=res["probe_cv"]["binary"], folds=FOLDS, seed=SEED),
                note="logistic regression on L2-normalised CLIP embeddings; tau chosen on all-data out-of-fold probabilities (best F1)")
    asset.PROBE_JSON.write_text(json.dumps(meta, indent=1), encoding="utf-8", newline="\n")
    res["tau_final"] = tau_final
    return res


def warmup_and_memory(runs=3):
    code = ("import time, psutil;t = time.perf_counter();from ml.photo.asset import get_asset_model;get_asset_model();"
            "print(time.perf_counter() - t, psutil.Process().memory_info().rss / 2**20)")
    out = []
    for _ in range(runs):
        o = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=config.ROOT).stdout.split()
        out.append(dict(seconds=float(o[0]), rss_mb=float(o[1])))
    return out


def full_warmup(runs=3):
    """api.warm_up() in fresh processes: portfolio + face detector + CLIP; peak RSS."""
    code = ("import time, psutil;t = time.perf_counter();from ml import api;api.warm_up();"
            "print(time.perf_counter() - t, psutil.Process().memory_info().rss / 2**20)")
    out = []
    for _ in range(runs):
        o = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=config.ROOT).stdout.split()
        out.append(dict(seconds=float(o[0]), rss_mb=float(o[1])))
    return out


def main():
    am = asset.get_asset_model()
    res = evaluate(am)
    res["clip_load"] = warmup_and_memory()
    res["warm_up_total"] = full_warmup()
    RES.write_text(json.dumps(res, indent=1), encoding="utf-8", newline="\n")
    print(json.dumps({k: res[k] for k in ("zero_shot", "probe_cv", "held_out", "false_match", "tau_final")}, indent=1)[:3000])


if __name__ == "__main__":
    main()
