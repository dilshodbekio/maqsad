"""Evaluate and train the asset check (CLIP zero-shot vs linear probe) -> ml/artifacts/eval_asset.json + the probe artifact.

    .venv/Scripts/python -m ml.eval.eval_asset

Data: the committed library (5 purposes x 20 + the "other" negatives; faces_test images are not used - they show people with
equipment, so their class is ambiguous).

The library contains series of photos of one object shot by the same author in one session (near-duplicates for CLIP). Plain
k-fold would put siblings of a test image into the training set, so **every split here is grouped by author**
(`manifest.csv` author = group): outer CV, the inner CV that chooses the threshold, the all-data out-of-fold threshold and the
"held-out" images. The CV is repeated with 5 seeds and reported as mean and range.

Protocol: stratified-grouped 5-fold; the "matches the loan purpose" threshold tau is chosen INSIDE every training fold (inner
5-fold out-of-fold probabilities, best F1), never on the test fold. The binary task pairs every image with each of the 5 loan
purposes (positive = the image shows that purpose). C = 10 and the choice "probe vs zero-shot" were made on this same CV (there is
no untouched test set; the numbers are optimistic by that amount). The deployed probe is fitted on all images with tau from the
all-data grouped out-of-fold probabilities.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

from ml.photo import asset, config, imageio
from ml.photo.transforms import TRANSFORMS

RES = config.ARTIFACTS / "eval_asset.json"
SEEDS, C, FOLDS = (0, 1, 2, 3, 4), 10.0, 5
CLS = list(asset.CLASSES)                         # 5 purposes + other
GRID = np.round(np.linspace(0.05, 0.95, 91), 2)
ROBUST = ("JPEG q=30", "resize 50%", "screenshot", "rotate 5 deg")


def library():
    rows = list(csv.DictReader(open(config.ASSETS / "manifest.csv", encoding="utf-8")))
    return [r for r in rows if r["purpose"] in CLS]


def group_of(r) -> str:
    """Photos of one author are one group (series of the same object / session)."""
    return (r["author"] or r["file"]).strip().lower()


def splitter(seed):
    return StratifiedGroupKFold(FOLDS, shuffle=True, random_state=seed)


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


def oof_probe_probs(X, y, g, seed):
    """Out-of-fold probe probabilities inside a set (inner grouped 5-fold)."""
    P = np.zeros((len(y), len(CLS)))
    for tr, va in splitter(seed).split(X, y, g):
        P[va] = probs_of(fit(X[tr], y[tr]), X[va])
    return P


def tau_probe(X, y, g, seed):
    return best_tau(*expanded(oof_probe_probs(X, y, g, seed), y))


def cv_once(seed, X, y, g, Pz, trans_emb):
    n = len(y)
    pred = np.zeros(n, int)
    P_all = np.zeros((n, len(CLS)))
    taus_p, taus_z = [], []
    bin_p, bin_z = [[], []], [[], []]
    fm = {"probe": [0, 0, 0, 0], "zero_shot": [0, 0, 0, 0]}        # [other fp, other n, wrong-asset fp, wrong-asset n]
    rob = {t: dict(probe_correct=0, probe_same=0, zs_correct=0, zs_same=0) for t in ROBUST}
    for tr, te in splitter(seed).split(X, y, g):
        assert not set(g[tr]) & set(g[te])                          # no author in both train and test
        clf = fit(X[tr], y[tr])
        tp_ = tau_probe(X[tr], y[tr], g[tr], seed)                  # nested: training fold only
        tz_ = best_tau(*expanded(Pz[tr], y[tr]))
        taus_p.append(tp_)
        taus_z.append(tz_)
        P = probs_of(clf, X[te])
        P_all[te], pred[te] = P, P.argmax(1)
        for store, probs, tau in ((bin_p, P, tp_), (bin_z, Pz[te], tz_)):
            s, l = expanded(probs, y[te])
            store[0].append(s >= tau)
            store[1].append(l)
        for name, probs, tau in (("probe", P, tp_), ("zero_shot", Pz[te], tz_)):
            for local, i in enumerate(te):
                for k in range(5):
                    if y[i] == k:
                        continue
                    hit = probs[local, k] >= tau
                    if y[i] == len(CLS) - 1:
                        fm[name][0] += hit
                        fm[name][1] += 1
                    else:
                        fm[name][2] += hit
                        fm[name][3] += 1
        for t in ROBUST:
            Xt = trans_emb[t][te]
            pp, pz = probs_of(clf, Xt).argmax(1), asset_zs(Xt).argmax(1)
            rob[t]["probe_correct"] += int((pp == y[te]).sum())
            rob[t]["probe_same"] += int((pp == P.argmax(1)).sum())
            rob[t]["zs_correct"] += int((pz == y[te]).sum())
            rob[t]["zs_same"] += int((pz == Pz[te].argmax(1)).sum())

    def binary(store):
        return prf(np.concatenate(store[0]).astype(float), np.concatenate(store[1]), 0.5)

    return dict(seed=seed, pred=pred, P=P_all, acc=float((pred == y).mean()),
                per_class={c: float((pred[y == i] == i).mean()) for i, c in enumerate(CLS)},
                taus_probe=taus_p, taus_zs=taus_z, probe_binary=binary(bin_p), zs_binary=binary(bin_z),
                false_match={k: dict(other_rate=v[0] / v[1], other_pairs=v[1], wrong_asset_rate=v[2] / v[3], wrong_asset_pairs=v[3])
                             for k, v in fm.items()}, robustness=rob)


_ZS = None


def asset_zs(X):
    return _ZS(X)


def spread(vals):
    return dict(mean=float(np.mean(vals)), min=float(np.min(vals)), max=float(np.max(vals)))


def evaluate(am):
    global _ZS
    _ZS = am.zero_shot_probs
    rows = library()
    items = [(r["file"], r["purpose"]) for r in rows]
    y = np.array([CLS.index(p) for _, p in items])
    g = np.array([group_of(r) for r in rows])
    imgs = [imageio.decode((config.ASSETS / f).read_bytes()) for f, _ in items]
    X = np.stack([am.embed(im) for im in imgs])
    n = len(y)
    reserved = set(json.loads(config.INDEX_PATH.read_text(encoding="utf-8"))["reserved"])
    res = dict(n=n, counts={c: int((y == i).sum()) for i, c in enumerate(CLS)}, c=C, folds=FOLDS, seeds=list(SEEDS),
               n_groups=int(len(set(g))), group_rule="author in manifest.csv")
    Pz = am.zero_shot_probs(X)
    zs_pred = Pz.argmax(1)
    trans_emb = {}
    for t in ROBUST:
        rng = np.random.default_rng(1)
        trans_emb[t] = np.stack([am.embed(TRANSFORMS[t](im, rng)) for im in imgs])

    runs = [cv_once(s, X, y, g, Pz, trans_emb) for s in SEEDS]
    first = runs[0]
    res["zero_shot"] = dict(acc=float((zs_pred == y).mean()),
                            per_class={c: float((zs_pred[y == i] == i).mean()) for i, c in enumerate(CLS)},
                            binary={k: spread([r["zs_binary"][k] for r in runs]) for k in ("precision", "recall", "f1")},
                            binary_seed0=first["zs_binary"], taus_seed0=first["taus_zs"])
    res["probe_cv"] = dict(acc=spread([r["acc"] for r in runs]),
                           per_class={c: float(np.mean([r["per_class"][c] for r in runs])) for c in CLS},
                           binary={k: spread([r["probe_binary"][k] for r in runs]) for k in ("precision", "recall", "f1")},
                           binary_seed0=first["probe_binary"], taus_seed0=first["taus_probe"],
                           per_seed=[dict(seed=r["seed"], acc=r["acc"], **{k: r["probe_binary"][k] for k in ("precision", "recall", "f1", "tp", "fp", "fn")})
                                     for r in runs])
    res["false_match"] = {m: {k: float(np.mean([r["false_match"][m][k] for r in runs])) for k in ("other_rate", "wrong_asset_rate")}
                          | {k: runs[0]["false_match"][m][k] for k in ("other_pairs", "wrong_asset_pairs")} for m in ("probe", "zero_shot")}
    res["robustness"] = {t: dict(n=n, **{k: v / n for k, v in first["robustness"][t].items()}) for t in ROBUST}
    res["misclassified"] = [dict(file=items[i][0], true=CLS[y[i]], probe=CLS[first["pred"][i]], score=round(float(first["P"][i].max()), 3),
                                 zero_shot=CLS[zs_pred[i]]) for i in range(n) if first["pred"][i] != y[i] or zs_pred[i] != y[i]]

    # ---------------- the images left out of the portfolio index: train only on authors that do not appear among them
    held = np.array([f in reserved for f, _ in items])
    held_groups = set(g[held])
    tr = np.where(~held & ~np.isin(g, list(held_groups)))[0]
    te = np.where(held)[0]
    clf = fit(X[tr], y[tr])
    tau = tau_probe(X[tr], y[tr], g[tr], 0)
    tauz = best_tau(*expanded(Pz[tr], y[tr]))
    Ph = probs_of(clf, X[te])
    sp, lp = expanded(Ph, y[te])
    sz, lz = expanded(Pz[te], y[te])
    res["held_out"] = dict(n=int(len(te)), n_train=int(len(tr)), n_dropped_same_author=int((~held & np.isin(g, list(held_groups))).sum()),
                           probe_acc=float((Ph.argmax(1) == y[te]).mean()), zero_shot_acc=float((Pz[te].argmax(1) == y[te]).mean()),
                           tau_probe=tau, tau_zero_shot=tauz, probe_binary=prf(sp, lp, tau), zero_shot_binary=prf(sz, lz, tauz),
                           wrong=[dict(file=items[i][0], true=CLS[y[i]], probe=CLS[Ph[k].argmax()], zero_shot=CLS[Pz[i].argmax()])
                                  for k, i in enumerate(te) if Ph[k].argmax() != y[i] or Pz[i].argmax() != y[i]])

    # ---------------- deployed probe: all images, tau from all-data grouped out-of-fold probabilities
    tau_final = tau_probe(X, y, g, 0)
    clf = fit(X, y)
    W = np.zeros((len(CLS), X.shape[1]))
    b = np.full(len(CLS), -50.0)
    W[clf.classes_], b[clf.classes_] = clf.coef_, clf.intercept_
    np.savez_compressed(asset.PROBE_NPZ, W=W.astype(np.float32), b=b.astype(np.float32))
    meta = dict(model=asset.MODEL_NAME, pretrained=asset.PRETRAINED, classes=CLS, tau=tau_final, C=C, n_train=n,
                counts=res["counts"], cv=dict(acc=res["probe_cv"]["acc"], binary=res["probe_cv"]["binary"], folds=FOLDS, seeds=list(SEEDS),
                                              grouped_by="author"),
                note="logistic regression on L2-normalised CLIP embeddings; tau = best F1 on all-data grouped out-of-fold probabilities")
    asset.PROBE_JSON.write_text(json.dumps(meta, indent=1), encoding="utf-8", newline="\n")
    res["tau_final"] = tau_final
    return res


def _subprocess_runs(code, runs=3):
    out = []
    for _ in range(runs):
        o = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=config.ROOT).stdout.split()
        out.append(dict(seconds=float(o[0]), rss_mb=float(o[1])))
    return out


def warmup_and_memory(runs=3):
    return _subprocess_runs("import time, psutil;t = time.perf_counter();from ml.photo.asset import get_asset_model;get_asset_model();"
                            "print(time.perf_counter() - t, psutil.Process().memory_info().rss / 2**20)", runs)


def full_warmup(runs=3):
    """api.warm_up() in fresh processes: portfolio + face detector + CLIP; RSS after it."""
    return _subprocess_runs("import time, psutil;t = time.perf_counter();from ml import api;api.warm_up();"
                            "print(time.perf_counter() - t, psutil.Process().memory_info().rss / 2**20)", runs)


def main():
    am = asset.get_asset_model()
    res = evaluate(am)
    res["clip_load"] = warmup_and_memory()
    res["warm_up_total"] = full_warmup()
    RES.write_text(json.dumps(res, indent=1), encoding="utf-8", newline="\n")
    print(json.dumps({k: res[k] for k in ("n_groups", "zero_shot", "held_out", "false_match", "tau_final")}, indent=1)[:2500])
    print(json.dumps(res["probe_cv"]["per_seed"], indent=1))


if __name__ == "__main__":
    main()
