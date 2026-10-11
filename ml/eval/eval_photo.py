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

from ml.generator.config import PURPOSES
from ml.photo import config, hashing, imageio
from ml.photo.transforms import TRANSFORMS

SEED = 20260410
MIN_PRECISION = 0.999
LATENCY_IMAGE = "tikuv_dastgohi_12.jpg"
GRID = [i / 128 for i in range(0, 65)]            # max normalised distance candidates


def library() -> list[dict]:
    import csv
    rows = list(csv.DictReader(open(config.ASSETS / "manifest.csv", encoding="utf-8")))
    return [r for r in rows if r["purpose"] in PURPOSES]       # the 5 asset purposes (not faces_test, not the CLIP "other" negatives)


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


def faces_eval() -> dict | None:
    """Detected vs hand-counted faces on the local-only faces_test images (ml/assets/faces_test/, git-ignored: images of real
    people are not redistributed); then the detectors are re-run on the blurred output. None if the folder is not there."""
    import csv
    from ml.photo.faces import FaceDetector, blur_faces
    counts = config.ASSETS.parent / "faces_test_manifest.csv"        # committed metadata: file, source URL, licence, hand count (no images)
    if not counts.exists() or not (config.FACES / "faces_test_00.jpg").exists():
        return None
    det = FaceDetector()
    truth = {r["file"]: (int(r["faces_visible"]), r["note"]) for r in csv.DictReader(open(counts, encoding="utf-8"))}
    rows = []
    for f, (n_true, note) in truth.items():
        rgb = imageio.decode((config.FACES / f).read_bytes())
        blurred, n_det = blur_faces(det, rgb)
        rows.append(dict(file=f, truth=n_true, detected=n_det, note=note,
                         left_yunet=len(det.detect(blurred)), left_haar=len(det.detect_haar(blurred)),
                         haar_before=len(det.detect_haar(rgb))))
    tp = sum(min(r["truth"], r["detected"]) for r in rows)
    return dict(rows=rows, backend=det.backend, truth=sum(r["truth"] for r in rows),
                detected=sum(r["detected"] for r in rows), tp=tp,
                fp=sum(max(0, r["detected"] - r["truth"]) for r in rows),
                fn=sum(max(0, r["truth"] - r["detected"]) for r in rows),
                left_yunet=sum(r["left_yunet"] for r in rows), left_haar=sum(r["left_haar"] for r in rows),
                haar_before=sum(r["haar_before"] for r in rows))


def latency_eval(n=500) -> dict:
    import time

    import cv2

    from ml import api
    api.get_queue()
    rgb = imageio.decode((config.ASSETS / LATENCY_IMAGE).read_bytes())          # a committed photo without people
    h, w = rgb.shape[:2]
    f = config.MAX_SIDE / max(h, w)
    big = cv2.resize(rgb, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    data = imageio.encode_jpeg(big)
    loan = str(api.get_queue()[0]["loan_id"])
    for _ in range(10):
        api.check_photo(loan, data)
    ms, wall = [], []
    for _ in range(n):
        t0 = time.perf_counter()
        r = api.check_photo(loan, data)
        wall.append((time.perf_counter() - t0) * 1000)
        ms.append(r["latency_ms"])
    q = lambda a, p: float(np.percentile(a, p))
    return dict(n=n, size=big.shape[:2], kb=len(data) / 1024, p50=q(ms, 50), p95=q(ms, 95),
                wall_p50=q(wall, 50), wall_p95=q(wall, 95))


def index_stats() -> dict:
    import csv
    from collections import Counter
    raw = json.loads(config.INDEX_PATH.read_text(encoding="utf-8"))
    c = Counter(i["purpose"] for i in raw["images"])
    cl = Counter(i["purpose"] for i in raw["images"] for _ in i["loans"])
    lib = Counter(r["purpose"] for r in library())
    return dict(stats=raw["stats"], per_purpose={p: (lib[p], c[p], cl[p]) for p in lib},
                size_kb=sum(p.stat().st_size for p in config.ASSETS.glob("*.jpg")) // 1024)


START, END = "<!-- photo:start -->", "<!-- photo:end -->"
RES_PHOTO = config.ARTIFACTS / "eval_photo.json"
SCRIPT = "`python -m ml.eval.eval_photo`"


def pct(x):
    return f"{100 * x:.1f}%"


def render(dup, faces, lat, idx) -> str:
    cfg = json.loads(config.DUP_PATH.read_text(encoding="utf-8"))
    v = cfg["variant"]
    L = []
    w = L.append
    w(START)
    w("## Photo module (openly licensed images)")
    w("")
    w("**All numbers are on a small library of openly licensed photos (CC / public domain) and are produced by "
      f"{SCRIPT}. They say nothing about real borrower photos; real accuracy is measured in the pilot.**")
    w("")
    w(f"- Library: {dup['n_images']} asset photos (Wikimedia Commons; `ml/assets/photos/manifest.csv`) "
      f"(the CLIP `other` negatives are not counted here); {idx['size_kb']} KB for all committed JPEGs in `ml/assets/photos/` including the `other` negatives. "
      "No committed image contains a recognisable face (faces were removed or pixelated); a few images show people from behind, far away or partially "
      "(`asalarichilik_03`, `asalarichilik_13`, `chorva_14`, `parranda_17`).")
    w("- Portfolio photo index (`ml/artifacts/photo_index_v1.json`): "
      f"{idx['stats']['images']} images on {idx['stats']['loans']} loans of the demo portfolio; "
      f"{idx['stats']['shared_images']} images are shared by 2-3 loans (the generator's photo-reuse groups). "
      "Photos are matched to the loan's purpose.")
    w("")
    w("| Purpose | library images | in index | loans with a photo |")
    w("|---|---|---|---|")
    for p, (a, b, c) in idx["per_purpose"].items():
        w(f"| {p} | {a} | {b} | {c} |")
    w("")
    w("### Duplicate detection under transformations")
    w("")
    w("Hashes: pHash + dHash (64 bit each) of the whole image and of its central 80% crop; distance = hamming/128, "
      "for the crop-tolerant variant the minimum over whole/central combinations. Positives = a transformed copy "
      "against its own original. Hard negatives = the same transformed copy against a *different* image of the same "
      "purpose (e.g. two different sewing machines). The threshold is the largest distance with pooled precision >= "
      f"{cfg['min_precision']} on the tuning half; numbers below are on the other half "
      f"({dup['n_test']} images; {dup['n_tune']} for tuning).")
    w("")
    w("| Variant | max distance (tuned on half A) | recall (held-out) | precision (held-out) | false positives |")
    w("|---|---|---|---|---|")
    for k, d in dup["variants"].items():
        p = d["pooled"]
        w(f"| {k} | {d['max_distance']:.3f} | {pct(p['recall'])} | {pct(p['precision'])} | {p['fp']} / {p['n_neg']} |")
    w("")
    d = dup["variants"][v]
    w(f"**Deployed: `{v}`, max distance {cfg['max_distance']:.4f} (similarity >= {cfg['similarity_threshold']:.4f}).** "
      f"The deployed variant and threshold were re-tuned on all {dup['n_images']} images (stricter threshold, "
      "more hard negatives) after the held-out numbers were fixed; the table above shows how the tune-on-one-half "
      "procedure generalises. With the deployed setting no two single-loan images of the portfolio index match each other. "
      f"Held-out per-transform numbers for the {v} variant (threshold tuned on half A):")
    w("")
    w("| Transform | positives | recall | negatives | false positives | precision |")
    w("|---|---|---|---|---|---|")
    for t, m in d["per"].items():
        w(f"| {t} | {m['n_pos']} | {pct(m['recall'])} | {m['n_neg']} | {m['fp']} | {pct(m['precision'])} |")
    w("")
    w("### Face blur (YuNet, OpenCV zoo, MIT; Haar cascade as fallback)")
    w("")
    w(f"Measured on {len(faces['rows'])} CC-licensed images containing people, counted by eye (faces that are very small, dark or out of focus are not counted). "
      "**The images are not redistributed, for privacy reasons.** To reproduce, download them from the URLs in `ml/assets/faces_test_manifest.csv` "
      "(title, author, licence, source URL and the hand count per image) into `ml/assets/faces_test/` (git-ignored) and run `ml/eval/eval_photo.py`; "
      "without the images the script keeps the numbers stored in `ml/artifacts/eval_photo.json` (measured once from these images).")
    w("")
    w("| Item | Value |")
    w("|---|---|")
    w(f"| Faces present (hand count) | {faces['truth']} |")
    w(f"| Faces detected by `{faces['backend']}` | {faces['detected']} (true {faces['tp']}, false {faces['fp']}, missed {faces['fn']}) |")
    w(f"| Face recall / precision (count based) | {pct(faces['tp'] / faces['truth'])} / {pct(faces['tp'] / faces['detected'])} |")
    w(f"| Detectable faces left after blur (YuNet re-run) | {faces['left_yunet']} |")
    w(f"| Detectable faces left after blur (independent Haar cascade; it found {faces['haar_before']} before blur) | {faces['left_haar']} |")
    w("")
    w("Detection is count based (no box annotations), so a miss and a false alarm on the same image can cancel out.")
    w("")
    w("### Latency")
    w("")
    w("| Item | Value |")
    w("|---|---|")
    w(f"| `check_photo` p50 / p95 ({lat['n']} calls, warm, {lat['size'][1]}x{lat['size'][0]} px, {lat['kb']:.0f} KB JPEG; "
      "committed photo without people; decode, hashing, duplicate search, face detection, CLIP asset check, re-encode) | "
      f"{lat['p50']:.0f} ms / {lat['p95']:.0f} ms |")
    w(f"| same, wall clock | {lat['wall_p50']:.0f} ms / {lat['wall_p95']:.0f} ms |")
    w("| Cost per check | 0 (local, no external API) |")
    w("")
    w("### Known failure cases")
    w("")
    w(f"- **Heavy crops:** crop 20% is found only {pct(d['per']['crop 20%']['recall'])} of the time, crop 10% "
      f"{pct(d['per']['crop 10%']['recall'])} (random crop position). Hashing is not crop-proof.")
    w("- **Different photo of the same thing** (the same cow from another angle, the same machine in another room) "
      "is not a duplicate by hash; recognising an individual animal or machine is out of scope.")
    w("- **Photo of a screen** is only covered by the simulated 'screenshot' transform (border + colour shift); moire "
      "from a real screen photo and perspective distortion were not tested.")
    w("- **Larger rotation / mirroring** were not tested (only 5 degrees).")
    w("- **Faces:** a decorated bull head was detected as a face once (`faces_test_04`); a dark, out-of-focus face in "
      "the background (`faces_test_08`) is not counted and was not detected, so it is not blurred. Side and tiny faces are the weak spot "
      "of any detector; the 'faces left' numbers above only show that *detectable* faces are gone.")
    w("- **Small library:** hard negatives come from ~10 images per purpose and half; with a real portfolio of "
      "thousands of photos the false-positive rate at this threshold must be re-measured.")
    w("- The photo result does not change the money-trail risk; it can only raise the queue colour via the fusion policy (section 5).")
    w(END)
    return "\n".join(L) + "\n"


def write_config():
    """Write ml/artifacts/photo_dup_v1.json from the evaluation (variant + threshold; then rebuild the index)."""
    dup = dup_eval()
    v = choose_variant(dup)
    d = dup["deploy"][v]
    config.DUP_PATH.write_text(json.dumps(dict(
        variant=v, max_distance=d["max_distance"], similarity_threshold=round(1 - d["max_distance"], 4),
        tuned_on_images=dup["n_images"], min_precision=MIN_PRECISION, seed=SEED,
        note="deployed variant/threshold tuned on all library images at pooled precision >= min_precision; "
             "held-out numbers in docs/metrics.md (Photo module)"), indent=1), encoding="utf-8", newline="\n")
    print("wrote", config.DUP_PATH, v, d)


def main():
    dup = dup_eval()
    cfg = json.loads(config.DUP_PATH.read_text(encoding="utf-8"))
    assert cfg["variant"] == choose_variant(dup) and abs(cfg["max_distance"] - dup["deploy"][cfg["variant"]]["max_distance"]) < 1e-9,         "ml/artifacts/photo_dup_v1.json is out of date - run `python -m ml.eval.eval_photo --write-config`"
    faces = faces_eval()
    if faces is None:
        old = json.loads(RES_PHOTO.read_text(encoding="utf-8")) if RES_PHOTO.exists() else {}
        faces = old.get("faces")
        print("faces_test images not found in ml/assets/faces_test/: face-blur part skipped, stored numbers kept" if faces else
              "faces_test images not found: face-blur part skipped (no stored numbers)")
    out = dict(dup=dup, faces=faces, lat=latency_eval(), idx=index_stats())
    RES_PHOTO.write_text(json.dumps(out, indent=1), encoding="utf-8")
    from ml.eval.build_metrics import main as build_metrics
    build_metrics()


if __name__ == "__main__":
    import sys
    write_config() if "--write-config" in sys.argv else main()
