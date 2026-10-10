"""Perceptual hashes (pure numpy / OpenCV, 64 bit each) and the duplicate distance.

Per image: pHash (DCT, 8x8 low frequencies vs median) and dHash (horizontal gradients), each of the whole image
('full') and of its central 80% ('ctr'), so that a cropped copy can still match. Which distance variant is used,
and its threshold, is chosen on a tuning half of the library in ml/eval/eval_photo.py (artifacts/photo_dup_v1.json).
"""
from __future__ import annotations

import cv2
import numpy as np

KEYS = ("p", "d", "pc", "dc")
VARIANTS = ("p", "pd", "pd_crop")


def _gray(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def _bits(b: np.ndarray) -> int:
    return int("".join("1" if x else "0" for x in b.ravel()), 2)


def phash(gray: np.ndarray) -> int:
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    low = cv2.dct(small)[:8, :8].ravel()
    return _bits(low > np.median(low[1:]))


def dhash(gray: np.ndarray) -> int:
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA).astype(np.int16)
    return _bits(small[:, 1:] > small[:, :-1])


def hash_image(rgb: np.ndarray) -> dict[str, int]:
    g = _gray(rgb)
    h, w = g.shape
    c = g[int(h * 0.1):int(h * 0.9), int(w * 0.1):int(w * 0.9)]
    return dict(p=phash(g), d=dhash(g), pc=phash(c), dc=dhash(c))


def to_json(h: dict[str, int]) -> dict[str, str]:
    return {k: f"{v:016x}" for k, v in h.items()}


def from_json(h: dict[str, str]) -> dict[str, int]:
    return {k: int(v, 16) for k, v in h.items()}


def _ham(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def distance(variant: str, a: dict[str, int], b: dict[str, int]) -> float:
    """Normalised distance 0..1 (0 = identical hashes)."""
    if variant == "p":
        return _ham(a["p"], b["p"]) / 64
    full = lambda x, y: (_ham(x["p"], y["p"]) + _ham(x["d"], y["d"])) / 128
    if variant == "pd":
        return full(a, b)
    if variant == "pd_crop":                 # crop tolerant: compare whole <-> central crop too
        ca = {"p": a["pc"], "d": a["dc"]}
        cb = {"p": b["pc"], "d": b["dc"]}
        return min(full(a, b), full(ca, b), full(a, cb), full(ca, cb))
    raise ValueError(variant)


def similarity(variant: str, a: dict[str, int], b: dict[str, int]) -> float:
    return 1.0 - distance(variant, a, b)
