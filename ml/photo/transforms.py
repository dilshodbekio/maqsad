"""Image transforms used to test duplicate detection (evaluation and tests). All take/return RGB uint8."""
from __future__ import annotations

import cv2
import numpy as np


def crop(rgb, frac, rng):
    """Keep (1 - frac) of each side at a random position (NOT the centre)."""
    h, w = rgb.shape[:2]
    ch, cw = int(h * (1 - frac)), int(w * (1 - frac))
    y, x = int(rng.integers(0, h - ch + 1)), int(rng.integers(0, w - cw + 1))
    return rgb[y:y + ch, x:x + cw].copy()


def jpeg(rgb, q):
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def resize(rgb, f):
    return cv2.resize(rgb, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)


def rotate(rgb, deg):
    h, w = rgb.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return cv2.warpAffine(rgb, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def brightness(rgb, f):
    return np.clip(rgb.astype(np.float32) * f, 0, 255).astype(np.uint8)


def screenshot(rgb):
    """Photo shown on a screen and captured: downscale, UI-like bars and borders, colour shift, JPEG."""
    im = resize(rgb, 0.6)
    h, w = im.shape[:2]
    bar, side = max(12, int(h * 0.08)), max(4, int(w * 0.03))
    out = np.full((h + bar + 6, w + 2 * side, 3), 242, np.uint8)
    out[bar:bar + h, side:side + w] = im
    out[:bar] = (225, 228, 232)
    for k in range(3):                                          # fake title / buttons
        x0 = side + 10 + k * 60
        cv2.rectangle(out, (x0, bar // 3), (x0 + 40, 2 * bar // 3), (110, 110, 120), -1)
    out[-6:] = (200, 200, 205)
    shift = np.array([0.96, 1.0, 1.05], np.float32)
    out = np.clip(out.astype(np.float32) * shift, 0, 255).astype(np.uint8)
    return jpeg(out, 70)


TRANSFORMS = {
    "crop 10%": lambda im, rng: crop(im, 0.10, rng),
    "crop 20%": lambda im, rng: crop(im, 0.20, rng),
    "JPEG q=30": lambda im, rng: jpeg(im, 30),
    "resize 50%": lambda im, rng: resize(im, 0.5),
    "rotate 5 deg": lambda im, rng: rotate(im, 5.0 if rng.random() < 0.5 else -5.0),
    "brightness +30%": lambda im, rng: brightness(im, 1.3),
    "brightness -30%": lambda im, rng: brightness(im, 0.7),
    "screenshot": lambda im, rng: screenshot(im),
}
