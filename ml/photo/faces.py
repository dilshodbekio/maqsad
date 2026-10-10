"""Face detection (OpenCV YuNet ONNX, Haar cascade fallback) and strong blur. In-memory only."""
from __future__ import annotations

import threading

import cv2
import numpy as np

from . import config

SCORE_THRESHOLD = 0.6
MIN_SIDE = 16                  # images thinner than this are not searched for faces
DETECT_SIDE = 960              # YuNet runs on a copy whose longest side is at most this (speed)
MAX_PASSES = 3                 # blur -> re-detect -> blur again until no detectable face is left


class FaceDetector:
    def __init__(self):
        self._lock = threading.Lock()
        self.backend = "haar"
        self._yunet = None
        if config.YUNET_PATH.exists():
            try:
                self._yunet = cv2.FaceDetectorYN.create(str(config.YUNET_PATH), "", (320, 320), SCORE_THRESHOLD, 0.3, 5000)
                self.backend = "yunet"
            except Exception:
                self._yunet = None
        self._haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def detect(self, rgb: np.ndarray) -> list[tuple[int, int, int, int]]:
        h, w = rgb.shape[:2]
        if min(h, w) < MIN_SIDE:                      # too small / too thin for any detector
            return []
        if self._yunet is not None:
            s = min(1.0, DETECT_SIDE / max(h, w))
            if min(h, w) * s < MIN_SIDE:               # a very wide / tall image would become too thin after scaling
                s = 1.0 if min(h, w) >= MIN_SIDE else s
            small = cv2.resize(rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else rgb
            bgr = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)
            try:
                with self._lock:
                    self._yunet.setInputSize((bgr.shape[1], bgr.shape[0]))
                    _, faces = self._yunet.detect(bgr)
            except cv2.error:                          # degenerate geometry the network cannot take: no faces found
                return []
            if faces is None:
                return []
            return [tuple(int(v / s) for v in f[:4]) for f in faces if np.isfinite(f[:4]).all()]
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        found = self._haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return [tuple(int(v) for v in f) for f in found]

    def detect_haar(self, rgb: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Second, independent detector (used to double-check that blurred output has no face left)."""
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        return [tuple(int(v) for v in f) for f in self._haar.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))]


def blur_boxes(rgb: np.ndarray, boxes, grow: float = 0.35) -> np.ndarray:
    """Heavy pixelation + blur of every box (grown by `grow` on each side). Returns a new array."""
    out = rgb.copy()
    H, W = out.shape[:2]
    for x, y, w, h in boxes:
        gx, gy = int(w * grow), int(h * grow)
        x0, y0, x1, y1 = max(0, x - gx), max(0, y - gy), min(W, x + w + gx), min(H, y + h + gy)
        if x1 <= x0 or y1 <= y0:
            continue
        roi = out[y0:y1, x0:x1]
        small = cv2.resize(roi, (max(2, (x1 - x0) // 14), max(2, (y1 - y0) // 14)), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (3, 3), 0)            # blur on the tiny version: cheap and just as anonymising
        out[y0:y1, x0:x1] = cv2.resize(small, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    return out


def blur_faces(detector: FaceDetector, rgb: np.ndarray) -> tuple[np.ndarray, int]:
    """(blurred image, number of faces found in the original). Re-detects on the result and blurs again."""
    boxes = detector.detect(rgb)
    n_faces, out, grow = len(boxes), rgb, 0.35
    for _ in range(MAX_PASSES):
        if not boxes:
            break
        out = blur_boxes(out, boxes, grow)
        boxes = detector.detect(out)
        grow += 0.25
    return out, n_faces
