"""check_photo: validate, hash, find duplicates, blur faces, geo / time checks. Everything in memory.

The uploaded original is never written anywhere (no temp files, no logging of pixels); only a perceptual hash is
kept, and only inside the caller's sandbox (in RAM). The result does NOT influence the money-trail risk.
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from math import asin, cos, radians, sin, sqrt
from typing import Optional

from ml.errors import LoanNotFoundError

from . import config, hashing, imageio
from .faces import FaceDetector, blur_faces
from .index import get_index

MAX_SANDBOXES, MAX_PER_SANDBOX = 1000, 200
_lock = threading.Lock()
_sandboxes: "OrderedDict[str, list[dict]]" = OrderedDict()      # sandbox_id -> [{loan_id, hash}] (RAM only)


@lru_cache(maxsize=1)
def _detector() -> FaceDetector:
    return FaceDetector()


def reset_sandbox(sandbox_id: str) -> None:
    """Forget every photo uploaded in this sandbox (idempotent)."""
    with _lock:
        _sandboxes.pop(sandbox_id, None)


def _km(lat1, lon1, lat2, lon2) -> float:
    a = sin(radians(lat2 - lat1) / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * asin(sqrt(a))


def _utc_naive(dt: datetime) -> datetime:
    return dt.astimezone(timezone.utc).replace(tzinfo=None) if dt.tzinfo else dt


def _geo_ok(lat, lon, loan) -> Optional[bool]:
    if lat is None or lon is None:
        return None
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return False
    return _km(lat, lon, float(loan["borrower_lat"]), float(loan["borrower_lon"])) <= config.GEO_MAX_KM


def _time_ok(captured_at, loan) -> Optional[bool]:
    if captured_at is None:
        return None
    t = _utc_naive(captured_at)
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    if t > now + timedelta(minutes=config.FUTURE_SKEW_MIN):
        return False
    return t >= loan["disbursed_at"].to_pydatetime()


def _sandbox_matches(sandbox_id, h, loan_id) -> list[dict]:
    idx = get_index()
    with _lock:
        mine = list(_sandboxes.get(sandbox_id, ()))
    out = []
    for e in mine:
        d = hashing.distance(idx.variant, h, e["hash"])
        if d <= idx.max_distance and e["loan_id"] != loan_id:
            out.append(dict(loan_id=e["loan_id"], similarity=round(1 - d, 4)))
    return out


def _remember(sandbox_id, loan_id, h):
    with _lock:
        lst = _sandboxes.setdefault(sandbox_id, [])
        _sandboxes.move_to_end(sandbox_id)
        lst.append(dict(loan_id=loan_id, hash=h))
        del lst[:-MAX_PER_SANDBOX]
        while len(_sandboxes) > MAX_SANDBOXES:
            _sandboxes.popitem(last=False)


def check_photo(loan_id: str, image_bytes: bytes, lat: Optional[float] = None, lon: Optional[float] = None,
                captured_at: Optional[datetime] = None, sandbox_id: Optional[str] = None) -> dict:
    from ml.money_trail.service import get_portfolio
    t0 = time.perf_counter()
    P = get_portfolio()
    P._check(loan_id)
    loan = P.loans.loc[loan_id]
    rgb = imageio.decode(image_bytes)                                # InvalidImageError on bad input
    h = hashing.hash_image(rgb)                                      # hash of the original pixels, in memory

    dups = get_index().matches(h, exclude_loan=loan_id)
    if sandbox_id is not None:
        dups = sorted(dups + _sandbox_matches(sandbox_id, h, loan_id), key=lambda x: (-x["similarity"], x["loan_id"]))
        _remember(sandbox_id, loan_id, h)

    blurred, n_faces = blur_faces(_detector(), rgb)
    jpeg = imageio.encode_jpeg(blurred)                              # re-encoding drops EXIF / metadata
    return dict(
        asset_match=None,                                            # CLIP asset check comes later
        duplicate_of=dups,
        geo_ok=_geo_ok(lat, lon, loan), time_ok=_time_ok(captured_at, loan),
        faces_blurred=n_faces > 0,                                   # True = faces were found and blurred
        faces_count=n_faces, blurred_image=jpeg,
        latency_ms=round((time.perf_counter() - t0) * 1000, 2))
