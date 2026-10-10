"""check_photo: validate, hash, find duplicates, blur faces, geo / time checks. Everything in memory.

The uploaded original is never written anywhere (no temp files, no logging of pixels); only a perceptual hash is
kept, and only inside the caller's sandbox (in RAM). The result does NOT influence the money-trail risk.
"""
from __future__ import annotations

import threading
import time
import warnings
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from math import asin, cos, radians, sin, sqrt
from typing import Optional

from ml.errors import LoanNotFoundError
from ml.sandbox import validate_sandbox_id

from . import config, hashing, imageio
from .faces import FaceDetector, blur_faces
from .index import get_index

MAX_SANDBOXES, MAX_PER_SANDBOX = 1000, 200
_lock = threading.Lock()
# sandbox_id -> {"n": uploads so far, "entries": [{seq, loan_id, hash, geo_ok, distance_km, time_ok}]} (RAM only)
_sandboxes: "OrderedDict[str, dict]" = OrderedDict()


_init_lock = threading.Lock()
_detector_obj: Optional[FaceDetector] = None


def _detector() -> FaceDetector:
    """The shared face detector, created once (thread-safe)."""
    global _detector_obj
    if _detector_obj is None:
        with _init_lock:
            if _detector_obj is None:
                _detector_obj = FaceDetector()
    return _detector_obj


def reset_sandbox(sandbox_id: Optional[str]) -> None:
    """Forget every photo uploaded in this sandbox (idempotent). `None` clears ALL sandboxes.
    The portfolio index is never touched."""
    validate_sandbox_id(sandbox_id)
    with _lock:
        if sandbox_id is None:
            _sandboxes.clear()
        else:
            _sandboxes.pop(sandbox_id, None)


def _km(lat1, lon1, lat2, lon2) -> float:
    a = sin(radians(lat2 - lat1) / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * asin(sqrt(a))


def _utc_naive(dt: datetime) -> datetime:
    return dt.astimezone(timezone.utc).replace(tzinfo=None) if dt.tzinfo else dt


km_between = _km


def _geo(lat, lon, loan) -> tuple[Optional[bool], Optional[float]]:
    """(geo_ok, distance_km): (None, None) without coordinates, (False, None) for impossible ones."""
    if lat is None or lon is None:
        return None, None
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return False, None
    km = _km(lat, lon, float(loan["borrower_lat"]), float(loan["borrower_lon"]))
    return km <= config.GEO_MAX_KM, km


def _time_ok(captured_at, loan) -> Optional[bool]:
    if captured_at is None:
        return None
    t = _utc_naive(captured_at)
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    if t > now + timedelta(minutes=config.FUTURE_SKEW_MIN):
        return False
    return t >= loan["disbursed_at"].to_pydatetime()


def _asset_match(rgb, purpose) -> Optional[dict]:
    """CLIP + probe asset check, or None if the optional CLIP stack / weights are not available (warned once)."""
    global _asset_warned
    try:
        from .asset import get_asset_model
        return get_asset_model().asset_match(rgb, purpose)
    except (ImportError, OSError, RuntimeError) as e:
        if not _asset_warned:
            _asset_warned = True
            warnings.warn(f"asset check disabled: {type(e).__name__}: {e}")
        return None


_asset_warned = False


def _match_and_remember(sandbox_id, loan_id, h, geo_ok, distance_km, time_ok, asset=None) -> list[dict]:
    """ATOMIC (one critical section): the earlier uploads of this sandbox that duplicate `h`, then this upload is
    stored. Two concurrent uploads of the same photo therefore always see each other in exactly one direction."""
    idx = get_index()
    with _lock:
        sb = _sandboxes.setdefault(sandbox_id, dict(n=0, entries=[]))
        _sandboxes.move_to_end(sandbox_id)
        out = []
        for e in sb["entries"]:
            d = hashing.distance(idx.variant, h, e["hash"])
            if d <= idx.max_distance and e["loan_id"] != loan_id:
                out.append(dict(loan_id=e["loan_id"], similarity=round(1 - d, 4)))
        sb["n"] += 1
        sb["entries"].append(dict(seq=sb["n"], loan_id=loan_id, hash=h, geo_ok=geo_ok, distance_km=distance_km,
                                  time_ok=time_ok, asset=asset))
        del sb["entries"][:-MAX_PER_SANDBOX]
        while len(_sandboxes) > MAX_SANDBOXES:
            _sandboxes.popitem(last=False)
    return out


def sandbox_signals(sandbox_id: Optional[str]) -> dict[str, list[dict]]:
    """Fusion signals (ml/fusion.py) of everything uploaded in this sandbox: {loan_id: [signal, ...]}.
    Duplicates are computed now, against the portfolio index and against the sandbox's other uploads
    (a photo uploaded for two loans escalates both). Other sandboxes and portfolio mode are unaffected."""
    from ml.fusion import photo_signals
    if validate_sandbox_id(sandbox_id) is None:
        return {}
    with _lock:
        entries = list(_sandboxes.get(sandbox_id, {}).get("entries", ()))
    if not entries:
        return {}
    idx, out = get_index(), {}
    for e in entries:
        dups = idx.matches(e["hash"], exclude_loan=e["loan_id"])
        for o in entries:
            d = hashing.distance(idx.variant, e["hash"], o["hash"])
            if o is not e and o["loan_id"] != e["loan_id"] and d <= idx.max_distance:
                dups.append(dict(loan_id=o["loan_id"], similarity=round(1 - d, 4)))
        dups = sorted({d["loan_id"]: d for d in sorted(dups, key=lambda x: x["similarity"])}.values(),
                      key=lambda x: (-x["similarity"], x["loan_id"]))
        sig = photo_signals(ref=f"photo:sandbox:{e['loan_id']}:{e['seq']}", photo_label=f"sandbox upload #{e['seq']}",
                            duplicates=dups, geo_ok=e["geo_ok"], distance_km=e["distance_km"], time_ok=e["time_ok"],
                            asset_match=e.get("asset"), uploaded=True)
        if sig:
            out.setdefault(e["loan_id"], []).extend(sig)
    return out


def check_photo(loan_id: str, image_bytes: bytes, lat: Optional[float] = None, lon: Optional[float] = None,
                captured_at: Optional[datetime] = None, sandbox_id: Optional[str] = None) -> dict:
    from ml.money_trail.service import get_portfolio
    t0 = time.perf_counter()
    validate_sandbox_id(sandbox_id)                                  # InvalidRequestError before anything else
    P = get_portfolio()
    P._check(loan_id)
    loan = P.loans.loc[loan_id]
    rgb = imageio.decode(image_bytes)                                # InvalidImageError on bad input
    h = hashing.hash_image(rgb)                                      # hash of the original pixels, in memory

    geo_ok, km = _geo(lat, lon, loan)
    time_ok = _time_ok(captured_at, loan)
    blurred, n_faces = blur_faces(_detector(), rgb)
    jpeg = imageio.encode_jpeg(blurred)                              # re-encoding drops EXIF / metadata
    asset = _asset_match(rgb, str(loan["purpose"]))                  # on the (unblurred) pixels, in memory
    dups = get_index().matches(h, exclude_loan=loan_id)
    if sandbox_id is not None:
        dups = sorted(dups + _match_and_remember(sandbox_id, loan_id, h, geo_ok, km, time_ok, asset),
                      key=lambda x: (-x["similarity"], x["loan_id"]))
    return dict(
        asset_match=asset,                                           # {label, score, matches_purpose, label_uz} | None
        duplicate_of=dups,
        geo_ok=geo_ok, time_ok=time_ok,
        faces_blurred=n_faces > 0,                                   # True = faces were found and blurred
        faces_count=n_faces, blurred_image=jpeg,
        latency_ms=round((time.perf_counter() - t0) * 1000, 2))
