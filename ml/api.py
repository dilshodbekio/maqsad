"""Public interface of the ML layer. The backend imports ONLY from here.

Contract: docs/api_contract.md. Signatures are stable; implementations are filled in
module by module. Until a function is implemented it raises NotImplementedError —
it never returns fake data. Wired: get_queue / get_loan / score_loan (money-trail model).
check_photo / reset_sandbox: photo module (no CLIP asset match yet).
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from ml.errors import InvalidImageError, LoanNotFoundError  # noqa: F401  (re-exported for the backend)
from ml.money_trail.model import MODEL_VERSION
from ml.money_trail.service import get_portfolio


def get_queue() -> list[dict]:
    """Risk queue, sorted by risk descending.

    Each item: {loan_id, borrower_name_masked, purpose, amount_uzs, region,
                risk: float 0..1, color: "green"|"yellow"|"red", top_reason: str (Uzbek)}

    The first call generates the demo portfolio (test-split config, seed 1337) in memory and scores it
    (~3 s); later calls are served from memory.
    """
    return get_portfolio().queue()


def get_loan(loan_id: str) -> dict:
    """Full loan view.

    Returns: {loan, transactions[], photos[], invoice | None,
              risk, color, reasons: [{code, text_uz, weight, evidence_refs}],
              evidence: [{ref, kind: "txn"|"invoice"|"graph", summary_uz, data}], model_version}
    `photos`: [{photo_id, path (repo-relative), purpose, captured_at, lat, lon, duplicate_of: [{loan_id, similarity}]}]
    for the subset of loans that have a photo, [] otherwise. Raises: LoanNotFoundError
    """
    return get_portfolio().loan(loan_id)


def score_loan(loan_id: str) -> dict:
    """Score one loan (live model inference on its cached features).

    Returns: {risk: float 0..1, color, reasons: [{code, text_uz, weight}],
              model_version, latency_ms}
    Raises: LoanNotFoundError
    """
    return get_portfolio().score(loan_id)


def check_photo(
    loan_id: str,
    image_bytes: bytes,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    captured_at: Optional[datetime] = None,
    sandbox_id: Optional[str] = None,
) -> dict:
    """Check a borrower's asset photo.

    The original image is never stored: faces are blurred, EXIF is dropped by re-encoding,
    and only the blurred JPEG + a perceptual hash are kept.

    sandbox_id: demo visitor's sandbox. Duplicate search covers the portfolio index
    + photos from THIS sandbox only. Sandbox photos are never added to the portfolio
    index and are dropped by reset_sandbox(). None = production/portfolio mode (nothing is remembered).

    Returns: {asset_match: {label, score} | None   (None until the CLIP step is added),
              duplicate_of: [{loan_id, similarity}],
              geo_ok: bool | None, time_ok: bool | None,
              faces_blurred: bool (True = faces were found and blurred), faces_count: int,
              blurred_image: bytes (JPEG, EXIF removed; internal - backend stores it,
                             not exposed over HTTP),
              latency_ms}
    Does not change the money-trail risk. Raises: LoanNotFoundError, InvalidImageError
    """
    from ml.photo.checker import check_photo as _check
    return _check(loan_id, image_bytes, lat, lon, captured_at, sandbox_id)


def reset_sandbox(sandbox_id: str) -> None:
    """Forget every photo hash uploaded in this demo sandbox (idempotent). Contract addition."""
    from ml.photo.checker import reset_sandbox as _reset
    _reset(sandbox_id)
