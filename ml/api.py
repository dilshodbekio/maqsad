"""Public interface of the ML layer. The backend imports ONLY from here.

Contract: docs/api_contract.md. Signatures are stable; implementations are filled in
module by module. Until a function is implemented it raises NotImplementedError —
it never returns fake data (the backend shows its own STUB banner meanwhile).
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

MODEL_VERSION = "not-ready"


class LoanNotFoundError(KeyError):
    """loan_id does not exist in the portfolio. Backend → HTTP 404."""


class InvalidImageError(ValueError):
    """Image is empty, corrupted, not an image, or too large. Backend → HTTP 400/413."""


def get_queue() -> list[dict]:
    """Risk queue, sorted by risk descending.

    Each item: {loan_id, borrower_name_masked, purpose, amount_uzs, region,
                risk: float 0..1, color: "green"|"yellow"|"red", top_reason: str (Uzbek)}
    """
    raise NotImplementedError("get_queue: money-trail model not wired yet")


def get_loan(loan_id: str) -> dict:
    """Full loan view.

    Returns: {loan, transactions[], photos[], invoice | None,
              risk, color, reasons: [{code, text_uz, weight}], evidence: [...], model_version}
    Raises: LoanNotFoundError
    """
    raise NotImplementedError("get_loan: not wired yet")


def score_loan(loan_id: str) -> dict:
    """Score one loan.

    Returns: {risk: float 0..1, color, reasons: [{code, text_uz, weight}],
              model_version, latency_ms}
    Raises: LoanNotFoundError
    """
    raise NotImplementedError("score_loan: money-trail model not wired yet")


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
    index and are dropped when the sandbox is reset. None = production/portfolio mode.

    Returns: {asset_match: {label, score} | None,
              duplicate_of: [{loan_id, similarity}],
              geo_ok: bool | None, time_ok: bool | None,
              faces_blurred: bool, faces_count: int,
              blurred_image: bytes (JPEG, EXIF removed; internal — backend stores it,
                             not exposed over HTTP),
              latency_ms}
    Raises: LoanNotFoundError, InvalidImageError
    """
    raise NotImplementedError("check_photo: photo module not wired yet")
