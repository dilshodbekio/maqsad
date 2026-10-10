"""Public interface of the ML layer. The backend imports ONLY from here.

Contract: docs/api_contract.md. Signatures are stable; implementations are filled in
module by module. Until a function is implemented it raises NotImplementedError —
it never returns fake data. Wired: get_queue / get_loan / score_loan (money-trail model).
Not wired yet: check_photo.
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
    `photos` is [] until the photo module is wired. Raises: LoanNotFoundError
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
