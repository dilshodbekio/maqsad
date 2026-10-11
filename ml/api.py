"""Public interface of the ML layer. The backend imports ONLY from here.

Contract: docs/api_contract.md. Signatures are stable; implementations are filled in
module by module. Until a function is implemented it raises NotImplementedError —
it never returns fake data. Wired: get_queue / get_loan / score_loan (money-trail model).
check_photo / reset_sandbox: photo module (incl. the CLIP asset check).
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from ml.errors import InvalidImageError, InvalidRequestError, LoanNotFoundError, ScenarioError  # noqa: F401  (re-exported for the backend)
from ml.money_trail.model import MODEL_VERSION
from ml.money_trail.service import get_portfolio


def get_queue(sandbox_id: Optional[str] = None) -> list[dict]:
    """Risk queue, sorted by priority descending (colour rank + risk).

    Each item: {loan_id, borrower_name_masked, purpose, amount_uzs, region,
                risk: float 0..1 (money-trail only, never changed by photos),
                color: "green"|"yellow"|"red" (after evidence fusion), top_reason: str (Uzbek),
                base_color (money-trail colour), priority (sort key = colour rank + risk),
                escalations: [{code, min_color, text_uz, evidence_refs, raises_color}]}
    Sorted by priority, descending. `sandbox_id` (optional): the view of that demo sandbox, including the
    escalations caused by photos uploaded there; other sandboxes and the default view are unaffected.

    The first call generates the demo portfolio (demo split, seed 7) in memory and scores it
    (~6-7 s with imports); later calls are served from memory. Prefer `warm_up()` at service start.
    """
    return get_portfolio().queue(sandbox_id)


def get_loan(loan_id: str, sandbox_id: Optional[str] = None) -> dict:
    """Full loan view.

    Returns: {loan, transactions[], photos[], invoice | None,
              risk, color (after fusion), base_color, priority, escalations[],
              reasons: [{code, text_uz, weight, evidence_refs}] (photo-policy reasons first: weight null, source "fusion_policy"),
              evidence: [{ref, kind: "txn"|"invoice"|"graph"|"photo", summary_uz, data}], model_version}
    `photos`: [{photo_id, path (repo-relative), purpose, captured_at, lat, lon,
    duplicate_of: [{loan_id, similarity}], source: "portfolio"}]
    for the subset of loans that have a photo, [] otherwise. Raises: LoanNotFoundError
    """
    return get_portfolio().loan(loan_id, sandbox_id)


def score_scenario(scenario: dict) -> dict:
    """Score a loan + transactions built by the user (a judge), live, with the same model and reasons as `score_loan`.

    Input: {purpose, amount_uzs, region?, snapshot_day,
            supplier: {mode: "existing", supplier_id} | {mode: "new", oked, registered_days_before},
            transactions: [{day, type, amount_uzs, counterparty_kind?}]}   (schema: docs/api_contract.md)
    Returns: the `score_loan` shape + `percentile` (0-100 within the demo portfolio) + `features` (computed values).
    The cached portfolio is never modified. Raises: ScenarioError (Uzbek message; backend -> HTTP 400)
    """
    from ml.money_trail.scenario import score_scenario as _score
    return _score(scenario)


def get_scenario_presets() -> list[dict]:
    """4 editable starting scenarios [{id, title_uz, description_uz, scenario}]. Inputs only: scores are computed live."""
    from ml.money_trail.scenario import get_scenario_presets as _presets
    return _presets()


def get_color_thresholds() -> dict:
    """Colour rule of the served portfolio: red = top 10% by risk, yellow = next 10%, green = rest.

    Returns {red, yellow (risk scores at those ranks), n_red, n_yellow (by risk), n_loans, rule} plus the effect of the
    photo policy on the default queue: n_escalated_to_red (loans raised to red), n_escalated_to_yellow (green -> yellow),
    n_red_total, n_yellow_total and `note` - the 20% budget is exceeded by exactly these escalations.
    """
    P = get_portfolio()
    c = P.fused_counts()
    note = (f"{P.thresholds['n_red']} red by risk + {c['n_escalated_to_red']} escalated to red by the photo policy = "
            f"{c['n_red_total']} red; {c['n_yellow_total']} yellow ({P.thresholds['n_yellow']} by risk, "
            f"{c['n_escalated_to_yellow']} green loans raised to yellow, the rest moved to red)")
    return {**P.thresholds, **c, "note": note}


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

    Returns: {asset_match: {label, score, matches_purpose, label_uz} | None   (None only if the CLIP stack is unavailable),
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


def warm_up() -> dict:
    """Load everything a first request would otherwise pay for: the demo portfolio (features, scores, queue), the face
    detector and CLIP (asset check). Call it once at service start. Returns the seconds each part took."""
    import time
    t0 = time.perf_counter()
    get_portfolio()
    t1 = time.perf_counter()
    from ml.photo.checker import _detector
    _detector()
    t2 = time.perf_counter()
    clip, clip_error = False, None
    try:
        from ml.photo.asset import get_asset_model
        get_asset_model()
        clip = True
    except (ImportError, OSError, RuntimeError) as e:        # cached + logged once; check_photo then runs without asset_match
        clip_error = str(e)
    t3 = time.perf_counter()
    return dict(portfolio_s=round(t1 - t0, 2), face_detector_s=round(t2 - t1, 2), clip_s=round(t3 - t2, 2),
                total_s=round(t3 - t0, 2), clip_loaded=clip, clip_error=clip_error)


def reset_sandbox(sandbox_id: Optional[str]) -> None:
    """Forget every photo hash uploaded in this demo sandbox (idempotent); `None` clears ALL sandboxes.
    The portfolio photo index is never touched. Contract addition."""
    from ml.photo.checker import reset_sandbox as _reset
    _reset(sandbox_id)
