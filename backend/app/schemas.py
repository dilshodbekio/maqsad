"""Pydantic schemas — 1:1 with docs/api_contract.md (+ agreed changes in docs/api_contract_changes.md).

Top-level contract fields are strict.

Inner shapes (Loan, Transaction, Photo, Invoice, Evidence) are TEMPORARY (contract change #5):
they are permissive (extra fields allowed, most fields optional) and live only in this file.
When the ml generator's sample JSON arrives, its shape wins — change it here and nowhere else.
"""
from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

Color = Literal["green", "yellow", "red"]
Risk = Annotated[float, Field(ge=0.0, le=1.0)]
LoanId = Annotated[str, Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_\-]+$")]
LOAN_ID_PATTERN = r"^[A-Za-z0-9_\-]+$"


class Reason(BaseModel):
    code: str
    text_uz: str
    weight: float


# ---------- inner shapes (TEMPORARY, see module docstring) ----------

class _Flexible(BaseModel):
    model_config = ConfigDict(extra="allow")


class Loan(_Flexible):
    loan_id: str
    borrower_name_masked: Optional[str] = None
    purpose: Optional[str] = None
    amount_uzs: Optional[int] = None
    disbursed_at: Optional[str] = None
    region: Optional[str] = None
    branch_id: Optional[str] = None
    supplier_id: Optional[str] = None


class Transaction(_Flexible):
    txn_id: str
    ts: str
    type: str  # disbursement | cash_withdrawal | supplier_payment | p2p_out | p2p_in | external_transfer | income
    amount_uzs: int
    counterparty: Optional[str] = None


class Photo(_Flexible):
    photo_id: str
    url: Optional[str] = None
    captured_at: Optional[str] = None
    source: Optional[str] = None


class Invoice(_Flexible):
    supplier_id: Optional[str] = None
    supplier_name: Optional[str] = None
    oked_code: Optional[str] = None
    oked_matches_purpose: Optional[bool] = None
    amount_uzs: Optional[int] = None


class Evidence(_Flexible):
    ref: str
    kind: str  # txn | photo | invoice | graph
    summary_uz: Optional[str] = None
    data: Any = None


# ---------- endpoints ----------

class Health(BaseModel):
    ml_mode: Literal["real", "stub"]
    model_version: str


class QueueItem(BaseModel):
    loan_id: str
    borrower_name_masked: str
    purpose: str
    amount_uzs: int
    region: str
    risk: Risk
    color: Color
    top_reason: str


class LoanDetail(BaseModel):
    loan: Loan
    transactions: List[Transaction]
    photos: List[Photo]
    invoice: Optional[Invoice] = None
    risk: Risk
    color: Color
    reasons: List[Reason]
    evidence: List[Evidence]
    model_version: str


class ScoreRequest(BaseModel):
    loan_id: LoanId


class ScoreResponse(BaseModel):
    risk: Risk
    color: Color
    reasons: List[Reason]
    model_version: str
    latency_ms: float


class AssetMatch(BaseModel):
    label: str
    score: float


class Duplicate(BaseModel):
    loan_id: str
    similarity: float


class CheckPhotoFields(BaseModel):
    """Non-file fields of /check_photo (multipart form fields, primary; contract changes #2, #9)."""
    loan_id: LoanId
    lat: Optional[float] = Field(default=None, ge=-90, le=90)
    lon: Optional[float] = Field(default=None, ge=-180, le=180)
    captured_at: Optional[datetime] = None  # client time, only compared with server time


class CheckPhotoJSON(CheckPhotoFields):
    """Fallback JSON body for /check_photo: image as base64 (optionally a data: URL)."""
    image_base64: str = Field(min_length=1)


class CheckPhotoResponse(BaseModel):
    asset_match: AssetMatch
    duplicate_of: List[Duplicate]
    geo_ok: Optional[bool] = None  # None = no geolocation given
    time_ok: Optional[bool] = None  # None = no client time given
    faces_blurred: bool
    latency_ms: float


class DraftReportRequest(BaseModel):
    loan_id: LoanId


class Fact(BaseModel):
    claim: str
    evidence_ref: str
    unverified: bool = False  # contract change #1


class DraftReport(BaseModel):
    text_uz: str
    facts: List[Fact]
    model: str
    latency_ms: float
    cost_usd: Optional[float] = None  # None = price unknown (never guessed)


class DecisionRequest(BaseModel):
    loan_id: LoanId
    officer_id: str = Field(min_length=1, max_length=64)
    action: Literal["approve", "reject", "visit"]
    comment: str = Field(default="", max_length=2000)


class DecisionResponse(BaseModel):
    audit_id: int


class AuditEntry(BaseModel):
    ts: str
    actor: str
    action: str
    ai_recommendation: Optional[str] = None
    model_version: Optional[str] = None
    inputs_hash: str


class ResetResponse(BaseModel):
    ok: bool


class ErrorResponse(BaseModel):
    detail: str
