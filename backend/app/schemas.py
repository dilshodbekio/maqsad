"""Pydantic schemas — 1:1 with docs/api_contract.md (v1.4).

Inner shapes (Loan, Transaction, Photo, Invoice, Evidence, LoanReason) match the REAL ml.api output exactly
(get_loan / check_photo of ml money_trail_v1 + photo module). Unknown extra keys from ml are dropped
(pydantic default); backend/tests/test_ml_integration.py fails if ml starts returning keys not declared here,
so a shape change is noticed instead of silently lost.
"""
from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

Color = Literal["green", "yellow", "red"]
# risk is a model SCORE in 0..1 (ranking), NOT a probability — UI shows it as "xavf bahosi", never as %
Risk = Annotated[float, Field(ge=0.0, le=1.0)]
LoanId = Annotated[str, Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_\-]+$")]
LOAN_ID_PATTERN = r"^[A-Za-z0-9_\-]+$"
FunctionStatus = Literal["real", "stub", "not_ready", "missing"]


class Reason(BaseModel):
    code: str
    text_uz: str
    weight: float


class LoanReason(BaseModel):
    code: str
    text_uz: str
    weight: Optional[float] = None  # None for evidence-fusion policy reasons
    evidence_refs: List[str] = []  # each is an evidence[].ref of the same loan
    source: Optional[Literal["fusion_policy"]] = None


class Escalation(BaseModel):
    """Evidence-fusion policy: a photo finding that raises the queue colour (never the risk)."""
    code: str  # PHOTO_DUPLICATE | PHOTO_FAR | PHOTO_TIME | PHOTO_ASSET_MISMATCH
    min_color: Color
    text_uz: str
    evidence_refs: List[str] = []
    raises_color: bool


# ---------- inner shapes of GET /loan/{id} (= ml.api.get_loan output) ----------

class Loan(BaseModel):
    loan_id: str
    borrower_id: Optional[str] = None
    borrower_name_masked: Optional[str] = None
    purpose: Optional[str] = None
    amount_uzs: Optional[int] = None
    term_months: Optional[int] = None
    disbursed_at: Optional[str] = None
    snapshot_date: Optional[str] = None
    region: Optional[str] = None
    branch_id: Optional[str] = None
    supplier_id: Optional[str] = None
    invoice_amount_uzs: Optional[int] = None


class Transaction(BaseModel):
    txn_id: str
    ts: str
    type: str  # disbursement | cash_withdrawal | supplier_payment | p2p_out | p2p_in | external_transfer | income
    amount_uzs: int
    counterparty_id: Optional[str] = None
    counterparty_kind: Optional[str] = None


class Duplicate(BaseModel):
    loan_id: str
    similarity: float


class AssetMatch(BaseModel):
    """CLIP + linear probe (contract v1.5)."""
    label: str
    score: float
    matches_purpose: bool
    label_uz: str


class CheckPhotoResponse(BaseModel):
    asset_match: Optional[AssetMatch] = None  # None only if CLIP is unavailable
    duplicate_of: List[Duplicate]
    geo_ok: Optional[bool] = None  # None = no geolocation given
    time_ok: Optional[bool] = None  # None = no client time given
    faces_blurred: bool
    faces_count: Optional[int] = None
    latency_ms: float


class Photo(BaseModel):
    photo_id: str
    # portfolio photo (ml): path is repo-relative under ml/assets/; backend adds url = /api/asset/...
    path: Optional[str] = None
    purpose: Optional[str] = None
    captured_at: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    duplicate_of: List[Duplicate] = []
    asset_match: Optional[AssetMatch] = None  # optional: ml leaves it out for portfolio photos (in-sample, v1.5)
    # added by backend
    url: Optional[str] = None
    source: Literal["portfolio", "borrower_upload"] = "portfolio"
    client_captured_at: Optional[str] = None  # sandbox upload: client clock (server time is captured_at)
    check: Optional[CheckPhotoResponse] = None  # sandbox upload: result of /check_photo


class Invoice(BaseModel):
    supplier_name: Optional[str] = None
    supplier_oked: Optional[str] = None
    oked_matches_purpose: Optional[bool] = None
    invoice_amount_uzs: Optional[int] = None


class Evidence(BaseModel):
    ref: str
    kind: Literal["txn", "invoice", "graph", "photo"]
    summary_uz: Optional[str] = None
    data: Any = None


# ---------- endpoints ----------

class Health(BaseModel):
    ml_mode: Literal["real", "stub"]
    model_version: str
    warmup: Literal["pending", "running", "done", "failed"]
    functions: Dict[str, FunctionStatus]
    clip_loaded: bool = False  # ml.api.warm_up(): asset check available


class QueueItem(BaseModel):
    loan_id: str
    borrower_name_masked: str
    purpose: str
    amount_uzs: int
    region: str
    risk: Risk
    color: Color  # after evidence fusion (>= base_color)
    top_reason: str
    base_color: Color  # money-trail colour
    priority: float  # queue sort key: colour rank (green 0, yellow 1, red 2) + risk
    escalations: List[Escalation] = []


class QueueCounts(BaseModel):
    red: int
    yellow: int
    green: int


class QueuePage(BaseModel):
    """GET /queue (v1.3): one page of the caller's sandbox queue, in ml priority order."""
    items: List[QueueItem]
    total: int  # items matching the filters (before limit / offset)
    counts: QueueCounts  # whole queue of the sandbox, BEFORE filters


class QueueFacets(BaseModel):
    regions: List[str]
    purposes: List[str]


class ColorLegend(BaseModel):
    """= ml.api.get_color_thresholds(): colour rule of the served portfolio (default view, no sandbox)."""
    red: Risk  # risk score at the red rank boundary
    yellow: Risk  # risk score at the yellow rank boundary
    n_red: int  # red by risk
    n_yellow: int  # yellow by risk
    n_loans: int
    rule: str  # English, for logs / docs; the UI builds its own Uzbek text from the numbers
    n_escalated_to_red: int
    n_escalated_to_yellow: int
    n_red_total: int
    n_yellow_total: int
    note: str


class LoanDetail(BaseModel):
    loan: Loan
    transactions: List[Transaction]
    photos: List[Photo]
    invoice: Optional[Invoice] = None
    risk: Risk
    color: Color
    base_color: Color
    priority: float
    escalations: List[Escalation] = []
    reasons: List[LoanReason]
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


class CheckPhotoFields(BaseModel):
    """Non-file fields of /check_photo (multipart form fields, primary; contract changes #2, #9)."""
    loan_id: LoanId
    lat: Optional[float] = Field(default=None, ge=-90, le=90)
    lon: Optional[float] = Field(default=None, ge=-180, le=180)
    captured_at: Optional[datetime] = None  # client time, only compared with server time


class CheckPhotoJSON(CheckPhotoFields):
    """Fallback JSON body for /check_photo: image as base64 (optionally a data: URL)."""
    image_base64: str = Field(min_length=1)


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


class SessionInfo(BaseModel):
    """GET /session (v1.4): demo officer id derived from the sandbox cookie (production: from the bank's SSO)."""
    officer_id: str


class ResetResponse(BaseModel):
    ok: bool


class ErrorResponse(BaseModel):
    detail: str


# ---------- scenario (contract v1.6): hypothetical loan, never written to the portfolio ----------

PURPOSES = ("tikuv_dastgohi", "chorva", "issiqxona", "parranda", "asalarichilik")
SCENARIO_TX_TYPES = ("cash_withdrawal", "supplier_payment", "p2p_out", "p2p_in", "external_transfer", "income")
SCENARIO_MAX_TX = 60
SCENARIO_MAX_AMOUNT = 10 ** 11
SCENARIO_MAX_SINGLE_X, SCENARIO_MAX_INFLOW_X = 10, 20  # same limits as ml/money_trail/scenario.py
SCENARIO_INFLOWS = {"p2p_in", "income"}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ScenarioExistingSupplier(_Strict):
    mode: Literal["existing"]
    supplier_id: Annotated[str, Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_\-]+$")]


class ScenarioNewSupplier(_Strict):
    mode: Literal["new"]
    oked: Annotated[str, Field(pattern=r"^[0-9]{2}\.[0-9]{2}$")]
    registered_days_before: Annotated[StrictInt, Field(ge=0, le=40_000)]


class ScenarioTransaction(_Strict):
    day: Annotated[StrictInt, Field(ge=0, le=364)]
    type: Literal["cash_withdrawal", "supplier_payment", "p2p_out", "p2p_in", "external_transfer", "income"]
    amount_uzs: Annotated[StrictInt, Field(ge=1, le=SCENARIO_MAX_AMOUNT)]
    counterparty_kind: Optional[Literal["person", "business", "external", "supplier", "none"]] = None


class ScenarioRequest(_Strict):
    purpose: Literal["tikuv_dastgohi", "chorva", "issiqxona", "parranda", "asalarichilik"]
    amount_uzs: Annotated[StrictInt, Field(ge=1, le=SCENARIO_MAX_AMOUNT)]
    invoice_amount_uzs: Optional[Annotated[StrictInt, Field(ge=1, le=SCENARIO_MAX_AMOUNT)]] = None
    region: Optional[Annotated[str, Field(min_length=1, max_length=64)]] = None
    snapshot_day: Annotated[StrictInt, Field(ge=2, le=364)]
    supplier: Annotated[Union[ScenarioExistingSupplier, ScenarioNewSupplier], Field(discriminator="mode")]
    transactions: Annotated[List[ScenarioTransaction], Field(max_length=SCENARIO_MAX_TX)]

    @model_validator(mode="after")
    def _limits(self):
        loan = self.amount_uzs
        cap = SCENARIO_MAX_SINGLE_X * loan
        if self.invoice_amount_uzs is not None and self.invoice_amount_uzs > cap:
            raise ValueError("invoice_amount_uzs kredit summasidan 10 baravardan katta bo'lishi mumkin emas")
        for i, t in enumerate(self.transactions, 1):
            if t.day > self.snapshot_day:
                raise ValueError(f"{i}-tranzaksiya kuni snapshot_day dan katta bo'lishi mumkin emas")
            if t.amount_uzs > cap:
                raise ValueError(f"{i}-tranzaksiya summasi kredit summasidan 10 baravardan katta")
        inflow = sum(t.amount_uzs for t in self.transactions if t.type in SCENARIO_INFLOWS)
        if inflow > SCENARIO_MAX_INFLOW_X * loan:
            raise ValueError("kirimlar yig'indisi kredit summasidan 20 baravardan katta")
        return self


class ScenarioScore(BaseModel):
    risk: Risk
    color: Color
    reasons: List[Reason]
    model_version: str
    latency_ms: float
    percentile: float
    features: Dict[str, Any]


class ScenarioPreset(BaseModel):
    id: str
    title_uz: str
    description_uz: str
    scenario: Dict[str, Any]
