"""Maqsad API: FastAPI app serving /api/* (docs/api_contract.md + docs/api_contract_changes.md)
and the frontend build (frontend/dist) at /.

Run:  uvicorn backend.app.main:create_app --factory --port 8000   (from the repo root)
"""
from __future__ import annotations

import base64
import binascii
import hmac
import json
import logging
import uuid
from http import HTTPStatus
from pathlib import Path as FsPath
from typing import Annotated, List, Optional

from fastapi import APIRouter, FastAPI, Header, HTTPException, Path, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from pydantic import TypeAdapter, ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import llm
from .config import Settings
from .db import Database, canonical, sha256_hex, utc_now
from .ml_adapter import MLAdapter, MLBadInput, MLFailure, MLNotFound, load_adapter
from .ratelimit import RateLimiter
from .schemas import (
    LOAN_ID_PATTERN, AuditEntry, CheckPhotoFields, CheckPhotoJSON, CheckPhotoResponse, DecisionRequest,
    DecisionResponse, DraftReport, DraftReportRequest, Health, LoanDetail, QueueItem, ResetResponse,
    ScoreRequest, ScoreResponse,
)

log = logging.getLogger("maqsad.api")

SANDBOX_COOKIE = "sandbox_id"
SANDBOX_MAX_AGE_S = 30 * 24 * 3600
UPLOAD_PATHS = {"/api/check_photo"}

MSG_TOO_MANY = "Juda ko'p so'rov. Bir daqiqadan keyin qayta urinib ko'ring."
MSG_BAD_TYPE = "Faqat JPEG, PNG yoki WebP rasm qabul qilinadi."
MSG_EMPTY = "Rasm fayli bo'sh."
MSG_NO_IMAGE = "Rasm yuborilmadi: 'image' maydoniga JPEG, PNG yoki WebP fayl biriktiring."
MSG_REPORT_NOT_READY = "Dalolatnoma qoralamasi hozircha tayyor emas"
HTTP_MSG_UZ = {
    400: "So'rov noto'g'ri.",
    403: "Ruxsat yo'q.",
    404: "Topilmadi.",
    405: "Bu metod qo'llab-quvvatlanmaydi.",
    413: "So'rov hajmi juda katta.",
    429: MSG_TOO_MANY,
}

LoanIdPath = Annotated[str, Path(min_length=1, max_length=64, pattern=LOAN_ID_PATTERN)]
LoanIdQuery = Annotated[str, Query(min_length=1, max_length=64, pattern=LOAN_ID_PATTERN)]


def sniff_image(data: bytes) -> Optional[str]:
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def _client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _is_https(request: Request) -> bool:
    return request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https"


def _validated(model, data):
    """Validate an ml result against the contract; a mismatch is an ml failure (502), not a 500."""
    try:
        return TypeAdapter(model).validate_python(data)
    except ValidationError as e:
        log.error("ml response does not match contract (%s): %s", model, e)
        raise MLFailure(f"ml javobi shartnomaga mos emas ({e.error_count()} ta xato)") from e


CHECK_PHOTO_OPENAPI = {
    "requestBody": {
        "required": True,
        "content": {
            "multipart/form-data": {"schema": {
                "type": "object",
                "required": ["loan_id", "image"],
                "properties": {
                    "image": {"type": "string", "format": "binary"},
                    "loan_id": {"type": "string"},
                    "lat": {"type": "number"},
                    "lon": {"type": "number"},
                    "captured_at": {"type": "string", "format": "date-time"},
                },
            }},
            "application/json": {"schema": CheckPhotoJSON.model_json_schema()},
        },
    }
}


def create_app(settings: Optional[Settings] = None, adapter: Optional[MLAdapter] = None) -> FastAPI:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = settings or Settings.from_env()
    adapter = adapter or load_adapter()  # REQUIRE_REAL_ML=1 + no ml.api -> raises, service does not start
    db = Database(settings.data_dir)
    limiter = RateLimiter()

    app = FastAPI(title="Maqsad API", version="0.1.0")
    app.state.settings, app.state.ml, app.state.db, app.state.limiter = settings, adapter, db, limiter

    # ---------- middleware: sandbox cookie, upload size, ML mode headers ----------

    @app.middleware("http")
    async def sandbox_and_limits(request: Request, call_next):
        sid = request.cookies.get(SANDBOX_COOKIE, "")
        try:
            sid = str(uuid.UUID(sid))
            new_cookie = False
        except ValueError:
            sid, new_cookie = str(uuid.uuid4()), True
        request.state.sandbox_id = sid

        if request.url.path in UPLOAD_PATHS:
            cl = request.headers.get("content-length", "")
            # base64 is ~4/3 of the file; +64 KB for form fields / JSON overhead
            if cl.isdigit() and int(cl) > settings.max_upload_bytes * 4 // 3 + 65536:
                return JSONResponse({"detail": _too_large_msg(settings)}, status_code=413)

        response = await call_next(request)
        if new_cookie:
            response.set_cookie(SANDBOX_COOKIE, sid, max_age=SANDBOX_MAX_AGE_S, httponly=True,
                                samesite="lax", secure=_is_https(request))
        response.headers["X-ML-Mode"] = adapter.mode
        response.headers["X-Model-Version"] = adapter.model_version
        return response

    # ---------- errors: always {detail: "<Uzbek message>"} ----------

    @app.exception_handler(RequestValidationError)
    async def on_validation(_: Request, exc: RequestValidationError):
        parts = []
        for err in exc.errors()[:5]:
            loc = ".".join(str(x) for x in err.get("loc", ()) if x not in ("body", "query", "path"))
            parts.append(f"{loc or 'body'}: {err.get('msg')}")
        return JSONResponse({"detail": "So'rov ma'lumotlari noto'g'ri — " + "; ".join(parts)}, status_code=422)

    @app.exception_handler(StarletteHTTPException)
    async def on_http(_: Request, exc: StarletteHTTPException):
        detail = exc.detail
        if not isinstance(detail, str) or detail == _phrase(exc.status_code):
            # default English phrase ("Not Found", ...) -> Uzbek
            detail = HTTP_MSG_UZ.get(exc.status_code, str(detail))
        return JSONResponse({"detail": detail}, status_code=exc.status_code, headers=getattr(exc, "headers", None))

    @app.exception_handler(MLNotFound)
    async def on_ml_not_found(_: Request, exc: MLNotFound):
        return JSONResponse({"detail": f"Kredit topilmadi: {str(exc).strip(chr(39))}"}, status_code=404)

    @app.exception_handler(MLBadInput)
    async def on_ml_bad_input(_: Request, exc: MLBadInput):
        return JSONResponse({"detail": f"Model kiritilgan ma'lumotni qabul qilmadi: {exc}"}, status_code=422)

    @app.exception_handler(MLFailure)
    async def on_ml_failure(_: Request, exc: MLFailure):
        return JSONResponse({"detail": "Model xizmatida xatolik yuz berdi, keyinroq qayta urinib ko'ring."},
                            status_code=502)

    @app.exception_handler(Exception)
    async def on_unexpected(_: Request, exc: Exception):
        log.exception("unhandled error")
        return JSONResponse({"detail": "Ichki xatolik yuz berdi."}, status_code=500)

    def rate_check(request: Request, bucket: str) -> None:
        sid = request.state.sandbox_id
        if not (limiter.allow(f"{bucket}:sb:{sid}", settings.rate_limit_per_min)
                and limiter.allow(f"{bucket}:ip:{_client_ip(request)}", settings.rate_limit_ip_per_min)):
            raise HTTPException(429, MSG_TOO_MANY)

    # ---------- /api ----------

    api = APIRouter(prefix="/api")

    @api.get("/health", response_model=Health)
    def health():
        return {"ml_mode": adapter.mode, "model_version": adapter.model_version}

    @api.get("/queue", response_model=List[QueueItem])
    def queue():
        items = _validated(List[QueueItem], adapter.get_queue())
        return sorted(items, key=lambda x: -x.risk)

    @api.get("/loan/{loan_id}", response_model=LoanDetail)
    def loan(loan_id: LoanIdPath, request: Request):
        data = dict(adapter.get_loan(loan_id))
        own = [{
            "photo_id": p["photo_id"],
            "url": f"/api/photo/{p['photo_id']}" if p["image_file"] else None,
            "captured_at": p["server_ts"],
            "source": "borrower_upload",
            "client_captured_at": p["client_ts"],
            "lat": p["lat"],
            "lon": p["lon"],
            "check": json.loads(p["result_json"]),
        } for p in db.photo_list(request.state.sandbox_id, loan_id)]
        data["photos"] = list(data.get("photos") or []) + own
        return _validated(LoanDetail, data)

    @api.post("/score_loan", response_model=ScoreResponse)
    def score(body: ScoreRequest):
        return _validated(ScoreResponse, adapter.score_loan(body.loan_id))

    @api.post("/check_photo", response_model=CheckPhotoResponse, openapi_extra=CHECK_PHOTO_OPENAPI)
    async def check_photo(request: Request):
        rate_check(request, "check_photo")
        ctype = request.headers.get("content-type", "")
        if ctype.startswith("multipart/form-data"):
            form = await request.form(max_files=1, max_fields=10)
            upload = form.get("image")
            if upload is None or isinstance(upload, str):
                raise HTTPException(400, MSG_NO_IMAGE)
            image = await upload.read(settings.max_upload_bytes + 1)
            raw = {k: form.get(k) for k in ("loan_id", "lat", "lon", "captured_at")}
            raw = {k: v for k, v in raw.items() if isinstance(v, str) and v.strip() != ""}
            try:
                fields = CheckPhotoFields.model_validate(raw)
            except ValidationError as e:
                raise RequestValidationError(e.errors()) from e
        elif ctype.startswith("application/json"):
            try:
                payload = await request.json()
            except ValueError as e:
                raise HTTPException(400, "JSON noto'g'ri formatda.") from e
            try:
                fields = CheckPhotoJSON.model_validate(payload)
            except ValidationError as e:
                raise RequestValidationError(e.errors()) from e
            b64 = fields.image_base64.split(",", 1)[1] if fields.image_base64.startswith("data:") else fields.image_base64
            if len(b64) > settings.max_upload_bytes * 4 // 3 + 8:
                raise HTTPException(413, _too_large_msg(settings))
            try:
                image = base64.b64decode(b64, validate=True)
            except (binascii.Error, ValueError) as e:
                raise HTTPException(400, "image_base64 noto'g'ri (base64 emas).") from e
        else:
            raise HTTPException(400, "So'rov multipart/form-data (asosiy) yoki application/json bo'lishi kerak.")

        if len(image) == 0:
            raise HTTPException(400, MSG_EMPTY)
        if len(image) > settings.max_upload_bytes:
            raise HTTPException(413, _too_large_msg(settings))
        if sniff_image(image) is None:
            raise HTTPException(400, MSG_BAD_TYPE)

        sid = request.state.sandbox_id
        server_ts = utc_now()  # server time is the primary timestamp
        result, blurred = await run_in_threadpool(
            adapter.check_photo, fields.loan_id, image, fields.lat, fields.lon, fields.captured_at, sid)
        resp = _validated(CheckPhotoResponse, result)
        photo_id = uuid.uuid4().hex
        client_ts = fields.captured_at.isoformat() if fields.captured_at else None
        db.photo_add(photo_id=photo_id, sandbox_id=sid, loan_id=fields.loan_id, server_ts=server_ts,
                     client_ts=client_ts, lat=fields.lat, lon=fields.lon, result=resp.model_dump(),
                     blurred_image=blurred)
        db.audit_append(
            sandbox_id=sid, loan_id=fields.loan_id, actor="borrower", action="photo_upload",
            ai_recommendation=None, model_version=adapter.model_version,
            inputs_hash=sha256_hex(image + canonical({"loan_id": fields.loan_id, "lat": fields.lat,
                                                      "lon": fields.lon, "captured_at": client_ts,
                                                      "server_ts": server_ts})),
        )
        return resp

    @api.get("/photo/{photo_id}", include_in_schema=False)
    def photo(photo_id: Annotated[str, Path(pattern=r"^[0-9a-f]{32}$")], request: Request):
        # only the owner sandbox can read its photo; anyone else gets the same 404 as a missing photo
        p = db.photo_file(request.state.sandbox_id, photo_id)
        if p is None:
            raise HTTPException(404, "Rasm topilmadi.")
        with p.open("rb") as f:
            head = f.read(16)
        return FileResponse(p, media_type=sniff_image(head) or "application/octet-stream",
                            headers={"Cache-Control": "private, no-store"})

    @api.post("/draft_report", response_model=DraftReport)
    def draft_report(body: DraftReportRequest, request: Request):
        rate_check(request, "draft_report")
        data = adapter.get_loan(body.loan_id)
        if not settings.llm_configured:
            raise HTTPException(503, f"{MSG_REPORT_NOT_READY}: LLM ulanmagan. Kredit sahifasi undan mustaqil ishlaydi.")
        try:
            return llm.draft_report(settings, body.loan_id, data.get("evidence") or [], data.get("reasons") or [], db)
        except llm.LLMError as e:
            raise HTTPException(502, f"{MSG_REPORT_NOT_READY}: LLM xatosi. Keyinroq qayta urinib ko'ring.") from e

    @api.post("/decision", response_model=DecisionResponse)
    def decision(body: DecisionRequest, request: Request):
        # the decision is human; the AI recommendation is stored separately next to it
        try:
            score = adapter.score_loan(body.loan_id)  # MLNotFound -> 404
            ai_rec = f"{score['color']} (risk={float(score['risk']):.3f})"
            model_version = str(score.get("model_version") or adapter.model_version)
        except MLFailure:
            score, ai_rec, model_version = None, None, adapter.model_version
        audit_id = db.audit_append(
            sandbox_id=request.state.sandbox_id, loan_id=body.loan_id, actor=body.officer_id,
            action=body.action, ai_recommendation=ai_rec, model_version=model_version,
            inputs_hash=sha256_hex(canonical({"request": body.model_dump(), "ai": score})),
            comment=body.comment,
        )
        return {"audit_id": audit_id}

    @api.get("/audit", response_model=List[AuditEntry])
    def audit(loan_id: LoanIdQuery, request: Request):
        adapter.get_loan(loan_id)  # 404 for unknown loan
        return db.audit_list(request.state.sandbox_id, loan_id)

    @api.post("/admin/reset", response_model=ResetResponse)
    def admin_reset(x_admin_token: Annotated[Optional[str], Header()] = None):
        if not settings.admin_token or not x_admin_token or not hmac.compare_digest(
                x_admin_token.encode(), settings.admin_token.encode()):
            raise HTTPException(403, "Ruxsat yo'q: X-Admin-Token noto'g'ri yoki ADMIN_TOKEN sozlanmagan.")
        db.reset_sandbox_data()
        adapter.reset()
        limiter.reset()
        log.warning("admin reset done")
        return {"ok": True}

    app.include_router(api)

    # ---------- frontend (SPA) ----------

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str):
        if full_path == "api" or full_path.startswith("api/"):
            raise HTTPException(404, "API endpoint topilmadi.")
        dist = settings.frontend_dist
        index = dist / "index.html"
        if not index.is_file():
            if full_path == "":
                return {"service": "maqsad", "detail": "Frontend build topilmadi (frontend/dist). API: /api/health, /docs"}
            raise HTTPException(404)
        if full_path:
            candidate = (dist / full_path).resolve()
            if candidate.is_file() and dist in candidate.parents:
                return FileResponse(candidate)
            if FsPath(full_path).suffix:  # missing asset -> real 404, not index.html
                raise HTTPException(404)
        return FileResponse(index, headers={"Cache-Control": "no-cache"})

    return app


def _too_large_msg(settings: Settings) -> str:
    return f"Rasm hajmi {settings.max_upload_bytes // (1024 * 1024)} MB dan oshmasligi kerak."


def _phrase(status_code: int) -> str:
    try:
        return HTTPStatus(status_code).phrase
    except ValueError:
        return ""
