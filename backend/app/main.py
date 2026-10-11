"""Maqsad API: FastAPI app serving /api/* (docs/api_contract.md v1.4) and the frontend build
(frontend/dist) at /.

Run:  uvicorn backend.app.main:create_app --factory --port 8000   (from the repo root)
"""
from __future__ import annotations

import base64
import binascii
import hmac
import json
import logging
import time
import uuid
from http import HTTPStatus
from pathlib import Path as FsPath
from pathlib import PurePosixPath
from typing import Annotated, List, Optional

from fastapi import APIRouter, FastAPI, Header, HTTPException, Path, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from pydantic import TypeAdapter, ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.formparsers import MultiPartParser

from . import i18n, llm
from .config import Settings
from .i18n import Msg
from .draft_queue import MSG_BUSY, PRIORITY_PREWARM, DraftQueue, Prewarm, QueueFull
from .db import Database, canonical, sha256_hex, utc_now
from .ml_adapter import (MLAdapter, MLBadInput, MLFailure, MLInvalidImage, MLInvalidRequest, MLNotFound, MLNotReady,
                         load_adapter)
from .limits import MSG_BODY_TOO_LARGE, OTHER_BODY_LIMIT, BodyLimitMiddleware
from .ratelimit import RateLimiter
from .schemas import (
    LOAN_ID_PATTERN, AuditEntry, CheckPhotoFields, CheckPhotoJSON, CheckPhotoResponse, Color, ColorLegend,
    DecisionRequest, DecisionResponse, DraftJob, DraftReport, DraftReportRequest, Health, LoanDetail, QueueFacets, QueueItem,
    PrewarmStatus, QueuePage, ResetResponse, ScenarioPreset, ScenarioRequest, ScenarioScore, ScoreRequest, ScoreResponse,
    SessionInfo,
)

log = logging.getLogger("maqsad.api")

SANDBOX_COOKIE = "sandbox_id"
SANDBOX_MAX_AGE_S = 30 * 24 * 3600
UPLOAD_PATHS = {"/api/check_photo"}
REGEN_IP_PER_MIN = 3  # regenerate: global per-IP cap (each one is a paid LLM call)
QUEUE_DEFAULT_LIMIT = 50
QUEUE_MAX_LIMIT = 200
PREWARM_DEFAULT_N = 40

MSG_TOO_MANY = Msg("too_many")
MSG_BAD_TYPE = Msg("bad_type")
MSG_EMPTY = Msg("empty_image")
MSG_NO_IMAGE = Msg("no_image")
MSG_BAD_IMAGE = Msg("bad_image")
ASSET_SUFFIXES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
HTTP_MSG = {code: Msg(f"http_{code}") for code in (400, 403, 404, 405, 413, 429)}

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
    """The rightmost X-Forwarded-For entry (appended by our own proxy; the left ones are client-supplied and can be forged),
    else the socket peer."""
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        last = fwd.rsplit(",", 1)[-1].strip()
        if last:
            return last
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
    adapter.start_warmup()  # background thread: /api/health answers while ml builds its portfolio
    db = Database(settings.data_dir)
    limiter = RateLimiter()
    drafts = DraftQueue(settings, db, rpm=settings.llm_rpm)
    prewarm = Prewarm()

    app = FastAPI(title="Maqsad API", version="0.1.0")
    app.state.settings, app.state.ml, app.state.db, app.state.limiter = settings, adapter, db, limiter
    app.state.drafts, app.state.prewarm = drafts, prewarm

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
        request.state.lang = i18n.parse_lang(request.headers.get("accept-language"))

        response = await call_next(request)
        if new_cookie:
            response.set_cookie(SANDBOX_COOKIE, sid, max_age=SANDBOX_MAX_AGE_S, httponly=True,
                                samesite="lax", secure=_is_https(request))
        response.headers["X-ML-Mode"] = adapter.mode
        response.headers["X-Model-Version"] = adapter.model_version
        return response

    # Outermost middleware: counts the bytes actually received. base64 is ~4/3 of the file; +64 KB for form fields / JSON overhead.
    photo_body_limit = settings.max_upload_bytes * 4 // 3 + 65536

    def body_limit_for(path: str):
        if path in UPLOAD_PATHS:
            return photo_body_limit, _too_large_msg(settings)
        return OTHER_BODY_LIMIT, MSG_BODY_TOO_LARGE

    app.add_middleware(BodyLimitMiddleware, limit_for=body_limit_for)
    # The original photo must never reach the disk: Starlette spools an uploaded part to a temp file above this size.
    for attr in ("max_file_size", "spool_max_size"):
        if hasattr(MultiPartParser, attr):
            setattr(MultiPartParser, attr, max(getattr(MultiPartParser, attr), photo_body_limit + 1))

    # ---------- errors: always {detail: "<message in the caller's language (Accept-Language, default uz)>"} ----------

    def lang_of(request: Request) -> str:
        return getattr(request.state, "lang", i18n.DEFAULT_LANG)

    @app.exception_handler(RequestValidationError)
    async def on_validation(request: Request, exc: RequestValidationError):
        msg = Msg("validation", details=lambda lg: i18n.error_details(exc.errors(), lg))
        return JSONResponse({"detail": msg.render(lang_of(request))}, status_code=422)

    @app.exception_handler(StarletteHTTPException)
    async def on_http(request: Request, exc: StarletteHTTPException):
        detail = exc.detail
        if not isinstance(detail, str) or detail == _phrase(exc.status_code):
            # default English phrase ("Not Found", ...) -> a message in the caller's language
            detail = HTTP_MSG.get(exc.status_code, str(detail))
        return JSONResponse({"detail": i18n.render(detail, lang_of(request))}, status_code=exc.status_code,
                            headers=getattr(exc, "headers", None))

    @app.exception_handler(MLNotFound)
    async def on_ml_not_found(request: Request, exc: MLNotFound):
        msg = Msg("loan_not_found", loan_id=str(exc).strip(chr(39)))
        return JSONResponse({"detail": msg.render(lang_of(request))}, status_code=404)

    @app.exception_handler(MLNotReady)
    async def on_ml_not_ready(request: Request, exc: MLNotReady):
        return JSONResponse({"detail": Msg("ml_not_ready", name=str(exc)).render(lang_of(request))}, status_code=503)

    @app.exception_handler(MLInvalidImage)
    async def on_ml_invalid_image(request: Request, exc: MLInvalidImage):
        if exc.too_large:
            return JSONResponse({"detail": i18n.render(_too_large_msg(settings), lang_of(request))}, status_code=413)
        return JSONResponse({"detail": i18n.render(MSG_BAD_IMAGE, lang_of(request))}, status_code=400)

    @app.exception_handler(MLInvalidRequest)
    async def on_ml_invalid_request(request: Request, exc: MLInvalidRequest):
        msg = Msg("request_rejected", detail=lambda lg: i18n.pick(getattr(exc, "messages", None), lg, str(exc)))
        return JSONResponse({"detail": msg.render(lang_of(request))}, status_code=400)

    @app.exception_handler(MLBadInput)
    async def on_ml_bad_input(request: Request, exc: MLBadInput):
        msg = Msg("model_rejected", detail=lambda lg: i18n.pick(getattr(exc, "messages", None), lg, str(exc)))
        return JSONResponse({"detail": msg.render(lang_of(request))}, status_code=422)

    @app.exception_handler(MLFailure)
    async def on_ml_failure(request: Request, exc: MLFailure):
        return JSONResponse({"detail": Msg("ml_failure").render(lang_of(request))}, status_code=502)

    @app.exception_handler(Exception)
    async def on_unexpected(request: Request, exc: Exception):
        log.exception("unhandled error")
        return JSONResponse({"detail": Msg("internal").render(lang_of(request))}, status_code=500)

    def rate_check(request: Request, bucket: str) -> None:
        sid = request.state.sandbox_id
        if not (limiter.allow(f"{bucket}:sb:{sid}", settings.rate_limit_per_min)
                and limiter.allow(f"{bucket}:ip:{_client_ip(request)}", settings.rate_limit_ip_per_min)):
            raise HTTPException(429, MSG_TOO_MANY)

    # ---------- /api ----------

    api = APIRouter(prefix="/api")

    @api.get("/health", response_model=Health)
    def health():
        return {"ml_mode": adapter.mode, "model_version": adapter.model_version,
                "warmup": adapter.warmup_state, "functions": dict(adapter.status),
                "clip_loaded": adapter.clip_loaded}

    @api.get("/session", response_model=SessionInfo)
    def session(request: Request):
        return {"officer_id": demo_officer_id(request.state.sandbox_id)}

    def full_queue(request: Request) -> List[QueueItem]:
        items = _validated(List[QueueItem], adapter.get_queue(request.state.sandbox_id))
        return sorted(items, key=lambda x: -x.priority)  # ml already sorts; keep the contract even if it did not

    @api.get("/queue", response_model=QueuePage)
    def queue(
        request: Request,
        limit: Annotated[int, Query(ge=1, le=QUEUE_MAX_LIMIT)] = QUEUE_DEFAULT_LIMIT,
        offset: Annotated[int, Query(ge=0)] = 0,
        color: Optional[Color] = None,
        region: Annotated[Optional[str], Query(max_length=64)] = None,
        purpose: Annotated[Optional[str], Query(max_length=64)] = None,
    ):
        items = full_queue(request)
        counts = {c: 0 for c in ("red", "yellow", "green")}
        for it in items:
            counts[it.color] += 1  # whole sandbox queue, before filters
        matched = [it for it in items
                   if (not color or it.color == color)
                   and (not region or it.region == region)
                   and (not purpose or it.purpose == purpose)]
        return {"items": matched[offset:offset + limit], "total": len(matched), "counts": counts}

    @api.get("/queue/facets", response_model=QueueFacets)
    def queue_facets(request: Request):
        items = full_queue(request)
        return {"regions": sorted({it.region for it in items}), "purposes": sorted({it.purpose for it in items})}

    @api.get("/queue/legend", response_model=ColorLegend)
    def queue_legend():
        return _validated(ColorLegend, adapter.get_color_thresholds())

    @api.get("/loan/{loan_id}", response_model=LoanDetail)
    def loan(loan_id: LoanIdPath, request: Request):
        data = dict(adapter.get_loan(loan_id, request.state.sandbox_id))
        portfolio = [{**ph, "url": _asset_url(ph.get("path")), "source": "portfolio"}
                     for ph in data.get("photos") or []]
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
        data["photos"] = portfolio + own
        return _validated(LoanDetail, data)

    @api.post("/score_loan", response_model=ScoreResponse)
    def score(body: ScoreRequest):
        return _validated(ScoreResponse, adapter.score_loan(body.loan_id))

    @api.get("/scenario_presets", response_model=List[ScenarioPreset])
    def scenario_presets():
        return _validated(List[ScenarioPreset], adapter.get_scenario_presets())

    @api.post("/score_scenario", response_model=ScenarioScore)
    async def score_scenario(request: Request):
        """Hypothetical scenario scored live by ml; nothing is stored (no audit, no portfolio change)."""
        try:
            payload = await request.json()
        except ValueError:
            raise HTTPException(400, Msg("json_required")) from None
        try:
            scenario = ScenarioRequest.model_validate(payload)
        except ValidationError as e:
            raise HTTPException(400, Msg("scenario_invalid", details=lambda lg, errs=e.errors(): scenario_error_text(errs, lg))) from None
        result = await run_in_threadpool(adapter.score_scenario, scenario.model_dump(mode="json"))
        return _validated(ScenarioScore, result)

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
                raise HTTPException(400, Msg("json_malformed")) from e
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
                raise HTTPException(400, Msg("base64_invalid")) from e
        else:
            raise HTTPException(400, Msg("content_type"))

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
            raise HTTPException(404, Msg("photo_not_found"))
        with p.open("rb") as f:
            head = f.read(16)
        return FileResponse(p, media_type=sniff_image(head) or "application/octet-stream",
                            headers={"Cache-Control": "private, no-store"})

    def localized_view(view: dict, lang: str):
        """(evidence, reasons) of a loan with every text in `lang` only: this is what the LLM sees (and what the cache key covers)."""
        evidence = []
        for e in view.get("evidence") or []:
            e = dict(e)
            summary = e.pop("summary", None) or {}
            e["summary"] = summary.get(lang) or summary.get("uz") or e.get("summary_uz")
            e.pop("summary_uz", None)
            evidence.append(e)
        reasons = []
        for r in view.get("reasons") or []:
            r = dict(r)
            text = r.pop("text", None) or {}
            r["text"] = text.get(lang) or text.get("uz") or r.get("text_uz")
            r.pop("text_uz", None)
            reasons.append(r)
        return evidence, reasons

    def draft_inputs(loan_id: str, sandbox_id: Optional[str], lang: str = i18n.DEFAULT_LANG):
        """(cache scope, cache key, evidence, reasons) in `lang`. Scope is "shared" unless this sandbox added evidence to the loan
        (an uploaded photo / escalation): then the draft depends on the sandbox and is cached under its id.
        The key covers the language, so each language has its own cached draft."""
        view = adapter.get_loan(loan_id, sandbox_id)
        evidence, reasons = localized_view(view, lang)

        def key_of(ev, rs):
            return sha256_hex(canonical({"evidence": ev, "reasons": rs, "model": settings.llm_model,
                                         "prompt": llm.PROMPT_VERSION, "lang": lang}))

        key, scope = key_of(evidence, reasons), Database.SHARED_SCOPE
        if sandbox_id is not None and db.photo_list(sandbox_id, loan_id):
            base = localized_view(adapter.get_loan(loan_id, None), lang)
            if key_of(*base) != key:
                scope = sandbox_id
        return scope, key, evidence, reasons

    def cached_draft(scope: str, loan_id: str, key: str, lookup_started: float, lang: str = i18n.DEFAULT_LANG) -> Optional[dict]:
        hit = db.draft_cache_get(scope, loan_id, key, lang)
        if hit is None:
            return None
        return {**hit, "cached": True, "cost_usd": 0.0, "input_tokens": None, "output_tokens": None,
                "latency_ms": round((time.perf_counter() - lookup_started) * 1000, 1)}

    @api.post("/draft_report", response_model=DraftReport,
              responses={202: {"model": DraftJob, "description": "Not cached: queued, poll /draft_report/job/{job_id}"}})
    def draft_report(body: DraftReportRequest, request: Request):
        rate_check(request, "draft_report")
        sid = request.state.sandbox_id
        t0 = time.perf_counter()
        if not settings.llm_configured:
            adapter.get_loan(body.loan_id, sid)  # 404 for an unknown loan first
            raise HTTPException(503, Msg("report_no_llm"))
        lang = request.state.lang
        scope, key, evidence, reasons = draft_inputs(body.loan_id, sid, lang)
        if body.regenerate:
            if not (limiter.allow(f"regenerate:sb:{sid}", 1)
                    and limiter.allow(f"regenerate:ip:{_client_ip(request)}", REGEN_IP_PER_MIN)):
                raise HTTPException(429, Msg("regen_limit"))
            scope = sid  # a regenerated draft goes to this sandbox's cache only; the shared (pre-warmed) cache is never overwritten
        else:
            for lookup_scope in dict.fromkeys((sid, scope)):  # this sandbox's own regenerated draft wins over the shared one
                hit = cached_draft(lookup_scope, body.loan_id, key, t0, lang)
                if hit is not None:
                    return hit

        def lookup():
            return db.draft_cache_get(sid, body.loan_id, key, lang) or db.draft_cache_get(scope, body.loan_id, key, lang)

        try:
            ticket = drafts.submit(
                sandbox_id=sid, scope=scope, loan_id=body.loan_id, key=key, evidence=evidence, reasons=reasons, lang=lang,
                cache_lookup=None if body.regenerate else lookup)
        except QueueFull:
            raise HTTPException(503, MSG_BUSY) from None
        if ticket is None:  # the draft appeared in the cache while we were deciding
            return cached_draft(sid, body.loan_id, key, t0, lang) or cached_draft(scope, body.loan_id, key, t0, lang)
        return JSONResponse(drafts.ticket_view(ticket), status_code=202)

    @api.get("/draft_report/job/{job_id}", response_model=DraftJob)
    def draft_job(job_id: Annotated[str, Path(pattern=r"^[0-9a-f]{32}$")], request: Request):
        view = drafts.view(job_id, request.state.sandbox_id)
        if view is None:  # unknown, expired, lost in a restart, or another sandbox's job: the client resubmits
            raise HTTPException(404, Msg("job_not_found"))
        if view.get("error"):
            view = {**view, "error": i18n.render(view["error"], request.state.lang)}
        return view

    @api.post("/decision", response_model=DecisionResponse)
    def decision(body: DecisionRequest, request: Request):
        # the decision is human; the AI recommendation is stored separately next to it. It is what the officer
        # saw on the loan page: the sandbox view's colour after evidence fusion (+ the money-trail colour if raised)
        try:
            view = adapter.get_loan(body.loan_id, request.state.sandbox_id)  # MLNotFound -> 404
            score = {k: view.get(k) for k in ("risk", "color", "base_color", "model_version")}
            score["escalations"] = [e.get("code") for e in view.get("escalations") or [] if e.get("raises_color")]
            base = f", base={score['base_color']}" if score["base_color"] and score["base_color"] != score["color"] else ""
            ai_rec = f"{score['color']} (risk={float(score['risk']):.3f}{base})"
            model_version = str(score.get("model_version") or adapter.model_version)
        except MLFailure:
            score, ai_rec, model_version = None, None, adapter.model_version
        # the actor is the officer of this sandbox, set on the server; a client-supplied officer_id is ignored
        actor = demo_officer_id(request.state.sandbox_id)
        audit_id = db.audit_append(
            sandbox_id=request.state.sandbox_id, loan_id=body.loan_id, actor=actor,
            action=body.action, ai_recommendation=ai_rec, model_version=model_version,
            inputs_hash=sha256_hex(canonical({"request": body.model_dump(exclude={"officer_id"}), "actor": actor, "ai": score})),
            comment=body.comment,
        )
        return {"audit_id": audit_id}

    @api.get("/audit", response_model=List[AuditEntry])
    def audit(loan_id: LoanIdQuery, request: Request):
        adapter.get_loan(loan_id, request.state.sandbox_id)  # 404 for unknown loan
        return db.audit_list(request.state.sandbox_id, loan_id)

    def admin_check(token: Optional[str]) -> None:
        if not settings.admin_token or not token or not hmac.compare_digest(
                token.encode(), settings.admin_token.encode()):
            raise HTTPException(403, Msg("admin_forbidden"))

    @api.post("/admin/prewarm", response_model=PrewarmStatus, status_code=202)
    def admin_prewarm(x_admin_token: Annotated[Optional[str], Header()] = None,
                      n: Annotated[int, Query(ge=1, le=200)] = PREWARM_DEFAULT_N):
        """Shared-cache drafts for the top-n loans of the default queue (red + escalated first), lowest queue priority."""
        admin_check(x_admin_token)
        if not settings.llm_configured:
            raise HTTPException(503, Msg("report_no_llm_short"))

        def loan_ids() -> List[str]:
            items = _validated(List[QueueItem], adapter.get_queue(None))
            return [it.loan_id for it in sorted(items, key=lambda x: -x.priority)[:n]]

        def handle(loan_id: str, on_finish) -> str:
            scope, key, evidence, reasons = draft_inputs(loan_id, None)
            ticket = drafts.submit(sandbox_id="prewarm", scope=scope, loan_id=loan_id, key=key, evidence=evidence,
                                   reasons=reasons, priority=PRIORITY_PREWARM, on_finish=on_finish,
                                   cache_lookup=lambda: db.draft_cache_get(scope, loan_id, key))
            return "cached" if ticket is None else "queued"

        prewarm.start(loan_ids, handle)  # already running: the status below shows that run
        return prewarm.status()

    @api.get("/admin/prewarm", response_model=PrewarmStatus)
    def admin_prewarm_status(x_admin_token: Annotated[Optional[str], Header()] = None):
        admin_check(x_admin_token)
        return prewarm.status()

    @api.post("/admin/reset", response_model=ResetResponse)
    def admin_reset(x_admin_token: Annotated[Optional[str], Header()] = None, clear_shared_cache: bool = False):
        admin_check(x_admin_token)
        adapter.reset_sandbox(None)  # ml: None = every sandbox (contract v1.2)
        db.reset_sandbox_data(clear_shared_cache=clear_shared_cache)  # shared (pre-warmed) drafts stay unless asked
        drafts.forget_tickets()
        limiter.reset()
        log.warning("admin reset done")
        return {"ok": True}

    @api.get("/asset/{asset_path:path}", include_in_schema=False)
    def asset(asset_path: str):
        """Only the portfolio photos listed in ml's photo index (photos[].url); everything else is 404."""
        if asset_path not in adapter.portfolio_asset_paths():
            raise HTTPException(404, Msg("file_not_found"))
        root = settings.ml_assets_dir.resolve()
        try:
            target = (root / asset_path).resolve()
        except (OSError, ValueError):
            raise HTTPException(404, Msg("file_not_found")) from None
        media_type = ASSET_SUFFIXES.get(target.suffix.lower())
        if root not in target.parents or media_type is None or not target.is_file():
            raise HTTPException(404, Msg("file_not_found"))
        return FileResponse(target, media_type=media_type, headers={"Cache-Control": "public, max-age=86400"})

    app.include_router(api)

    # ---------- frontend (SPA) ----------

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str):
        if full_path == "api" or full_path.startswith("api/"):
            raise HTTPException(404, Msg("api_not_found"))
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


def scenario_error_text(errors: list, lang: str = "uz") -> str:
    """Pydantic errors of a scenario -> short text in `lang` (first 4 problems)."""
    return i18n.error_details(errors, lang, limit=4, root="v_scenario") + "."


def _asset_url(path: Optional[str]) -> Optional[str]:
    """'ml/assets/photos/x.jpg' (repo-relative, from ml) -> '/api/asset/photos/x.jpg'; anything else -> None."""
    if not path:
        return None
    parts = PurePosixPath(path.replace("\\", "/")).parts
    if len(parts) < 3 or parts[:2] != ("ml", "assets") or ".." in parts:
        return None
    return "/api/asset/" + "/".join(parts[2:])


def demo_officer_id(sandbox_id: str) -> str:
    """Demo only: a stable 'Xodim-XXXX' per sandbox. A hash, so the sandbox id (the key to its photos) never leaks."""
    return "Xodim-" + sha256_hex(sandbox_id.encode())[:4].upper()


def _too_large_msg(settings: Settings) -> Msg:
    return Msg("too_large", mb=settings.max_upload_bytes // (1024 * 1024))


def _phrase(status_code: int) -> str:
    try:
        return HTTPStatus(status_code).phrase
    except ValueError:
        return ""
