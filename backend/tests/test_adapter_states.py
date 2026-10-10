"""ml_adapter: per-function status, warm-up fallback, error mapping, admin reset, asset serving."""
import builtins
import json
import threading
import types
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from backend.app import llm, ml_stub
from backend.app.config import Settings
from backend.app.main import create_app
from backend.app.ml_adapter import MLAdapter, load_adapter

from .conftest import png_bytes

RESULT = {"asset_match": None, "duplicate_of": [], "geo_ok": None, "time_ok": None,
          "faces_blurred": False, "faces_count": 0, "latency_ms": 1.0}


def _fake_ml(**fns):
    m = types.ModuleType("fake_ml")
    m.MODEL_VERSION = "fake-1"
    m.get_queue, m.get_loan, m.score_loan = ml_stub.get_queue, ml_stub.get_loan, ml_stub.score_loan
    m.check_photo = lambda *a, **k: dict(RESULT)
    m.reset_sandbox = lambda sandbox_id: None
    for name, fn in fns.items():
        setattr(m, name, fn)
    return m


def _not_implemented(*a, **k):
    raise NotImplementedError("not wired yet")


def _upload(c, lid):
    return c.post("/api/check_photo", data={"loan_id": lid}, files={"image": ("a.png", png_bytes(), "image/png")})


# ---------- per-function status ----------

def test_not_implemented_function_is_503_and_shown_in_health(make_client):
    c = make_client(adapter=MLAdapter(_fake_ml(check_photo=_not_implemented), "real"))
    h = c.get("/api/health").json()
    assert h["ml_mode"] == "real" and h["warmup"] == "done"
    assert h["functions"]["check_photo"] == "not_ready"  # found by the warm-up probe
    assert h["functions"]["get_queue"] == "real"
    lid = c.get("/api/queue").json()["items"][0]["loan_id"]
    r = _upload(c, lid)
    assert r.status_code == 503
    assert r.json()["detail"].startswith("Bu funksiya hali ulanmagan")


def test_missing_function_is_503(make_client):
    m = _fake_ml()
    del m.score_loan
    c = make_client(adapter=MLAdapter(m, "real"))
    assert c.get("/api/health").json()["functions"]["score_loan"] == "missing"
    lid = c.get("/api/queue").json()["items"][0]["loan_id"]
    assert c.post("/api/score_loan", json={"loan_id": lid}).status_code == 503


def test_get_queue_failure_switches_to_full_stub(make_client):
    c = make_client(adapter=MLAdapter(_fake_ml(get_queue=_not_implemented), "real"))
    h = c.get("/api/health").json()
    assert h["ml_mode"] == "stub" and h["model_version"] == "stub" and h["warmup"] == "failed"
    assert set(h["functions"].values()) == {"stub"}
    q = c.get("/api/queue")
    assert q.status_code == 200 and q.json()["total"] == 40  # stub fixtures, STUB banner in UI


def test_get_queue_failure_with_require_real_is_fatal():
    fatal = []
    a = MLAdapter(_fake_ml(get_queue=_not_implemented), "real", require_real=True, on_fatal=fatal.append)
    a.warmup()
    assert a.warmup_state == "failed" and a.mode == "real"
    assert len(fatal) == 1 and "get_queue" in fatal[0]


def test_warmup_runs_in_background_and_health_does_not_block(tmp_path):
    release = threading.Event()

    def slow_queue():
        release.wait(10)
        return ml_stub.get_queue()

    adapter = MLAdapter(_fake_ml(get_queue=slow_queue), "real")
    app = create_app(settings=replace(Settings.from_env(), data_dir=tmp_path / "data"), adapter=adapter)
    c = TestClient(app)
    h = c.get("/api/health").json()  # answers while get_queue() is still running
    assert h["warmup"] == "running" and h["ml_mode"] == "real"
    release.set()
    assert adapter.wait_warmup(10)
    assert c.get("/api/health").json()["warmup"] == "done"


@pytest.mark.parametrize("exc", [ImportError("no ml"), OSError("libgomp.so.1: cannot open shared object file"),
                                 SyntaxError("unterminated string literal")])
def test_import_errors_fall_back_to_stub(monkeypatch, exc):
    monkeypatch.delenv("ML_FORCE_STUB", raising=False)
    monkeypatch.delenv("REQUIRE_REAL_ML", raising=False)
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "ml.api":
            raise exc
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert load_adapter().mode == "stub"
    monkeypatch.setenv("REQUIRE_REAL_ML", "1")
    with pytest.raises(RuntimeError, match="REQUIRE_REAL_ML"):
        load_adapter()


# ---------- image errors ----------

@pytest.mark.parametrize("message,status", [("cannot decode image: UnidentifiedImageError", 400),
                                            ("image larger than 10 MB", 413)])
def test_invalid_image_error_mapping(make_client, message, status):
    def check_photo(*a, **k):
        raise ValueError(message)  # ml.errors.InvalidImageError is a ValueError

    c = make_client(adapter=MLAdapter(_fake_ml(check_photo=check_photo), "real"))
    r = _upload(c, c.get("/api/queue").json()["items"][0]["loan_id"])
    assert r.status_code == status
    assert "MB" in r.json()["detail"] if status == 413 else "Rasmni o'qib bo'lmadi" in r.json()["detail"]


# ---------- admin reset -> ml.reset_sandbox ----------

def test_admin_reset_calls_ml_reset_sandbox_none(make_client):
    resets = []
    c = make_client(adapter=MLAdapter(_fake_ml(reset_sandbox=resets.append), "real"), admin_token="t")
    resets.clear()  # drop the warm-up probe call
    lid = c.get("/api/queue").json()["items"][0]["loan_id"]
    _upload(c, lid)
    _upload(TestClient(c.app), lid)
    assert c.post("/api/admin/reset", headers={"X-Admin-Token": "t"}).status_code == 200
    assert resets == [None]  # one call: None = every sandbox (contract v1.2)


def test_sandbox_id_passed_to_queue_and_loan(make_client):
    seen = []

    def get_queue(sandbox_id=None):
        seen.append(("queue", sandbox_id))
        return ml_stub.get_queue()

    def get_loan(loan_id, sandbox_id=None):
        seen.append(("loan", sandbox_id))
        return ml_stub.get_loan(loan_id)

    c = make_client(adapter=MLAdapter(_fake_ml(get_queue=get_queue, get_loan=get_loan), "real"))
    seen.clear()  # warm-up / probes
    lid = c.get("/api/queue").json()["items"][0]["loan_id"]
    c.get(f"/api/loan/{lid}")
    sid = c.cookies.get("sandbox_id")
    assert seen == [("queue", sid), ("loan", sid)]


def test_old_ml_signature_without_sandbox_still_works(make_client):
    m = _fake_ml(get_queue=lambda: ml_stub.get_queue(), get_loan=lambda loan_id: ml_stub.get_loan(loan_id))
    c = make_client(adapter=MLAdapter(m, "real"))
    lid = c.get("/api/queue").json()["items"][0]["loan_id"]
    assert c.get(f"/api/loan/{lid}").status_code == 200


# ---------- portfolio photos from ml/assets ----------

@pytest.fixture
def assets(tmp_path):
    root = tmp_path / "ml_assets"
    (root / "photos").mkdir(parents=True)
    (root / "photos" / "chorva_00.jpg").write_bytes(b"\xff\xd8\xff" + b"JPEGDATA")
    (root / "photos" / "manifest.csv").write_text("file,purpose\n", encoding="utf-8")
    (tmp_path / "secret.jpg").write_bytes(b"\xff\xd8\xffSECRET")
    return root


def test_portfolio_photo_url_and_asset_serving(make_client, assets):
    def get_loan(loan_id):
        d = ml_stub.get_loan(loan_id)
        return {**d, "photos": [{"photo_id": "PH1", "path": "ml/assets/photos/chorva_00.jpg", "purpose": "chorva",
                                 "captured_at": "2026-08-01T10:00:00", "lat": 41.3, "lon": 69.2,
                                 "duplicate_of": [{"loan_id": "L0002", "similarity": 0.97}]}]}

    adapter = MLAdapter(_fake_ml(get_loan=get_loan), "real")
    adapter._asset_files = frozenset({"photos/chorva_00.jpg"})  # what ml's photo index would list
    c = make_client(adapter=adapter, ml_assets_dir=assets)
    lid = c.get("/api/queue").json()["items"][0]["loan_id"]
    ph = c.get(f"/api/loan/{lid}").json()["photos"][0]
    assert ph["url"] == "/api/asset/photos/chorva_00.jpg" and ph["source"] == "portfolio"
    assert ph["path"] == "ml/assets/photos/chorva_00.jpg" and ph["duplicate_of"][0]["similarity"] == 0.97
    r = c.get(ph["url"])
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg" and r.content.endswith(b"JPEGDATA")


@pytest.mark.parametrize("path", [
    "/api/asset/..%2Fsecret.jpg",                       # escapes ml/assets
    "/api/asset/photos/..%2F..%2Fsecret.jpg",
    "/api/asset/%2e%2e/%2e%2e/backend/app/main.py",
    "/api/asset/photos/manifest.csv",                   # inside, but not an image
    "/api/asset/photos/missing.jpg",
    "/api/asset/photos",                                # directory
    "/api/asset/C:%2FWindows%2Fwin.ini",                # absolute path
])
def test_asset_path_traversal_is_404(make_client, assets, path):
    adapter = MLAdapter(_fake_ml(), "real")
    adapter._asset_files = frozenset({"photos/chorva_00.jpg", "photos/manifest.csv"})
    c = make_client(adapter=adapter, ml_assets_dir=assets)
    r = c.get(path)
    assert r.status_code == 404, (path, r.status_code)
    assert b"SECRET" not in r.content


def test_asset_not_in_photo_index_is_404_even_if_file_exists(make_client, assets):
    (assets / "photos" / "person.jpg").write_bytes(bytes([255, 216, 255]) + b"PERSON")  # on disk, not in ml's index
    adapter = MLAdapter(_fake_ml(), "real")
    adapter._asset_files = frozenset({"photos/chorva_00.jpg"})
    c = make_client(adapter=adapter, ml_assets_dir=assets)
    assert c.get("/api/asset/photos/chorva_00.jpg").status_code == 200
    r = c.get("/api/asset/photos/person.jpg")
    assert r.status_code == 404 and b"PERSON" not in r.content


def test_asset_serves_nothing_in_stub_mode(client):
    assert client.get("/api/asset/photos/chorva_00.jpg").status_code == 404


def test_asset_index_files_from_real_ml_index():
    pytest.importorskip("ml.photo.index")
    from ml.photo.index import get_index
    files = MLAdapter(_fake_ml(), "real").portfolio_asset_paths()
    assert files == {f"photos/{im['file']}" for im in get_index().images} and files


def test_asset_url_only_for_ml_assets_paths():
    from backend.app.main import _asset_url
    assert _asset_url("ml/assets/photos/x.jpg") == "/api/asset/photos/x.jpg"
    assert _asset_url("ml/artifacts/money_trail_v1.txt") is None
    assert _asset_url("ml/assets/../artifacts/x.jpg") is None
    assert _asset_url(None) is None


# ---------- LLM: evidence_ref only from evidence[].ref ----------

def test_llm_ref_must_be_an_evidence_ref(make_client, monkeypatch):
    c = make_client(llm_base_url="http://llm.invalid/v1", llm_model="m")
    lid = c.get("/api/queue").json()["items"][0]["loan_id"]
    d = c.get(f"/api/loan/{lid}").json()
    ev = d["evidence"][0]
    seen = {}

    def chat(settings, messages):
        seen["user"] = json.loads(messages[1]["content"])
        return json.dumps({"text_uz": "t", "facts": [
            {"claim": ev["summary_uz"], "evidence_ref": ev["ref"]},
            {"claim": "Sabab", "evidence_ref": d["reasons"][0]["code"]},  # a reason code, not an evidence ref
            {"claim": "Dalil", "evidence_ref": ""},
        ]}), 10, 5

    monkeypatch.setattr(llm, "_chat", chat)
    r = c.post("/api/draft_report", json={"loan_id": lid})
    assert r.status_code == 200, r.text
    assert [f["unverified"] for f in r.json()["facts"]] == [False, True, True]
    assert seen["user"]["allowed_refs"] == [e["ref"] for e in d["evidence"]]


# ---------- warm_up (contract v1.5), clip_loaded in health ----------

def test_warm_up_called_and_clip_loaded_in_health(make_client):
    calls = []

    def warm_up():
        calls.append(1)
        return {"portfolio_s": 1.0, "clip_loaded": True, "clip_error": None}

    c = make_client(adapter=MLAdapter(_fake_ml(warm_up=warm_up), "real"))
    assert calls == [1]
    h = c.get("/api/health").json()
    assert h["clip_loaded"] is True and h["warmup"] == "done"


def test_clip_not_loaded_is_reported_and_service_keeps_working(make_client):
    c = make_client(adapter=MLAdapter(_fake_ml(warm_up=lambda: {"clip_loaded": False, "clip_error": "no weights"}), "real"))
    h = c.get("/api/health").json()
    assert h["clip_loaded"] is False and h["warmup"] == "done"
    assert c.app.state.ml.clip_error == "no weights"


def test_warm_up_crash_does_not_break_service(make_client):
    def warm_up():
        raise RuntimeError("boom")

    c = make_client(adapter=MLAdapter(_fake_ml(warm_up=warm_up), "real"))
    h = c.get("/api/health").json()
    assert h["clip_loaded"] is False and h["warmup"] == "done" and h["ml_mode"] == "real"


def test_no_warm_up_function_means_clip_false(make_client):
    c = make_client(adapter=MLAdapter(_fake_ml(), "real"))
    assert c.get("/api/health").json()["clip_loaded"] is False


# ---------- InvalidRequestError / ScenarioError -> 400 ----------

class InvalidRequestError(ValueError):  # same class name as ml.errors.InvalidRequestError
    pass


class ScenarioError(ValueError):
    pass


@pytest.mark.parametrize("exc", [InvalidRequestError, ScenarioError])
def test_invalid_request_and_scenario_errors_are_400_uzbek(make_client, exc):
    def check_photo(loan_id, *a, **k):
        if loan_id.startswith("__"):
            return dict(RESULT)
        raise exc("sandbox_id noto'g'ri")

    c = make_client(adapter=MLAdapter(_fake_ml(check_photo=check_photo), "real"))
    r = _upload(c, c.get("/api/queue").json()["items"][0]["loan_id"])
    assert r.status_code == 400
    assert "sandbox_id noto'g'ri" in r.json()["detail"]


def test_check_photo_asset_match_shape(make_client):
    am = {"label": "chorva", "score": 0.4, "matches_purpose": False, "label_uz": "avtomobil"}
    c = make_client(adapter=MLAdapter(_fake_ml(check_photo=lambda *a, **k: {**RESULT, "asset_match": am}), "real"))
    r = _upload(c, c.get("/api/queue").json()["items"][0]["loan_id"])
    assert r.status_code == 200 and r.json()["asset_match"] == am
    bad = {"label": "chorva", "score": 0.4}  # old shape: not valid any more -> ml contract error (502)
    c = make_client(adapter=MLAdapter(_fake_ml(check_photo=lambda *a, **k: {**RESULT, "asset_match": bad}), "real"))
    assert _upload(c, c.app.state.ml.get_queue()[0]["loan_id"]).status_code == 502


def test_loan_photos_without_asset_match_is_valid(make_client):
    def get_loan(loan_id, sandbox_id=None):
        return {**ml_stub.get_loan(loan_id), "photos": [{"photo_id": "P1", "path": "ml/assets/photos/a.jpg",
                                                         "purpose": "chorva", "duplicate_of": []}]}
    c = make_client(adapter=MLAdapter(_fake_ml(get_loan=get_loan), "real"))
    lid = c.get("/api/queue").json()["items"][0]["loan_id"]
    ph = c.get(f"/api/loan/{lid}").json()["photos"][0]
    assert ph["asset_match"] is None
