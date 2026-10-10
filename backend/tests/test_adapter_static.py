import json
import types

import pytest

from backend.app import ml_stub
from backend.app.ml_adapter import MLAdapter, load_adapter
from backend.fixtures import make_fixtures

from .conftest import png_bytes


def _fake_ml(check_photo, **overrides):
    m = types.ModuleType("fake_ml")
    m.MODEL_VERSION = "fake-1"
    m.get_queue, m.get_loan, m.score_loan = ml_stub.get_queue, ml_stub.get_loan, ml_stub.score_loan
    m.check_photo = check_photo
    for name, fn in overrides.items():
        setattr(m, name, fn)
    return m


PROBE = "__maqsad_probe__"  # warm-up probes every function with this loan_id


RESULT = {"asset_match": {"label": "tikuv_dastgohi", "score": 0.9}, "duplicate_of": [],
          "geo_ok": None, "time_ok": None, "faces_blurred": True, "latency_ms": 1.0}


def test_require_real_ml_refuses_stub(monkeypatch):
    monkeypatch.setenv("REQUIRE_REAL_ML", "1")
    monkeypatch.setenv("ML_FORCE_STUB", "1")
    with pytest.raises(RuntimeError, match="REQUIRE_REAL_ML"):
        load_adapter()


def test_real_ml_without_sandbox_and_blur(make_client):
    calls = []

    def check_photo(loan_id, image_bytes, lat=None, lon=None, captured_at=None):  # old signature
        if loan_id != PROBE:
            calls.append(loan_id)
        return dict(RESULT)

    c = make_client(adapter=MLAdapter(_fake_ml(check_photo), "real"))
    lid = c.get("/api/queue").json()["items"][0]["loan_id"]
    r = c.post("/api/check_photo", data={"loan_id": lid}, files={"image": ("a.png", png_bytes(), "image/png")})
    assert r.status_code == 200, r.text
    assert calls == [lid]
    h = c.get("/api/health").json()
    assert (h["ml_mode"], h["model_version"], h["warmup"]) == ("real", "fake-1", "done")
    assert h["functions"] == {"get_queue": "real", "get_loan": "real", "score_loan": "real",
                              "check_photo": "real", "reset_sandbox": "missing",
                              "get_color_thresholds": "missing"}
    photo = c.get(f"/api/loan/{lid}").json()["photos"][0]
    assert photo["url"] is None  # no blurred_image -> nothing stored
    assert list(c.app.state.db.photo_dir.iterdir()) == []


def test_real_ml_with_sandbox_and_blur(make_client):
    seen = {}

    def check_photo(loan_id, image_bytes, lat=None, lon=None, captured_at=None, sandbox_id=None):
        seen["sandbox_id"] = sandbox_id
        return {**RESULT, "blurred_image": b"BLURRED" + image_bytes}

    c = make_client(adapter=MLAdapter(_fake_ml(check_photo), "real"))
    lid = c.get("/api/queue").json()["items"][0]["loan_id"]
    img = png_bytes()
    r = c.post("/api/check_photo", data={"loan_id": lid}, files={"image": ("a.png", img, "image/png")})
    assert r.status_code == 200, r.text
    assert "blurred_image" not in r.json()
    assert seen["sandbox_id"] == c.cookies.get("sandbox_id")
    url = c.get(f"/api/loan/{lid}").json()["photos"][0]["url"]
    assert c.get(url).content == b"BLURRED" + img
    # original bytes never written to disk
    stored = [p.read_bytes() for p in c.app.state.db.photo_dir.iterdir()]
    assert stored == [b"BLURRED" + img]


def test_photo_visible_only_to_owner_sandbox(make_client):
    from fastapi.testclient import TestClient

    def check_photo(loan_id, image_bytes, lat=None, lon=None, captured_at=None, sandbox_id=None):
        return {**RESULT, "blurred_image": image_bytes}

    owner = make_client(adapter=MLAdapter(_fake_ml(check_photo), "real"))
    lid = owner.get("/api/queue").json()["items"][0]["loan_id"]
    owner.post("/api/check_photo", data={"loan_id": lid}, files={"image": ("a.png", png_bytes(), "image/png")})
    url = owner.get(f"/api/loan/{lid}").json()["photos"][0]["url"]

    r = owner.get(url)
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.headers["cache-control"] == "private, no-store"

    other = TestClient(owner.app)  # another visitor: new sandbox cookie
    assert other.get(f"/api/loan/{lid}").json()["photos"] == []
    r = other.get(url)
    assert r.status_code == 404 and r.json()["detail"] == "Rasm topilmadi."
    anon = TestClient(owner.app, cookies={"sandbox_id": "not-a-uuid"})  # forged/invalid cookie
    assert anon.get(url).status_code == 404
    assert owner.get("/api/photo/" + "0" * 32).status_code == 404  # unknown id, same answer
    assert owner.get("/api/photo/..%2Fmaqsad.sqlite3").status_code in (404, 422)  # id must be 32 hex


def test_ml_exception_is_502(make_client):
    def check_photo(*a, **k):
        raise RuntimeError("model crashed")

    c = make_client(adapter=MLAdapter(_fake_ml(check_photo), "real"))
    lid = c.get("/api/queue").json()["items"][0]["loan_id"]
    r = c.post("/api/check_photo", data={"loan_id": lid}, files={"image": ("a.png", png_bytes(), "image/png")})
    assert r.status_code == 502
    assert "Model xizmatida xatolik" in r.json()["detail"]


def test_ml_contract_mismatch_is_502(make_client):
    c = make_client(adapter=MLAdapter(_fake_ml(lambda *a, **k: {"oops": 1}), "real"))
    lid = c.get("/api/queue").json()["items"][0]["loan_id"]
    r = c.post("/api/check_photo", data={"loan_id": lid}, files={"image": ("a.png", png_bytes(), "image/png")})
    assert r.status_code == 502


def test_fixtures_are_generated_not_hand_edited():
    on_disk = json.loads(make_fixtures.OUT_PATH.read_text(encoding="utf-8"))
    assert on_disk == json.loads(json.dumps(make_fixtures.build(on_disk["seed"], on_disk["n_loans"]),
                                            ensure_ascii=False))


def test_spa_served_when_dist_exists(make_client, tmp_path):
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<html>maqsad</html>", encoding="utf-8")
    (dist / "assets" / "app.js").write_text("console.log(1)", encoding="utf-8")
    c = make_client(frontend_dist=dist)
    assert c.get("/").text == "<html>maqsad</html>"
    assert c.get("/loan/L0001").text == "<html>maqsad</html>"  # SPA route
    assert c.get("/assets/app.js").text == "console.log(1)"
    assert c.get("/assets/missing.js").status_code == 404
    (tmp_path / "secret.txt").write_text("SECRET", encoding="utf-8")
    assert "SECRET" not in c.get("/%2e%2e/secret.txt").text  # never escapes dist


def test_root_without_dist(client):
    r = client.get("/")
    assert r.status_code == 200 and r.json()["service"] == "maqsad"
