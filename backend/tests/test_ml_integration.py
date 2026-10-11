"""Integration with the REAL ml.api (money_trail model + photo module).

Skipped when the ml dependencies are not installed. Runs in CI (ml/requirements.txt installed) and in .venv.
Also guards the schema: if ml starts returning keys that backend/app/schemas.py does not declare, these
tests fail, so a shape change is noticed instead of being silently dropped by the response model.
"""
from dataclasses import replace

import pytest

try:
    import ml.api as ml_api
except Exception as e:  # noqa: BLE001 - ImportError / OSError / SyntaxError (env-dependent)
    pytest.skip(f"real ml.api not importable: {e!r}", allow_module_level=True)

from fastapi.testclient import TestClient

from backend.app.config import REPO_ROOT, Settings
from backend.app.main import create_app
from backend.app.ml_adapter import FUNCTIONS, MLAdapter
from backend.app.schemas import (CheckPhotoResponse, Escalation, Evidence, Invoice, Loan, LoanDetail, LoanReason,
                                 Photo, QueueItem, Transaction)


@pytest.fixture(scope="module")
def adapter():
    a = MLAdapter(ml_api, "real")
    a.warmup()
    assert (a.mode, a.warmup_state) == ("real", "done"), a.warmup_error
    return a


@pytest.fixture
def client(adapter, tmp_path):
    settings = replace(Settings.from_env(), data_dir=tmp_path / "data", frontend_dist=tmp_path / "no_dist",
                       admin_token="t", rate_limit_per_min=1000)
    return TestClient(create_app(settings=settings, adapter=adapter))


@pytest.fixture(scope="module")
def photo_loan():
    """A portfolio loan that has a photo, and that photo's index entry."""
    from ml.photo.index import get_index
    idx = get_index()
    im = next(i for i in idx.images if len(i["loans"]) == 1)
    return im["loans"][0], im


def _all_items(client) -> list:
    """Every queue item of the client's sandbox, page by page (max limit 200)."""
    out, total = [], None
    while total is None or len(out) < total:
        page = client.get("/api/queue", params={"limit": 200, "offset": len(out)}).json()
        total = page["total"]
        out += page["items"]
    return out


def _keys_declared(model, items):
    undeclared = {k for item in items for k in item} - set(model.model_fields)
    assert not undeclared, f"ml returns keys not declared in {model.__name__}: {sorted(undeclared)}"


def test_health_all_real(client):
    h = client.get("/api/health").json()
    assert isinstance(h.pop("clip_loaded"), bool)  # True where the CLIP weights are available
    assert h == {"ml_mode": "real", "model_version": ml_api.MODEL_VERSION, "warmup": "done",
                 "functions": {f: "real" for f in FUNCTIONS}}


def test_ml_output_matches_schemas(adapter, photo_loan):
    q = adapter.get_queue()
    _keys_declared(QueueItem, q[:50])
    _keys_declared(Escalation, [e for i in q for e in i["escalations"]])
    escalated = next(i["loan_id"] for i in q if i["escalations"])  # fusion reasons (weight None) + photo evidence
    for lid in (q[0]["loan_id"], q[-1]["loan_id"], photo_loan[0], escalated):
        raw = adapter.get_loan(lid)
        assert set(raw) == set(LoanDetail.model_fields)
        LoanDetail.model_validate(raw)
        _keys_declared(Loan, [raw["loan"]])
        _keys_declared(Transaction, raw["transactions"])
        _keys_declared(LoanReason, raw["reasons"])
        _keys_declared(Evidence, raw["evidence"])
        _keys_declared(Photo, raw["photos"])
        if raw["invoice"] is not None:
            _keys_declared(Invoice, [raw["invoice"]])
        refs = {e["ref"] for e in raw["evidence"]}
        assert all(set(r["evidence_refs"]) <= refs for r in raw["reasons"])
    assert adapter.get_loan(photo_loan[0])["photos"], "loan from the photo index has no photos"


def test_queue_loan_score_end_to_end(client):
    q = client.get("/api/queue")
    assert q.status_code == 200
    items = q.json()["items"]
    assert len(items) > 0 and [i["priority"] for i in items] == sorted((i["priority"] for i in items), reverse=True)
    assert all(i["color"] in ("green", "yellow", "red") and i["base_color"] for i in items)
    lid = items[0]["loan_id"]
    d = client.get(f"/api/loan/{lid}").json()
    assert d["model_version"] == ml_api.MODEL_VERSION and d["risk"] == items[0]["risk"]
    s = client.post("/api/score_loan", json={"loan_id": lid}).json()
    assert s["model_version"] == ml_api.MODEL_VERSION and s["color"] == d["color"]
    assert client.get("/api/loan/NO_SUCH_LOAN").status_code == 404


def test_portfolio_photo_is_served_from_ml_assets(client, photo_loan):
    lid, im = photo_loan
    ph = client.get(f"/api/loan/{lid}").json()["photos"][0]
    assert ph["source"] == "portfolio" and ph["path"] == f"ml/assets/photos/{im['file']}"
    assert ph["url"] == f"/api/asset/photos/{im['file']}"
    r = client.get(ph["url"])
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert r.content == (REPO_ROOT / ph["path"]).read_bytes()


def test_check_photo_real_with_asset_image(client, photo_loan):
    owner, im = photo_loan
    img = (REPO_ROOT / "ml" / "assets" / "photos" / im["file"]).read_bytes()
    other = next(i["loan_id"] for i in client.get("/api/queue").json()["items"] if i["loan_id"] != owner)
    r = client.post("/api/check_photo", data={"loan_id": other, "lat": "41.3", "lon": "69.2",
                                               "captured_at": "2026-10-10T10:00:00Z"},
                    files={"image": (im["file"], img, "image/jpeg")})
    assert r.status_code == 200, r.text
    d = r.json()
    CheckPhotoResponse.model_validate(d)
    assert owner in {x["loan_id"] for x in d["duplicate_of"]}  # same photo already used by another loan
    assert isinstance(d["faces_count"], int) and isinstance(d["geo_ok"], bool) and isinstance(d["time_ok"], bool)

    # only the blurred, re-encoded JPEG is stored, and only the uploading sandbox can read it
    up = [p for p in client.get(f"/api/loan/{other}").json()["photos"] if p["source"] == "borrower_upload"]
    assert len(up) == 1 and up[0]["url"].startswith("/api/photo/")
    stored = client.get(up[0]["url"])
    assert stored.status_code == 200 and stored.headers["content-type"] == "image/jpeg"
    assert stored.content != img
    assert TestClient(client.app).get(up[0]["url"]).status_code == 404


def test_check_photo_corrupt_jpeg_is_400(client):
    lid = client.get("/api/queue").json()["items"][0]["loan_id"]
    r = client.post("/api/check_photo", data={"loan_id": lid},
                    files={"image": ("x.jpg", b"\xff\xd8\xff\xe0" + b"\x00" * 64, "image/jpeg")})
    assert r.status_code == 400
    assert "Rasmni o'qib bo'lmadi" in r.json()["detail"]


def test_admin_reset_with_real_ml(client, photo_loan):
    owner, im = photo_loan
    img = (REPO_ROOT / "ml" / "assets" / "photos" / im["file"]).read_bytes()
    q = [i["loan_id"] for i in client.get("/api/queue").json()["items"] if i["loan_id"] != owner]
    client.post("/api/check_photo", data={"loan_id": q[0]}, files={"image": ("a.jpg", img, "image/jpeg")})
    assert client.post("/api/admin/reset", headers={"X-Admin-Token": "t"}).json() == {"ok": True}
    # after reset the sandbox upload is forgotten: only portfolio duplicates remain
    r = client.post("/api/check_photo", data={"loan_id": q[1]}, files={"image": ("a.jpg", img, "image/jpeg")})
    assert q[0] not in {x["loan_id"] for x in r.json()["duplicate_of"]}


def test_escalated_loan_served_by_api(client):
    """Loans with fusion reasons (weight null) and photo evidence must not 502."""
    lid = next(i["loan_id"] for i in _all_items(client) if i["escalations"])
    d = client.get(f"/api/loan/{lid}")
    assert d.status_code == 200, d.text
    d = d.json()
    assert any(r.get("source") == "fusion_policy" and r["weight"] is None for r in d["reasons"])
    assert any(e["kind"] == "photo" for e in d["evidence"])


def test_sandbox_upload_escalates_only_own_view_until_reset(client, photo_loan):
    owner, im = photo_loan
    img = (REPO_ROOT / "ml" / "assets" / "photos" / im["file"]).read_bytes()
    q = client.get("/api/queue", params={"color": "green"}).json()["items"]
    target = next(i for i in q if i["loan_id"] != owner and i["color"] == "green" and not i["escalations"])
    r = client.post("/api/check_photo", data={"loan_id": target["loan_id"]},
                    files={"image": ("a.jpg", img, "image/jpeg")})
    assert r.status_code == 200 and r.json()["duplicate_of"]

    mine = client.get(f"/api/loan/{target['loan_id']}").json()
    assert mine["color"] == "red" and mine["base_color"] == "green"
    assert "PHOTO_DUPLICATE" in {e["code"] for e in mine["escalations"]}
    assert mine["risk"] == target["risk"]  # the photo never changes risk

    other = TestClient(client.app).get(f"/api/loan/{target['loan_id']}").json()
    assert other["color"] == "green" and other["escalations"] == []  # other sandboxes unaffected

    assert client.post("/api/admin/reset", headers={"X-Admin-Token": "t"}).status_code == 200
    after = client.get(f"/api/loan/{target['loan_id']}").json()
    assert after["color"] == "green" and after["escalations"] == []


def test_queue_page_counts_match_color_thresholds(client):
    """Fresh sandbox = default view: queue counts equal ml's colour budget (incl. photo escalations)."""
    page = client.get("/api/queue").json()
    lg = client.get("/api/queue/legend")
    assert lg.status_code == 200, lg.text
    lg = lg.json()
    assert len(page["items"]) == 50 and page["total"] == lg["n_loans"]
    assert page["counts"] == {"red": lg["n_red_total"], "yellow": lg["n_yellow_total"],
                              "green": lg["n_loans"] - lg["n_red_total"] - lg["n_yellow_total"]}
    red = client.get("/api/queue", params={"color": "red", "limit": 200}).json()
    assert red["total"] == lg["n_red_total"] and all(i["color"] == "red" for i in red["items"])


def test_queue_facets_real(client):
    f = client.get("/api/queue/facets").json()
    assert set(f["purposes"]) == {"tikuv_dastgohi", "chorva", "issiqxona", "parranda", "asalarichilik"}
    assert len(f["regions"]) > 1 and f["regions"] == sorted(f["regions"])
    region = f["regions"][0]
    page = client.get("/api/queue", params={"region": region}).json()
    assert page["total"] > 0 and all(i["region"] == region for i in page["items"])
