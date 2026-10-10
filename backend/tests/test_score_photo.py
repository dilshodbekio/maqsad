import base64

from .conftest import png_bytes


# ---------- /api/score_loan ----------

def test_score_loan(client, loan_id):
    r = client.post("/api/score_loan", json={"loan_id": loan_id})
    assert r.status_code == 200
    d = r.json()
    assert set(d) == {"risk", "color", "reasons", "model_version", "latency_ms"}
    assert 0 <= d["risk"] <= 1 and d["color"] in ("green", "yellow", "red")
    assert d["model_version"] == "stub"


def test_score_loan_empty_body(client):
    assert client.post("/api/score_loan", json={}).status_code == 422
    assert client.post("/api/score_loan").status_code == 422


def test_score_loan_unknown(client):
    assert client.post("/api/score_loan", json={"loan_id": "L9999"}).status_code == 404


# ---------- /api/check_photo ----------

def _upload(client, loan_id, data=None, name="a.png", mime="image/png", **fields):
    return client.post("/api/check_photo", data={"loan_id": loan_id, **fields},
                       files={"image": (name, data if data is not None else png_bytes(), mime)})


def test_check_photo_multipart(client, loan_id):
    r = _upload(client, loan_id, lat="41.31", lon="69.28", captured_at="2026-10-10T10:00:00Z")
    assert r.status_code == 200, r.text
    d = r.json()
    assert set(d) == {"asset_match", "duplicate_of", "geo_ok", "time_ok", "faces_blurred", "faces_count", "latency_ms"}
    assert d["asset_match"] is None  # no classifier yet: optional, not faked
    assert d["geo_ok"] is True
    assert d["duplicate_of"] == []
    # stored as a sandbox photo; stub returns no blurred image -> nothing on disk, no url
    photos = client.get(f"/api/loan/{loan_id}").json()["photos"]
    assert len(photos) == 1 and photos[0]["url"] is None and photos[0]["source"] == "borrower_upload"


def test_check_photo_json_base64_fallback(client, loan_id):
    b64 = base64.b64encode(png_bytes()).decode()
    r = client.post("/api/check_photo", json={"loan_id": loan_id, "image_base64": f"data:image/png;base64,{b64}"})
    assert r.status_code == 200, r.text
    assert r.json()["geo_ok"] is None and r.json()["time_ok"] is None


def test_check_photo_duplicate_only_within_sandbox(client):
    from fastapi.testclient import TestClient
    q = client.get("/api/queue").json()["items"]
    a, b = q[0]["loan_id"], q[1]["loan_id"]
    img = png_bytes((1, 2, 3))
    assert _upload(client, a, img).json()["duplicate_of"] == []
    assert _upload(client, b, img).json()["duplicate_of"] == [{"loan_id": a, "similarity": 1.0}]
    other = TestClient(client.app)  # new visitor -> new sandbox
    assert _upload(other, b, img).json()["duplicate_of"] == []


def test_check_photo_wrong_type(client, loan_id):
    r = _upload(client, loan_id, b"hello, not an image", name="a.txt", mime="text/plain")
    assert r.status_code == 400
    assert "JPEG, PNG yoki WebP" in r.json()["detail"]


def test_check_photo_empty_file(client, loan_id):
    r = _upload(client, loan_id, b"")
    assert r.status_code == 400


def test_check_photo_too_large(make_client, loan_id):
    small = make_client(max_upload_bytes=1024)
    big = png_bytes() + b"\x00" * 2048
    r = _upload(small, loan_id, big)
    assert r.status_code == 413
    assert "MB" in r.json()["detail"]


def test_check_photo_missing_image(client, loan_id):
    r = client.post("/api/check_photo", data={"loan_id": loan_id}, files={"other": ("x", b"x")})
    assert r.status_code == 400


def test_check_photo_missing_loan_id(client):
    r = client.post("/api/check_photo", files={"image": ("a.png", png_bytes(), "image/png")})
    assert r.status_code == 422


def test_check_photo_unknown_loan(client):
    assert _upload(client, "L9999").status_code == 404


def test_check_photo_bad_coords(client, loan_id):
    assert _upload(client, loan_id, lat="123").status_code == 422


def test_check_photo_rate_limit(make_client, loan_id):
    c = make_client(rate_limit_per_min=2)
    assert _upload(c, loan_id, png_bytes((1, 1, 1))).status_code == 200
    assert _upload(c, loan_id, png_bytes((2, 2, 2))).status_code == 200
    r = _upload(c, loan_id, png_bytes((3, 3, 3)))
    assert r.status_code == 429
    assert "Juda ko'p so'rov" in r.json()["detail"]
