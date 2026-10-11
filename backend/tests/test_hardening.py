"""Open-demo hardening: client IP, regenerate scope and limits, request-size limit, decision actor, rate-limiter memory."""
import json
import tempfile
import time

import pytest

from backend.app import llm
from backend.app.main import _client_ip
from backend.app.ratelimit import RateLimiter

from .conftest import draft, png_bytes

LLM = dict(llm_base_url="http://llm.invalid/v1", llm_model="m")


def _fake_llm(monkeypatch):
    calls = []

    def fake(settings, messages):
        calls.append(1)
        return json.dumps({"text_uz": f"Matn {len(calls)}", "facts": []}), 10, 5

    monkeypatch.setattr(llm, "_chat", fake)
    return calls


# ---------- 1. client IP ----------

class _Req:
    def __init__(self, headers, host="10.0.0.9"):
        self.headers = headers
        self.client = type("C", (), {"host": host})() if host else None


def test_client_ip_takes_the_rightmost_forwarded_entry():
    assert _client_ip(_Req({"x-forwarded-for": "1.1.1.1, 2.2.2.2, 203.0.113.7"})) == "203.0.113.7"
    assert _client_ip(_Req({"x-forwarded-for": "203.0.113.7"})) == "203.0.113.7"
    assert _client_ip(_Req({})) == "10.0.0.9"
    assert _client_ip(_Req({}, host=None)) == "unknown"
    assert _client_ip(_Req({"x-forwarded-for": "1.1.1.1, "})) == "10.0.0.9"


def test_forged_left_forwarded_for_does_not_dodge_the_ip_limit(make_client):
    c = make_client(rate_limit_per_min=1000, rate_limit_ip_per_min=3)
    lid = c.get("/api/queue").json()["items"][0]["loan_id"]
    codes = []
    for i in range(5):
        c.cookies.clear()  # a new sandbox every time, a different forged left value every time
        r = c.post("/api/check_photo", data={"loan_id": lid}, files={"image": ("a.png", png_bytes((i, 0, 0)), "image/png")},
                   headers={"X-Forwarded-For": f"9.9.9.{i}, 198.51.100.1"})
        codes.append(r.status_code)
    assert codes == [200, 200, 200, 429, 429]


# ---------- 2. regenerate ----------

def test_regenerate_never_overwrites_the_shared_cache(make_client, monkeypatch):
    c = make_client(**LLM)
    calls = _fake_llm(monkeypatch)
    lid = c.get("/api/queue").json()["items"][0]["loan_id"]
    shared = draft(c, lid)
    assert shared["text_uz"] == "Matn 1"
    mine = draft(c, lid, regenerate=True)
    assert mine["text_uz"] == "Matn 2" and len(calls) == 2
    assert draft(c, lid)["text_uz"] == "Matn 2"  # my own sandbox sees my regenerated draft
    c.cookies.clear()  # another visitor
    other = draft(c, lid)
    assert other["cached"] is True and other["text_uz"] == "Matn 1"  # still the shared one
    assert len(calls) == 2


def test_regenerate_is_limited_per_ip_across_sandboxes(make_client, monkeypatch):
    c = make_client(**LLM)
    _fake_llm(monkeypatch)
    lid = c.get("/api/queue").json()["items"][0]["loan_id"]
    draft(c, lid)
    codes = []
    for _ in range(5):
        c.cookies.clear()  # fresh sandbox each time: the per-sandbox limit cannot help
        codes.append(c.post("/api/draft_report", json={"loan_id": lid, "regenerate": True}).status_code)
    assert codes == [202, 202, 202, 429, 429]
    c.cookies.clear()
    assert c.post("/api/draft_report", json={"loan_id": lid, "regenerate": True},
                  headers={"X-Forwarded-For": "198.51.100.77"}).status_code == 202  # another IP is unaffected


# ---------- 3. request-size limit ----------

def test_other_post_over_64kb_is_413(client, loan_id):
    r = client.post("/api/score_loan", json={"loan_id": loan_id, "pad": "x" * 70_000})
    assert r.status_code == 413 and "KB" in r.json()["detail"]
    assert client.post("/api/score_loan", json={"loan_id": loan_id}).status_code == 200


def test_photo_over_limit_is_413_by_actual_bytes_even_without_content_length(client, loan_id):
    limit = client.app.state.settings.max_upload_bytes * 4 // 3 + 65536
    sent = []

    def chunks():  # chunked: no Content-Length, the bytes are counted as they arrive
        yield b'{"loan_id": "' + loan_id.encode() + b'", "image_base64": "'
        for _ in range(limit // 65536 + 2):
            sent.append(1)
            yield b"A" * 65536

    r = client.post("/api/check_photo", content=chunks(), headers={"Content-Type": "application/json"})
    assert r.status_code == 413
    assert "MB" in r.json()["detail"]


def test_declared_oversize_is_413_immediately(client, loan_id):
    big = b"\x89PNG\r\n\x1a\n" + b"0" * (client.app.state.settings.max_upload_bytes * 4 // 3 + 70000)
    r = client.post("/api/check_photo", data={"loan_id": loan_id}, files={"image": ("a.png", big, "image/png")})
    assert r.status_code == 413


def test_photo_upload_is_never_spooled_to_disk(client, loan_id, monkeypatch):
    rolled = []
    orig = tempfile.SpooledTemporaryFile.rollover
    monkeypatch.setattr(tempfile.SpooledTemporaryFile, "rollover", lambda self: (rolled.append(1), orig(self))[1])
    data = png_bytes() + b"\0" * (3 * 1024 * 1024)  # 3 MB: above Starlette's default 1 MB spool size
    r = client.post("/api/check_photo", data={"loan_id": loan_id}, files={"image": ("a.png", data, "image/png")})
    assert r.status_code == 200, r.text
    assert rolled == []


# ---------- 4. decision actor ----------

def test_decision_actor_comes_from_the_sandbox_not_the_body(client, loan_id):
    officer = client.get("/api/session").json()["officer_id"]
    r = client.post("/api/decision", json={"loan_id": loan_id, "officer_id": "Boss-Forged", "action": "visit"})
    assert r.status_code == 200
    assert client.post("/api/decision", json={"loan_id": loan_id, "action": "approve"}).status_code == 200  # field is optional
    actors = [e["actor"] for e in client.get(f"/api/audit?loan_id={loan_id}").json()]
    assert actors == [officer, officer]


# ---------- 5. rate limiter memory ----------

def test_ratelimiter_drops_idle_keys(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    rl = RateLimiter(window_s=60)
    for i in range(500):
        assert rl.allow(f"ip:{i}", 5)
    assert len(rl._hits) == 500
    now[0] += 61
    assert rl.allow("ip:new", 5)
    assert list(rl._hits) == ["ip:new"]
    for _ in range(4):
        assert rl.allow("ip:new", 5)
    assert not rl.allow("ip:new", 5)  # the limit itself still works
