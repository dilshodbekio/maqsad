"""Draft queue: parallel requests, dedup, 429 retry, timeout, shared vs sandbox cache, pre-warm, job privacy."""
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from backend.app import draft_queue, llm, ml_stub
from backend.app.draft_queue import DraftQueue, MSG_BUSY
from backend.app.ml_adapter import MLAdapter

from .conftest import draft, png_bytes

LLM = {"llm_base_url": "http://llm.invalid/v1", "llm_model": "m"}


def _result(loan_id):
    return {"text_uz": f"Matn {loan_id}", "facts": [], "model": "m", "latency_ms": 1.0, "cost_usd": None,
            "input_tokens": 10, "output_tokens": 5}


class FakeLLM:
    """Replaces llm.draft_report: records the call order, optionally blocks the first call until released."""

    def __init__(self, monkeypatch, block_first=False):
        self.calls, self.gate, self.first_started = [], threading.Event(), threading.Event()
        self.block_first = block_first
        if not block_first:
            self.gate.set()
        monkeypatch.setattr(llm, "draft_report", self)

    def __call__(self, settings, loan_id, evidence, reasons, db=None):
        self.calls.append(loan_id)
        if len(self.calls) == 1:
            self.first_started.set()
        self.gate.wait(10)
        return _result(loan_id)


def _loans(c, n):
    return [i["loan_id"] for i in c.get("/api/queue", params={"limit": n}).json()["items"]]


def _post(c, loan_id, **body):
    return c.post("/api/draft_report", json={"loan_id": loan_id, **body})


def _wait(pred, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.01)
    return False


def _idle(c):
    """Wait until the app's queue has nothing queued or running: no task may outlive its test (the LLM is patched)."""
    def idle():
        st = c.app.state.drafts.stats()
        return st["queued"] == 0 and not st["running"]
    assert _wait(idle), "draft queue did not drain"


def _poll_done(c, job_id, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        j = c.get(f"/api/draft_report/job/{job_id}").json()
        if j["status"] in ("done", "error"):
            return j
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_ten_parallel_requests_all_finish_without_errors(make_client, monkeypatch):
    c = make_client(**LLM)
    fake = FakeLLM(monkeypatch)
    ids = _loans(c, 10)

    def one(lid):
        client = TestClient(c.app)  # its own sandbox
        return draft(client, lid)

    with ThreadPoolExecutor(10) as ex:
        results = list(ex.map(one, ids))
    assert all("text_uz" in r and "error" not in r for r in results)
    assert sorted(fake.calls) == sorted(ids)  # one LLM call per loan, all through the queue


def test_waiting_requests_see_queued_with_position(make_client, monkeypatch):
    c = make_client(**LLM)
    fake = FakeLLM(monkeypatch, block_first=True)
    a, b, d = _loans(c, 3)
    jobs = [_post(c, x).json() for x in (a, b, d)]
    assert [r["status"] for r in jobs][0] in ("queued", "running")
    assert fake.first_started.wait(5)
    views = [c.get(f"/api/draft_report/job/{j['job_id']}").json() for j in jobs]
    assert [v["status"] for v in views] == ["running", "queued", "queued"]
    assert [v["position"] for v in views] == [0, 2, 3]  # the running call counts as one ahead
    assert all(v["eta_s"] is not None for v in views[1:])
    fake.gate.set()
    assert all(_poll_done(c, j["job_id"])["status"] == "done" for j in jobs)


def test_same_key_requests_share_one_task(make_client, monkeypatch):
    c = make_client(**LLM)
    fake = FakeLLM(monkeypatch, block_first=True)
    other = TestClient(c.app)
    lid = _loans(c, 1)[0]
    j1, j2 = _post(c, lid).json(), _post(other, lid).json()
    assert j1["job_id"] != j2["job_id"]  # each requester has their own ticket
    assert fake.first_started.wait(5)
    fake.gate.set()
    r1, r2 = _poll_done(c, j1["job_id"]), _poll_done(other, j2["job_id"])
    assert r1["status"] == r2["status"] == "done" and r1["result"]["text_uz"] == r2["result"]["text_uz"]
    assert fake.calls == [lid]  # one LLM call for both


def test_job_is_visible_only_to_its_sandbox(make_client, monkeypatch):
    c = make_client(**LLM)
    FakeLLM(monkeypatch)
    job = _post(c, _loans(c, 1)[0]).json()
    stranger = TestClient(c.app)
    assert stranger.get(f"/api/draft_report/job/{job['job_id']}").status_code == 404
    assert c.get(f"/api/draft_report/job/{job['job_id']}").status_code == 200
    assert c.get(f"/api/draft_report/job/{'0' * 32}").status_code == 404
    assert c.get("/api/draft_report/job/not-an-id").status_code == 422


def test_429_keeps_the_job_queued_and_retries(make_client, monkeypatch):
    monkeypatch.setattr(draft_queue, "BACKOFF_MIN_S", 0.05)
    monkeypatch.setattr(draft_queue, "BACKOFF_MAX_S", 0.2)
    c = make_client(**LLM)
    attempts = []

    def chat(settings, messages):
        attempts.append(1)
        if len(attempts) <= 2:
            raise llm.LLMRateLimited("HTTP 429", 0.01)
        import json
        return json.dumps({"text_uz": "ok", "facts": []}), 10, 5

    monkeypatch.setattr(llm, "_chat", chat)
    job = _post(c, _loans(c, 1)[0]).json()
    seen = set()
    end = time.monotonic() + 10
    while time.monotonic() < end:
        j = c.get(f"/api/draft_report/job/{job['job_id']}").json()
        seen.add(j["status"])
        if j["status"] in ("done", "error"):
            break
        time.sleep(0.01)
    assert j["status"] == "done" and j["result"]["text_uz"] == "ok"
    assert "error" not in seen and len(attempts) == 3


def test_job_older_than_the_timeout_reports_the_busy_message(make_client, monkeypatch):
    c = make_client(**LLM)
    c.app.state.drafts.job_timeout_s = 0.2
    fake = FakeLLM(monkeypatch, block_first=True)
    job = _post(c, _loans(c, 1)[0]).json()
    assert fake.first_started.wait(5)
    time.sleep(0.3)
    j = c.get(f"/api/draft_report/job/{job['job_id']}").json()
    assert j["status"] == "error" and j["error"] == MSG_BUSY and "band" in j["error"]
    fake.gate.set()
    _idle(c)


def test_too_long_a_queue_is_the_busy_message(make_client, monkeypatch):
    c = make_client(**LLM)
    c.app.state.drafts.max_queue = 2
    fake = FakeLLM(monkeypatch, block_first=True)
    ids = _loans(c, 5)
    codes = []
    for lid in ids:
        r = _post(c, lid)
        codes.append(r.status_code)
        if lid == ids[0]:
            assert fake.first_started.wait(5)  # the first leaves the queue and runs
    assert codes == [202, 202, 202, 503, 503]
    assert MSG_BUSY in r.json()["detail"]
    fake.gate.set()
    _idle(c)


def test_regenerate_once_a_minute_per_sandbox(make_client, monkeypatch):
    c = make_client(**LLM)
    FakeLLM(monkeypatch)
    lid = _loans(c, 1)[0]
    draft(c, lid)
    assert _post(c, lid, regenerate=True).status_code == 202
    r = _post(c, lid, regenerate=True)
    assert r.status_code == 429 and "daqiqada" in r.json()["detail"]
    _idle(c)


# ---------- shared vs sandbox cache ----------

def _adapter_with_photo_evidence():
    uploaded = set()
    m = types.ModuleType("fake_ml")
    m.MODEL_VERSION = "fake-1"
    m.get_queue, m.score_loan = ml_stub.get_queue, ml_stub.score_loan
    m.reset_sandbox = lambda sandbox_id: None

    def get_loan(loan_id, sandbox_id=None):
        d = ml_stub.get_loan(loan_id)
        if sandbox_id in uploaded:  # the sandbox's own photo adds evidence
            d = {**d, "evidence": list(d["evidence"]) + [
                {"ref": f"photo:sandbox:{loan_id}:1", "kind": "photo", "summary_uz": "Rasm yuklandi", "data": {}}]}
        return d

    def check_photo(loan_id, image_bytes, lat=None, lon=None, captured_at=None, sandbox_id=None):
        if sandbox_id:  # the adapter probes check_photo at warm-up without a sandbox
            uploaded.add(sandbox_id)
        return {"asset_match": None, "duplicate_of": [], "geo_ok": None, "time_ok": None, "faces_blurred": False,
                "faces_count": 0, "latency_ms": 1.0}

    m.get_loan, m.check_photo = get_loan, check_photo
    return MLAdapter(m, "real")


def test_shared_cache_without_sandbox_evidence_sandbox_cache_with(make_client, monkeypatch):
    c = make_client(adapter=_adapter_with_photo_evidence(), **LLM)
    fake = FakeLLM(monkeypatch)
    b = TestClient(c.app)
    lid = _loans(c, 1)[0]
    first = draft(c, lid)
    assert first["cached"] is False
    assert draft(b, lid)["cached"] is True  # another sandbox, no extra evidence: the shared draft
    assert fake.calls == [lid]
    # sandbox A uploads a photo: its evidence differs -> its own draft (not the shared one)
    assert c.post("/api/check_photo", data={"loan_id": lid},
                  files={"image": ("a.png", png_bytes(), "image/png")}).status_code == 200
    mine = draft(c, lid)
    assert mine["cached"] is False and fake.calls == [lid, lid]
    assert draft(c, lid)["cached"] is True  # cached under A's sandbox now
    assert draft(b, lid)["cached"] is True and len(fake.calls) == 2  # B still gets the shared draft
    assert draft(TestClient(c.app), lid)["text_uz"] == first["text_uz"]


def test_cache_survives_a_restart(make_client, monkeypatch):
    c = make_client(**LLM)
    fake = FakeLLM(monkeypatch)
    lid = _loans(c, 1)[0]
    draft(c, lid)
    c2 = make_client(**LLM)  # same data_dir, a brand-new app and queue
    assert draft(c2, lid)["cached"] is True and fake.calls == [lid]


def test_admin_reset_clears_sandbox_cache_but_keeps_shared_unless_asked(make_client, monkeypatch):
    c = make_client(adapter=_adapter_with_photo_evidence(), admin_token="tok", **LLM)
    FakeLLM(monkeypatch)
    lid = _loans(c, 1)[0]
    draft(c, lid)  # shared
    c.post("/api/check_photo", data={"loan_id": lid}, files={"image": ("a.png", png_bytes(), "image/png")})
    draft(c, lid)  # A's own
    h = {"X-Admin-Token": "tok"}
    assert c.post("/api/admin/reset", headers=h).status_code == 200
    db = c.app.state.db

    def count():
        with db.conn() as cn:
            return cn.execute("SELECT COUNT(*) FROM draft_cache").fetchone()[0]

    assert count() == 1  # the shared draft stays, the sandbox one is gone
    assert c.post("/api/admin/reset?clear_shared_cache=true", headers=h).status_code == 200
    assert count() == 0


# ---------- pre-warm ----------

def test_prewarm_needs_the_admin_token(make_client):
    c = make_client(admin_token="tok", **LLM)
    assert c.post("/api/admin/prewarm").status_code == 403
    assert c.post("/api/admin/prewarm", headers={"X-Admin-Token": "bad"}).status_code == 403
    assert c.get("/api/admin/prewarm").status_code == 403
    c2 = make_client(admin_token="", **LLM)
    assert c2.post("/api/admin/prewarm", headers={"X-Admin-Token": ""}).status_code == 403


def test_prewarm_fills_the_shared_cache_and_skips_cached(make_client, monkeypatch):
    c = make_client(admin_token="tok", **LLM)
    fake = FakeLLM(monkeypatch)
    h = {"X-Admin-Token": "tok"}
    top = _loans(c, 4)
    draft(c, top[0])  # already cached
    assert c.post("/api/admin/prewarm?n=4", headers=h).status_code == 202
    assert _wait(lambda: not c.get("/api/admin/prewarm", headers=h).json()["running"])
    st = c.get("/api/admin/prewarm", headers=h).json()
    assert (st["total"], st["skipped"], st["done"], st["failed"]) == (4, 1, 3, 0)
    assert sorted(fake.calls) == sorted(top)  # one call each, none repeated
    assert all(draft(TestClient(c.app), lid)["cached"] for lid in top)  # any visitor gets them instantly


def test_user_requests_go_before_prewarm_tasks(make_client, monkeypatch):
    c = make_client(admin_token="tok", **LLM)
    fake = FakeLLM(monkeypatch, block_first=True)
    h = {"X-Admin-Token": "tok"}
    top = _loans(c, 6)
    assert c.post("/api/admin/prewarm?n=3", headers=h).status_code == 202
    assert fake.first_started.wait(5)  # a pre-warm call is running (it is never interrupted)
    user_loan = top[5]
    job = _post(c, user_loan).json()
    v = c.get(f"/api/draft_report/job/{job['job_id']}").json()
    assert v["position"] == 2  # behind the running call only, ahead of the other pre-warm tasks
    fake.gate.set()
    assert _poll_done(c, job["job_id"])["status"] == "done"
    assert _wait(lambda: not c.get("/api/admin/prewarm", headers=h).json()["running"])
    assert fake.calls[1] == user_loan  # right after the running pre-warm call, before the remaining ones


def test_user_joining_a_prewarm_task_moves_it_up(make_client, monkeypatch):
    c = make_client(admin_token="tok", **LLM)
    fake = FakeLLM(monkeypatch, block_first=True)
    h = {"X-Admin-Token": "tok"}
    top = _loans(c, 3)
    c.post("/api/admin/prewarm?n=3", headers=h)
    assert fake.first_started.wait(5)
    job = _post(c, top[2]).json()  # the last pre-warm loan: same key -> same task, now user priority
    assert c.get(f"/api/draft_report/job/{job['job_id']}").json()["position"] == 2
    fake.gate.set()
    assert _poll_done(c, job["job_id"])["status"] == "done"
    assert _wait(lambda: not c.get("/api/admin/prewarm", headers=h).json()["running"])  # no tasks outlive the test
    assert fake.calls.count(top[2]) == 1


# ---------- the sliding window (fake clock, no sleeping) ----------

def test_window_never_exceeds_rpm_and_eta_follows_it():
    now = [1000.0]
    q = DraftQueue(types.SimpleNamespace(), None, rpm=4, clock=lambda: now[0])
    assert q._slot_wait(now[0]) == 0
    q._calls.extend([now[0] - 50, now[0] - 40, now[0] - 30, now[0] - 20])  # 4 calls in the last minute
    assert q._slot_wait(now[0]) == pytest.approx(10.0)  # the oldest leaves the window in 10 s
    assert q._eta(now[0], 1) == 10
    assert q._eta(now[0], 2) == 20  # the 2nd needs the next one to leave: 20 s
    now[0] += 11
    assert q._slot_wait(now[0]) == 0
    q._blocked_until = now[0] + 7  # provider back-off after a 429
    assert q._slot_wait(now[0]) == pytest.approx(7.0)


def test_exhausted_daily_quota_fails_waiting_jobs_at_once_and_refuses_new_ones(make_client, monkeypatch):
    c = make_client(**LLM)
    calls = []

    def chat(settings, messages):
        calls.append(1)
        raise llm.LLMRateLimited("HTTP 429 (daily quota)", 6549.0)  # "retry in 1h49m": longer than a job may wait

    monkeypatch.setattr(llm, "_chat", chat)
    a = _loans(c, 1)[0]
    r1 = _poll_done(c, _post(c, a).json()["job_id"])
    assert r1["status"] == "error" and r1["error"] == MSG_BUSY
    assert len(calls) == 1
    refused = _post(c, _loans(c, 3)[2])
    assert refused.status_code == 503 and MSG_BUSY in refused.json()["detail"]
    assert len(calls) == 1


def test_retry_after_parsing():
    from types import SimpleNamespace as R
    assert llm._retry_after(R(headers={"retry-after": "7"}, text="")) == 7
    assert llm._retry_after(R(headers={}, text="... Please retry in 2.105190017s.")) == pytest.approx(2.105190017)
    assert llm._retry_after(R(headers={}, text="Please retry in 1h49m9.552035437s.")) == pytest.approx(6549.552, abs=0.01)
    assert llm._retry_after(R(headers={}, text="Please retry in 3m.")) == 180
    assert llm._retry_after(R(headers={}, text="no hint")) == llm.DEFAULT_RETRY_AFTER_S
