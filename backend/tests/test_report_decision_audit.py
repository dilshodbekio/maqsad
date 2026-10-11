import json
import sqlite3

import pytest
from fastapi.testclient import TestClient

from backend.app import llm

from .conftest import draft

# ---------- /api/draft_report ----------


def test_draft_report_llm_not_configured(client, loan_id):
    r = client.post("/api/draft_report", json={"loan_id": loan_id})
    assert r.status_code == 503
    assert "tayyor emas" in r.json()["detail"]


def test_draft_report_fact_check(make_client, monkeypatch, loan_id):
    c = make_client(llm_base_url="http://llm.invalid/v1", llm_model="test-model")
    ev = c.get(f"/api/loan/{loan_id}").json()["evidence"]
    txn = next(e for e in ev if e["kind"] == "txn")
    amount = txn["data"]["amount_uzs"]
    llm_answer = {
        "text_uz": "Qoralama matni.",
        "facts": [
            {"claim": f"Kredit summasi {amount:,} so'm".replace(",", " "), "evidence_ref": txn["ref"]},
            {"claim": "Summa 123 456 789 so'm", "evidence_ref": txn["ref"]},  # number not in evidence
            {"claim": "Mavjud bo'lmagan dalil", "evidence_ref": "txn:NOPE"},  # unknown ref
        ],
    }
    monkeypatch.setattr(llm, "_chat", lambda settings, messages: (json.dumps(llm_answer), 100, 50))
    d = draft(c, loan_id)
    assert set(d) == {"text_uz", "facts", "model", "latency_ms", "cost_usd", "cached", "input_tokens", "output_tokens",
                      "generated_at", "lang"}
    assert [f["unverified"] for f in d["facts"]] == [False, True, True]
    assert d["model"] == "test-model"
    assert d["cost_usd"] is None  # price not configured -> not guessed


def test_draft_report_llm_error_is_a_failed_job(make_client, monkeypatch, loan_id):
    c = make_client(llm_base_url="http://llm.invalid/v1", llm_model="test-model")
    monkeypatch.setattr(llm, "_chat", lambda settings, messages: ("not json", None, None))
    j = draft(c, loan_id)
    assert j["status"] == "error" and "tayyor emas" in j["error"]


def test_draft_report_bad_input(client):
    assert client.post("/api/draft_report", json={}).status_code == 422
    assert client.post("/api/draft_report", json={"loan_id": "L9999"}).status_code == 404


# ---------- /api/decision + /api/audit ----------

def test_decision_and_audit(client, loan_id):
    r = client.post("/api/decision", json={"loan_id": loan_id, "officer_id": "xodim-1",
                                           "action": "visit", "comment": "joyiga borish"})
    assert r.status_code == 200
    assert isinstance(r.json()["audit_id"], int)
    rows = client.get(f"/api/audit?loan_id={loan_id}").json()
    assert len(rows) == 1
    row = rows[0]
    assert set(row) == {"ts", "actor", "action", "ai_recommendation", "model_version", "inputs_hash"}
    assert row["actor"] == "xodim-1" and row["action"] == "visit"
    assert row["ai_recommendation"].split(" ")[0] in ("green", "yellow", "red")
    assert row["model_version"] == "stub"


def test_decision_bad_action(client, loan_id):
    r = client.post("/api/decision", json={"loan_id": loan_id, "officer_id": "x", "action": "guilty"})
    assert r.status_code == 422


def test_decision_empty_body_and_unknown_loan(client):
    assert client.post("/api/decision", json={}).status_code == 422
    r = client.post("/api/decision", json={"loan_id": "L9999", "officer_id": "x", "action": "approve"})
    assert r.status_code == 404


def test_audit_bad_input(client):
    assert client.get("/api/audit").status_code == 422
    assert client.get("/api/audit?loan_id=L9999").status_code == 404


def test_audit_sandbox_isolation(client, loan_id):
    client.post("/api/decision", json={"loan_id": loan_id, "officer_id": "a", "action": "approve"})
    other = TestClient(client.app)
    assert other.get(f"/api/audit?loan_id={loan_id}").json() == []
    assert len(client.get(f"/api/audit?loan_id={loan_id}").json()) == 1


def test_audit_hash_chain_and_append_only(client, loan_id):
    for action in ("visit", "approve"):
        client.post("/api/decision", json={"loan_id": loan_id, "officer_id": "a", "action": action})
    db = client.app.state.db
    assert db.audit_verify()
    with sqlite3.connect(db.path) as c:
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            c.execute("UPDATE audit SET action = 'reject'")
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            c.execute("DELETE FROM audit")
        # tampering that bypasses the trigger is detected by the chain
        c.execute("DROP TRIGGER audit_no_update")
        c.execute("UPDATE audit SET action = 'reject' WHERE id = 1")
    assert not db.audit_verify()


# ---------- /api/admin/reset ----------

def test_admin_reset_requires_token(make_client):
    c = make_client(admin_token="s3cret")
    assert c.post("/api/admin/reset").status_code == 403
    assert c.post("/api/admin/reset", headers={"X-Admin-Token": "wrong"}).status_code == 403


def test_admin_reset_disabled_without_admin_token(client):
    assert client.post("/api/admin/reset", headers={"X-Admin-Token": ""}).status_code == 403


def test_admin_reset_clears_photos(make_client, loan_id):
    from .conftest import png_bytes
    c = make_client(admin_token="s3cret")
    c.post("/api/check_photo", data={"loan_id": loan_id}, files={"image": ("a.png", png_bytes(), "image/png")})
    assert len(c.get(f"/api/loan/{loan_id}").json()["photos"]) == 1
    r = c.post("/api/admin/reset", headers={"X-Admin-Token": "s3cret"})
    assert r.status_code == 200 and r.json() == {"ok": True}
    assert c.get(f"/api/loan/{loan_id}").json()["photos"] == []


# ---------- draft cache + reasoning_effort + price ----------

def _counting_llm(monkeypatch):
    calls = []

    def fake(settings, messages):
        calls.append(settings.llm_reasoning_effort)
        return json.dumps({"text_uz": "Matn", "facts": []}), 1000, 500

    monkeypatch.setattr(llm, "_chat", fake)
    return calls


def test_draft_is_cached_and_regenerate_bypasses(make_client, monkeypatch, loan_id):
    c = make_client(llm_base_url="http://llm.invalid/v1", llm_model="m",
                    llm_price_in_per_mtok=1.0, llm_price_out_per_mtok=2.0)
    calls = _counting_llm(monkeypatch)
    first = draft(c, loan_id)
    assert first["cached"] is False and first["cost_usd"] == pytest.approx(0.002)  # 1000/1e6*1 + 500/1e6*2
    assert (first["input_tokens"], first["output_tokens"]) == (1000, 500)
    again = draft(c, loan_id)
    assert again["cached"] is True and again["cost_usd"] == 0.0 and again["input_tokens"] is None and again["text_uz"] == first["text_uz"]
    assert len(calls) == 1
    fresh = draft(c, loan_id, regenerate=True)
    assert fresh["cached"] is False and len(calls) == 2
    assert draft(c, loan_id)["cached"] is True  # the regenerated draft replaced the cached one
    c.cookies.clear()  # another sandbox: the loan has no sandbox evidence, so the draft is SHARED
    assert draft(c, loan_id)["cached"] is True
    assert len(calls) == 2


def test_failed_draft_is_not_cached(make_client, monkeypatch, loan_id):
    c = make_client(llm_base_url="http://llm.invalid/v1", llm_model="m")
    monkeypatch.setattr(llm, "_chat", lambda s, m: ("not json", None, None))
    assert draft(c, loan_id)["status"] == "error"
    calls = _counting_llm(monkeypatch)
    assert draft(c, loan_id)["cached"] is False
    assert len(calls) == 1


def test_reasoning_effort_is_sent_when_set(monkeypatch):
    from dataclasses import replace
    from backend.app.config import Settings
    sent = {}

    class R:
        status_code, is_error = 200, False
        def json(self):
            return {"choices": [{"message": {"content": "{}"}}], "usage": {}}

    monkeypatch.setattr(llm.httpx, "post", lambda url, json, **k: sent.update(json) or R())
    base = replace(Settings.from_env(), llm_base_url="http://x/v1", llm_model="m", llm_api_key="")
    llm._chat(replace(base, llm_reasoning_effort="minimal"), [])
    assert sent["reasoning_effort"] == "minimal"
    sent.clear()
    llm._chat(replace(base, llm_reasoning_effort=""), [])
    assert "reasoning_effort" not in sent


def test_price_env_names(monkeypatch):
    from backend.app.config import Settings
    monkeypatch.setenv("LLM_PRICE_IN_PER_1M", "0.5")
    monkeypatch.setenv("LLM_PRICE_OUT_PER_1M", "2.5")
    s = Settings.from_env()
    assert (s.llm_price_in_per_mtok, s.llm_price_out_per_mtok) == (0.5, 2.5)
