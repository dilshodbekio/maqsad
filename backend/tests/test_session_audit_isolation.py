"""Loan page support: demo officer id per sandbox (/api/session) and the audit view per sandbox.
The hash chain stays global (one chain over every sandbox); only the view is filtered."""
import re
import sqlite3

from fastapi.testclient import TestClient

from .conftest import png_bytes


def _decide(c: TestClient, loan_id: str, action: str) -> int:
    officer = c.get("/api/session").json()["officer_id"]
    r = c.post("/api/decision", json={"loan_id": loan_id, "officer_id": officer, "action": action})
    assert r.status_code == 200
    return r.json()["audit_id"]


def test_session_officer_id_is_stable_per_sandbox(client):
    r = client.get("/api/session")
    assert r.status_code == 200
    officer = r.json()["officer_id"]
    assert re.fullmatch(r"Xodim-[0-9A-F]{4}", officer)
    assert client.get("/api/session").json()["officer_id"] == officer  # same cookie -> same officer
    assert set(r.json()) == {"officer_id"}  # the sandbox id itself is never returned


def test_session_differs_between_sandboxes(client):
    other = TestClient(client.app)
    ids = {client.get("/api/session").json()["officer_id"], other.get("/api/session").json()["officer_id"]}
    assert len(ids) == 2  # 16 bits of a hash per sandbox: a collision for two fixed uuids is not expected


def test_session_wrong_method(client):
    assert client.post("/api/session").status_code == 405


def test_audit_view_is_per_sandbox_chain_is_global(client, loan_id):
    a, b = client, TestClient(client.app)
    a_officer = a.get("/api/session").json()["officer_id"]
    b_officer = b.get("/api/session").json()["officer_id"]
    id_a1 = _decide(a, loan_id, "visit")
    id_b1 = _decide(b, loan_id, "reject")
    up = b.post("/api/check_photo", data={"loan_id": loan_id}, files={"image": ("x.png", png_bytes(), "image/png")})
    assert up.status_code == 200
    id_a2 = _decide(a, loan_id, "approve")

    rows_a = a.get(f"/api/audit?loan_id={loan_id}").json()
    rows_b = b.get(f"/api/audit?loan_id={loan_id}").json()
    assert [(r["actor"], r["action"]) for r in rows_a] == [(a_officer, "visit"), (a_officer, "approve")]
    assert [(r["actor"], r["action"]) for r in rows_b] == [(b_officer, "reject"), ("borrower", "photo_upload")]
    # a third visitor sees nothing for this loan
    assert TestClient(client.app).get(f"/api/audit?loan_id={loan_id}").json() == []

    # one global chain: every row links to the previous row, whichever sandbox wrote it
    db = client.app.state.db
    assert db.audit_verify()
    with sqlite3.connect(db.path) as c:
        rows = c.execute("SELECT id, sandbox_id, prev_hash, hash FROM audit ORDER BY id").fetchall()
    assert [r[0] for r in rows] == [id_a1, id_b1, id_a1 + 2, id_a2]
    assert len({r[1] for r in rows}) == 2
    for prev, cur in zip(rows, rows[1:]):
        assert cur[2] == prev[3]


def test_audit_view_per_sandbox_for_other_loans(client, loan_id):
    other_loan = next(i["loan_id"] for i in client.get("/api/queue").json()["items"] if i["loan_id"] != loan_id)
    _decide(client, loan_id, "visit")
    # the decision on one loan does not show up on another loan of the same sandbox
    assert client.get(f"/api/audit?loan_id={other_loan}").json() == []


def test_decision_records_the_colour_the_officer_saw(client, loan_id, monkeypatch):
    """With an escalation the page shows the fused colour: the audit keeps it, plus the money-trail colour."""
    ml = client.app.state.ml
    real_get_loan = ml.get_loan

    def escalated(lid, sandbox_id=None):
        d = dict(real_get_loan(lid, sandbox_id))
        d.update(color="red", base_color="yellow",
                 escalations=[{"code": "PHOTO_DUPLICATE", "min_color": "red", "text_uz": "x",
                               "evidence_refs": [], "raises_color": True}])
        return d

    monkeypatch.setattr(ml, "get_loan", escalated)
    risk = real_get_loan(loan_id, None)["risk"]
    _decide(client, loan_id, "visit")
    row = client.get(f"/api/audit?loan_id={loan_id}").json()[0]
    assert row["ai_recommendation"] == f"red (risk={risk:.3f}, base=yellow)"


def test_decision_without_escalation_keeps_plain_format(client, loan_id):
    _decide(client, loan_id, "approve")
    rec = client.get(f"/api/audit?loan_id={loan_id}").json()[0]["ai_recommendation"]
    assert re.fullmatch(r"(red|yellow|green) \(risk=\d\.\d{3}\)", rec)
