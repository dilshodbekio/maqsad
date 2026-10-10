def test_health_stub(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {
        "ml_mode": "stub", "model_version": "stub", "warmup": "done",
        "functions": {f: "stub" for f in ("get_queue", "get_loan", "score_loan", "check_photo", "reset_sandbox")},
    }
    assert r.headers["X-ML-Mode"] == "stub"
    assert "sandbox_id" in r.cookies


def test_health_wrong_method(client):
    r = client.post("/api/health")
    assert r.status_code == 405
    assert r.json()["detail"] == "Bu metod qo'llab-quvvatlanmaydi."


def test_unknown_api_path_is_json_404(client):
    r = client.get("/api/nope")
    assert r.status_code == 404
    assert r.json() == {"detail": "API endpoint topilmadi."}


def test_queue_sorted_and_complete(client):
    r = client.get("/api/queue")
    assert r.status_code == 200
    items = r.json()
    assert len(items) == 40
    assert [i["risk"] for i in items] == sorted((i["risk"] for i in items), reverse=True)
    assert set(items[0]) == {"loan_id", "borrower_name_masked", "purpose", "amount_uzs", "region",
                             "risk", "color", "top_reason"}
    assert {i["color"] for i in items} == {"green", "yellow", "red"}
    assert len({i["purpose"] for i in items}) == 5


def test_queue_wrong_method(client):
    assert client.post("/api/queue").status_code == 405


def test_loan_detail(client, loan_id):
    r = client.get(f"/api/loan/{loan_id}")
    assert r.status_code == 200
    d = r.json()
    for key in ("loan", "transactions", "photos", "invoice", "risk", "color", "reasons", "evidence", "model_version"):
        assert key in d
    assert d["model_version"] == "stub"
    assert d["loan"]["loan_id"] == loan_id
    assert all({"code", "text_uz", "weight", "evidence_refs"} <= set(x) for x in d["reasons"])
    refs = {e["ref"] for e in d["evidence"]}
    assert all(set(x["evidence_refs"]) <= refs for x in d["reasons"])
    assert all(e["kind"] in ("txn", "invoice", "graph") for e in d["evidence"])
    assert all({"counterparty_id", "counterparty_kind"} <= set(t) for t in d["transactions"])


def test_loan_unknown_is_404(client):
    r = client.get("/api/loan/L9999")
    assert r.status_code == 404
    assert "Kredit topilmadi" in r.json()["detail"]


def test_loan_bad_id_is_422(client):
    r = client.get("/api/loan/bad%20id!")
    assert r.status_code == 422
    assert r.json()["detail"].startswith("So'rov ma'lumotlari noto'g'ri")
