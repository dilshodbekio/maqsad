"""/api/scenario_presets and /api/score_scenario: backend validation (400), ml error mapping, real-ml round trip."""
import copy
import types
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from backend.app import ml_stub
from backend.app.config import Settings
from backend.app.main import create_app
from backend.app.ml_adapter import MLAdapter

GOOD = {
    "purpose": "chorva", "amount_uzs": 50_000_000, "invoice_amount_uzs": 50_000_000, "region": None,
    "snapshot_day": 60,
    "supplier": {"mode": "new", "oked": "01.41", "registered_days_before": 900},
    "transactions": [{"day": 1, "type": "cash_withdrawal", "amount_uzs": 42_000_000},
                     {"day": 5, "type": "supplier_payment", "amount_uzs": 5_000_000}],
}
SCORE = {"risk": 0.5, "color": "yellow", "reasons": [{"code": "x", "text_uz": "Sabab", "weight": 0.1}],
         "model_version": "fake-1", "latency_ms": 1.0, "percentile": 50.0, "features": {"a": 1.0}}
PRESETS = [{"id": "p1", "title_uz": "Bir", "description_uz": "Tavsif", "scenario": GOOD}]


class ScenarioError(ValueError):  # same name as ml.errors.ScenarioError: the adapter maps it by name
    pass


def _fake_ml(score=None):
    m = types.ModuleType("fake_ml")
    m.MODEL_VERSION = "fake-1"
    m.get_queue, m.get_loan, m.score_loan = ml_stub.get_queue, ml_stub.get_loan, ml_stub.score_loan
    m.check_photo = lambda *a, **k: {"asset_match": None, "duplicate_of": [], "geo_ok": None, "time_ok": None,
                                     "faces_blurred": False, "faces_count": 0, "latency_ms": 1.0}
    m.reset_sandbox = lambda sandbox_id: None
    m.calls = []

    def score_scenario(scn):
        m.calls.append(scn)
        if not scn:
            raise ScenarioError("bo'sh")
        if score:
            return score(scn)
        return dict(SCORE)

    m.score_scenario = score_scenario
    m.get_scenario_presets = lambda: copy.deepcopy(PRESETS)
    return m


@pytest.fixture
def fake(make_client):
    m = _fake_ml()
    c = make_client(adapter=MLAdapter(m, "real"))
    return c, m


def _bad(**changes):
    s = copy.deepcopy(GOOD)
    s.update(changes)
    return s


def test_presets_and_score_pass_through(fake):
    c, m = fake
    assert [{k: v for k, v in p.items() if v is not None} for p in c.get("/api/scenario_presets").json()] == PRESETS
    r = c.post("/api/score_scenario", json=GOOD)
    assert r.status_code == 200, r.text
    assert r.json()["risk"] == 0.5 and r.json()["percentile"] == 50.0
    assert m.calls[-1]["supplier"] == GOOD["supplier"]
    h = c.get("/api/health").json()["functions"]
    assert h["score_scenario"] == "real" and h["get_scenario_presets"] == "real"


@pytest.mark.parametrize("scn", [
    _bad(purpose="noma'lum"),
    _bad(extra_key=1),
    _bad(snapshot_day=1),
    _bad(snapshot_day=365),
    _bad(amount_uzs=0),
    _bad(amount_uzs="50000000"),
    _bad(amount_uzs=True),
    _bad(amount_uzs=1.5),
    _bad(invoice_amount_uzs=600_000_000),
    _bad(supplier={"mode": "other"}),
    _bad(supplier={"mode": "new", "oked": "1.4", "registered_days_before": 1}),
    _bad(supplier={"mode": "new", "oked": "01.41", "registered_days_before": 1, "supplier_id": "S1"}),
    _bad(transactions=[{"day": 1, "type": "cash_withdrawal", "amount_uzs": 600_000_000}]),
    _bad(transactions=[{"day": 61, "type": "cash_withdrawal", "amount_uzs": 1}]),
    _bad(transactions=[{"day": 1, "type": "gift", "amount_uzs": 1}]),
    _bad(transactions=[{"day": 1, "type": "cash_withdrawal", "amount_uzs": 1, "x": 1}]),
    _bad(transactions=[{"day": 1, "type": "income", "amount_uzs": 500_000_000}] * 3),  # inflows > 20x the loan
    _bad(transactions=[{"day": 1, "type": "cash_withdrawal", "amount_uzs": 1}] * 61),
    "matn", [], None,
])
def test_invalid_scenario_is_400_in_uzbek(fake, scn):
    c, m = fake
    n = len(m.calls)
    r = c.post("/api/score_scenario", json=scn)
    assert r.status_code == 400, r.text
    detail = r.json()["detail"]
    assert isinstance(detail, str) and detail
    assert len(m.calls) == n  # rejected before ml


def test_missing_key_and_bad_body(fake):
    c, _ = fake
    s = copy.deepcopy(GOOD)
    del s["purpose"]
    assert c.post("/api/score_scenario", json=s).status_code == 400
    hdr = {"content-type": "application/json"}
    assert c.post("/api/score_scenario", content=b"", headers=hdr).status_code == 400
    assert c.post("/api/score_scenario", content=b"{not json", headers=hdr).status_code == 400


def test_optional_fields_may_be_omitted(fake):
    c, _ = fake
    s = copy.deepcopy(GOOD)
    del s["invoice_amount_uzs"], s["region"]
    s["supplier"] = {"mode": "existing", "supplier_id": "S1500747"}
    s["transactions"] = []
    assert c.post("/api/score_scenario", json=s).status_code == 200


def test_ml_scenario_error_is_400(make_client):
    def boom(scn):
        raise ScenarioError("Noma'lum hudud: 'X'")
    c = make_client(adapter=MLAdapter(_fake_ml(score=boom), "real"))
    r = c.post("/api/score_scenario", json=GOOD)
    assert r.status_code == 400 and "Noma'lum hudud" in r.json()["detail"]


def test_ml_contract_mismatch_is_502(make_client):
    c = make_client(adapter=MLAdapter(_fake_ml(score=lambda s: {"risk": 5}), "real"))
    assert c.post("/api/score_scenario", json=GOOD).status_code == 502


def test_stub_mode_is_503(client):
    assert client.get("/api/health").json()["functions"]["score_scenario"] == "missing"
    assert client.get("/api/scenario_presets").status_code == 503
    assert client.post("/api/score_scenario", json=GOOD).status_code == 503
    assert client.post("/api/score_scenario", json=_bad(purpose="x")).status_code == 400  # validation comes first


# ---------- real ml ----------

ml_api = pytest.importorskip("ml.api", reason="real ml.api not importable")


@pytest.fixture(scope="module")
def real_client(tmp_path_factory):
    a = MLAdapter(ml_api, "real")
    a.warmup()
    assert a.mode == "real", a.warmup_error
    tmp = tmp_path_factory.mktemp("scn")
    settings = replace(Settings.from_env(), data_dir=tmp / "data", frontend_dist=tmp / "no_dist", admin_token="t")
    return TestClient(create_app(settings=settings, adapter=a))


def test_real_presets_score_with_200(real_client):
    presets = real_client.get("/api/scenario_presets").json()
    assert {p["id"] for p in presets} >= {"honest_purchase", "fast_cash_out", "supplier_ring", "asset_resale"}
    for p in presets:
        r = real_client.post("/api/score_scenario", json=p["scenario"])
        assert r.status_code == 200, (p["id"], r.text)
        body = r.json()
        assert 0 <= body["risk"] <= 1 and body["color"] in ("green", "yellow", "red")


def test_real_fast_cash_out_scores_above_honest(real_client):
    by_id = {p["id"]: p["scenario"] for p in real_client.get("/api/scenario_presets").json()}
    honest = real_client.post("/api/score_scenario", json=by_id["honest_purchase"]).json()["risk"]
    cash = real_client.post("/api/score_scenario", json=by_id["fast_cash_out"]).json()["risk"]
    assert cash > honest


def test_real_ml_rejections_are_400(real_client):
    r = real_client.post("/api/score_scenario", json=_bad(region="Atlantida"))
    assert r.status_code == 400 and "hudud" in r.json()["detail"]
    r = real_client.post("/api/score_scenario", json=_bad(supplier={"mode": "existing", "supplier_id": "NOPE"}))
    assert r.status_code == 400
    # an outflow above the available balance is rejected by ml, never capped
    r = real_client.post("/api/score_scenario", json=_bad(
        transactions=[{"day": 1, "type": "cash_withdrawal", "amount_uzs": 400_000_000}]))
    assert r.status_code == 400


def test_real_scenario_is_not_stored(real_client):
    before = real_client.get("/api/queue").json()["total"]
    real_client.post("/api/score_scenario", json=GOOD)
    assert real_client.get("/api/queue").json()["total"] == before
