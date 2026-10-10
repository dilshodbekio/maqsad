"""Smoke tests of the public ML interface (ml.api) backed by the money-trail model."""
from __future__ import annotations

import json

import pytest

from ml import api
from ml.money_trail.features import FEATURE_COLUMNS
from ml.money_trail.model import META_PATH

META = json.loads(META_PATH.read_text(encoding="utf-8"))
QUEUE_KEYS = {"loan_id", "borrower_name_masked", "purpose", "amount_uzs", "region", "risk", "color", "top_reason",
              "base_color", "priority", "escalations"}
RANK = {"green": 0, "yellow": 1, "red": 2}
FORBIDDEN = {"is_fraud", "typology", "profile", "stealth", "ring_id", "fraud_visible"}
ACCUSATORY = ("firibgar", "aybdor", "jinoyat", "o'g'ri", "soxta")


@pytest.fixture(scope="module")
def queue():
    return api.get_queue()


def test_queue_sorted_and_colors_by_rank(queue):
    thr = api.get_color_thresholds()
    assert len(queue) == 5000 and (thr["n_red"], thr["n_yellow"], thr["n_loans"]) == (500, 500, 5000)
    pri = [q["priority"] for q in queue]
    assert pri == sorted(pri, reverse=True)                                    # sorted by priority (colour rank + risk)
    assert all(0.0 <= q["risk"] <= 1.0 for q in queue)
    by_risk = sorted(queue, key=lambda q: -q["risk"])                          # money-trail colours: by risk rank
    assert [q["base_color"] for q in by_risk[:500]] == ["red"] * 500
    assert [q["base_color"] for q in by_risk[500:1000]] == ["yellow"] * 500
    assert {q["base_color"] for q in by_risk[1000:]} == {"green"}
    assert by_risk[499]["risk"] >= thr["red"] > by_risk[500]["risk"] >= thr["yellow"] > by_risk[1000]["risk"]
    for q in queue:
        assert set(q) == QUEUE_KEYS and q["top_reason"]
        assert RANK[q["color"]] >= RANK[q["base_color"]]                      # fusion never lowers a colour
        assert q["priority"] == pytest.approx(RANK[q["color"]] + q["risk"])
    assert len({q["loan_id"] for q in queue}) == len(queue)


def test_top_reasons_are_neutral(queue):
    for q in queue[:500]:
        assert not any(w in q["top_reason"].lower() for w in ACCUSATORY), q["top_reason"]


@pytest.mark.parametrize("pos", [0, 1, 700, 2500, 4999])
def test_get_loan(queue, pos):
    item = queue[pos]
    d = api.get_loan(item["loan_id"])
    assert {"loan", "transactions", "photos", "invoice", "risk", "color", "reasons", "evidence",
            "model_version"} <= set(d)
    assert d["model_version"] == "money_trail_v2" and isinstance(d["photos"], list)
    assert d["risk"] == item["risk"] and d["color"] == item["color"] and d["base_color"] == item["base_color"]
    assert d["loan"]["loan_id"] == item["loan_id"] and d["transactions"]
    assert set(d["invoice"]) == {"supplier_name", "supplier_oked", "oked_matches_purpose", "invoice_amount_uzs"}
    assert not FORBIDDEN & set(d["loan"]) and not any(k.startswith("photo_") for k in d["loan"])
    model_reasons = [r for r in d["reasons"] if r.get("source") != "fusion_policy"]
    if d["base_color"] == "red":
        assert model_reasons
    assert len(model_reasons) <= 3
    refs = {e["ref"] for e in d["evidence"]}
    for r in d["reasons"]:
        assert r["text_uz"] and set(r["evidence_refs"]) <= refs
    for r in model_reasons:
        assert 0 < r["weight"] <= 1 and r["code"] in FEATURE_COLUMNS
    assert d["escalations"] == item["escalations"] and d["priority"] == item["priority"]
    for e in d["evidence"]:
        assert e["kind"] in ("txn", "invoice", "graph") and e["summary_uz"] and e["data"]
    raised = [e for e in d["escalations"] if e["raises_color"]]
    expected = raised[0]["text_uz"] if raised else (model_reasons[0]["text_uz"] if model_reasons else item["top_reason"])
    assert item["top_reason"] == expected


def test_score_loan_contract_and_consistency(queue):
    for item in (queue[0], queue[1234], queue[-1]):
        s = api.score_loan(item["loan_id"])
        assert set(s) == {"risk", "color", "reasons", "model_version", "latency_ms"}
        assert s["risk"] == pytest.approx(item["risk"], abs=1e-9) and s["color"] == item["base_color"]  # money-trail only
        assert s["latency_ms"] > 0 and s["model_version"] == "money_trail_v2"
        assert all(set(r) == {"code", "text_uz", "weight"} for r in s["reasons"])


def test_unknown_loan():
    for fn in (api.get_loan, api.score_loan):
        with pytest.raises(api.LoanNotFoundError):
            fn("L-does-not-exist")


def test_features_are_money_trail_only():
    banned = ("photo", "lat", "lon", "geo", "captured")
    assert not [c for c in FEATURE_COLUMNS if any(b in c for b in banned)]
    assert not set(FEATURE_COLUMNS) & FORBIDDEN
    assert META["features"] == FEATURE_COLUMNS


def test_invalid_loan_id_types():
    for fn in (api.get_loan, api.score_loan):
        for bad in (None, ["L1"], {"a": 1}, 5, b"L1"):
            with pytest.raises(api.LoanNotFoundError):
                fn(bad)


@pytest.mark.slow
def test_reasons_do_not_repeat_a_fact_and_texts_are_sane(queue):
    from ml.money_trail.reasons import DUPLICATE_GROUPS
    for item in queue[:300] + queue[2000:2100]:
        d = api.get_loan(item["loan_id"])
        codes = [r["code"] for r in d["reasons"] if r.get("source") != "fusion_policy"]
        for g in DUPLICATE_GROUPS:
            assert len(g & set(codes)) <= 1, codes
        for r in d["reasons"]:
            t = r["text_uz"]
            assert "% i" not in t and "%i" not in t and "atigi 0 kun" not in t and "muntazam" not in t, t


def test_template_edge_cases():
    from ml.money_trail.features import NEVER
    from ml.money_trail.reasons import TEMPLATES as T
    assert "atigi" not in T["sup_age_at_first_payment_days"](518, {}) and "atigi" in T["sup_age_at_first_payment_days"](20, {})
    assert "atigi" not in T["sup_active_span_days"](338, {}) and "atigi" in T["sup_active_span_days"](40, {})
    assert "atigi" not in T["sup_pay_share"](0.85, {}) and "atigi" in T["sup_pay_share"](0.1, {})
    assert "to'lov tushmagan" in T["sup_age_at_first_payment_days"](-1, {})
    assert "atigi 0" not in T["sup_active_span_days"](-1, {}) and "hali harakat" in T["sup_active_span_days"](-1, {})
    assert T["invoice_gap"](-1, {}) == "Hisob-faktura mavjud emas" and "to'lanmagan" in T["invoice_gap"](1.0, {})
    assert "100 foizdan ko'pini" in T["sup_cash_share"](1.3, {}) and "100 foizdan ko'pini" in T["sup_exit_share"](1.5, {})
    assert "100 foizini" in T["sup_cash_share"](1.0, {}) and "foizdan" not in T["sup_cash_share"](1.0, {})        # exactly 100% is a percentage
    assert "78 foizi" in T["cash_share_d1"](0.78, {}) and "24 soat" in T["cash_share_d1"](0.78, {})
    assert "hisob-fakturasida ko'rsatilgan" in T["sup_in_degree"](7, {})
    assert "naqd yechish qayd etilmagan" in T["first_cash_days"](NEVER, {})
