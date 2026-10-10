"""Smoke tests of the public ML interface (ml.api) backed by the money-trail model."""
from __future__ import annotations

import json

import pytest

from ml import api
from ml.money_trail.features import FEATURE_COLUMNS
from ml.money_trail.model import META_PATH

META = json.loads(META_PATH.read_text(encoding="utf-8"))
RED, YELLOW = META["thresholds"]["red"], META["thresholds"]["yellow"]
QUEUE_KEYS = {"loan_id", "borrower_name_masked", "purpose", "amount_uzs", "region", "risk", "color", "top_reason"}
FORBIDDEN = {"is_fraud", "typology", "profile", "stealth", "ring_id", "fraud_visible"}
ACCUSATORY = ("firibgar", "aybdor", "jinoyat", "o'g'ri", "soxta")


@pytest.fixture(scope="module")
def queue():
    return api.get_queue()


def test_queue_sorted_and_colors_match_thresholds(queue):
    assert len(queue) == 5000
    risks = [q["risk"] for q in queue]
    assert risks == sorted(risks, reverse=True)
    assert all(0.0 <= r <= 1.0 for r in risks)
    for q in queue:
        assert set(q) == QUEUE_KEYS
        want = "red" if q["risk"] >= RED else "yellow" if q["risk"] >= YELLOW else "green"
        assert q["color"] == want
        assert q["top_reason"]
    assert {q["color"] for q in queue} == {"green", "yellow", "red"}
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
    assert d["model_version"] == "money_trail_v1" and isinstance(d["photos"], list)
    assert d["risk"] == item["risk"] and d["color"] == item["color"]
    assert d["loan"]["loan_id"] == item["loan_id"] and d["transactions"]
    assert set(d["invoice"]) == {"supplier_name", "supplier_oked", "oked_matches_purpose", "invoice_amount_uzs"}
    assert not FORBIDDEN & set(d["loan"]) and not any(k.startswith("photo_") for k in d["loan"])
    if d["color"] == "red":
        assert d["reasons"]
    assert len(d["reasons"]) <= 3
    refs = {e["ref"] for e in d["evidence"]}
    for r in d["reasons"]:
        assert r["text_uz"] and 0 < r["weight"] <= 1 and r["code"] in FEATURE_COLUMNS
        assert set(r["evidence_refs"]) <= refs
    for e in d["evidence"]:
        assert e["kind"] in ("txn", "invoice", "graph") and e["summary_uz"] and e["data"]
    if d["reasons"]:
        assert d["reasons"][0]["text_uz"] == item["top_reason"]


def test_score_loan_contract_and_consistency(queue):
    for item in (queue[0], queue[1234], queue[-1]):
        s = api.score_loan(item["loan_id"])
        assert set(s) == {"risk", "color", "reasons", "model_version", "latency_ms"}
        assert s["risk"] == pytest.approx(item["risk"], abs=1e-9) and s["color"] == item["color"]
        assert s["latency_ms"] > 0 and s["model_version"] == "money_trail_v1"
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
