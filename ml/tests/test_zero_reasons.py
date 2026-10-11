"""Part 0c: no reason states a zero value or a fact that does not push the risk up."""
from __future__ import annotations

import re

import numpy as np
import pytest

from ml import api
from ml.money_trail.features import NEVER
from ml.money_trail.reasons import TEMPLATES, ZERO_HIDDEN, build_reasons, is_skipped
from ml.money_trail.service import get_portfolio

ZERO_TEXT = re.compile(r"(?<![\d.,])0 (foiz|ta |kun)|\b0\.0 kun|atigi 0")


@pytest.fixture(scope="module")
def P():
    return get_portfolio()


def test_is_skipped_rules():
    assert is_skipped("cash_share_d7", 0.0) and is_skipped("cash_share_d7", 0.004) and not is_skipped("cash_share_d7", 0.006)
    assert is_skipped("p2p_out_count", 0) and not is_skipped("p2p_out_count", 1)
    assert is_skipped("income_rate_per_month", -1.0) and not is_skipped("income_rate_per_month", 0.0)   # 'no income' is a fact
    assert is_skipped("sup_active_span_days", 0.2) and not is_skipped("sup_active_span_days", -1) and not is_skipped("sup_active_span_days", 40)
    assert is_skipped("invoice_gap", 0.0) and not is_skipped("invoice_gap", -1) and not is_skipped("invoice_gap", 0.9)
    assert not is_skipped("sup_pay_share", 0.0)                                                         # 'no payment' is a fact
    assert all(c in TEMPLATES for c in ZERO_HIDDEN)


def test_zero_valued_contributions_are_never_shown():
    feats = ["cash_share_d1", "p2p_out_count", "invoice_gap", "sup_pay_share"]
    x = np.array([0.0, 0.0, 0.0, 0.0])
    r = build_reasons(feats, x, np.array([0.5, 0.4, 0.3, 0.2]), {"supplier_oked": "01.41"}, with_refs=False)
    assert [i["code"] for i in r] == ["sup_pay_share"]                  # the only one that states a (non-zero-quantity) fact
    assert r[0]["text_uz"] == "Ta'minotchiga to'lov o'tkazilmagan"


def test_only_risk_raising_features_are_candidates():
    feats = ["cash_share_d1", "sup_cash_share"]
    r = build_reasons(feats, np.array([0.9, 0.9]), np.array([-0.5, 0.0]), {"supplier_oked": "01.41"}, with_refs=False)
    assert r == []


def test_no_reason_in_the_queue_or_in_sampled_loans_states_a_zero(P):
    q = api.get_queue()
    assert not [x["top_reason"] for x in q if ZERO_TEXT.search(x["top_reason"])]
    for item in q[::25]:
        for r in api.get_loan(item["loan_id"])["reasons"]:
            assert not ZERO_TEXT.search(r["text_uz"]), r["text_uz"]
            assert r.get("weight") is None or r["weight"] > 0                     # weights are shares of a positive push


def test_honest_preset_has_no_zero_reason_and_scenario_reasons_are_clean():
    for p in api.get_scenario_presets():
        for r in api.score_scenario(p["scenario"])["reasons"]:
            assert not ZERO_TEXT.search(r["text_uz"]), (p["id"], r["text_uz"])
    honest = next(p for p in api.get_scenario_presets() if p["id"] == "honest_purchase")
    reasons = api.score_scenario(honest["scenario"])["reasons"]
    assert all("0 foiz" not in r["text_uz"] for r in reasons)


def test_zero_day_facts_use_dedicated_texts():
    assert "kunning o'zida" in TEMPLATES["first_cash_days"](0.0, {})
    assert "kunning o'zida" in TEMPLATES["first_sup_pay_days"](0.2, {})
    assert "kunida ro'yxatdan o'tgan" in TEMPLATES["sup_age_at_first_payment_days"](0.0, {})
    assert "yechish qayd etilmagan" in TEMPLATES["first_cash_days"](NEVER, {})
