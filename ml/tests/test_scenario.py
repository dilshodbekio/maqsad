"""Scenario scoring: judges build a loan + transactions, the frozen model scores it live."""
from __future__ import annotations

import copy

import pytest

from ml import api
from ml.errors import ScenarioError
from ml.money_trail.features import FEATURE_COLUMNS
from ml.money_trail.scenario import _features, validate
from ml.money_trail.service import get_portfolio


@pytest.fixture(scope="module")
def P():
    return get_portfolio()


@pytest.fixture(scope="module")
def presets():
    return {p["id"]: p for p in api.get_scenario_presets()}


def test_presets_score_without_error(presets):
    assert len(presets) == 4
    for p in presets.values():
        assert p["title_uz"] and p["description_uz"]
        r = api.score_scenario(p["scenario"])
        assert set(r) == {"risk", "color", "reasons", "model_version", "latency_ms", "percentile", "features"}
        assert 0 <= r["risk"] <= 1 and r["color"] in ("green", "yellow", "red") and 0 <= r["percentile"] <= 100
        assert r["model_version"] == "money_trail_v2" and r["reasons"]
        assert all(set(x) == {"code", "text_uz", "weight"} for x in r["reasons"])
        assert set(r["features"]) == set(FEATURE_COLUMNS)


def test_presets_use_a_real_demo_supplier_for_the_ring(presets, P):
    sup = presets["supplier_ring"]["scenario"]["supplier"]
    assert sup["mode"] == "existing" and sup["supplier_id"] in P.suppliers.index


def test_moving_cash_out_from_day_1_to_day_30_lowers_risk(presets):
    scn = copy.deepcopy(presets["fast_cash_out"]["scenario"])
    cash = next(t for t in scn["transactions"] if t["type"] == "cash_withdrawal")
    assert cash["day"] == 1
    early = api.score_scenario(scn)
    cash["day"] = 30
    late = api.score_scenario(scn)
    assert late["risk"] < early["risk"]
    assert late["features"]["cash_share_d1"] < early["features"]["cash_share_d1"]
    assert late["features"]["first_cash_days"] == 30 and early["features"]["first_cash_days"] == 1


def test_fast_cash_out_scores_higher_than_honest_purchase(presets):
    assert api.score_scenario(presets["fast_cash_out"]["scenario"])["risk"] > \
        api.score_scenario(presets["honest_purchase"]["scenario"])["risk"]


def test_features_equal_the_full_portfolio_copy(presets, P):
    for p in presets.values():
        v = validate(p["scenario"], P)
        assert (_features(P, v) - _features(P, v, full=True)).abs().max() == 0


def test_demo_portfolio_unchanged_after_scenarios(presets, P):
    before_q = api.get_queue()
    shapes = (P.loans.shape, P.tx.shape, P.suppliers.shape, P.features.shape, len(P.risk))
    for p in presets.values():
        api.score_scenario(p["scenario"])
    assert api.get_queue() == before_q
    assert shapes == (P.loans.shape, P.tx.shape, P.suppliers.shape, P.features.shape, len(P.risk))
    assert "L-SCENARIO" not in P.loans.index and "S-SCENARIO" not in P.suppliers.index
    assert not P.tx["account_id"].astype(str).str.contains("SCENARIO").any()


def test_latency_under_500_ms(presets):
    for p in presets.values():
        api.score_scenario(p["scenario"])                       # warm
    for p in presets.values():
        assert api.score_scenario(p["scenario"])["latency_ms"] < 500


def base():
    return dict(purpose="chorva", amount_uzs=10_000_000, snapshot_day=30,
                supplier=dict(mode="new", oked="01.41", registered_days_before=100),
                transactions=[dict(day=1, type="cash_withdrawal", amount_uzs=1_000_000)])


@pytest.mark.parametrize("mutate", [
    lambda s: s.update(purpose="uchar_likop"),
    lambda s: s.update(purpose=None),
    lambda s: s.update(amount_uzs=0),
    lambda s: s.update(amount_uzs=-5),
    lambda s: s.update(amount_uzs="10"),
    lambda s: s.update(amount_uzs=True),
    lambda s: s.update(amount_uzs=1.5),
    lambda s: s.update(snapshot_day=731),
    lambda s: s.update(snapshot_day=-1),
    lambda s: s.update(region="Mars"),
    lambda s: s["supplier"].update(mode="other"),
    lambda s: s.update(supplier="S1"),
    lambda s: s.update(supplier=dict(mode="existing", supplier_id="S-nope")),
    lambda s: s.update(supplier=dict(mode="existing", supplier_id=["x"])),
    lambda s: s["supplier"].update(oked="abc"),
    lambda s: s["supplier"].update(registered_days_before=-1),
    lambda s: s["transactions"][0].update(type="teleport"),
    lambda s: s["transactions"][0].update(amount_uzs=0),
    lambda s: s["transactions"][0].update(day=31),
    lambda s: s["transactions"][0].update(day=-1),
    lambda s: s["transactions"][0].update(day=1.5),
    lambda s: s["transactions"][0].update(counterparty_kind="alien"),
    lambda s: s.update(transactions=[dict(day=1, type="income", amount_uzs=1000)] * 61),
    lambda s: s.update(transactions="none"),
    lambda s: s.update(transactions=[5]),
    lambda s: s["transactions"][0].update(amount_uzs=10_000_001),          # more than the account holds
])
def test_invalid_scenarios_raise(mutate):
    s = base()
    mutate(s)
    with pytest.raises(ScenarioError) as e:
        api.score_scenario(s)
    assert isinstance(e.value, ValueError) and str(e.value)


def test_not_a_dict_raises():
    for bad in (None, [], "x", 5):
        with pytest.raises(ScenarioError):
            api.score_scenario(bad)


def test_overdraft_is_rejected_with_a_clear_uzbek_message_not_capped():
    s = base()
    s["transactions"] = [dict(day=1, type="cash_withdrawal", amount_uzs=6_000_000),
                         dict(day=2, type="supplier_payment", amount_uzs=5_000_000)]
    with pytest.raises(ScenarioError, match="hisobda mablag' yetarli emas"):
        api.score_scenario(s)


def test_inflow_on_the_same_day_covers_an_outflow_and_valid_edge_values():
    s = base()
    s["transactions"] = [dict(day=5, type="cash_withdrawal", amount_uzs=10_500_000),
                         dict(day=5, type="p2p_in", amount_uzs=1_000_000)]
    s["snapshot_day"] = 5
    assert api.score_scenario(s)["risk"] >= 0
    s2 = base()
    s2.update(snapshot_day=730, transactions=[])
    assert api.score_scenario(s2)["features"]["first_cash_days"] == 400       # no cash withdrawal at all


def test_young_loan_without_transactions_is_scored():
    s = base()
    s.update(snapshot_day=0, transactions=[])
    r = api.score_scenario(s)
    assert 0 <= r["risk"] <= 1


def test_existing_supplier_in_degree_includes_the_scenario_loan(presets, P):
    scn = presets["supplier_ring"]["scenario"]
    sid = scn["supplier"]["supplier_id"]
    n_demo = int((P.loans["supplier_id"] == sid).sum())
    assert api.score_scenario(scn)["features"]["sup_in_degree"] == n_demo + 1


def test_scenario_tracks_the_inputs_not_a_lookup(presets):
    """Different inputs -> different scores (no hard-coded answers)."""
    s = copy.deepcopy(presets["fast_cash_out"]["scenario"])
    risks = set()
    for day in (1, 3, 7, 14, 30):
        s["transactions"][0]["day"] = day
        risks.add(round(api.score_scenario(s)["risk"], 6))
    assert len(risks) >= 3
