"""Extra scenario tests: randomised equivalence with the full-portfolio copy, every documented validation rule,
percentile / reasons properties, and "no hidden state" checks."""
from __future__ import annotations

import copy
import random
import re

import numpy as np
import pytest

from ml import api
from ml.errors import ScenarioError
from ml.generator.config import PURPOSES, REGIONS
from ml.money_trail import scenario as S
from ml.money_trail.features import FEATURE_COLUMNS
from ml.money_trail.service import get_portfolio


@pytest.fixture(scope="module")
def P():
    return get_portfolio()


def base():
    return dict(purpose="chorva", amount_uzs=10_000_000, snapshot_day=30,
                supplier=dict(mode="new", oked="01.41", registered_days_before=100),
                transactions=[dict(day=1, type="cash_withdrawal", amount_uzs=1_000_000)])


def random_scenario(rng: random.Random, P) -> dict:
    amount = rng.randint(1, 200) * 1_000_000
    snap = rng.choice([2, 3, 5, 30, 90, 200, 364])
    if rng.random() < 0.5:
        sup = dict(mode="existing", supplier_id=rng.choice(list(P.suppliers.index)))
    else:
        sup = dict(mode="new", oked=f"{rng.randint(1, 99):02d}.{rng.randint(0, 99):02d}",
                   registered_days_before=rng.choice([0, 10, 500, 5000]))
    budget, txs = amount, []                          # outflows total <= loan amount, so always affordable
    for _ in range(rng.randint(0, 12)):
        typ = rng.choice(S.TX_TYPES)
        a = rng.randint(1, max(1, amount // 10))
        if typ not in S.INFLOWS:
            if a > budget:
                continue
            budget -= a
        t = dict(day=rng.randint(0, snap), type=typ, amount_uzs=a)
        if rng.random() < 0.3:
            t["counterparty_kind"] = rng.choice(S.CP_KINDS)
        txs.append(t)
    scn = dict(purpose=rng.choice(PURPOSES), amount_uzs=amount, snapshot_day=snap, supplier=sup, transactions=txs)
    if rng.random() < 0.5:
        scn["region"] = rng.choice([r[0] for r in REGIONS])
    return scn


# ---------------------------------------------------------------- equivalence with the full copy
@pytest.mark.slow
@pytest.mark.parametrize("seed", range(10))
def test_random_scenarios_features_equal_full_copy(P, seed):
    scn = random_scenario(random.Random(seed), P)
    v = S.validate(scn, P)
    fast, full = S._features(P, v), S._features(P, v, full=True)
    diff = (fast.astype(float) - full.astype(float)).abs().max()
    assert diff == 0, (scn, (fast.astype(float) - full.astype(float)).abs().sort_values().tail(3))


@pytest.mark.slow
def test_random_existing_supplier_scenarios_cover_all_branches(P):
    """Several different existing suppliers (incl. one without any demo loan, if present) match the full copy."""
    with_loans = set(P.loans["supplier_id"])
    sups = list(P.suppliers.index)
    no_loan = [s for s in sups if s not in with_loans][:1]
    with_l = sorted(with_loans)[:3]
    for sid in no_loan + with_l:
        scn = base()
        scn["supplier"] = dict(mode="existing", supplier_id=sid)
        scn["transactions"].append(dict(day=2, type="supplier_payment", amount_uzs=5_000_000))
        v = S.validate(scn, P)
        assert (S._features(P, v).astype(float) - S._features(P, v, full=True).astype(float)).abs().max() == 0


def test_score_scenario_is_deterministic(P):
    scn = random_scenario(random.Random(99), P)
    a, b = api.score_scenario(scn), api.score_scenario(copy.deepcopy(scn))
    for r in (a, b):
        r.pop("latency_ms")
    assert a == b


def test_input_is_not_mutated(P):
    scn = random_scenario(random.Random(5), P)
    before = copy.deepcopy(scn)
    api.score_scenario(scn)
    assert scn == before


def test_unknown_keys_are_rejected_and_listed():
    for where, mutate, name in (
            ("ssenariyda", lambda s: s.update(label="fraud", risk=0.99), "label, risk"),
            ("ta'minotchida", lambda s: s["supplier"].update(foo=1), "foo"),
            ("1-tranzaksiyada", lambda s: s["transactions"][0].update(bar="x"), "bar"),
            ("ssenariyda", lambda s: s.update(counterpartykind="x"), "counterpartykind")):
        s = base()
        mutate(s)
        with pytest.raises(ScenarioError) as e:
            api.score_scenario(s)
        assert where in str(e.value) and name in str(e.value), str(e.value)


def test_existing_supplier_rejects_new_supplier_fields():
    P = get_portfolio()
    sid = sorted(set(P.loans["supplier_id"]))[0]
    b = base(); b["supplier"] = dict(mode="existing", supplier_id=sid, oked="99.99", registered_days_before=1)
    with pytest.raises(ScenarioError, match="oked"):
        api.score_scenario(b)


# ---------------------------------------------------------------- validation rules
@pytest.mark.parametrize("mutate", [
    lambda s: s.pop("purpose"),
    lambda s: s.pop("amount_uzs"),
    lambda s: s.pop("snapshot_day"),
    lambda s: s.pop("supplier"),
    lambda s: s.pop("transactions"),
    lambda s: s.update(amount_uzs=10 ** 11 + 1),
    lambda s: s.update(amount_uzs=float("nan")),
    lambda s: s.update(amount_uzs=float("inf")),
    lambda s: s.update(amount_uzs=None),
    lambda s: s.update(snapshot_day=True),
    lambda s: s.update(snapshot_day="30"),
    lambda s: s.update(snapshot_day=None),
    lambda s: s.update(region=5),
    lambda s: s.update(region=""),
    lambda s: s.update(region="mars"),
    lambda s: s["supplier"].pop("mode"),
    lambda s: s["supplier"].update(mode=None),
    lambda s: s["supplier"].pop("oked"),
    lambda s: s["supplier"].update(oked=146),
    lambda s: s["supplier"].update(oked="1.41"),
    lambda s: s["supplier"].update(oked="01.411"),
    lambda s: s["supplier"].update(oked="01-41"),
    lambda s: s["supplier"].update(oked=" 01.41"),
    lambda s: s["supplier"].pop("registered_days_before"),
    lambda s: s["supplier"].update(registered_days_before=40_001),
    lambda s: s["supplier"].update(registered_days_before=2.5),
    lambda s: s["supplier"].update(registered_days_before="5"),
    lambda s: s["supplier"].update(registered_days_before=True),
    lambda s: s.update(supplier=dict(mode="existing")),
    lambda s: s.update(supplier=dict(mode="existing", supplier_id=None)),
    lambda s: s.update(supplier=dict(mode="existing", supplier_id=5)),
    lambda s: s.update(supplier=dict(mode="existing", supplier_id="")),
    lambda s: s.update(supplier=None),
    lambda s: s.update(supplier=[]),
    lambda s: s.update(transactions=None),
    lambda s: s.update(transactions={}),
    lambda s: s.update(transactions=(dict(day=1, type="income", amount_uzs=5),)),
    lambda s: s.update(transactions=[None]),
    lambda s: s.update(transactions=[[]]),
    lambda s: s.update(transactions=[dict(day=1, amount_uzs=5)]),                     # type missing
    lambda s: s.update(transactions=[dict(day=1, type=None, amount_uzs=5)]),
    lambda s: s.update(transactions=[dict(type="income", amount_uzs=5)]),             # day missing
    lambda s: s.update(transactions=[dict(day=1, type="income")]),                    # amount missing
    lambda s: s.update(transactions=[dict(day=1, type="income", amount_uzs=-1)]),
    lambda s: s.update(transactions=[dict(day=1, type="income", amount_uzs=10 ** 11 + 1)]),
    lambda s: s.update(transactions=[dict(day=1, type="income", amount_uzs="5")]),
    lambda s: s.update(transactions=[dict(day=True, type="income", amount_uzs=5)]),
    lambda s: s.update(transactions=[dict(day=1, type="income", amount_uzs=5, counterparty_kind="")]),
    lambda s: s.update(transactions=[dict(day=1, type="income", amount_uzs=5, counterparty_kind=3)]),
    lambda s: s.update(transactions=[dict(day=1, type="income", amount_uzs=5, counterparty_kind="Person")]),
    lambda s: s.update(transactions=[dict(day=1, type="Income", amount_uzs=5)]),
    lambda s: s.update(transactions=[dict(day=1, type="disbursement", amount_uzs=5)]),   # not user-enterable
    lambda s: s.update(transactions=[dict(day=1, type="income", amount_uzs=5)] * 100),
])
def test_more_invalid_scenarios_raise_scenario_error(mutate):
    s = base()
    mutate(s)
    with pytest.raises(ScenarioError) as e:
        api.score_scenario(s)
    assert isinstance(e.value, ValueError) and str(e.value).strip()


def test_scenario_error_is_exported_by_api():
    assert api.ScenarioError is ScenarioError and issubclass(ScenarioError, ValueError)


def test_transaction_limit_boundary_60_ok_61_rejected():
    s = base()
    s["transactions"] = [dict(day=1, type="income", amount_uzs=1000)] * 60
    assert 0 <= api.score_scenario(s)["risk"] <= 1
    s["transactions"] = [dict(day=1, type="income", amount_uzs=1000)] * 61
    with pytest.raises(ScenarioError, match="60"):
        api.score_scenario(s)


def test_day_boundaries_0_and_snapshot_day_are_valid_and_snapshot_plus_one_is_not():
    for day in (0, 30):
        s = base()
        s["transactions"][0]["day"] = day
        assert 0 <= api.score_scenario(s)["risk"] <= 1
    s = base()
    s["transactions"][0]["day"] = 31
    with pytest.raises(ScenarioError):
        api.score_scenario(s)


def test_snapshot_and_amount_and_registration_boundaries():
    for snap in (2, 364):
        s = base(); s.update(snapshot_day=snap); s["transactions"][0]["day"] = 0
        api.score_scenario(s)
    for snap in (0, 1, 365, 730):
        s = base(); s.update(snapshot_day=snap)
        with pytest.raises(ScenarioError, match="bir yilgacha"):
            api.score_scenario(s)
    s = base(); s.update(amount_uzs=1, transactions=[]); api.score_scenario(s)
    s = base(); s.update(amount_uzs=10 ** 11, transactions=[]); api.score_scenario(s)
    s = base(); s["supplier"]["registered_days_before"] = 0; api.score_scenario(s)
    s = base(); s["supplier"]["registered_days_before"] = 40_000; api.score_scenario(s)


def test_integer_valued_floats_and_numpy_ints_are_accepted():
    s = base()
    s["amount_uzs"] = 10_000_000.0
    s["snapshot_day"] = np.int64(30)
    s["transactions"][0]["amount_uzs"] = np.float64(1_000_000)
    assert api.score_scenario(s)["risk"] == api.score_scenario(base())["risk"]


@pytest.mark.slow
def test_every_documented_region_is_accepted(P):
    for name in [r[0] for r in REGIONS]:
        s = base(); s["region"] = name
        assert 0 <= api.score_scenario(s)["risk"] <= 1
    s = base(); s["region"] = None
    api.score_scenario(s)


@pytest.mark.parametrize("kind", ["person", "business", "external", "supplier", "none"])
def test_every_documented_counterparty_kind_is_accepted(kind):
    s = base()
    s["transactions"][0]["counterparty_kind"] = kind
    api.score_scenario(s)
    s["transactions"][0]["counterparty_kind"] = None            # None == omitted
    api.score_scenario(s)


@pytest.mark.parametrize("typ", S.TX_TYPES)
def test_every_documented_transaction_type_is_accepted(typ):
    s = base()
    s["transactions"] = [dict(day=2, type=typ, amount_uzs=1_000_000)]
    api.score_scenario(s)


def test_outflow_exactly_equal_to_balance_is_allowed_and_one_more_is_not():
    s = base()
    s["transactions"] = [dict(day=1, type="cash_withdrawal", amount_uzs=10_000_000)]
    api.score_scenario(s)
    s["transactions"] = [dict(day=1, type="cash_withdrawal", amount_uzs=10_000_001)]
    with pytest.raises(ScenarioError, match="yetarli emas"):
        api.score_scenario(s)


def test_inflow_ordering_is_independent_of_the_list_order():
    """Same-day inflow covers a same-day outflow whether it is listed before or after it; an inflow on a LATER day does not."""
    out = dict(day=5, type="cash_withdrawal", amount_uzs=10_500_000)
    inn = dict(day=5, type="p2p_in", amount_uzs=1_000_000)
    for order in ([out, inn], [inn, out]):
        s = base(); s["snapshot_day"] = 5; s["transactions"] = [dict(t) for t in order]
        api.score_scenario(s)
    s = base(); s["snapshot_day"] = 6
    s["transactions"] = [dict(out), dict(day=6, type="p2p_in", amount_uzs=1_000_000)]
    with pytest.raises(ScenarioError, match="yetarli emas"):
        api.score_scenario(s)


def test_balance_error_names_the_offending_transaction_and_day():
    s = base()
    s["transactions"] = [dict(day=1, type="income", amount_uzs=1), dict(day=7, type="p2p_out", amount_uzs=99_000_000)]
    with pytest.raises(ScenarioError) as e:
        api.score_scenario(s)
    assert "2-tranzaksiya" in str(e.value) and "7-kun" in str(e.value)


def test_transaction_order_in_the_list_does_not_change_the_score():
    s = base()
    s["snapshot_day"] = 20
    s["transactions"] = [dict(day=1, type="cash_withdrawal", amount_uzs=2_000_000),
                         dict(day=4, type="supplier_payment", amount_uzs=3_000_000),
                         dict(day=9, type="income", amount_uzs=500_000),
                         dict(day=12, type="p2p_out", amount_uzs=400_000)]
    r = api.score_scenario(s)
    s["transactions"] = list(reversed(s["transactions"]))
    r2 = api.score_scenario(s)
    assert r["risk"] == r2["risk"] and r["features"] == r2["features"]


# ---------------------------------------------------------------- output properties
def test_percentile_is_in_range_and_monotonic_in_risk(P):
    rng = random.Random(7)
    out = [api.score_scenario(random_scenario(rng, P)) for _ in range(25)]
    for r in out:
        assert 0 <= r["percentile"] <= 100 and isinstance(r["percentile"], float)
    out.sort(key=lambda r: r["risk"])
    pcts = [r["percentile"] for r in out]
    assert pcts == sorted(pcts)


def test_percentile_matches_the_documented_definition(P):
    r = api.score_scenario(base())
    share = 100.0 * float((P.sorted_risk <= r["risk"]).sum()) / len(P.sorted_risk)
    assert r["percentile"] == pytest.approx(share, abs=0.051)


def test_percentile_extremes_cover_the_range(P):
    pcts = []
    for p in api.get_scenario_presets():
        pcts.append(api.score_scenario(p["scenario"])["percentile"])
    assert len(set(pcts)) > 1                                # not a constant


def test_color_agrees_with_portfolio_thresholds(P):
    from ml.money_trail.colors import color_of
    rng = random.Random(3)
    for _ in range(15):
        r = api.score_scenario(random_scenario(rng, P))
        assert r["color"] == color_of(r["risk"], P.thresholds)


def test_reasons_use_the_same_templates_and_are_clean(P):
    rng = random.Random(11)
    for _ in range(20):
        r = api.score_scenario(random_scenario(rng, P))
        for x in r["reasons"]:
            assert set(x) == {"code", "text_uz", "text", "weight"}
            assert isinstance(x["text_uz"], str) and x["text_uz"].strip()
            assert isinstance(x["weight"], float)
            assert x["code"] in set(FEATURE_COLUMNS) or x["code"]
            assert not re.search(r"atigi 0(?!\d)", x["text_uz"]), x["text_uz"]
            assert not re.search(r"\b(nan|None|inf)\b", x["text_uz"]), x["text_uz"]
            assert not any(w in x["text_uz"].lower() for w in ("firibgar", "aybdor", "jinoyat"))
        w = [x["weight"] for x in r["reasons"]]
        assert w == sorted(w, reverse=True)


def test_features_are_finite_floats_for_random_scenarios(P):
    rng = random.Random(13)
    for _ in range(15):
        f = api.score_scenario(random_scenario(rng, P))["features"]
        assert set(f) == set(FEATURE_COLUMNS)
        assert all(isinstance(v, float) and np.isfinite(v) for v in f.values())


def test_scenario_calls_leave_portfolio_state_untouched(P):
    before_q = api.get_queue()
    before_tx, before_loans = P.tx.copy(), P.loans.copy()
    rng = random.Random(21)
    for _ in range(8):
        api.score_scenario(random_scenario(rng, P))
    assert api.get_queue() == before_q
    assert P.tx.equals(before_tx) and P.loans.equals(before_loans)
    assert "L-SCENARIO" not in {x["loan_id"] for x in api.get_queue()}
    with pytest.raises(api.LoanNotFoundError):
        api.get_loan("L-SCENARIO")


def test_failed_scenario_does_not_poison_later_calls():
    good = api.score_scenario(base())["risk"]
    bad = base(); bad["transactions"][0]["amount_uzs"] = 10 ** 10
    with pytest.raises(ScenarioError):
        api.score_scenario(bad)
    assert api.score_scenario(base())["risk"] == good


def test_presets_are_stable_and_inputs_only():
    a, b = api.get_scenario_presets(), api.get_scenario_presets()
    assert a == b and [p["id"] for p in a] == ["honest_purchase", "fast_cash_out", "supplier_ring", "asset_resale"]
    for p in a:
        assert set(p) == {"id", "title_uz", "description_uz", "title", "description", "scenario"}
        assert not ({"risk", "color", "percentile"} & set(p["scenario"]))
    a[0]["scenario"]["amount_uzs"] = 1                       # mutating a returned preset must not leak
    assert api.get_scenario_presets()[0]["scenario"]["amount_uzs"] != 1


def test_preset_ordering_of_risk_is_sensible():
    r = {p["id"]: api.score_scenario(p["scenario"])["risk"] for p in api.get_scenario_presets()}
    assert r["honest_purchase"] == min(r.values())
