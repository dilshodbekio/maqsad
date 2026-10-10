"""Scenario honesty fixes (reasons only state what the user entered), snapshot range, unknown keys, absurd inputs;
fusion evidence merge, one reason ordering for queue and loan, colour-budget reporting."""
from __future__ import annotations

import copy
import random

import pytest

from ml import api, fusion
from ml.errors import ScenarioError
from ml.money_trail import scenario as S
from ml.money_trail.service import get_portfolio
from ml.photo import config
from ml.photo.index import get_index

NO_REASON = "Sezilarli xavf belgisi aniqlanmadi"
RANK = fusion.COLOR_RANK


@pytest.fixture(autouse=True)
def clean_sandboxes():
    api.reset_sandbox(None)
    yield
    api.reset_sandbox(None)


@pytest.fixture(scope="module")
def P():
    return get_portfolio()


@pytest.fixture(scope="module")
def presets():
    return {p["id"]: p for p in api.get_scenario_presets()}


def base():
    return dict(purpose="chorva", amount_uzs=10_000_000, snapshot_day=30,
                supplier=dict(mode="new", oked="01.41", registered_days_before=100),
                transactions=[dict(day=1, type="cash_withdrawal", amount_uzs=8_000_000),
                              dict(day=4, type="supplier_payment", amount_uzs=1_000_000)])


# ---------------------------------------------------------------- invoice is optional
def test_without_invoice_the_invoice_gap_uses_the_no_invoice_convention_and_is_never_a_reason(presets):
    scn = copy.deepcopy(presets["fast_cash_out"]["scenario"])
    with_invoice = api.score_scenario(scn)
    assert with_invoice["features"]["invoice_gap"] == pytest.approx(0.9)
    assert "invoice_gap" in [r["code"] for r in with_invoice["reasons"]]        # the user entered an invoice
    scn.pop("invoice_amount_uzs")
    no_invoice = api.score_scenario(scn)
    assert no_invoice["features"]["invoice_gap"] == -1
    assert "invoice_gap" not in [r["code"] for r in no_invoice["reasons"]]
    assert not any("Hisob-faktura" in r["text_uz"] for r in no_invoice["reasons"])


def test_invoice_amount_is_used_and_validated():
    s = base()
    s["invoice_amount_uzs"] = 20_000_000
    assert api.score_scenario(s)["features"]["invoice_gap"] == pytest.approx(1 - 1_000_000 / 20_000_000)
    for bad in (0, -1, "10", True, 1.5, float("nan"), 10 * 10_000_000 + 1, 10 ** 11 + 1):
        s = base()
        s["invoice_amount_uzs"] = bad
        with pytest.raises(ScenarioError):
            api.score_scenario(s)
    s = base()
    s["invoice_amount_uzs"] = None                                                  # same as absent
    assert api.score_scenario(s)["features"]["invoice_gap"] == -1


def test_presets_carry_an_explicit_editable_invoice(presets):
    for p in presets.values():
        assert p["scenario"]["invoice_amount_uzs"] == p["scenario"]["amount_uzs"]


# ---------------------------------------------------------------- new supplier: no supplier-ledger reasons
def test_new_supplier_never_gets_reasons_about_its_own_cash_out():
    rng = random.Random(5)
    for _ in range(40):
        s = base()
        s["supplier"] = dict(mode="new", oked=f"{rng.randint(1, 99):02d}.{rng.randint(0, 99):02d}",
                             registered_days_before=rng.choice([0, 30, 800]))
        s["snapshot_day"] = rng.choice([5, 60, 200])
        s["transactions"] = [dict(day=rng.randint(0, 5), type="supplier_payment", amount_uzs=rng.randint(1, 8) * 1_000_000),
                             dict(day=rng.randint(0, 5), type="cash_withdrawal", amount_uzs=rng.randint(1, 2) * 1_000_000)]
        r = api.score_scenario(s)
        assert not {x["code"] for x in r["reasons"]} & S.NEW_SUPPLIER_HIDDEN


def test_existing_supplier_still_shows_its_real_ledger_reasons(presets):
    r = api.score_scenario(presets["supplier_ring"]["scenario"])
    assert {x["code"] for x in r["reasons"]} & {"sup_cash_share", "sup_exit_share"}


def test_hiding_reasons_does_not_change_the_score(presets):
    scn = copy.deepcopy(presets["honest_purchase"]["scenario"])
    a = api.score_scenario(scn)
    scn["supplier"]["registered_days_before"] += 0
    assert api.score_scenario(scn)["risk"] == a["risk"]
    assert a["features"]["sup_cash_share"] == 0 and a["features"]["sup_active_span_days"] >= 0   # still model inputs


# ---------------------------------------------------------------- absurd inputs, snapshot range, unknown keys
def test_single_amount_over_10x_the_loan_is_rejected_but_exactly_10x_is_not():
    s = base()
    s["transactions"] = [dict(day=2, type="income", amount_uzs=10 * 10_000_000 + 1)]
    with pytest.raises(ScenarioError, match="10 baravar"):
        api.score_scenario(s)
    s["transactions"] = [dict(day=2, type="income", amount_uzs=10 * 10_000_000)]
    assert 0 <= api.score_scenario(s)["risk"] <= 1


def test_total_inflows_over_20x_the_loan_are_rejected():
    s = base()
    s["transactions"] = [dict(day=d, type="income", amount_uzs=10 * 10_000_000) for d in (1, 2)] + \
                        [dict(day=3, type="p2p_in", amount_uzs=1)]
    with pytest.raises(ScenarioError, match="20 baravar"):
        api.score_scenario(s)
    s["transactions"] = [dict(day=d, type="income", amount_uzs=10 * 10_000_000) for d in (1, 2)]   # exactly 20x
    assert 0 <= api.score_scenario(s)["risk"] <= 1


@pytest.mark.parametrize("day", [0, 1, 365, 400, 730, -3])
def test_snapshot_day_outside_the_training_range_is_rejected_in_uzbek(day):
    s = base()
    s["snapshot_day"] = day
    s["transactions"] = []
    with pytest.raises(ScenarioError) as e:
        api.score_scenario(s)
    assert "bir yilgacha" in str(e.value) and "2..364" in str(e.value)


@pytest.mark.parametrize("day", [2, 3, 364])
def test_snapshot_day_inside_the_training_range_is_accepted(day):
    s = base()
    s["snapshot_day"] = day
    s["transactions"] = [dict(day=2, type="cash_withdrawal", amount_uzs=1_000)]
    assert 0 <= api.score_scenario(s)["risk"] <= 1


def test_unknown_keys_anywhere_are_listed():
    cases = [(lambda s: s.update(invoice=5, foo=1), "foo, invoice"),
             (lambda s: s["supplier"].update(supplier_id="S1"), "supplier_id"),
             (lambda s: s["transactions"][1].update(note="x"), "note"),
             (lambda s: s.update(supplier=dict(mode="existing", supplier_id="S1", oked="01.41")), "oked")]
    for mutate, names in cases:
        s = base()
        mutate(s)
        with pytest.raises(ScenarioError) as e:
            api.score_scenario(s)
        assert names in str(e.value) and "Mumkin" in str(e.value)


def test_oked_must_use_ascii_digits():
    s = base()
    s["supplier"]["oked"] = "٠١.٤١"
    with pytest.raises(ScenarioError):
        api.score_scenario(s)


# ---------------------------------------------------------------- fusion: one reason, both evidence entries
def image(name) -> bytes:
    return (config.ASSETS / name).read_bytes()


def test_sandbox_upload_evidence_is_kept_next_to_the_portfolio_evidence_with_one_reason():
    idx = get_index()
    shared = next(i for i in idx.images if len(i["loans"]) > 1)
    loan, partner = shared["loans"][0], shared["loans"][1]
    single = next(i for i in idx.images if len(i["loans"]) == 1)
    other_owner = single["loans"][0]
    api.check_photo(loan, image(single["file"]), sandbox_id="A")
    d = api.get_loan(loan, "A")
    dup = [r for r in d["reasons"] if r["code"] == "PHOTO_DUPLICATE"]
    assert len(dup) == 1                                                          # one reason ...
    assert partner in dup[0]["text_uz"] and other_owner in dup[0]["text_uz"]      # ... that names both partners
    assert dup[0]["text_uz"].startswith("Bu kreditdagi rasm") and "yuklangan rasm" in dup[0]["text_uz"]
    refs = dup[0]["evidence_refs"]
    assert len(refs) == 2 and sum(r.startswith("photo:sandbox:") for r in refs) == 1
    photo_items = [e for e in d["evidence"] if e["kind"] == "photo"]
    assert {e["ref"] for e in photo_items} == set(refs)                           # ... backed by both evidence entries
    assert next(e for e in photo_items if e["ref"].startswith("photo:sandbox:"))["data"]["uploaded"] is True
    assert next(e for e in photo_items if not e["ref"].startswith("photo:sandbox:"))["data"]["uploaded"] is False
    assert [e["code"] for e in d["escalations"]].count("PHOTO_DUPLICATE") == 1
    # the default view and other sandboxes only have the portfolio evidence
    d0 = api.get_loan(loan)
    assert [r["code"] for r in d0["reasons"]].count("PHOTO_DUPLICATE") == 1
    assert other_owner not in next(r for r in d0["reasons"] if r["code"] == "PHOTO_DUPLICATE")["text_uz"]
    assert len([e for e in d0["evidence"] if e["kind"] == "photo"]) == 1
    assert len([e for e in api.get_loan(loan, "B")["evidence"] if e["kind"] == "photo"]) == 1


# ---------------------------------------------------------------- one reason ordering for queue and loan
def _expected_top(d):
    return d["reasons"][0]["text_uz"] if d["reasons"] else NO_REASON


def _check_top_reasons(items):
    for it in items:
        d = api.get_loan(it["loan_id"], _check_top_reasons.sandbox)
        assert it["top_reason"] == _expected_top(d), (it["loan_id"], it["top_reason"], d["reasons"][:2])


def _upload_some(P):
    idx = get_index()
    now_far = []
    q = api.get_queue()
    for k, x in enumerate([x for x in q if x["loan_id"] not in idx.by_loan][:: len(q) // 12][:12]):
        row = P.loans.loc[x["loan_id"]]
        photo = idx.images[k * 7 % len(idx.images)]["file"] if k % 2 == 0 else idx.reserved[k % len(idx.reserved)]
        api.check_photo(x["loan_id"], image(photo), lat=float(row["borrower_lat"]) + (1.0 if k % 3 == 0 else 0.0),
                        lon=float(row["borrower_lon"]), sandbox_id="A")
        now_far.append(x["loan_id"])


def test_top_reason_equals_first_reason_for_escalated_and_sampled_loans_default_and_sandbox(P):
    rng = random.Random(2)
    for sandbox in (None, "A"):
        if sandbox:
            _upload_some(P)
        q = api.get_queue(sandbox)
        sample = [x for x in q if x["escalations"]] + rng.sample(q, 150)
        _check_top_reasons.sandbox = sandbox
        _check_top_reasons(sample)


@pytest.mark.slow
def test_top_reason_equals_first_reason_over_the_whole_demo_queue_default_view_and_one_sandbox(P):
    for sandbox in (None, "A"):
        if sandbox:
            _upload_some(P)
        _check_top_reasons.sandbox = sandbox
        _check_top_reasons(api.get_queue(sandbox))


def test_non_raising_policy_reasons_come_after_the_money_trail_reasons():
    q = api.get_queue()
    red = next(x for x in q if x["base_color"] == "red" and x["escalations"] and not any(e["raises_color"] for e in x["escalations"]))
    d = api.get_loan(red["loan_id"])
    codes = [r["code"] for r in d["reasons"]]
    n_policy = len(d["escalations"])
    assert all(c.startswith("PHOTO_") for c in codes[-n_policy:]) and not codes[0].startswith("PHOTO_")
    assert red["top_reason"] == d["reasons"][0]["text_uz"]


def test_order_reasons_unit():
    esc = [dict(raises_color=False), dict(raises_color=True)]
    pol = [dict(code="B"), dict(code="A")]
    out = fusion.order_reasons(pol, [dict(code="M1"), dict(code="M2")], esc)
    assert [r["code"] for r in out] == ["A", "M1", "M2", "B"]


# ---------------------------------------------------------------- colour budget reporting
def test_color_thresholds_report_the_photo_escalations(P):
    q = api.get_queue()
    t = api.get_color_thresholds()
    esc_red = sum(1 for x in q if x["color"] == "red" and x["base_color"] != "red")
    esc_yellow = sum(1 for x in q if x["color"] == "yellow" and x["base_color"] == "green")
    assert (t["n_red"], t["n_yellow"]) == (500, 500)                                  # thresholds unchanged
    assert t["n_escalated_to_red"] == esc_red and t["n_escalated_to_yellow"] == esc_yellow
    assert t["n_red_total"] == sum(1 for x in q if x["color"] == "red") == 500 + esc_red
    assert t["n_yellow_total"] == sum(1 for x in q if x["color"] == "yellow")
    assert f"{esc_red} escalated to red by the photo policy" in t["note"] and t["note"].startswith("500 red by risk")
    assert esc_red > 0 and t["n_red_total"] > t["n_red"]                              # the budget overrun is visible, not hidden
