"""Evidence fusion: photo signals escalate the queue colour (policy, no learned model); never lower a colour."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from ml import api, fusion
from ml.money_trail.service import get_portfolio
from ml.photo import config
from ml.photo.index import get_index

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
def idx():
    return get_index()


def image(name) -> bytes:
    return (config.ASSETS / name).read_bytes()


def quiet_green_loans(idx, n=2):
    """Green, un-escalated loans without a portfolio photo."""
    q = api.get_queue()
    out = [x["loan_id"] for x in reversed(q) if x["color"] == "green" and not x["escalations"] and x["loan_id"] not in idx.by_loan]
    return out[:n]


def single_photo(idx):
    return next(i for i in idx.images if len(i["loans"]) == 1)


def view(sandbox_id, loan_id):
    q = {x["loan_id"]: x for x in api.get_queue(sandbox_id)}
    return q[loan_id], api.get_loan(loan_id, sandbox_id)


# ---------------------------------------------------------------- policy
def test_policy_table():
    assert fusion.POLICY["PHOTO_DUPLICATE"]["min_color"] == "red"
    assert fusion.POLICY["PHOTO_FAR"]["min_color"] == "yellow" and fusion.POLICY["PHOTO_TIME"]["min_color"] == "yellow"
    assert fusion.POLICY["PHOTO_ASSET_MISMATCH"]["min_color"] == "yellow" and fusion.POLICY["PHOTO_ASSET_MISMATCH"]["active"] is True
    assert all(v["min_color"] in RANK for v in fusion.POLICY.values())


@pytest.mark.parametrize("base", ["green", "yellow", "red"])
def test_apply_policy_never_lowers_a_colour(base):
    for codes in ([], ["PHOTO_FAR"], ["PHOTO_TIME"], ["PHOTO_DUPLICATE"], ["PHOTO_FAR", "PHOTO_DUPLICATE", "PHOTO_TIME"]):
        sig = [dict(code=c, text_uz="x", evidence_ref="photo:p") for c in codes]
        res = fusion.apply_policy(base, sig)
        want = max([RANK[base]] + [RANK[fusion.POLICY[c]["min_color"]] for c in codes])
        assert RANK[res["color"]] == want >= RANK[base]


def test_inactive_policy_codes_produce_no_signal():
    sig = fusion.photo_signals(ref="photo:x", photo_label="x", duplicates=[], geo_ok=True, distance_km=1.0, time_ok=True,
                               asset_match={"label": "other", "score": 0.0}, uploaded=True)
    assert sig == []
    dup = fusion.photo_signals(ref="photo:x", photo_label="x", duplicates=[dict(loan_id="L1", similarity=1.0)],
                               geo_ok=False, distance_km=12.0, time_ok=False, uploaded=True)
    assert [s["code"] for s in dup] == ["PHOTO_DUPLICATE", "PHOTO_FAR", "PHOTO_TIME"]
    assert "L1" in dup[0]["text_uz"] and dup[0]["evidence"]["kind"] == "photo"


def test_priority_is_colour_rank_plus_risk():
    assert fusion.priority("red", 0.1) > fusion.priority("yellow", 0.99) > fusion.priority("green", 0.99)


# ---------------------------------------------------------------- sandbox escalation
def test_duplicate_upload_raises_colour_in_that_sandbox_only_and_reset_restores(P, idx):
    loan, = quiet_green_loans(idx, 1)
    photo = single_photo(idx)
    owner = photo["loans"][0]
    default_before = api.get_queue()
    item0, d0 = view(None, loan)
    assert item0["color"] == "green" and d0["color"] == "green"

    api.check_photo(loan, image(photo["file"]), sandbox_id="A")
    a_item, a_loan = view("A", loan)
    assert a_item["color"] == "red" and a_loan["color"] == "red" and a_item["base_color"] == "green"
    assert a_item["risk"] == item0["risk"] == a_loan["risk"]                      # money-trail risk is untouched
    assert api.score_loan(loan)["risk"] == item0["risk"] and api.score_loan(loan)["color"] == "green"
    esc = a_loan["escalations"]
    assert [e["code"] for e in esc] == ["PHOTO_DUPLICATE"] and esc[0]["raises_color"] and esc[0]["min_color"] == "red"
    # policy reason first: Uzbek, factual, names the other loan, evidence ref to the photo, weight null
    r0 = a_loan["reasons"][0]
    assert r0["code"] == "PHOTO_DUPLICATE" and r0["weight"] is None and r0["source"] == "fusion_policy"
    assert r0["text_uz"] == f"Yuklangan rasm {owner} kreditidagi rasm bilan bir xil"
    assert r0["evidence_refs"][0].startswith("photo:sandbox:") and any(e["ref"] == r0["evidence_refs"][0] and e["kind"] == "photo" for e in a_loan["evidence"])
    assert a_item["top_reason"] == r0["text_uz"]
    assert not any(w in r0["text_uz"].lower() for w in ("firibgar", "aybdor", "jinoyat"))
    # the loan moved up in A's queue, the rest of the queue keeps its order
    qa = api.get_queue("A")
    pos_a = [x["loan_id"] for x in qa].index(loan)
    assert pos_a < [x["loan_id"] for x in default_before].index(loan)
    assert [x["priority"] for x in qa] == sorted([x["priority"] for x in qa], reverse=True)
    assert [x["loan_id"] for x in qa if x["loan_id"] != loan] == [x["loan_id"] for x in default_before if x["loan_id"] != loan]

    # other sandboxes and the default view are unaffected
    assert api.get_queue("B") == default_before == api.get_queue()
    assert view("B", loan)[1]["color"] == "green" and view(None, loan)[1]["color"] == "green"
    assert api.get_queue("never-used") == default_before

    api.reset_sandbox("A")                                                         # restores A's view
    assert api.get_queue("A") == default_before
    assert view("A", loan)[1]["color"] == "green" and view("A", loan)[1]["escalations"] == []


def test_far_and_wrong_time_raise_to_yellow_and_duplicate_to_red(P, idx):
    la, lb, lc = quiet_green_loans(idx, 3)
    row = lambda l: P.loans.loc[l]
    new = image(idx.reserved[0])                                                   # not a duplicate of anything
    now = datetime.now(timezone.utc)
    api.check_photo(la, new, lat=float(row(la)["borrower_lat"]) + 1.0, lon=float(row(la)["borrower_lon"]), sandbox_id="A")
    api.check_photo(lb, image(idx.reserved[1]), captured_at=now + timedelta(days=2), sandbox_id="A")
    api.check_photo(lc, image(idx.reserved[2]), lat=float(row(lc)["borrower_lat"]), lon=float(row(lc)["borrower_lon"]),
                    captured_at=now - timedelta(minutes=1), sandbox_id="A")            # all fine -> no escalation
    ia, da = view("A", la)
    ib, db = view("A", lb)
    ic, dc = view("A", lc)
    assert (ia["color"], [e["code"] for e in da["escalations"]]) == ("yellow", ["PHOTO_FAR"])
    assert "km uzoqda" in da["reasons"][0]["text_uz"] and da["reasons"][0]["text_uz"].startswith("Rasm olingan joy")
    assert (ib["color"], [e["code"] for e in db["escalations"]]) == ("yellow", ["PHOTO_TIME"])
    assert (ic["color"], dc["escalations"]) == ("green", [])
    # a duplicate on top of a far photo -> red, both reasons listed, strongest first
    api.check_photo(la, image(single_photo(idx)["file"]), lat=float(row(la)["borrower_lat"]) + 1.0,
                    lon=float(row(la)["borrower_lon"]), sandbox_id="A")
    _, da2 = view("A", la)
    assert da2["color"] == "red" and [e["code"] for e in da2["escalations"]] == ["PHOTO_DUPLICATE", "PHOTO_FAR"]


def test_two_uploads_of_the_same_new_photo_escalate_both_loans_only_in_that_sandbox(P, idx):
    la, lb = quiet_green_loans(idx, 2)
    new = image(idx.reserved[3])
    api.check_photo(la, new, sandbox_id="A")
    assert view("A", la)[0]["color"] == "green"                                    # nothing to duplicate yet
    api.check_photo(lb, new, sandbox_id="A")
    ia, da = view("A", la)
    ib, db = view("A", lb)
    assert ia["color"] == ib["color"] == "red"
    assert lb in da["reasons"][0]["text_uz"] and la in db["reasons"][0]["text_uz"]
    assert view("B", la)[0]["color"] == "green" and view(None, lb)[0]["color"] == "green"


def test_no_escalation_lowers_a_colour_in_any_view(P, idx):
    # escalate a red loan, a yellow loan and green loans in sandbox A and check every item of the view
    q = api.get_queue()
    red = next(x["loan_id"] for x in q if x["base_color"] == "red" and not x["escalations"] and x["loan_id"] not in idx.by_loan)
    yellow = next(x["loan_id"] for x in q if x["base_color"] == "yellow" and not x["escalations"] and x["loan_id"] not in idx.by_loan)
    green = quiet_green_loans(idx, 1)[0]
    now = datetime.now(timezone.utc)
    for loan, name in ((red, idx.reserved[4]), (yellow, idx.reserved[5]), (green, idx.reserved[6])):
        row = P.loans.loc[loan]
        api.check_photo(loan, image(name), lat=float(row["borrower_lat"]) + 2.0, lon=float(row["borrower_lon"]),
                        captured_at=now + timedelta(days=1), sandbox_id="A")
    default = {x["loan_id"]: x for x in api.get_queue()}
    for x in api.get_queue("A"):
        assert RANK[x["color"]] >= RANK[x["base_color"]]
        assert RANK[x["color"]] >= RANK[default[x["loan_id"]]["color"]]
        assert x["risk"] == default[x["loan_id"]]["risk"]
    assert view("A", red)[0]["color"] == "red" and view("A", yellow)[0]["color"] == "yellow"
    assert view("A", green)[0]["color"] == "yellow"
    # an already-red loan keeps its money-trail top reason, but the escalations are still listed
    ir, dr = view("A", red)
    assert [e["raises_color"] for e in dr["escalations"]] == [False, False] and ir["top_reason"] == api.get_queue()[
        [x["loan_id"] for x in api.get_queue()].index(red)]["top_reason"]


# ---------------------------------------------------------------- portfolio photos
def test_portfolio_photo_flags_drive_the_default_queue(P, idx):
    q = {x["loan_id"]: x for x in api.get_queue()}
    shared = [i for i in idx.images if len(i["loans"]) > 1]
    assert shared
    for im in shared:
        for lid in im["loans"]:
            assert q[lid]["color"] == "red" and "PHOTO_DUPLICATE" in [e["code"] for e in q[lid]["escalations"]]
            text = next(e["text_uz"] for e in q[lid]["escalations"] if e["code"] == "PHOTO_DUPLICATE")
            assert text.startswith("Bu kreditdagi rasm") and all(o in text for o in im["loans"] if o != lid)
    escalated = [x for x in q.values() if x["escalations"]]
    assert escalated and all(x["loan_id"] in idx.by_loan for x in escalated)
    assert all(RANK[x["color"]] >= RANK[x["base_color"]] for x in q.values())
    assert {x["loan_id"] for x in q.values() if x["color"] != x["base_color"]} <= {x["loan_id"] for x in escalated}
    d = api.get_loan(shared[0]["loans"][0])
    assert d["reasons"][0]["source"] == "fusion_policy" and d["color"] == "red"
    assert any(e["kind"] == "photo" and e["ref"].startswith("photo:") for e in d["evidence"])
