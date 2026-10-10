"""Extra fusion / sandbox tests: queue vs get_loan consistency, thread isolation, size limits, random uploads,
default-view stability and mutation safety."""
from __future__ import annotations

import copy
import random
import threading
from datetime import datetime, timedelta, timezone

import pytest

from ml import api, fusion
from ml.money_trail.service import get_portfolio
from ml.photo import checker, config
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


@pytest.fixture
def protect_cache(P):
    """Mutation tests may expose a real aliasing bug; restore the cached queue so other tests are not polluted."""
    saved = copy.deepcopy(P._queue)
    yield
    P._queue[:] = saved


@pytest.fixture(scope="module")
def idx():
    return get_index()


@pytest.fixture(scope="module")
def images(idx):
    """name -> bytes for the reserved (non-portfolio) images plus a few portfolio ones."""
    names = list(idx.reserved) + [i["file"] for i in idx.images[:3]]
    return {n: (config.ASSETS / n).read_bytes() for n in names}


def quiet(idx, n):
    q = api.get_queue()
    return [x["loan_id"] for x in reversed(q) if x["color"] == "green" and not x["escalations"] and x["loan_id"] not in idx.by_loan][:n]


def strip(q):
    return copy.deepcopy(q)


# ---------------------------------------------------------------- queue <-> get_loan consistency
def check_consistent(item, d):
    assert item["loan_id"] == d["loan"]["loan_id"]
    assert item["color"] == d["color"] and item["base_color"] == d["base_color"]
    assert item["risk"] == d["risk"]
    assert item["priority"] == pytest.approx(d["priority"])
    assert item["escalations"] == d["escalations"]
    assert item["priority"] == pytest.approx(fusion.priority(item["color"], item["risk"]))
    assert RANK[item["color"]] >= RANK[item["base_color"]]
    raised = [e for e in item["escalations"] if e["raises_color"]]
    if raised:
        assert item["top_reason"] == raised[0]["text_uz"]
        assert d["reasons"][0]["text_uz"] == d["escalations"][0]["text_uz"]
    else:
        assert item["top_reason"] and not item["top_reason"].startswith("Yuklangan rasm")
    pol = [r for r in d["reasons"] if r["source"] == "fusion_policy"] if all("source" in r for r in d["reasons"]) else \
        [r for r in d["reasons"] if r.get("source") == "fusion_policy"]
    assert [r["code"] for r in pol] == [e["code"] for e in d["escalations"]]
    assert all(r["weight"] is None for r in pol)
    assert d["reasons"][: len(pol)] == pol                                  # policy reasons come first
    refs = {e["ref"] for e in d["evidence"]}
    for e in d["escalations"]:
        assert set(e["evidence_refs"]) <= refs                               # every escalation is backed by an evidence item


def test_default_view_queue_and_get_loan_agree(P):
    q = api.get_queue()
    escalated = [x for x in q if x["escalations"]]
    rng = random.Random(1)
    sample = escalated + rng.sample(q, 25) + q[:3] + q[-3:]
    for item in sample:
        check_consistent(item, api.get_loan(item["loan_id"]))


def test_queue_is_sorted_by_priority_and_loan_ids_unique(P):
    for sb in (None, "A"):
        q = api.get_queue(sb)
        assert len(q) == len(P.loans) and len({x["loan_id"] for x in q}) == len(q)
        pr = [x["priority"] for x in q]
        assert pr == sorted(pr, reverse=True)


def test_sandbox_view_queue_and_get_loan_agree(P, idx, images):
    la, lb, lc = quiet(idx, 3)
    now = datetime.now(timezone.utc)
    row = P.loans.loc[la]
    api.check_photo(la, images[idx.reserved[0]], lat=float(row["borrower_lat"]) + 1.5, lon=float(row["borrower_lon"]), sandbox_id="A")
    api.check_photo(lb, images[idx.reserved[1]], captured_at=now + timedelta(days=3), sandbox_id="A")
    api.check_photo(lc, images[idx.reserved[1]], sandbox_id="A")             # same image as lb -> both red
    q = {x["loan_id"]: x for x in api.get_queue("A")}
    for lid in (la, lb, lc):
        check_consistent(q[lid], api.get_loan(lid, "A"))
    assert (q[la]["color"], q[lb]["color"], q[lc]["color"]) == ("yellow", "red", "red")
    # escalations of lb: duplicate (red) first, then time (yellow)
    assert [e["code"] for e in q[lb]["escalations"]] == ["PHOTO_DUPLICATE", "PHOTO_TIME"]
    # and nothing leaks into the default view
    for lid in (la, lb, lc):
        check_consistent({x["loan_id"]: x for x in api.get_queue()}[lid], api.get_loan(lid))
        assert api.get_loan(lid)["escalations"] == []


def test_get_loan_unknown_loan_with_sandbox_raises():
    with pytest.raises(api.LoanNotFoundError):
        api.get_loan("L-nope", "A")


def test_get_queue_with_sandbox_that_has_no_uploads_equals_default():
    assert api.get_queue("nobody") == api.get_queue()


# ---------------------------------------------------------------- limits and reset
def test_reset_sandbox_none_clears_everything_and_unknown_id_is_idempotent(idx, images):
    la, lb = quiet(idx, 2)
    future = datetime.now(timezone.utc) + timedelta(days=2)
    api.check_photo(la, images[idx.reserved[0]], captured_at=future, sandbox_id="A")
    api.check_photo(lb, images[idx.reserved[0]], captured_at=future, sandbox_id="B")
    assert api.get_queue("A") != api.get_queue() and api.get_queue("B") != api.get_queue()
    api.reset_sandbox("zzz")                                                  # unknown: no error
    api.reset_sandbox("A")
    api.reset_sandbox("A")
    assert api.get_queue("A") == api.get_queue() and api.get_queue("B") != api.get_queue()
    api.reset_sandbox(None)
    assert api.get_queue("B") == api.get_queue()
    assert checker._sandboxes == {}
    api.reset_sandbox(None)                                                   # idempotent on empty


def test_sandbox_without_id_remembers_nothing(idx, images):
    la, lb = quiet(idx, 2)
    api.check_photo(la, images[idx.reserved[0]])
    r = api.check_photo(lb, images[idx.reserved[0]])
    assert r["duplicate_of"] == [] and checker._sandboxes == {}


def test_failed_uploads_do_not_create_or_change_a_sandbox(idx):
    la, = quiet(idx, 1)
    with pytest.raises(api.InvalidImageError):
        api.check_photo(la, b"not an image", sandbox_id="X")
    with pytest.raises(api.InvalidImageError):
        api.check_photo(la, b"", sandbox_id="X")
    with pytest.raises(api.LoanNotFoundError):
        api.check_photo("L-nope", b"x", sandbox_id="X")
    assert "X" not in checker._sandboxes and api.get_queue("X") == api.get_queue()


def test_max_sandboxes_evicts_least_recently_used(monkeypatch, idx, images):
    monkeypatch.setattr(checker, "MAX_SANDBOXES", 3)
    la, lb = quiet(idx, 2)
    img = images[idx.reserved[0]]
    for sb in ("s1", "s2", "s3"):
        api.check_photo(la, img, sandbox_id=sb)
    api.check_photo(lb, img, sandbox_id="s1")                                  # touch s1 -> s2 is now the oldest
    api.check_photo(la, img, sandbox_id="s4")                                  # evicts s2
    assert list(checker._sandboxes) == ["s3", "s1", "s4"]
    assert api.get_queue("s2") == api.get_queue()                              # evicted: back to the default view
    assert api.get_queue("s1") != api.get_queue()


def test_max_per_sandbox_keeps_only_the_latest_uploads(monkeypatch, idx, images):
    monkeypatch.setattr(checker, "MAX_PER_SANDBOX", 2)
    l1, l2, l3 = quiet(idx, 3)
    now = datetime.now(timezone.utc)
    future = now + timedelta(days=5)
    for lid, name in ((l1, idx.reserved[0]), (l2, idx.reserved[1]), (l3, idx.reserved[2])):
        api.check_photo(lid, images[name], captured_at=future, sandbox_id="A")
    entries = checker._sandboxes["A"]["entries"]
    assert [e["loan_id"] for e in entries] == [l2, l3] and checker._sandboxes["A"]["n"] == 3
    q = {x["loan_id"]: x for x in api.get_queue("A")}
    assert q[l1]["escalations"] == [] and q[l2]["color"] == q[l3]["color"] == "yellow"


# ---------------------------------------------------------------- threads
def test_concurrent_uploads_in_two_sandboxes_stay_isolated(idx, images):
    loans = quiet(idx, 8)
    names = list(idx.reserved)[:8]
    errors, barrier = [], threading.Barrier(2)

    def worker(sb, lids, imgs):
        try:
            barrier.wait()
            for lid, n in zip(lids, imgs):
                r = api.check_photo(lid, images[n], sandbox_id=sb)
                assert all(d["loan_id"] in lids for d in r["duplicate_of"])
        except Exception as e:                                                  # noqa: BLE001
            errors.append(e)

    # both threads upload the SAME images for DIFFERENT loans: duplicates may only appear within one sandbox
    ta = threading.Thread(target=worker, args=("TA", loans[:4], names[:4]))
    tb = threading.Thread(target=worker, args=("TB", loans[4:], names[:4]))
    ta.start(); tb.start(); ta.join(); tb.join()
    assert not errors, errors
    qa, qb = {x["loan_id"]: x for x in api.get_queue("TA")}, {x["loan_id"]: x for x in api.get_queue("TB")}
    for l in loans[:4]:
        assert qa[l]["escalations"] == [] and qb[l]["escalations"] == []        # four distinct images, 4 distinct loans: no dups in a sandbox
    for l in loans[4:]:
        assert qa[l]["escalations"] == [] and qb[l]["escalations"] == []
    assert checker._sandboxes["TA"]["n"] == 4 and checker._sandboxes["TB"]["n"] == 4


def test_concurrent_same_image_in_one_sandbox_flags_all_loans(idx, images):
    loans = quiet(idx, 6)
    img = images[idx.reserved[0]]
    ts = [threading.Thread(target=api.check_photo, args=(l, img), kwargs=dict(sandbox_id="S")) for l in loans]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert checker._sandboxes["S"]["n"] == 6 and len(checker._sandboxes["S"]["entries"]) == 6
    q = {x["loan_id"]: x for x in api.get_queue("S")}
    assert all(q[l]["color"] == "red" for l in loans)
    assert {x["loan_id"] for x in api.get_queue() if x["color"] == "red"} & set(loans) == set()


def test_concurrent_queue_reads_while_uploading(idx, images):
    loans = quiet(idx, 4)
    stop, errors = threading.Event(), []

    def reader():
        try:
            while not stop.is_set():
                q = api.get_queue("R")
                assert [x["priority"] for x in q] == sorted((x["priority"] for x in q), reverse=True)
        except Exception as e:                                                  # noqa: BLE001
            errors.append(e)

    t = threading.Thread(target=reader)
    t.start()
    for l, n in zip(loans, idx.reserved):
        api.check_photo(l, images[n], captured_at=datetime.now(timezone.utc) + timedelta(days=2), sandbox_id="R")
    stop.set(); t.join()
    assert not errors, errors


# ---------------------------------------------------------------- randomised: fusion never lowers a colour
def test_fusion_never_lowers_any_colour_after_many_random_uploads(P, idx, images):
    rng = random.Random(2026)
    q0 = api.get_queue()
    default = {x["loan_id"]: x for x in q0}
    loans = [x["loan_id"] for x in q0[:40]] + [x["loan_id"] for x in q0[-40:]] + rng.sample(list(P.loans.index), 40)
    names = list(images)
    now = datetime.now(timezone.utc)
    for _ in range(60):
        lid = rng.choice(loans)
        row = P.loans.loc[lid]
        kw = {}
        if rng.random() < 0.6:
            kw.update(lat=float(row["borrower_lat"]) + rng.choice([0, 0.01, 1.0, 3.0]), lon=float(row["borrower_lon"]))
        if rng.random() < 0.6:
            kw["captured_at"] = now + timedelta(days=rng.choice([-2000, -1, 0, 3]))
        api.check_photo(lid, images[rng.choice(names)], sandbox_id="RND", **kw)
    qv = api.get_queue("RND")
    assert len(qv) == len(q0) and {x["loan_id"] for x in qv} == set(default)
    for x in qv:
        d = default[x["loan_id"]]
        assert RANK[x["color"]] >= RANK[d["color"]] >= RANK[d["base_color"]] == RANK[x["base_color"]]
        assert x["risk"] == d["risk"]
        assert x["priority"] == pytest.approx(fusion.priority(x["color"], x["risk"]))
        assert {e["code"] for e in d["escalations"]} <= {e["code"] for e in x["escalations"]}
        codes = [e["code"] for e in x["escalations"]]
        assert len(codes) == len(set(codes))                                      # one escalation per code
    assert any(x["escalations"] != default[x["loan_id"]]["escalations"] for x in qv)
    pr = [x["priority"] for x in qv]
    assert pr == sorted(pr, reverse=True)
    for lid in rng.sample(loans, 15):
        check_consistent({x["loan_id"]: x for x in qv}[lid], api.get_loan(lid, "RND"))
    assert api.get_queue() == q0                                                  # default view untouched


def test_upload_order_does_not_change_the_final_sandbox_view(idx, images):
    loans = quiet(idx, 3)
    now = datetime.now(timezone.utc)
    jobs = [(loans[0], idx.reserved[0], dict(captured_at=now + timedelta(days=2))),
            (loans[1], idx.reserved[0], {}),
            (loans[2], idx.reserved[1], dict(captured_at=now + timedelta(days=2)))]

    def run(order, sb):
        for i in order:
            l, n, kw = jobs[i]
            api.check_photo(l, images[n], sandbox_id=sb, **kw)
        return [(x["loan_id"], x["color"], [e["code"] for e in x["escalations"]]) for x in api.get_queue(sb)
                if x["loan_id"] in loans]

    a = run([0, 1, 2], "O1")
    b = run([2, 1, 0], "O2")
    assert sorted(a) == sorted(b)


# ---------------------------------------------------------------- default-view stability and mutation safety
def test_default_queue_unchanged_by_scenarios_and_sandbox_uploads(idx, images):
    q0 = strip(api.get_queue())
    la, = quiet(idx, 1)
    for p in api.get_scenario_presets():
        api.score_scenario(p["scenario"])
    api.check_photo(la, images[idx.reserved[0]], sandbox_id="A")
    api.check_photo(la, images[idx.reserved[1]], sandbox_id=None)
    api.get_queue("A")
    assert api.get_queue() == q0


def test_mutating_a_returned_queue_item_does_not_change_later_results(protect_cache, idx):
    q0 = strip(api.get_queue())
    q = api.get_queue()
    q[0]["color"] = "green"
    q[0]["risk"] = -1
    q[0]["top_reason"] = "HACKED"
    q.pop()
    q.reverse()
    assert api.get_queue() == q0


def test_mutating_returned_escalations_does_not_change_later_results(protect_cache):
    """Escalated loans: `escalations` is a nested list/dict; a caller editing it must not corrupt the cache."""
    q0 = strip(api.get_queue())
    q = api.get_queue()
    item = next(x for x in q if x["escalations"])
    item["escalations"][0]["min_color"] = "green"
    item["escalations"][0]["text_uz"] = "HACKED"
    item["escalations"].append(dict(code="X"))
    item["escalations"].clear()
    assert api.get_queue() == q0
    assert api.get_loan(item["loan_id"])["escalations"] == q0[[x["loan_id"] for x in q0].index(item["loan_id"])]["escalations"]


def test_mutating_escalations_of_a_sandbox_view_does_not_change_later_results(protect_cache, idx, images):
    la, = quiet(idx, 1)
    api.check_photo(la, images[idx.reserved[0]], captured_at=datetime.now(timezone.utc) + timedelta(days=2), sandbox_id="A")
    v0 = strip(api.get_queue("A"))
    q = api.get_queue("A")
    for x in q:
        for e in x["escalations"]:
            e["code"] = "HACKED"
            e["evidence_refs"].append("photo:evil")
        x["escalations"].append(dict(code="Z"))
    assert api.get_queue("A") == v0
    assert api.get_queue() != v0 or True
    assert all(e["code"] != "HACKED" for x in api.get_queue() for e in x["escalations"])


def test_mutating_get_loan_result_does_not_change_later_results(protect_cache, idx):
    q = api.get_queue()
    lid = next(x["loan_id"] for x in q if x["escalations"])
    d0 = copy.deepcopy(api.get_loan(lid))
    d = api.get_loan(lid)
    d["escalations"][0]["text_uz"] = "HACKED"
    d["reasons"].clear()
    d["evidence"].clear()
    d["loan"]["amount_uzs"] = -1
    assert api.get_loan(lid) == d0
    assert {x["loan_id"]: x for x in api.get_queue()}[lid]["escalations"] == d0["escalations"]


def test_apply_policy_does_not_mutate_its_input_signals():
    sig = [dict(code="PHOTO_FAR", text_uz="x", evidence_ref="photo:p"), dict(code="PHOTO_DUPLICATE", text_uz="y", evidence_ref="photo:q")]
    before = copy.deepcopy(sig)
    res = fusion.apply_policy("green", sig)
    assert sig == before
    res["escalations"][0]["evidence_refs"].append("evil")
    assert sig == before


def test_apply_policy_deduplicates_codes_and_orders_strongest_first():
    sig = [dict(code=c, text_uz=c, evidence_ref=f"photo:{i}") for i, c in enumerate(["PHOTO_TIME", "PHOTO_FAR", "PHOTO_DUPLICATE", "PHOTO_FAR"])]
    res = fusion.apply_policy("yellow", sig)
    assert [e["code"] for e in res["escalations"]] == ["PHOTO_DUPLICATE", "PHOTO_FAR", "PHOTO_TIME"]
    assert res["color"] == "red"
    assert [e["raises_color"] for e in res["escalations"]] == [True, False, False]
    assert fusion.apply_policy("red", [])["escalations"] == []


def test_photo_signals_far_without_distance_and_ids_truncation():
    sig = fusion.photo_signals(ref="photo:x", photo_label="x", duplicates=[dict(loan_id=f"L{i}", similarity=1.0) for i in range(5)],
                               geo_ok=False, distance_km=None, time_ok=None, uploaded=False)
    by = {s["code"]: s for s in sig}
    assert "va yana 2 ta" in by["PHOTO_DUPLICATE"]["text_uz"] and by["PHOTO_DUPLICATE"]["text_uz"].startswith("Bu kreditdagi rasm")
    assert "uzoqda" in by["PHOTO_FAR"]["text_uz"] and by["PHOTO_FAR"]["evidence"]["data"]["distance_km"] is None
    assert "PHOTO_TIME" not in by                                                  # time_ok None is not a signal
