"""Zero-value reason rule over a larger sample of the demo queue (Part 0c)."""
from __future__ import annotations

import re

import pytest

from ml import api
from ml.money_trail.reasons import NO_REASON_UZ

ZERO = re.compile(r"(?<![\d.,])0 (foiz|ta |kun|so'm)|\b0\.0 kun|atigi 0|(?<![\d.,])0%|(?<![\d.,])0\.0+ ")


@pytest.fixture(scope="module")
def queue():
    return api.get_queue()


def test_all_top_reasons_in_the_queue_are_clean_and_non_empty(queue):
    for x in queue:
        assert x["top_reason"].strip()
        assert not ZERO.search(x["top_reason"]), (x["loan_id"], x["top_reason"])


def test_reasons_of_a_large_spread_sample_are_clean(queue):
    sample = queue[::7] + queue[:60] + queue[-30:]
    n_reasons = 0
    for x in sample:
        d = api.get_loan(x["loan_id"])
        rs = d["reasons"]
        assert x["top_reason"] == (rs[0]["text_uz"] if rs else NO_REASON_UZ)
        codes = [r["code"] for r in rs]
        assert len(codes) == len(set(codes))
        for r in rs:
            n_reasons += 1
            assert not ZERO.search(r["text_uz"]), (x["loan_id"], r["code"], r["text_uz"])
            assert r["weight"] is None or 0 < r["weight"] <= 1.0001
            assert r["text_uz"].strip()
    assert n_reasons > 100


def test_red_loans_have_at_least_one_reason(queue):
    for x in [q for q in queue if q["color"] == "red"][:80]:
        assert x["top_reason"] != NO_REASON_UZ, x["loan_id"]


def test_score_loan_reasons_are_clean_too(queue):
    for x in queue[::40]:
        for r in api.score_loan(x["loan_id"])["reasons"]:
            assert not ZERO.search(r["text_uz"]), r["text_uz"]
            assert r["weight"] > 0
