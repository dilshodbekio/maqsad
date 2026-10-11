"""Every reason of every demo loan cites evidence (a transaction, the invoice, the supplier graph, a photo or the checked period)."""
import pytest

from ml import api


def _check(step: int):
    q = api.get_queue()
    assert len(q) == 5000
    n = no_refs = unresolved = 0
    kinds = set()
    for it in q[::step]:
        d = api.get_loan(it["loan_id"])
        ev = {e["ref"]: e for e in d["evidence"]}
        kinds |= {e["kind"] for e in d["evidence"]}
        for r in d["reasons"]:
            n += 1
            no_refs += not r["evidence_refs"]
            unresolved += sum(ref not in ev for ref in r["evidence_refs"])
    assert n > 10000 // step
    assert no_refs == 0 and unresolved == 0
    assert "period" in kinds                                  # absence reasons (no income / no cash-out) cite the checked period


def test_every_reason_of_every_8th_demo_loan_has_resolvable_evidence():
    _check(8)


@pytest.mark.slow
def test_every_reason_of_the_whole_demo_queue_has_resolvable_evidence():
    _check(1)


def test_absence_reason_points_to_the_checked_period():
    for it in api.get_queue():
        d = api.get_loan(it["loan_id"])
        r = next((r for r in d["reasons"] if r["code"] == "income_rate_per_month" and "qayd etilmagan" in r["text_uz"]), None)
        if r is None:
            continue
        ref = next(x for x in r["evidence_refs"] if x.startswith("period:"))
        e = next(e for e in d["evidence"] if e["ref"] == ref)
        assert e["kind"] == "period" and e["data"]["start"] in e["summary_uz"] and e["data"]["end"] in e["summary_uz"]
        assert e["data"]["days"] >= 1 and e["data"]["start"] < e["data"]["end"]
        return
    raise AssertionError("no loan with an 'income not recorded' reason in the demo queue")
