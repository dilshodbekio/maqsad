"""Part 0: parallel uploads from different sandboxes lose / mix nothing; sandbox_id is validated everywhere."""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from ml import api
from ml.errors import InvalidRequestError
from ml.money_trail.service import get_portfolio
from ml.photo import checker, config
from ml.photo.index import get_index
from ml.sandbox import MAX_LEN, validate_sandbox_id


@pytest.fixture(autouse=True)
def clean():
    api.reset_sandbox(None)
    yield
    api.reset_sandbox(None)


@pytest.fixture(scope="module")
def P():
    return get_portfolio()


def image(name) -> bytes:
    return (config.ASSETS / name).read_bytes()


# ---------------------------------------------------------------- parallel uploads
def test_parallel_uploads_from_different_sandboxes_lose_and_mix_nothing(P):
    idx = get_index()
    loans = [l for l in P.loans.index if l not in idx.by_loan][:24]
    photos = [i["file"] for i in idx.images if len(i["loans"]) == 1][:24]
    jobs = [(f"sbx-{k % 8}", loans[k], photos[k]) for k in range(24)]      # 8 sandboxes x 3 uploads, all different loans
    barrier = threading.Barrier(8)
    errors = []

    def run(job):
        sb, loan, photo = job
        try:
            barrier.wait(timeout=30) if job is jobs[0] or job in jobs[:8] else None
            return api.check_photo(loan, image(photo), sandbox_id=sb)
        except Exception as e:                                          # noqa: BLE001
            errors.append(repr(e))

    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(run, jobs))
    assert not errors and all(r is not None for r in results)
    for k in range(8):
        entries = checker._sandboxes[f"sbx-{k}"]["entries"]
        mine = [j for j in jobs if j[0] == f"sbx-{k}"]
        assert sorted(e["loan_id"] for e in entries) == sorted(j[1] for j in mine)      # nothing lost, nothing foreign
        assert sorted(e["seq"] for e in entries) == [1, 2, 3] and checker._sandboxes[f"sbx-{k}"]["n"] == 3
    # every portfolio photo is its own image -> each upload sees exactly its portfolio owner, never another sandbox's loan
    for (sb, loan, photo), r in zip(jobs, results):
        owner = next(i["loans"][0] for i in idx.images if i["file"] == photo)
        assert [d["loan_id"] for d in r["duplicate_of"]] == [owner]
    # views are isolated too
    for k in range(8):
        escalated = {x["loan_id"] for x in api.get_queue(f"sbx-{k}") if x["escalations"]}
        assert {j[1] for j in jobs if j[0] == f"sbx-{k}"} <= escalated
        others = {j[1] for j in jobs if j[0] != f"sbx-{k}"}
        assert not others & (escalated - {x["loan_id"] for x in api.get_queue() if x["escalations"]})


def test_concurrent_uploads_of_the_same_photo_in_one_sandbox_see_each_other_once(P):
    idx = get_index()
    la, lb = [l for l in P.loans.index if l not in idx.by_loan][:2]
    new = image(idx.reserved[0])
    for _ in range(5):
        api.reset_sandbox("race")
        barrier = threading.Barrier(2)

        def up(loan):
            barrier.wait(timeout=30)
            return api.check_photo(loan, new, sandbox_id="race")

        with ThreadPoolExecutor(max_workers=2) as ex:
            ra, rb = list(ex.map(up, (la, lb)))
        seen = [d["loan_id"] for d in ra["duplicate_of"]] + [d["loan_id"] for d in rb["duplicate_of"]]
        assert sorted(seen) in ([la], [lb])                         # the later upload sees the earlier one; never both, never none
        assert checker._sandboxes["race"]["n"] == 2
        view = {x["loan_id"]: x for x in api.get_queue("race")}
        assert view[la]["color"] == view[lb]["color"] == "red"      # the view (computed later) escalates both


def test_concurrent_reset_and_upload_never_raise(P):
    idx = get_index()
    loan = [l for l in P.loans.index if l not in idx.by_loan][0]
    img = image(idx.reserved[1])
    stop = threading.Event()
    errors = []

    def uploader():
        while not stop.is_set():
            try:
                api.check_photo(loan, img, sandbox_id="x")
                api.get_queue("x")
            except Exception as e:                                  # noqa: BLE001
                errors.append(repr(e))

    t = threading.Thread(target=uploader)
    t.start()
    for _ in range(40):
        api.reset_sandbox("x")
        api.reset_sandbox(None)
    stop.set()
    t.join(timeout=60)
    assert not errors


def test_detector_is_created_once_under_a_race():
    checker._detector_obj = None
    with ThreadPoolExecutor(max_workers=8) as ex:
        objs = list(ex.map(lambda _: checker._detector(), range(16)))
    assert len({id(o) for o in objs}) == 1


# ---------------------------------------------------------------- sandbox_id validation
GOOD = ["a", "A-b_c.d:e", "sandbox-123", "x" * MAX_LEN, "0"]
BAD = ["", "x" * (MAX_LEN + 1), " a", "a b", "a/b", "a\\b", "ключ", "a\nb", "a;b", "<script>", "a%20b", "../etc", "a\x00b", 5, 1.5,
       b"abc", ["a"], {"a": 1}, ("a",), True, object()]


@pytest.mark.parametrize("sid", GOOD)
def test_valid_sandbox_ids_are_accepted(sid, P):
    assert validate_sandbox_id(sid) == sid
    api.get_queue(sid)
    api.reset_sandbox(sid)


def test_none_means_portfolio_mode():
    assert validate_sandbox_id(None) is None


@pytest.mark.parametrize("sid", BAD, ids=lambda v: repr(v)[:30])
def test_invalid_sandbox_ids_raise_invalid_request_error_everywhere(sid, P):
    idx = get_index()
    loan = next(iter(idx.by_loan))
    for call in (lambda: api.get_queue(sid), lambda: api.get_loan(loan, sid), lambda: api.reset_sandbox(sid),
                 lambda: api.check_photo(loan, image(idx.reserved[0]), sandbox_id=sid)):
        with pytest.raises(InvalidRequestError) as e:
            call()
        assert isinstance(e.value, ValueError) and str(e.value).strip()


def test_invalid_sandbox_id_is_exported_and_does_not_create_state(P):
    assert api.InvalidRequestError is InvalidRequestError
    n = len(checker._sandboxes)
    idx = get_index()
    with pytest.raises(InvalidRequestError):
        api.check_photo(next(iter(idx.by_loan)), image(idx.reserved[0]), sandbox_id="bad id")
    assert len(checker._sandboxes) == n
