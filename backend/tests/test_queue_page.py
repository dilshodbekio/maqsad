"""GET /api/queue (paging, filters, counts), /api/queue/facets, /api/queue/legend — contract v1.3."""
import types
from collections import Counter

import pytest

from backend.app import ml_stub
from backend.app.ml_adapter import MLAdapter


def _all(client, **params) -> list:
    return client.get("/api/queue", params={"limit": 200, **params}).json()["items"]


# ---------- /api/queue ----------

def test_counts_are_whole_queue_and_ignore_filters(client):
    full = _all(client)
    expected = {c: n for c, n in Counter(i["color"] for i in full).items()}
    for params in ({}, {"color": "red"}, {"region": full[0]["region"]}, {"purpose": "chorva", "color": "green"}):
        page = client.get("/api/queue", params=params).json()
        assert page["counts"] == {"red": expected.get("red", 0), "yellow": expected.get("yellow", 0),
                                  "green": expected.get("green", 0)}, params


@pytest.mark.parametrize("key", ["color", "region", "purpose"])
def test_single_filter(client, key):
    full = _all(client)
    for value in {i[key] for i in full}:
        page = client.get("/api/queue", params={key: value, "limit": 200}).json()
        want = [i["loan_id"] for i in full if i[key] == value]
        assert [i["loan_id"] for i in page["items"]] == want  # same (priority) order as the full queue
        assert page["total"] == len(want) > 0


def test_combined_filters(client):
    full = _all(client)
    region, purpose = full[0]["region"], full[0]["purpose"]
    page = client.get("/api/queue", params={"region": region, "purpose": purpose, "limit": 200}).json()
    want = [i["loan_id"] for i in full if i["region"] == region and i["purpose"] == purpose]
    assert [i["loan_id"] for i in page["items"]] == want and page["total"] == len(want)


def test_pages_concatenate_to_full_order(client):
    full = [i["loan_id"] for i in _all(client)]
    got, offset = [], 0
    while True:
        page = client.get("/api/queue", params={"limit": 7, "offset": offset}).json()
        assert page["total"] == len(full)
        if not page["items"]:
            break
        assert len(page["items"]) <= 7
        got += [i["loan_id"] for i in page["items"]]
        offset += 7
    assert got == full


def test_offset_past_end_is_empty_page(client):
    page = client.get("/api/queue", params={"offset": 10_000}).json()
    assert page["items"] == [] and page["total"] == 40


def test_unknown_region_or_purpose_is_empty_not_error(client):
    for params in ({"region": "Atlantida"}, {"purpose": "kosmodrom"}):
        r = client.get("/api/queue", params=params)
        assert r.status_code == 200
        assert r.json()["items"] == [] and r.json()["total"] == 0
        assert sum(r.json()["counts"].values()) == 40


def test_empty_filter_values_mean_no_filter(client):
    page = client.get("/api/queue", params={"region": "", "purpose": ""}).json()
    assert page["total"] == 40


@pytest.mark.parametrize("params", [
    {"limit": 0}, {"limit": 201}, {"limit": "abc"}, {"offset": -1}, {"color": "blue"}, {"region": "x" * 65},
])
def test_bad_query_is_422_in_uzbek(client, params):
    r = client.get("/api/queue", params=params)
    assert r.status_code == 422
    assert r.json()["detail"].startswith("So'rov ma'lumotlari noto'g'ri")


def _big_fake_ml(n=130):
    """ml returning more loans than the default page: stub items with distinct ids, ml (priority) order."""
    base = ml_stub.get_queue()
    items = [{**base[k % len(base)], "loan_id": f"X{k:04d}"} for k in range(n)]
    items.sort(key=lambda x: -x["priority"])
    m = types.ModuleType("fake_ml")
    m.MODEL_VERSION = "fake-1"
    m.get_queue = lambda sandbox_id=None: [dict(i) for i in items]
    m.get_loan, m.score_loan = ml_stub.get_loan, ml_stub.score_loan
    m.check_photo = lambda *a, **k: {}
    m.reset_sandbox = lambda sandbox_id: None
    return m, items


def test_default_limit_50_and_max_200(make_client):
    m, items = _big_fake_ml(260)
    c = make_client(adapter=MLAdapter(m, "real"))
    page = c.get("/api/queue").json()
    assert len(page["items"]) == 50 and page["total"] == 260
    assert [i["loan_id"] for i in page["items"]] == [i["loan_id"] for i in items[:50]]
    assert len(c.get("/api/queue", params={"limit": 200}).json()["items"]) == 200
    assert sum(page["counts"].values()) == 260


def test_queue_keeps_ml_order(make_client):
    """Order is ml's priority order; the backend does not re-rank (ties keep ml's order)."""
    m, items = _big_fake_ml(60)
    c = make_client(adapter=MLAdapter(m, "real"))
    got = [i["loan_id"] for i in c.get("/api/queue", params={"limit": 200}).json()["items"]]
    assert got == [i["loan_id"] for i in items]


# ---------- /api/queue/facets ----------

def test_facets_from_queue(client):
    full = _all(client)
    r = client.get("/api/queue/facets")
    assert r.status_code == 200
    assert r.json() == {"regions": sorted({i["region"] for i in full}),
                        "purposes": sorted({i["purpose"] for i in full})}


def test_facets_wrong_method(client):
    assert client.post("/api/queue/facets").status_code == 405


# ---------- /api/queue/legend ----------

def test_legend_stub_matches_queue_counts(client):
    r = client.get("/api/queue/legend")
    assert r.status_code == 200
    lg = r.json()
    counts = client.get("/api/queue").json()["counts"]
    assert lg["n_loans"] == sum(counts.values())
    assert (lg["n_red_total"], lg["n_yellow_total"]) == (counts["red"], counts["yellow"])  # stub: no fusion
    assert 0 <= lg["yellow"] <= lg["red"] <= 1
    assert lg["rule"].startswith("STUB")


def test_legend_not_ready_is_503(make_client):
    m, _ = _big_fake_ml(10)
    m.get_color_thresholds = lambda: (_ for _ in ()).throw(NotImplementedError("later"))
    c = make_client(adapter=MLAdapter(m, "real"))
    assert c.get("/api/health").json()["functions"]["get_color_thresholds"] == "not_ready"
    r = c.get("/api/queue/legend")
    assert r.status_code == 503 and r.json()["detail"].startswith("Bu funksiya hali ulanmagan")
    assert c.get("/api/queue").status_code == 200  # the queue itself still works


def test_legend_missing_is_503(make_client):
    m, _ = _big_fake_ml(10)  # no get_color_thresholds at all
    c = make_client(adapter=MLAdapter(m, "real"))
    assert c.get("/api/health").json()["functions"]["get_color_thresholds"] == "missing"
    assert c.get("/api/queue/legend").status_code == 503


def test_legend_bad_ml_shape_is_502(make_client):
    m, _ = _big_fake_ml(10)
    m.get_color_thresholds = lambda: {"red": 0.5}
    c = make_client(adapter=MLAdapter(m, "real"))
    r = c.get("/api/queue/legend")
    assert r.status_code == 502 and "Model xizmatida xatolik" in r.json()["detail"]
