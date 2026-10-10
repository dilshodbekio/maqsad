"""Asset check (OpenCLIP ViT-B-32 + linear probe): asset_match in check_photo and PHOTO_ASSET_MISMATCH fusion.
These tests need the CLIP weights (cache dir `ml/.cache/clip` or MAQSAD_CLIP_CACHE) and are the only ones that run CLIP."""
from __future__ import annotations

import csv
import json
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from ml import api, fusion
from ml.money_trail.service import get_portfolio
from ml.photo import asset, checker, config, imageio
from ml.photo.index import get_index

pytestmark = pytest.mark.clip


@pytest.fixture(autouse=True)
def clean():
    api.reset_sandbox(None)
    yield
    api.reset_sandbox(None)


@pytest.fixture(scope="module")
def P():
    return get_portfolio()


@pytest.fixture(scope="module")
def am():
    return asset.get_asset_model()


def image(name) -> bytes:
    return (config.ASSETS / name).read_bytes()


def manifest():
    return list(csv.DictReader(open(config.ASSETS / "manifest.csv", encoding="utf-8")))


def loan_of(P, purpose):
    idx = get_index()
    return next(l for l in P.loans.index if str(P.loans.at[l, "purpose"]) == purpose and l not in idx.by_loan)


# ---------------------------------------------------------------- artifacts and library
def test_probe_artifact_is_small_and_consistent():
    meta = json.loads(asset.PROBE_JSON.read_text(encoding="utf-8"))
    z = np.load(asset.PROBE_NPZ)
    assert meta["classes"] == list(asset.CLASSES) and meta["model"] == "ViT-B-32" and meta["pretrained"] == "openai"
    assert 0.05 <= meta["tau"] <= 0.95 and z["W"].shape == (6, 512) and z["b"].shape == (6,)
    assert asset.PROBE_NPZ.stat().st_size < 100_000 and meta["n_train"] >= 140


def test_library_has_a_negative_class_with_licences():
    rows = manifest()
    other = [r for r in rows if r["purpose"] == "other"]
    assert len(other) >= 40
    assert all(r["license"] and r["author"] and r["source_url"].startswith("https://commons.wikimedia.org/") for r in other)
    assert all(not r["license"].upper().count("NC") and "ND" not in r["license"].split() for r in other)


# ---------------------------------------------------------------- asset_match in check_photo
def test_asset_match_shape_and_values(P):
    r = api.check_photo(loan_of(P, "tikuv_dastgohi"), image("tikuv_dastgohi_00.jpg"))
    a = r["asset_match"]
    assert set(a) == {"label", "score", "matches_purpose", "label_uz"}
    assert a["label"] in asset.CLASSES and 0 < a["score"] <= 1 and isinstance(a["matches_purpose"], bool) and a["label_uz"]


def test_sewing_machine_matches_a_sewing_machine_loan_and_a_random_thing_does_not(P):
    loan = loan_of(P, "tikuv_dastgohi")
    sewing = [r["file"] for r in manifest() if r["purpose"] == "tikuv_dastgohi"]
    assert sum(api.check_photo(loan, image(f))["asset_match"]["matches_purpose"] for f in sewing[:6]) >= 5
    ok = api.check_photo(loan, image(sewing[5]))["asset_match"]
    assert ok["matches_purpose"] is True and ok["label"] == "tikuv_dastgohi" and ok["label_uz"] == "tikuv mashinasi"
    others = [r["file"] for r in manifest() if r["purpose"] == "other"]
    bad = [api.check_photo(loan, image(f))["asset_match"] for f in others[:8]]
    assert sum(not a["matches_purpose"] for a in bad) >= 7                       # random things do not match


def test_a_cow_does_not_match_a_sewing_machine_loan_but_matches_a_cattle_loan(P):
    cow = image("chorva_03.jpg")
    assert api.check_photo(loan_of(P, "tikuv_dastgohi"), cow)["asset_match"]["matches_purpose"] is False
    assert api.check_photo(loan_of(P, "chorva"), cow)["asset_match"]["matches_purpose"] is True


def test_non_asset_photo_is_named_in_uzbek(P):
    loan = loan_of(P, "chorva")
    names = {api.check_photo(loan, image(r["file"]))["asset_match"]["label_uz"] for r in manifest() if r["purpose"] == "other"}
    assert names & set(asset.OTHER_FINE)                                          # e.g. avtomobil, xona, odam ...


def test_asset_check_is_deterministic_and_thread_safe(P):
    loan = loan_of(P, "issiqxona")
    img = image("issiqxona_04.jpg")
    first = api.check_photo(loan, img)["asset_match"]
    with ThreadPoolExecutor(max_workers=4) as ex:
        res = list(ex.map(lambda _: api.check_photo(loan, img)["asset_match"], range(8)))
    assert all(r == first for r in res)


def test_asset_match_survives_typical_transforms(P):
    from ml.photo.transforms import TRANSFORMS
    loan = loan_of(P, "parranda")
    rgb = imageio.decode(image("parranda_00.jpg"))
    base = api.check_photo(loan, image("parranda_00.jpg"))["asset_match"]["label"]
    rng = np.random.default_rng(0)
    kept = sum(api.check_photo(loan, imageio.encode_jpeg(TRANSFORMS[t](rgb, rng)))["asset_match"]["label"] == base
               for t in ("JPEG q=30", "resize 50%", "screenshot", "rotate 5 deg"))
    assert kept >= 3


def test_latency_bound(P):
    loan = loan_of(P, "chorva")
    img = image("chorva_00.jpg")
    api.check_photo(loan, img)
    ms = []
    for _ in range(5):
        t = time.perf_counter()
        api.check_photo(loan, img)
        ms.append((time.perf_counter() - t) * 1000)
    assert float(np.median(ms)) < 1500, ms                                         # CPU, warm


# ---------------------------------------------------------------- fusion
def test_mismatch_escalates_to_at_least_yellow_in_that_sandbox_only(P):
    loan = next(x["loan_id"] for x in reversed(api.get_queue()) if x["color"] == "green" and not x["escalations"]
                and str(P.loans.at[x["loan_id"], "purpose"]) == "tikuv_dastgohi" and x["loan_id"] not in get_index().by_loan)
    other = next(r["file"] for r in manifest() if r["purpose"] == "other")
    r = api.check_photo(loan, image(other), sandbox_id="A")
    assert r["asset_match"]["matches_purpose"] is False
    d = api.get_loan(loan, "A")
    esc = [e for e in d["escalations"] if e["code"] == "PHOTO_ASSET_MISMATCH"]
    assert len(esc) == 1 and esc[0]["min_color"] == "yellow" and esc[0]["raises_color"]
    assert esc[0]["text_uz"] == f"Rasmda kredit maqsadiga mos aktiv aniqlanmadi (aniqlangan: {r['asset_match']['label_uz']})"
    assert RANK(d["color"]) >= 1 and d["risk"] == api.get_loan(loan)["risk"]
    assert d["reasons"][0]["code"] == "PHOTO_ASSET_MISMATCH" and d["reasons"][0]["weight"] is None
    assert any(e["kind"] == "photo" and e["data"]["asset_match"]["matches_purpose"] is False for e in d["evidence"])
    q = {x["loan_id"]: x for x in api.get_queue("A")}
    assert q[loan]["color"] in ("yellow", "red") and q[loan]["top_reason"] == esc[0]["text_uz"]
    assert api.get_loan(loan)["color"] == "green" and api.get_loan(loan, "B")["escalations"] == []   # others unaffected
    api.reset_sandbox("A")
    assert api.get_loan(loan, "A")["escalations"] == []


def test_a_matching_photo_does_not_escalate(P):
    loan = loan_of(P, "chorva")
    r = api.check_photo(loan, image("chorva_02.jpg"), sandbox_id="A")
    assert r["asset_match"]["matches_purpose"] is True
    assert "PHOTO_ASSET_MISMATCH" not in [e["code"] for e in api.get_loan(loan, "A")["escalations"]]


def test_mismatch_never_lowers_a_colour(P):
    q = api.get_queue()
    red = next(x["loan_id"] for x in q if x["base_color"] == "red" and not x["escalations"] and x["loan_id"] not in get_index().by_loan)
    other = next(r["file"] for r in manifest() if r["purpose"] == "other")
    api.check_photo(red, image(other), sandbox_id="A")
    for x in api.get_queue("A"):
        assert RANK(x["color"]) >= RANK(x["base_color"])
    assert api.get_loan(red, "A")["color"] == "red"


def RANK(c):
    return fusion.COLOR_RANK[c]


def test_policy_is_active_and_unit_signal():
    assert fusion.POLICY["PHOTO_ASSET_MISMATCH"]["active"] is True and fusion.POLICY["PHOTO_ASSET_MISMATCH"]["min_color"] == "yellow"
    bad = dict(label="other", score=0.9, matches_purpose=False, label_uz="avtomobil")
    sig = fusion.photo_signals(ref="photo:x", photo_label="x", duplicates=[], geo_ok=None, distance_km=None, time_ok=None,
                               asset_match=bad, uploaded=True)
    assert [s["code"] for s in sig] == ["PHOTO_ASSET_MISMATCH"]
    assert sig[0]["text_uz"] == "Rasmda kredit maqsadiga mos aktiv aniqlanmadi (aniqlangan: avtomobil)"
    assert fusion.photo_signals(ref="p", photo_label="x", duplicates=[], geo_ok=None, distance_km=None, time_ok=None,
                                asset_match=dict(bad, matches_purpose=True), uploaded=True) == []
    assert fusion.photo_signals(ref="p", photo_label="x", duplicates=[], geo_ok=None, distance_km=None, time_ok=None,
                                asset_match=None, uploaded=True) == []


# ---------------------------------------------------------------- portfolio photos, warm-up, graceful degradation
def test_portfolio_photos_carry_the_asset_check(P):
    idx = get_index()
    assert all(i.get("asset") for i in idx.images)
    lid = next(iter(idx.by_loan))
    ph = api.get_loan(lid)["photos"][0]
    assert "asset_match" not in ph                                    # not part of the backend's Photo schema (contract v1.4 shape)
    assert set(idx.photos_of(lid)[0]["asset_match"]) == {"label", "score", "matches_purpose", "label_uz"}


def test_warm_up_loads_clip():
    out = api.warm_up()
    assert out["clip_loaded"] is True and out["total_s"] >= 0 and set(out) >= {"portfolio_s", "face_detector_s", "clip_s"}
    assert asset._model is not None


def test_check_photo_degrades_gracefully_without_clip(P, monkeypatch):
    def boom():
        raise ImportError("torch missing")
    monkeypatch.setattr(asset, "get_asset_model", boom)
    monkeypatch.setattr(checker, "_asset_warned", False)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r = api.check_photo(loan_of(P, "chorva"), image("chorva_00.jpg"), sandbox_id="A")
        r2 = api.check_photo(loan_of(P, "chorva"), image("chorva_01.jpg"), sandbox_id="A")
    assert r["asset_match"] is None and r2["asset_match"] is None
    assert sum("asset check disabled" in str(x.message) for x in w) == 1                # warned once
    assert "PHOTO_ASSET_MISMATCH" not in [e["code"] for x in api.get_queue("A") for e in x["escalations"]]
