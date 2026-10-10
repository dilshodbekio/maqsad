"""Extra asset-check tests: consistency across index / sandbox view / check_photo, threshold semantics, per-purpose behaviour,
odd inputs, mutation safety, concurrency, escalation merge, warm_up."""
from __future__ import annotations

import copy
import csv
import io
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from PIL import Image

from ml import api, fusion
from ml.generator.config import PURPOSES
from ml.money_trail.service import get_portfolio
from ml.photo import asset, config, imageio
from ml.photo.index import get_index

pytestmark = pytest.mark.clip

KEYS = {"label", "score", "matches_purpose", "label_uz"}


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
    with open(config.ASSETS / "manifest.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def free_loan(P, purpose, skip=0):
    idx = get_index()
    ls = [l for l in P.loans.index if str(P.loans.at[l, "purpose"]) == purpose and l not in idx.by_loan]
    return ls[skip]


def png(mode, size, color=None) -> bytes:
    im = Image.new(mode, size, color)
    b = io.BytesIO()
    im.save(b, "PNG")
    return b.getvalue()


@pytest.fixture(scope="module")
def probs(am):
    """probe probabilities of 8 library images per class (embedding computed once)."""
    out = {}
    by = {}
    for r in manifest():
        by.setdefault(r["purpose"], []).append(r["file"])
    for cls, files in by.items():
        for f in files[:8]:
            e = am.embed(imageio.decode(image(f)))
            out[f] = (cls, am.probe_probs(e)[0])
    return out


# ---------------------------------------------------------------- consistency
def test_index_asset_equals_check_photo_for_the_same_image(P):
    idx = get_index()
    for im in [i for i in idx.images if len(i["loans"]) == 1][::6][:4]:
        loan = free_loan(P, im["purpose"])
        got = api.check_photo(loan, image(im["file"]))["asset_match"]
        assert got == im["asset"], (im["file"], got, im["asset"])


def test_portfolio_view_sandbox_view_and_check_photo_agree(P):
    idx = get_index()
    im = next(i for i in idx.images if len(i["loans"]) == 1 and i["purpose"] == "chorva")
    owner = im["loans"][0]
    assert "asset_match" not in api.get_loan(owner)["photos"][0]      # not in the HTTP photo shape; it lives in the index
    ph = idx.photos_of(owner)[0]["asset_match"]
    assert ph == im["asset"]
    wrong = free_loan(P, "tikuv_dastgohi", skip=2)                      # a cow uploaded for a sewing-machine loan
    r = api.check_photo(wrong, image(im["file"]), sandbox_id="S")["asset_match"]
    assert {k: r[k] for k in ("label", "score", "label_uz")} == {k: ph[k] for k in ("label", "score", "label_uz")}
    assert r["matches_purpose"] is False and ph["matches_purpose"] is True
    ev = [e for e in api.get_loan(wrong, "S")["evidence"] if e["kind"] == "photo" and e["data"].get("uploaded")]
    assert ev and all(e["data"]["asset_match"] == r for e in ev)


def test_asset_match_is_independent_of_sandbox_id_and_repeat(P):
    loan = free_loan(P, "asalarichilik")
    img = image("asalarichilik_03.jpg")
    a = api.check_photo(loan, img)["asset_match"]
    b = api.check_photo(loan, img, sandbox_id="Z")["asset_match"]
    c = api.check_photo(loan, img, sandbox_id="Z")["asset_match"]
    assert a == b == c and set(a) == KEYS


# ---------------------------------------------------------------- threshold semantics
def test_matches_purpose_iff_probability_of_purpose_ge_tau(am, probs):
    tau = json.loads(asset.PROBE_JSON.read_text(encoding="utf-8"))["tau"]
    assert am.tau == pytest.approx(tau)
    for f, (cls, p) in list(probs.items())[::3]:
        rgb = imageio.decode(image(f))
        for purpose in PURPOSES:
            got = am.asset_match(rgb, purpose)
            assert got["matches_purpose"] is bool(p[asset.CLASSES.index(purpose)] >= tau)
            assert got["label"] == asset.CLASSES[int(np.argmax(p))]
            assert got["score"] == pytest.approx(float(p.max()), abs=1e-4)


def test_probabilities_are_a_distribution(probs):
    for f, (cls, p) in probs.items():
        assert p.shape == (6,) and abs(p.sum() - 1) < 1e-9 and (p >= 0).all()


def test_matches_depends_on_the_loan_purpose_not_only_the_image(am):
    rgb = imageio.decode(image("chorva_01.jpg"))
    res = {p: am.asset_match(rgb, p)["matches_purpose"] for p in PURPOSES}
    assert res["chorva"] is True and sum(res.values()) == 1                 # exactly its own purpose


def test_unknown_purpose_raises_not_matches(am):
    with pytest.raises(ValueError):
        am.asset_match(imageio.decode(image("chorva_01.jpg")), "no_such_purpose")


# ---------------------------------------------------------------- per-purpose accept / reject
@pytest.mark.parametrize("purpose", PURPOSES)
def test_each_purpose_accepts_its_own_images_and_rejects_the_rest(purpose, probs, am):
    tau = am.tau
    k = asset.CLASSES.index(purpose)
    own = [p[k] >= tau for f, (c, p) in probs.items() if c == purpose]
    foreign = [p[k] >= tau for f, (c, p) in probs.items() if c not in (purpose, "other")]
    neg = [p[k] >= tau for f, (c, p) in probs.items() if c == "other"]
    assert len(own) >= 6 and len(foreign) >= 24 and len(neg) >= 6
    assert sum(own) / len(own) >= 0.75, (purpose, own)
    assert sum(foreign) / len(foreign) <= 0.15, (purpose, sum(foreign))
    assert sum(neg) / len(neg) <= 0.25, (purpose, sum(neg))


def test_the_five_purposes_are_distinguished_not_one_constant_label(probs):
    labels = {asset.CLASSES[int(np.argmax(p))] for f, (c, p) in probs.items()}
    assert labels >= set(PURPOSES) | {"other"}


# ---------------------------------------------------------------- labels
def test_label_uz_is_uzbek_and_never_a_raw_class_name(P):
    loan = free_loan(P, "chorva")
    seen = set()
    for r in manifest()[::4]:
        a = api.check_photo(loan, image(r["file"]))["asset_match"]
        assert isinstance(a["label_uz"], str) and a["label_uz"].strip()
        assert a["label_uz"] not in ("other", "tikuv_dastgohi", "chorva", "parranda", "asalarichilik")   # raw class names ('issiqxona' is itself an Uzbek word)
        assert a["label_uz"] in set(asset.LABEL_UZ.values()) | set(asset.OTHER_FINE)
        assert "_" not in a["label_uz"] and a["label_uz"] != "other"
        if a["label"] == "other":
            assert a["label_uz"] in asset.OTHER_FINE
        else:
            assert a["label_uz"] == asset.LABEL_UZ[a["label"]]
        seen.add(a["label_uz"])
    assert len(seen) >= 4


def test_label_tables_cover_all_classes():
    assert set(asset.LABEL_UZ) == set(asset.CLASSES) == set(asset.PROMPTS)
    assert all(v.strip() for v in asset.LABEL_UZ.values())


# ---------------------------------------------------------------- odd inputs
def _jpeg_arr(w, h, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    b = io.BytesIO()
    Image.fromarray(a).save(b, "JPEG")
    return b.getvalue()


ODD = {
    "1x1": lambda: png("RGB", (1, 1), (10, 200, 30)),
    "very_wide": lambda: _jpeg_arr(1600, 8),
    "very_tall": lambda: _jpeg_arr(8, 1600),
    "grayscale": lambda: png("L", (120, 90), 128),
    "palette": lambda: png("P", (64, 64), 3),
    "rgba": lambda: png("RGBA", (80, 60), (255, 0, 0, 90)),
    "1600px": lambda: _jpeg_arr(1600, 1200),
    "1bit": lambda: png("1", (50, 50), 1),
}


@pytest.mark.parametrize("name", list(ODD))
def test_odd_images_do_not_crash_and_give_a_valid_asset_match(name, P):
    loan = free_loan(P, "issiqxona")
    r = api.check_photo(loan, ODD[name]())
    a = r["asset_match"]
    assert set(a) == KEYS and a["label"] in asset.CLASSES and 0 < a["score"] <= 1 and isinstance(a["matches_purpose"], bool)
    assert a["label_uz"].strip() and isinstance(r["blurred_image"], bytes)


def test_oversized_image_still_raises_clear_error_with_clip_on(P):
    with pytest.raises(api.InvalidImageError):
        api.check_photo(free_loan(P, "chorva"), b"\xff\xd8" + b"0" * (config.MAX_BYTES + 1))
    with pytest.raises(api.InvalidImageError):
        api.check_photo(free_loan(P, "chorva"), b"not an image")


def test_different_images_give_different_asset_outputs(P):
    loan = free_loan(P, "chorva")
    outs = {json.dumps(api.check_photo(loan, image(f))["asset_match"], sort_keys=True)
            for f in ("chorva_00.jpg", "issiqxona_00.jpg", "parranda_00.jpg", "asalarichilik_00.jpg", "tikuv_dastgohi_00.jpg")}
    assert len(outs) == 5
    solid = [api.check_photo(loan, png("RGB", (100, 100), c))["asset_match"] for c in ((0, 0, 0), (255, 255, 255))]
    assert all(set(s) == KEYS for s in solid)


# ---------------------------------------------------------------- mutation safety
def test_mutating_a_returned_asset_match_does_not_corrupt_later_results(P):
    loan = free_loan(P, "chorva")
    img = image("chorva_00.jpg")
    first = api.check_photo(loan, img, sandbox_id="M")
    snap = copy.deepcopy(first["asset_match"])
    assert snap["matches_purpose"] is True
    first["asset_match"]["matches_purpose"] = False
    first["asset_match"]["label_uz"] = "HACKED"
    assert api.check_photo(loan, img)["asset_match"] == snap
    d = api.get_loan(loan, "M")      # the caller's mutation must not turn the sandbox's stored upload into a mismatch
    assert "PHOTO_ASSET_MISMATCH" not in [e["code"] for e in d["escalations"]]
    assert "HACKED" not in json.dumps(d, default=str)


def test_mutating_sandbox_view_does_not_change_the_sandbox_or_other_views(P):
    other = next(r["file"] for r in manifest() if r["purpose"] == "other")
    loan = free_loan(P, "tikuv_dastgohi")
    api.check_photo(loan, image(other), sandbox_id="A")
    d = api.get_loan(loan, "A")
    esc0 = copy.deepcopy(d["escalations"])
    for e in d["escalations"]:
        e["text_uz"] = "x"
        e["evidence_refs"].clear()
    for x in d["evidence"]:
        x["data"]["asset_match"] = None
    d["escalations"].clear()
    q = api.get_queue("A")
    for x in q:
        x["escalations"].clear()
    assert api.get_loan(loan, "A")["escalations"] == esc0
    assert any(x["loan_id"] == loan and x["escalations"] for x in api.get_queue("A"))
    ev = [e for e in api.get_loan(loan, "A")["evidence"] if e["kind"] == "photo"]
    assert ev and all(e["data"]["asset_match"] is not None for e in ev)


def test_sandbox_entries_hold_the_asset_but_it_is_not_shared_with_returned_dict(P):
    from ml.photo import checker
    loan = free_loan(P, "parranda")
    r = api.check_photo(loan, image("parranda_02.jpg"), sandbox_id="E")
    entry = checker._sandboxes["E"]["entries"][-1]
    assert entry["asset"] == r["asset_match"]
    r["asset_match"]["label_uz"] = "CHANGED"
    assert checker._sandboxes["E"]["entries"][-1]["asset"]["label_uz"] != "CHANGED"


# ---------------------------------------------------------------- concurrency with CLIP
def test_concurrent_clip_uploads_from_several_sandboxes_are_deterministic_and_isolated(P):
    files = ["chorva_00.jpg", "issiqxona_01.jpg", "parranda_01.jpg", "tikuv_dastgohi_02.jpg"]
    other = [r["file"] for r in manifest() if r["purpose"] == "other"][:4]
    jobs = []
    for k in range(8):
        f = (files + other)[k]
        purpose = f.rsplit("_", 1)[0] if k < 4 else "chorva"
        jobs.append((f"c{k}", free_loan(P, purpose, skip=k), f))
    seq = [api.check_photo(l, image(f))["asset_match"] for _, l, f in jobs]
    with ThreadPoolExecutor(max_workers=8) as ex:
        par = list(ex.map(lambda j: api.check_photo(j[1], image(j[2]), sandbox_id=j[0])["asset_match"], jobs))
    assert par == seq
    for sb, loan, f in jobs:
        view = {x["loan_id"]: x for x in api.get_queue(sb)}
        mism = [x["loan_id"] for x in view.values() if any(e["code"] == "PHOTO_ASSET_MISMATCH" for e in x["escalations"])]
        base = {x["loan_id"] for x in api.get_queue() if any(e["code"] == "PHOTO_ASSET_MISMATCH" for e in x["escalations"])}
        new = set(mism) - base
        assert new <= {loan}                                                # never another sandbox's loan


# ---------------------------------------------------------------- escalation merge
def test_asset_mismatch_merges_with_far_and_duplicate_into_documented_list(P):
    idx = get_index()
    other = next(r["file"] for r in manifest() if r["purpose"] == "other")
    loan = free_loan(P, "tikuv_dastgohi", skip=1)
    row = P.loans.loc[loan]
    lat, lon = float(row["borrower_lat"]) + 3.0, float(row["borrower_lon"])        # ~330 km away
    api.check_photo(loan, image(other), lat=lat, lon=lon, sandbox_id="F")
    d = api.get_loan(loan, "F")
    codes = [e["code"] for e in d["escalations"]]
    assert "PHOTO_ASSET_MISMATCH" in codes and "PHOTO_FAR" in codes and len(codes) == len(set(codes))
    assert [e["min_color"] for e in d["escalations"]] == sorted((e["min_color"] for e in d["escalations"]),
                                                                key=lambda c: -fusion.COLOR_RANK[c])
    assert d["color"] in ("yellow", "red") and fusion.COLOR_RANK[d["color"]] >= fusion.COLOR_RANK[d["base_color"]]
    # add a duplicate of a portfolio photo -> red
    dup_img = next(i for i in idx.images if len(i["loans"]) == 1)
    api.check_photo(loan, image(dup_img["file"]), sandbox_id="F")
    d2 = api.get_loan(loan, "F")
    assert d2["color"] == "red" and {"PHOTO_DUPLICATE", "PHOTO_FAR", "PHOTO_ASSET_MISMATCH"} <= {e["code"] for e in d2["escalations"]}
    assert d2["escalations"][0]["code"] == "PHOTO_DUPLICATE"                    # red first
    assert all(r["code"] in codes + ["PHOTO_DUPLICATE"] or r.get("weight") is not None for r in d2["reasons"])


def test_apply_policy_unit_asset_mismatch_never_lowers():
    sig = [dict(code="PHOTO_ASSET_MISMATCH", text_uz="t", evidence_ref="r")]
    for base in ("green", "yellow", "red"):
        out = fusion.apply_policy(base, sig)
        assert fusion.COLOR_RANK[out["color"]] >= fusion.COLOR_RANK[base]
        assert out["escalations"][0]["raises_color"] is (base == "green")
    dup = [dict(code="PHOTO_DUPLICATE", text_uz="a", evidence_ref="r1"), dict(code="PHOTO_ASSET_MISMATCH", text_uz="b", evidence_ref="r1"),
           dict(code="PHOTO_FAR", text_uz="c", evidence_ref="r1")]
    out = fusion.apply_policy("green", dup)
    assert out["color"] == "red" and [e["code"] for e in out["escalations"]] == ["PHOTO_DUPLICATE", "PHOTO_ASSET_MISMATCH", "PHOTO_FAR"]


# ---------------------------------------------------------------- warm_up
def test_warm_up_is_idempotent_and_does_not_reload_clip():
    first = api.warm_up()
    m = asset._model
    second = api.warm_up()
    assert asset._model is m and first["clip_loaded"] and second["clip_loaded"]
    assert second["clip_s"] <= max(1.0, first["clip_s"])
    assert asset.get_asset_model() is m
    assert set(second) == {"portfolio_s", "face_detector_s", "clip_s", "total_s", "clip_loaded", "clip_error"}


def test_get_asset_model_is_a_singleton_under_threads():
    with ThreadPoolExecutor(max_workers=8) as ex:
        ids = {id(m) for m in ex.map(lambda _: asset.get_asset_model(), range(16))}
    assert len(ids) == 1
