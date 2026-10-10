"""CLIP final fixes: (1) a failed CLIP load is cached, logged once, never downloads / retries / blocks - check_photo keeps working with
asset_match None; (2) one decision rule: matches_purpose = (label == purpose) and p >= tau, and the reason text never contradicts it.
Marked `clip` so that conftest does not replace the asset check with a stub."""
from __future__ import annotations

import csv
import inspect
import json
import logging
import time

import numpy as np
import pytest

from ml import api, fusion
from ml.money_trail.service import get_portfolio
from ml.photo import asset, checker, config, imageio
from ml.photo.index import get_index

pytestmark = pytest.mark.clip


@pytest.fixture(scope="module")
def P():
    return get_portfolio()


@pytest.fixture
def restore_asset_state():
    saved = (asset._model, asset._failed, checker._asset_warned)
    asset._reset_for_tests()
    checker._asset_warned = False
    yield
    asset._model, asset._failed, checker._asset_warned = saved
    api.reset_sandbox(None)


def image(name) -> bytes:
    return (config.ASSETS / name).read_bytes()


def a_loan(P, purpose="chorva"):
    idx = get_index()
    return next(l for l in P.loans.index if str(P.loans.at[l, "purpose"]) == purpose and l not in idx.by_loan)


# ---------------------------------------------------------------- (1) load failure
def test_load_failure_is_cached_logged_once_and_check_photo_keeps_working(P, monkeypatch, caplog, restore_asset_state):
    calls = []

    class Boom:
        def __init__(self, *a, **k):
            calls.append(1)
            raise OSError("simulated: weights missing")

    monkeypatch.setattr(asset, "AssetModel", Boom)
    caplog.set_level(logging.WARNING, logger="maqsad.asset")
    out = api.warm_up()
    assert out["clip_loaded"] is False and "simulated: weights missing" in out["clip_error"]
    loan = a_loan(P)
    times = []
    for k in range(6):
        t = time.perf_counter()
        r = api.check_photo(loan, image("chorva_00.jpg"), sandbox_id="F")
        times.append(time.perf_counter() - t)
        assert r["asset_match"] is None and r["faces_count"] >= 0 and r["blurred_image"][:2] == b"\xff\xd8"
    assert len(calls) == 1                                                       # never retried per upload
    assert max(times) < 1.5, times                                               # normal time, no blocking / timeout
    assert sum("asset check disabled" in rec.message for rec in caplog.records) == 1       # logged once
    esc = [e["code"] for x in api.get_queue("F") for e in x["escalations"]]
    assert "PHOTO_ASSET_MISMATCH" not in esc
    assert asset.load_error() and api.warm_up()["clip_error"] == out["clip_error"]
    assert len(calls) == 1


def test_missing_weights_never_trigger_a_download(P, monkeypatch, tmp_path, restore_asset_state):
    import open_clip

    def no_download(*a, **k):
        raise AssertionError("a download / load was attempted at runtime")

    monkeypatch.setattr(open_clip.openai, "load_openai_model", no_download)
    monkeypatch.setattr(asset, "weights_path", lambda: tmp_path / "ViT-B-32.pt")          # does not exist
    t = time.perf_counter()
    with pytest.raises(asset.AssetUnavailable, match="weights not found"):
        asset.get_asset_model()
    assert time.perf_counter() - t < 2
    r = api.check_photo(a_loan(P), image("chorva_00.jpg"))
    assert r["asset_match"] is None


def test_runtime_never_downloads_by_construction():
    sig = inspect.signature(asset.AssetModel.__init__)
    assert sig.parameters["allow_download"].default is False
    assert "HF_HUB_OFFLINE" in __import__("os").environ
    src = inspect.getsource(asset)
    assert src.count("allow_download=True") == 1 and "--download" in src               # only the CLI flag downloads


def test_asset_check_can_be_switched_off(P, monkeypatch, restore_asset_state):
    monkeypatch.setenv("MAQSAD_ASSET_CHECK", "off")
    out = api.warm_up()
    assert out["clip_loaded"] is False and "switched off" in out["clip_error"]
    t = time.perf_counter()
    assert api.check_photo(a_loan(P), image("chorva_00.jpg"))["asset_match"] is None
    assert time.perf_counter() - t < 1.5


def test_corrupt_probe_artifact_is_a_cached_failure_not_a_500(P, monkeypatch, tmp_path, restore_asset_state):
    bad = tmp_path / "probe.json"
    bad.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(asset, "PROBE_JSON", bad)
    out = api.warm_up()
    assert out["clip_loaded"] is False and "corrupt" in out["clip_error"]
    assert api.check_photo(a_loan(P), image("chorva_00.jpg"))["asset_match"] is None


# ---------------------------------------------------------------- (2) one consistent rule
@pytest.fixture(scope="module")
def pairs():
    am = asset.get_asset_model()
    rows = [r for r in csv.DictReader(open(config.ASSETS / "manifest.csv", encoding="utf-8")) if r["purpose"] in asset.CLASSES]
    embs = [am.embed(imageio.decode((config.ASSETS / r["file"]).read_bytes())) for r in rows]
    return am, [(r["file"], emb, p) for r, emb in zip(rows, embs) for p in asset.PURPOSES]


def test_label_matches_and_reason_text_never_contradict_over_all_eval_pairs(pairs):
    am, items = pairs
    assert len(items) == 143 * 5
    n_match = n_low = n_other = 0
    for f, emb, purpose in items:
        d = am.decide(emb, purpose)
        p_purpose = float(am.probe_probs(emb)[0][asset.CLASSES.index(purpose)])
        assert d["matches_purpose"] == (d["label"] == purpose and p_purpose >= am.tau), (f, purpose, d)
        sig = fusion.photo_signals(ref="photo:x", photo_label=f, duplicates=[], geo_ok=None, distance_km=None, time_ok=None,
                                   asset_match=d, uploaded=True, purpose=purpose)
        text = sig[0]["text_uz"] if sig else None
        if d["matches_purpose"]:
            n_match += 1
            assert d["label"] == purpose and sig == []                              # a match never produces a mismatch reason
        elif d["label"] == purpose:                                                  # right class, low confidence
            n_low += 1
            assert text and "mos aktiv aniqlanmadi" not in text and "ishonch past" in text and d["label_uz"] in text
        else:                                                                        # a different thing was found
            n_other += 1
            assert text == f"Rasmda kredit maqsadiga mos aktiv aniqlanmadi (aniqlangan: {d['label_uz']})"
            assert asset.LABEL_UZ[purpose] not in text                               # never names the loan's own asset as the finding
    assert n_match >= 90 and n_other > 400                                            # sanity: both branches were exercised
    print("matches", n_match, "low-confidence", n_low, "other", n_other)


def test_a_match_requires_the_most_likely_class_to_be_the_purpose(pairs):
    am, items = pairs
    for f, emb, purpose in items:
        p = am.probe_probs(emb)[0]
        d = am.decide(emb, purpose)
        if d["label"] != purpose:
            assert d["matches_purpose"] is False             # even if p(purpose) >= tau (the old rule would have said True)


def test_fusion_low_confidence_text_and_unchanged_default_text():
    low = dict(label="chorva", score=0.31, matches_purpose=False, label_uz="chorva (qoramol)")
    sig = fusion.photo_signals(ref="p", photo_label="x", duplicates=[], geo_ok=None, distance_km=None, time_ok=None,
                               asset_match=low, uploaded=True, purpose="chorva")
    assert sig[0]["code"] == "PHOTO_ASSET_MISMATCH" and "mos aktiv aniqlanmadi" not in sig[0]["text_uz"]
    assert sig[0]["text_uz"] == "Rasmdagi aktiv ishonchli aniqlanmadi (eng ehtimoliy: chorva (qoramol), ishonch past)"
    other = dict(label="other", score=0.9, matches_purpose=False, label_uz="avtomobil")
    sig = fusion.photo_signals(ref="p", photo_label="x", duplicates=[], geo_ok=None, distance_km=None, time_ok=None,
                               asset_match=other, uploaded=True, purpose="chorva")
    assert sig[0]["text_uz"] == "Rasmda kredit maqsadiga mos aktiv aniqlanmadi (aniqlangan: avtomobil)"
    # without a known purpose the default wording is kept (old callers)
    sig = fusion.photo_signals(ref="p", photo_label="x", duplicates=[], geo_ok=None, distance_km=None, time_ok=None,
                               asset_match=low, uploaded=True)
    assert "mos aktiv aniqlanmadi" in sig[0]["text_uz"]


def test_probe_meta_and_eval_agree_on_tau():
    meta = json.loads(asset.PROBE_JSON.read_text(encoding="utf-8"))
    ev = json.loads((config.ARTIFACTS / "eval_asset.json").read_text(encoding="utf-8"))
    assert meta["tau"] == ev["tau_final"] and 0.05 <= meta["tau"] <= 0.95


def test_sandbox_and_portfolio_reasons_use_the_rule_text(P, restore_asset_state):
    """End to end: a cow photographed for a sewing-machine loan names what was found; no 'matching asset' wording for a right-class photo."""
    loan = a_loan(P, "tikuv_dastgohi")
    r = api.check_photo(loan, image("chorva_03.jpg"), sandbox_id="R")
    assert r["asset_match"]["matches_purpose"] is False and r["asset_match"]["label"] != "tikuv_dastgohi"
    esc = [e for e in api.get_loan(loan, "R")["escalations"] if e["code"] == "PHOTO_ASSET_MISMATCH"]
    assert esc and esc[0]["text_uz"] == f"Rasmda kredit maqsadiga mos aktiv aniqlanmadi (aniqlangan: {r['asset_match']['label_uz']})"
    cattle = a_loan(P, "chorva")
    ok = api.check_photo(cattle, image("chorva_02.jpg"), sandbox_id="R")
    assert ok["asset_match"]["matches_purpose"] is True
    assert not [e for e in api.get_loan(cattle, "R")["escalations"] if e["code"] == "PHOTO_ASSET_MISMATCH"]
