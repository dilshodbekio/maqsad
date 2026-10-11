"""uz / en / ru versions of every ML text: same facts, same numbers, no untranslated or broken strings."""
import itertools
import re

import pytest

from ml import api, fusion
from ml.errors import ScenarioError
from ml.eval.audit_texts import _numbers, flags_lang
from ml.money_trail import reasons as R
from ml.money_trail.features import NEVER
from ml.photo import asset

LANGS = ("uz", "en", "ru")
CYR = re.compile(r"[А-Яа-яЁё]")
VALUES = [-1.0, 0.0, 0.003, 0.2, 0.5, 0.9, 1.0, 1.4, 3.0, 12.0, 70.0, 140.0, 500.0, float(NEVER)]
CTX = {"supplier_oked": "47.19"}
NEG_SENTINEL = {"big_inbound_day", "income_rate_per_month", "sup_active_span_days", "sup_age_at_first_payment_days", "invoice_gap"}   # features that use -1


def trio_ok(t, kind="reason"):
    assert isinstance(t, dict) and set(t) == set(LANGS), t
    assert all(isinstance(t[l], str) and t[l].strip() for l in LANGS), t
    assert _numbers(t["uz"]) == _numbers(t["en"]) == _numbers(t["ru"]), t
    assert not CYR.search(t["uz"]) and not CYR.search(t["en"]) and CYR.search(t["ru"]), t
    for lang in ("en", "ru"):
        assert flags_lang(t[lang], kind, lang) == [], (lang, t[lang], flags_lang(t[lang], kind, lang))


@pytest.mark.parametrize("code", sorted(R.TEMPLATES))
def test_every_template_has_en_and_ru_with_the_same_numbers(code):
    for v in VALUES:
        if R.is_skipped(code, v) or (v < 0 and code not in NEG_SENTINEL):
            continue                                  # a value that renders as zero is never shown as a reason
        trio_ok(R.render(code, v, CTX))


def test_reason_and_evidence_texts_do_not_accuse():
    bad = re.compile(r"guilty|fraud|criminal|crime|виновн|мошенн|преступ|aybdor|firibgar|jinoyat", re.I)
    for code in R.TEMPLATES:
        for v in VALUES:
            t = R.render(code, v, CTX)
            assert not any(bad.search(t[l]) for l in LANGS), t


def test_share_above_100_percent_is_not_shown_as_a_percentage_in_any_language():
    t = R.render("cash_share_d1", 1.4, CTX)
    assert "100 foizdan ko'p" in t["uz"] and "more than 100%" in t["en"].lower() and "более 100%" in t["ru"].lower()
    assert t["en"][0].isupper() and t["ru"][0].isupper()


def test_queue_loan_and_scenario_texts_are_trilingual_and_consistent():
    q = api.get_queue()
    assert all(it["top_reason_text"]["uz"] == it["top_reason"] for it in q[::25])
    for it in q[:60:3]:
        trio_ok(it["top_reason_text"])
        d = api.get_loan(it["loan_id"])
        assert d["reasons"][0]["text"] == it["top_reason_text"]              # top_reason == reasons[0], in every language
        for r in d["reasons"]:
            trio_ok(r["text"])
            assert r["text"]["uz"] == r["text_uz"]
        for e in d["evidence"]:
            trio_ok(e["summary"], "message")
            assert e["summary"]["uz"] == e["summary_uz"]
        for e in d["escalations"]:
            trio_ok(e["text"])
    for p in api.get_scenario_presets():
        trio_ok(p["title"], "message")
        trio_ok(p["description"], "message")
        assert p["title"]["uz"] == p["title_uz"] and p["description"]["uz"] == p["description_uz"]
        for x in api.score_scenario(p["scenario"])["reasons"]:
            trio_ok(x["text"])


def test_photo_escalations_are_trilingual_and_joined_per_language():
    am = dict(label="other", score=0.9, matches_purpose=False, label_uz="avtomobil", label_i18n=asset.label_i18n("avtomobil"))
    sig = fusion.photo_signals(ref="photo:x", photo_label="x", duplicates=[dict(loan_id=f"L150000{i}", similarity=1.0) for i in range(5)],
                               geo_ok=False, distance_km=123.4, time_ok=False, asset_match=am, uploaded=True, purpose="chorva")
    assert [s["code"] for s in sig] == ["PHOTO_DUPLICATE", "PHOTO_FAR", "PHOTO_ASSET_MISMATCH", "PHOTO_TIME"]
    for s in sig:
        trio_ok(s["text"])
        assert s["evidence"]["summary"] == s["text"]
    dup = sig[0]["text"]
    assert "va yana 2 ta" in dup["uz"] and "and 2 more" in dup["en"] and "и ещё 2" in dup["ru"]
    assert "123 km" in sig[1]["text"]["uz"] and "123 km" in sig[1]["text"]["en"] and "123 км" in sig[1]["text"]["ru"]
    assert "avtomobil" in sig[2]["text"]["uz"] and "car" in sig[2]["text"]["en"] and "автомобиль" in sig[2]["text"]["ru"]
    # the low-confidence wording when the most likely class is the loan's own asset
    am2 = dict(label="chorva", score=0.4, matches_purpose=False, label_uz="chorva (qoramol)", label_i18n=asset.label_i18n("chorva (qoramol)"))
    low = fusion.photo_signals(ref="p", photo_label="x", duplicates=[], geo_ok=None, distance_km=None, time_ok=None, asset_match=am2,
                               uploaded=False, purpose="chorva")[0]["text"]
    trio_ok(low)
    assert "ishonch past" in low["uz"] and "low confidence" in low["en"] and "низкая уверенность" in low["ru"]
    # two photos with the same code -> one escalation, texts joined in every language
    other = fusion.photo_signals(ref="photo:y", photo_label="y", duplicates=[], geo_ok=False, distance_km=55.0, time_ok=None, asset_match=None,
                                 uploaded=True, purpose="chorva")[0]
    e = fusion.apply_policy("green", [sig[1], other])["escalations"]
    assert len(e) == 1 and all(e[0]["text"][l].count("; ") == 1 for l in LANGS) and e[0]["text_uz"] == e[0]["text"]["uz"]


def test_asset_labels_exist_in_three_languages_for_every_class_and_fine_name():
    for label, uz in asset.LABEL_UZ.items():
        t = asset.label_i18n(uz)
        assert t["uz"] == uz and t["en"] != uz or label == "issiqxona"
        assert set(t) == set(LANGS) and CYR.search(t["ru"]) and not CYR.search(t["en"])
    for uz in asset.OTHER_FINE:
        t = asset.label_i18n(uz)
        assert set(t) == set(LANGS) and t["uz"] == uz and CYR.search(t["ru"]) and not CYR.search(t["en"]), uz
    assert asset.label_i18n("noma'lum")["en"] == "noma'lum"             # an unknown label is passed through, never lost


def test_scenario_errors_carry_three_languages_with_the_same_numbers():
    good = api.get_scenario_presets()[1]["scenario"]
    bad = [dict(good, snapshot_day=1000), dict(good, amount_uzs=0), dict(good, purpose="x"), dict(good, foo=1), dict(good, region="Mars"),
           dict(good, supplier=dict(mode="x")), dict(good, supplier=dict(mode="new", oked="abc", registered_days_before=1)),
           dict(good, transactions=[dict(day=1, type="teleport", amount_uzs=1)]),
           dict(good, transactions=[dict(day=1, type="cash_withdrawal", amount_uzs=10 ** 9)]),
           dict(good, transactions=[dict(day=1, type="cash_withdrawal", amount_uzs=40_000_000), dict(day=2, type="supplier_payment", amount_uzs=20_000_000)]),
           dict(good, transactions="none"), dict(good, invoice_amount_uzs=10 ** 10), "not a dict"]
    for scn in bad:
        with pytest.raises(ScenarioError) as e:
            api.score_scenario(scn)
        m = e.value.messages
        assert set(m) == set(LANGS) and str(e.value) == m["uz"]
        assert _numbers(m["uz"]) == _numbers(m["en"]) == _numbers(m["ru"]), m
        assert m["en"] != m["uz"] and m["ru"] != m["uz"] and CYR.search(m["ru"]) and not CYR.search(m["en"]), m


def test_invalid_sandbox_id_error_is_translated():
    from ml.errors import InvalidRequestError
    for bad in ("", "a" * 500, "bad id!", 5):
        with pytest.raises(InvalidRequestError) as e:
            api.check_photo("L1500001", b"x", sandbox_id=bad)
        assert set(e.value.messages) == set(LANGS) and CYR.search(e.value.messages["ru"]) and e.value.messages["en"] != e.value.messages["uz"]


def test_audit_checker_catches_what_it_is_meant_to_catch():
    assert "untranslated Uzbek: kredit" in flags_lang("The kredit was issued", "reason", "en")
    assert flags_lang("Loan 0% of the funds", "reason", "en") == ["zero value"]
    assert flags_lang("Some None value here", "reason", "ru")[0] == "None / nan"
    assert flags_lang("Cyrillic: Привет there friends", "reason", "en") == ["Cyrillic in English text"]
    assert flags_lang("", "reason", "ru") == ["empty translation"]
    assert flags_lang("The supplier Dala Ta'minot MChJ got money", "reason", "en") == []        # a company name is data, not prose
    assert _numbers("3 301 000 UZS, 2026-01-03, 6.8") == _numbers("3 301 000 сум, 2026-01-03, 6.8") != _numbers("3 301 001 сум, 2026-01-03, 6.8")


def test_same_sentence_from_several_photos_is_said_once_in_every_language():
    am = dict(label="other", score=0.9, matches_purpose=False, label_uz="avtomobil", label_i18n=asset.label_i18n("avtomobil"))
    one = lambda ref, ids: fusion.photo_signals(ref=ref, photo_label=ref, duplicates=[dict(loan_id=i, similarity=1.0) for i in ids], geo_ok=None,
                                                distance_km=None, time_ok=None, asset_match=None, uploaded=True, purpose="chorva")
    mism = lambda ref, a: fusion.photo_signals(ref=ref, photo_label=ref, duplicates=[], geo_ok=None, distance_km=None, time_ok=None, asset_match=a,
                                               uploaded=True, purpose="chorva")
    am2 = dict(am, label_uz="bino", label_i18n=asset.label_i18n("bino"))
    sigs = one("photo:a", ["L1500001"]) + one("photo:b", ["L1500001"]) + one("photo:c", ["L1500002"]) + mism("photo:d", am) + mism("photo:e", am) + mism("photo:f", am2)
    esc = {e["code"]: e for e in fusion.apply_policy("green", sigs)["escalations"]}
    for lang in LANGS:
        dup = esc["PHOTO_DUPLICATE"]["text"][lang].split("; ")
        assert len(dup) == len(set(dup)) == 2, dup                    # L1500001 (twice -> once) and L1500002
        mm = esc["PHOTO_ASSET_MISMATCH"]["text"][lang].split("; ")
        assert len(mm) == len(set(mm)) == 2, mm                       # car (twice -> once) and building
    assert esc["PHOTO_DUPLICATE"]["text_uz"] == esc["PHOTO_DUPLICATE"]["text"]["uz"]
    assert esc["PHOTO_DUPLICATE"]["evidence_refs"] == ["photo:a", "photo:b", "photo:c"]        # evidence items are all kept
    assert esc["PHOTO_ASSET_MISMATCH"]["evidence_refs"] == ["photo:d", "photo:e", "photo:f"]
