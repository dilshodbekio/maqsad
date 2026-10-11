"""Evidence fusion: photo signals -> queue priority. A documented, configurable POLICY, not a learned model.

There is no labelled data for photo signals, so nothing is fitted: each photo signal escalates a loan to a minimum
colour and nothing else. Escalation only RAISES priority, it never lowers the money-trail colour, and the
money-trail `risk` value is never changed. `priority` = colour rank + risk (the queue sort key), so an escalated
loan moves up among the loans of its new colour, ordered by its money-trail risk.

The policy lives in the one dict below; change it there (and re-run `python -m ml.eval.build_metrics`, which
prints it into docs/metrics.md).
"""
from __future__ import annotations

from ml.photo import config as photo_config
from ml.photo.asset import label_i18n

LANGS = ("uz", "en", "ru")

COLOR_RANK = {"green": 0, "yellow": 1, "red": 2}
RANK_COLOR = {v: k for k, v in COLOR_RANK.items()}

POLICY = {
    "PHOTO_DUPLICATE": dict(min_color="red", active=True,
                            when="the photo duplicates another loan's photo (in the portfolio index, or in the same sandbox)"),
    "PHOTO_FAR": dict(min_color="yellow", active=True,
                      when=f"geo_ok == False: the photo was taken more than {photo_config.GEO_MAX_KM:g} km from the borrower (assumed limit)"),
    "PHOTO_TIME": dict(min_color="yellow", active=True,
                       when="time_ok == False: captured before disbursement or in the future"),
    "PHOTO_ASSET_MISMATCH": dict(min_color="yellow", active=True,
                                 when="asset_match.matches_purpose == False: the photo does not show the asset the loan was for "
                                      "(CLIP + linear probe, `ml/photo/asset.py`)"),
}


def priority(color: str, risk: float) -> float:
    """Queue sort key (descending): colour rank first, then the unchanged money-trail risk."""
    return COLOR_RANK[color] + float(risk)


_MORE = {"uz": " va yana {n} ta", "en": " and {n} more", "ru": " и ещё {n}"}


def _ids(loan_ids: list[str], lang: str = "uz") -> str:
    shown = ", ".join(loan_ids[:3])
    return shown + (_MORE[lang].format(n=len(loan_ids) - 3) if len(loan_ids) > 3 else "")


def _texts(code: str, *, uploaded: bool, duplicates: list, distance_km, asset_match, purpose) -> dict:
    """The sentence of one triggered policy signal in uz / en / ru (same numbers in every language)."""
    max_km = f"{photo_config.GEO_MAX_KM:g}"
    ids = [d["loan_id"] for d in duplicates]
    if code == "PHOTO_DUPLICATE":
        return {
            "uz": f"{'Yuklangan rasm' if uploaded else 'Bu kreditdagi rasm'} {_ids(ids, 'uz')} kreditidagi rasm bilan bir xil",
            "en": f"{'The uploaded photo' if uploaded else 'The photo on this loan'} is identical to the photo of loan {_ids(ids, 'en')}",
            "ru": f"{'Загруженное фото' if uploaded else 'Фото по этому кредиту'} совпадает с фото по кредиту {_ids(ids, 'ru')}",
        }
    if code == "PHOTO_FAR":
        known = distance_km is not None
        km = f"{distance_km:.0f}" if known else ""
        return {
            "uz": f"Rasm olingan joy qarz oluvchining manzilidan {km + ' km uzoqda' if known else 'uzoqda'} (ruxsat etilgan masofa: {max_km} km)",
            "en": f"The place where the photo was taken is {km + ' km away from' if known else 'far from'} the borrower's address (allowed distance: {max_km} km)",
            "ru": f"Место съёмки фото находится {'в ' + km + ' км от' if known else 'далеко от'} адреса заёмщика (допустимое расстояние: {max_km} км)",
        }
    if code == "PHOTO_ASSET_MISMATCH":
        lab = asset_match.get("label_i18n") or label_i18n(asset_match["label_uz"])
        if purpose is not None and asset_match.get("label") == purpose:
            # the most likely class IS the loan's asset, only with low confidence: never say "no matching asset"
            return {
                "uz": f"Rasmdagi aktiv ishonchli aniqlanmadi (eng ehtimoliy: {lab['uz']}, ishonch past)",
                "en": f"The asset in the photo could not be identified reliably (most likely: {lab['en']}, low confidence)",
                "ru": f"Актив на фото не удалось определить надёжно (наиболее вероятно: {lab['ru']}, низкая уверенность)",
            }
        return {
            "uz": f"Rasmda kredit maqsadiga mos aktiv aniqlanmadi (aniqlangan: {lab['uz']})",
            "en": f"No asset matching the loan purpose was found in the photo (identified: {lab['en']})",
            "ru": f"На фото не обнаружен актив, соответствующий цели кредита (определено: {lab['ru']})",
        }
    return {  # PHOTO_TIME
        "uz": "Rasm olingan vaqt ishonchli emas (kredit berilishidan oldin yoki kelajakda ko'rsatilgan)",
        "en": "The time the photo was taken is not reliable (before disbursement or in the future)",
        "ru": "Время съёмки фото недостоверно (до выдачи кредита или в будущем)",
    }


def photo_signals(*, ref: str, photo_label: str, duplicates: list[dict], geo_ok, distance_km, time_ok,
                  asset_match=None, uploaded: bool, purpose: str | None = None) -> list[dict]:
    """Triggered policy signals of ONE photo: [{code, text_uz, text: {uz, en, ru}, evidence_ref, evidence}] (evidence = kind 'photo' item)."""
    out = []
    data = dict(photo=photo_label, uploaded=uploaded, duplicate_of=duplicates, geo_ok=geo_ok,
                distance_km=None if distance_km is None else round(float(distance_km), 1), time_ok=time_ok,
                asset_match=dict(asset_match) if asset_match else None)

    def add(code):
        if POLICY[code]["active"]:
            text = _texts(code, uploaded=uploaded, duplicates=duplicates, distance_km=distance_km, asset_match=asset_match, purpose=purpose)
            out.append(dict(code=code, text_uz=text["uz"], text=text, evidence_ref=ref,
                            evidence=dict(ref=ref, kind="photo", summary_uz=text["uz"], summary=text, data=data)))

    if duplicates:
        add("PHOTO_DUPLICATE")
    if geo_ok is False:
        add("PHOTO_FAR")
    if asset_match is not None and asset_match.get("matches_purpose") is False:
        add("PHOTO_ASSET_MISMATCH")
    if time_ok is False:
        add("PHOTO_TIME")
    return out


def apply_policy(base_color: str, signals: list[dict]) -> dict:
    """{color, escalations[]}: the final colour is never below `base_color`. One escalation (one reason) per policy
    code; if several photos trigger the same code (a portfolio photo and a sandbox upload) their texts are joined and
    ALL their evidence refs are kept."""
    final, by_code = COLOR_RANK[base_color], {}
    for s in signals:
        by_code.setdefault(s["code"], []).append(s)
    esc = []
    for code, group in by_code.items():
        minimum = COLOR_RANK[POLICY[code]["min_color"]]
        text = {}
        for lang in LANGS:
            texts = [(g.get("text") or {}).get(lang) or g["text_uz"] for g in group]     # a signal without translations: its Uzbek text
            texts = list(dict.fromkeys(texts))        # the same sentence from several photos is said once (order kept)
            text[lang] = "; ".join([texts[0]] + [t[:1].lower() + t[1:] for t in texts[1:]])
        esc.append(dict(code=code, min_color=POLICY[code]["min_color"], text_uz=text["uz"], text=text,
                        evidence_refs=list(dict.fromkeys(g["evidence_ref"] for g in group)),
                        raises_color=minimum > COLOR_RANK[base_color]))
        final = max(final, minimum)
    esc.sort(key=lambda e: (-COLOR_RANK[e["min_color"]], e["code"]))
    return dict(color=RANK_COLOR[final], escalations=esc)


def order_reasons(policy_reasons: list[dict], model_reasons: list[dict], escalations: list[dict]) -> list[dict]:
    """The one ordering of a loan's reasons (used by `get_loan().reasons` and by the queue's `top_reason`):
    policy reasons that RAISED the colour first, then the money-trail reasons, then the policy reasons that did not
    raise it (the colour was already that high). `policy_reasons[i]` belongs to `escalations[i]`."""
    raised = [r for r, e in zip(policy_reasons, escalations) if e["raises_color"]]
    rest = [r for r, e in zip(policy_reasons, escalations) if not e["raises_color"]]
    return raised + list(model_reasons) + rest


def policy_table() -> list[dict]:
    return [dict(code=c, **v) for c, v in POLICY.items()]
