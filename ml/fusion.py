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

COLOR_RANK = {"green": 0, "yellow": 1, "red": 2}
RANK_COLOR = {v: k for k, v in COLOR_RANK.items()}

POLICY = {
    "PHOTO_DUPLICATE": dict(min_color="red", active=True,
                            when="the photo duplicates another loan's photo (in the portfolio index, or in the same sandbox)"),
    "PHOTO_FAR": dict(min_color="yellow", active=True,
                      when=f"geo_ok == False: the photo was taken more than {photo_config.GEO_MAX_KM:g} km from the borrower (assumed limit)"),
    "PHOTO_TIME": dict(min_color="yellow", active=True,
                       when="time_ok == False: captured before disbursement or in the future"),
    "PHOTO_ASSET_MISMATCH": dict(min_color="yellow", active=False,
                                 when="asset_match below a threshold (switched on when the CLIP step is added)"),
}


def priority(color: str, risk: float) -> float:
    """Queue sort key (descending): colour rank first, then the unchanged money-trail risk."""
    return COLOR_RANK[color] + float(risk)


def _ids(loan_ids: list[str]) -> str:
    shown = ", ".join(loan_ids[:3])
    return shown + (f" va yana {len(loan_ids) - 3} ta" if len(loan_ids) > 3 else "")


def photo_signals(*, ref: str, photo_label: str, duplicates: list[dict], geo_ok, distance_km, time_ok,
                  asset_match=None, uploaded: bool) -> list[dict]:
    """Triggered policy signals of ONE photo: [{code, text_uz, evidence_ref, evidence}] (evidence = kind 'photo' item)."""
    subject = "Yuklangan rasm" if uploaded else "Bu kreditdagi rasm"
    out = []
    data = dict(photo=photo_label, uploaded=uploaded, duplicate_of=duplicates, geo_ok=geo_ok,
                distance_km=None if distance_km is None else round(float(distance_km), 1), time_ok=time_ok)

    def add(code, text):
        if POLICY[code]["active"]:
            out.append(dict(code=code, text_uz=text, evidence_ref=ref,
                            evidence=dict(ref=ref, kind="photo", summary_uz=text, data=data)))

    if duplicates:
        add("PHOTO_DUPLICATE", f"{subject} {_ids([d['loan_id'] for d in duplicates])} kreditidagi rasm bilan bir xil")
    if geo_ok is False:
        km = f"{distance_km:.0f} km uzoqda" if distance_km is not None else "uzoqda"
        add("PHOTO_FAR", f"Rasm olingan joy qarz oluvchining manzilidan {km} (ruxsat etilgan masofa: {photo_config.GEO_MAX_KM:g} km)")
    if time_ok is False:
        add("PHOTO_TIME", "Rasm olingan vaqt ishonchli emas (kredit berilishidan oldin yoki kelajakda ko'rsatilgan)")
    return out


def apply_policy(base_color: str, signals: list[dict]) -> dict:
    """{color, escalations[]}: the final colour is never below `base_color`; one escalation per policy code."""
    final, seen, esc = COLOR_RANK[base_color], set(), []
    for s in signals:
        code = s["code"]
        if code in seen:
            continue
        seen.add(code)
        minimum = COLOR_RANK[POLICY[code]["min_color"]]
        esc.append(dict(code=code, min_color=POLICY[code]["min_color"], text_uz=s["text_uz"],
                        evidence_refs=[s["evidence_ref"]], raises_color=minimum > COLOR_RANK[base_color]))
        final = max(final, minimum)
    esc.sort(key=lambda e: (-COLOR_RANK[e["min_color"]], e["code"]))
    return dict(color=RANK_COLOR[final], escalations=esc)


def policy_table() -> list[dict]:
    return [dict(code=c, **v) for c, v in POLICY.items()]
