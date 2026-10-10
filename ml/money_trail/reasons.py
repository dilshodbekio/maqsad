"""Reasons (top risk-raising features -> plain Uzbek) and the evidence they point to.

All user-facing wording lives in TEMPLATES (one entry per feature code) so it can be reviewed in one place.
Texts state facts about the money trail ("mablag'ning 78 foizi naqd yechilgan"); they never say a person is guilty.
Weights: share of each reason in the total risk-raising push of the loan (LightGBM pred_contrib), 0..1.
Reasons that repeat the same fact (e.g. several income or cash-share features) are collapsed to the strongest one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import NEVER

TOP_K = 3
MAX_REFS = 8
NO_REASON_UZ = "Sezilarli xavf belgisi aniqlanmadi"

# "atigi" (only) is used only below these values; above them the value is stated neutrally
LOW_PAY_SHARE, LOW_SUPPLIER_AGE_DAYS, LOW_ACTIVE_SPAN_DAYS = 0.25, 60, 100
SHARE_CAP = 1.5          # sup_cash_share / sup_exit_share are capped at 1.5 in the features


def _pct(v, suffix="") -> str:
    """'78 foiz' + Uzbek suffix ('i', 'ini', 'iga'). Values above 100% are shown as they are; the cap becomes
    '150 foiz va undan ko'p'."""
    v = max(float(v), 0.0)
    if v >= SHARE_CAP:
        return f"{SHARE_CAP * 100:.0f} foiz va undan ko'p{suffix}"
    return f"{v * 100:.0f} foiz{suffix}"


def _d(v) -> str:
    return f"{v:.0f}" if abs(v) >= 10 else f"{v:.1f}"


def _som(v) -> str:
    return f"{int(round(v)):,}".replace(",", " ") + " so'm"


# feature code -> template(value, context) -> Uzbek (Latin) sentence with the actual value
TEMPLATES = {
    "first_cash_days": lambda v, c: ("Kredit tushgandan beri naqd yechish qayd etilmagan" if v >= NEVER else
                                     f"Birinchi naqd yechish kredit tushgandan {_d(v)} kun o'tib bo'lgan"),
    "cash_share_d1": lambda v, c: f"Kredit tushgandan keyingi 24 soat ichida mablag'ning {_pct(v, 'i')} naqd yechilgan",
    "cash_share_d7": lambda v, c: f"Kredit tushgandan keyin 7 kun ichida mablag'ning {_pct(v, 'i')} naqd yechilgan",
    "cash_share_d30": lambda v, c: f"Kredit tushgandan keyin 30 kun ichida mablag'ning {_pct(v, 'i')} naqd yechilgan",
    "cash_share_all": lambda v, c: f"Kredit tushganidan beri mablag'ning {_pct(v, 'i')} naqd yechilgan",
    "sup_pay_share": lambda v, c: ("Ta'minotchiga to'lov o'tkazilmagan" if v <= 0 else
                                   f"Ta'minotchiga kredit summasining atigi {_pct(v, 'i')} o'tkazilgan" if v < LOW_PAY_SHARE else
                                   f"Ta'minotchiga kredit summasining {_pct(v, 'i')} o'tkazilgan"),
    "first_sup_pay_days": lambda v, c: ("Ta'minotchiga hali to'lov o'tkazilmagan" if v >= NEVER else
                                        f"Ta'minotchiga birinchi to'lov kredit tushgandan {_d(v)} kun o'tib qilingan"),
    "invoice_gap": lambda v, c: ("Hisob-faktura mavjud emas" if v < 0 else
                                 f"Hisob-fakturadagi summaning {_pct(v, 'i')} ta'minotchiga to'lanmagan"),
    "leave_share_30": lambda v, c: ("Dastlabki 30 kunda mablag'ning "
                                    f"{_pct(v, 'i')} bankdan chiqarilgan (naqd, boshqa hisob yoki shaxs)"),
    "p2p_out_count": lambda v, c: f"Shaxslarga {v:.0f} ta o'tkazma qilingan",
    "p2p_in_count": lambda v, c: f"Shaxslardan {v:.0f} ta o'tkazma qabul qilingan",
    "big_inbound_share": lambda v, c: f"Hisobga kredit summasining {_pct(v, 'iga')} teng bitta yirik kirim tushgan",
    "big_inbound_day": lambda v, c: ("Yirik kirim qayd etilmagan" if v < 0 else
                                     f"Eng yirik kirim kredit tushgandan {_d(v)} kun o'tib kelgan"),
    "big_inbound_person": lambda v, c: ("Eng yirik kirim jismoniy shaxsdan kelgan" if v >= 0.5 else
                                        "Eng yirik kirim tashkilotdan kelgan"),
    "post_inbound_exit": lambda v, c: f"Yirik kirimdan keyingi 5 kunda uning {_pct(v, 'i')} hisobdan chiqarilgan",
    "income_rate_per_month": lambda v, c: ("Kredit berilganidan beri tushum qayd etilmagan" if v <= 0 else
                                           f"Kredit davrida oyiga o'rtacha {v:.1f} ta tushum qayd etilgan"),
    "income_missing_flag": lambda v, c: "Kredit 90 kundan ko'proq oldin berilgan, lekin hali birorta tushum qayd etilmagan",
    "income_after_big_count": lambda v, c: ("Yirik kirimdan keyin tushum qayd etilmagan" if v <= 0 else
                                            f"Yirik kirimdan keyin {v:.0f} ta tushum qayd etilgan"),
    "income_gap_days": lambda v, c: f"Oxirgi tushumdan (yoki tushum kutilgan sanadan) beri {_d(v)} kun o'tgan",
    "income_stopped_flag": lambda v, c: "Yirik kirimdan keyin tushumlar to'xtagan",
    "sup_in_degree": lambda v, c: f"Shu ta'minotchi {v:.0f} ta kreditning hisob-fakturasida ko'rsatilgan",
    "sup_cash_share": lambda v, c: f"Ta'minotchi olgan mablag'ning {_pct(v, 'ini')} naqd yechgan",
    "sup_exit_share": lambda v, c: f"Ta'minotchi olgan mablag'ning {_pct(v, 'ini')} hisobidan chiqarib yuborgan",
    "sup_cash_lag_days": lambda v, c: ("Ta'minotchi naqd yechmagan" if v >= NEVER else
                                       f"Ta'minotchi pulni olgandan {_d(v)} kun o'tib naqd yechgan"),
    "sup_active_span_days": lambda v, c: ("Ta'minotchi hisobida hali harakat qayd etilmagan" if v < 0 else
                                          f"Ta'minotchi hisobi atigi {_d(v)} kun faol bo'lgan" if v < LOW_ACTIVE_SPAN_DAYS else
                                          f"Ta'minotchi hisobi {_d(v)} kun faol bo'lgan"),
    "sup_age_at_first_payment_days": lambda v, c: (
        "Ta'minotchiga hali to'lov tushmagan" if v < 0 else
        f"Ta'minotchi birinchi to'lovdan atigi {_d(v)} kun oldin ro'yxatdan o'tgan" if v < LOW_SUPPLIER_AGE_DAYS else
        f"Ta'minotchi birinchi to'lovdan {_d(v)} kun oldin ro'yxatdan o'tgan"),
    "sup_employee_link": lambda v, c: "Ta'minotchi bank xodimi bilan bog'langan",
    "emp_branch_match": lambda v, c: "Ta'minotchi bilan bog'langan xodim kredit berilgan filialda ishlaydi",
    "oked_mismatch": lambda v, c: (f"Ta'minotchining faoliyat turi (OKED {c['supplier_oked']}) "
                                   "kredit maqsadiga mos kelmaydi"),
}

# features that state (nearly) the same fact: only the strongest of a group is shown as a reason
DUPLICATE_GROUPS = [
    {"income_rate_per_month", "income_missing_flag", "income_after_big_count", "income_gap_days", "income_stopped_flag"},
    {"cash_share_d1", "cash_share_d7", "cash_share_d30", "cash_share_all", "first_cash_days"},
    {"sup_pay_share", "first_sup_pay_days", "invoice_gap"},
]
_GROUP_OF = {code: i for i, g in enumerate(DUPLICATE_GROUPS) for code in g}


TXN_TYPE_UZ = {
    "disbursement": "kredit tushgan", "cash_withdrawal": "naqd yechilgan",
    "supplier_payment": "ta'minotchiga o'tkazilgan", "p2p_out": "shaxsga o'tkazilgan",
    "p2p_in": "shaxsdan qabul qilingan", "external_transfer": "boshqa bankka o'tkazilgan",
    "income": "tushum qayd etilgan",
}


def _refs(code: str, c: dict) -> list[str]:
    """Evidence refs (transaction ids / invoice / graph) a feature is computed from."""
    t = c["txs"]
    typ = t["type"]
    ids = lambda m: list(t.loc[m, "txn_id"])
    sup = f"graph:{c['supplier_id']}"
    inv = f"invoice:{c['loan_id']}"
    cash = typ == "cash_withdrawal"
    if code == "first_cash_days":
        return ids(cash)[:1]
    if code in ("cash_share_d1", "cash_share_d7", "cash_share_d30"):
        return ids(cash & (t["day"] <= int(code.rsplit("_d", 1)[1])))
    if code == "cash_share_all":
        return ids(cash)
    if code in ("sup_pay_share", "first_sup_pay_days", "invoice_gap"):
        return ids(typ == "supplier_payment") + [inv]
    if code == "leave_share_30":
        return ids(typ.isin(["cash_withdrawal", "external_transfer", "p2p_out"]) & (t["day"] <= 30))
    if code in ("p2p_out_count", "p2p_in_count"):
        return ids(typ == code.replace("_count", ""))
    big = c.get("big_inbound_txn")
    if code in ("big_inbound_share", "big_inbound_day", "big_inbound_person"):
        return [big] if big else []
    if code == "post_inbound_exit":
        if not big:
            return []
        d0 = float(t.loc[t.txn_id == big, "day"].iloc[0])
        return [big] + ids(typ.isin(["cash_withdrawal", "external_transfer"]) & (t["day"] >= d0) & (t["day"] <= d0 + 5))
    if code in ("income_rate_per_month", "income_missing_flag", "income_after_big_count", "income_gap_days",
                "income_stopped_flag"):
        inc = t[typ == "income"].sort_values("ts", ascending=False)
        return list(inc["txn_id"])[:5] + ([big] if big and code in ("income_after_big_count", "income_stopped_flag") else [])
    if code == "oked_mismatch":
        return [inv]
    return [sup]          # supplier graph / employee link features


def build_reasons(model_features: list[str], x: np.ndarray, contrib: np.ndarray, c: dict,
                  with_refs: bool = True) -> list[dict]:
    """Top-K features pushing the risk UP -> [{code, text_uz, weight, evidence_refs}]. May be empty.
    `with_refs=False` skips the evidence lookup (then `c` only needs `supplier_oked`)."""
    pos = np.where(contrib > 1e-9)[0]
    if len(pos) == 0:
        return []
    total = float(contrib[pos].sum())
    out, used = [], set()
    for i in pos[np.argsort(-contrib[pos], kind="stable")]:
        code = model_features[i]
        group = _GROUP_OF.get(code)
        if group is not None and group in used:
            continue                                  # same fact already shown by a stronger feature
        used.add(group)
        item = dict(code=code, text_uz=TEMPLATES[code](float(x[i]), c),
                    weight=round(float(contrib[i]) / total, 3))
        if with_refs:
            item["evidence_refs"] = _refs(code, c)[:MAX_REFS]
        out.append(item)
        if len(out) == TOP_K:
            break
    return out


def build_evidence(refs: list[str], c: dict) -> list[dict]:
    """Evidence items {ref, kind: txn|invoice|graph, summary_uz, data} for the given refs (deduplicated, in order)."""
    t = c["txs"].set_index("txn_id")
    seen, out = set(), []
    for ref in refs:
        if ref in seen:
            continue
        seen.add(ref)
        if ref.startswith("invoice:"):
            inv = c["invoice"]
            ok = "mos keladi" if inv["oked_matches_purpose"] else "mos kelmaydi"
            out.append(dict(ref=ref, kind="invoice", data=inv, summary_uz=(
                f"Hisob-faktura {_som(inv['invoice_amount_uzs'])}, ta'minotchi {inv['supplier_name']} "
                f"(OKED {inv['supplier_oked']}); faoliyat turi kredit maqsadiga {ok}")))
        elif ref.startswith("graph:"):
            g = c["graph"]
            out.append(dict(ref=ref, kind="graph", data=g, summary_uz=(
                f"Ta'minotchi {g['supplier_name']}: {g['in_degree']} ta kreditning hisob-fakturasida ko'rsatilgan; olgan "
                f"mablag'ning {_pct(g['cash_share'], 'ini')} naqd yechgan; hisobi "
                + ("hali faol emas; " if g["active_span_days"] < 0 else f"{_d(g['active_span_days'])} kun faol; ") +
                f"bank xodimi bilan bog'lanish: {'bor' if g['employee_link'] else "yo'q"}")))
        elif ref in t.index:
            r = t.loc[ref]
            out.append(dict(ref=ref, kind="txn", data=dict(
                txn_id=ref, ts=r["ts"].isoformat(), type=r["type"], amount_uzs=int(r["amount_uzs"]),
                counterparty_id=None if pd.isna(r["counterparty_id"]) else r["counterparty_id"]),
                summary_uz=f"{r['ts']:%Y-%m-%d}: {_som(r['amount_uzs'])} {TXN_TYPE_UZ[r['type']]}"))
    return out
