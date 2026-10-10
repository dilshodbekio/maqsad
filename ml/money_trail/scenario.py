"""Scenario scoring: a judge builds a loan and its transactions, the frozen model v2 scores it live.

The scenario becomes one extra loan next to the demo portfolio's real context (never mutating the cached
portfolio): the same `build_features` and the same model / reason templates as for a portfolio loan.
For speed the context is restricted to what the scenario loan's features can see - the loans naming the same
supplier, their transactions and the supplier's ledger (every money-trail feature of a loan depends only on
its own transactions and on its supplier's loans / ledger); tests assert that this gives exactly the same features
as running on a full copy of the portfolio (`_features(..., full=True)`).

Assumptions (documented in docs/api_contract.md): the invoice amount equals the loan amount; the borrower's
supplier payments land on the supplier's account as `income` rows; a NEW supplier has no outflows of its own
(there is nothing the form can say about them); the loan's branch is the supplier's most common branch for an
existing supplier, otherwise the first branch of the chosen region (BR-00 without a region).
"""
from __future__ import annotations

import math
import re
import time

import numpy as np
import pandas as pd

from ml.errors import ScenarioError
from ml.generator.config import MATCHING_OKED, PURPOSES, REGIONS

from .colors import color_of
from .features import build_features
from .model import MODEL_VERSION
from .reasons import build_reasons

SCENARIO_LOAN_ID, SCENARIO_SUPPLIER_ID, SCENARIO_BORROWER_ID = "L-SCENARIO", "S-SCENARIO", "B-SCENARIO"
TX_TYPES = ("cash_withdrawal", "supplier_payment", "p2p_out", "p2p_in", "external_transfer", "income")
INFLOWS = {"p2p_in", "income"}
CP_KINDS = ("person", "business", "external", "supplier", "none")
DEFAULT_KIND = {"cash_withdrawal": "none", "supplier_payment": "supplier", "p2p_out": "person", "p2p_in": "person",
                "external_transfer": "external", "income": "business"}
MAX_TX, MAX_SNAPSHOT_DAY, MAX_AMOUNT = 60, 730, 10 ** 11
DAY = pd.Timedelta(days=1)


def _som(v) -> str:
    return f"{int(round(v)):,}".replace(",", " ") + " so'm"


def _int(v, what, lo=None, hi=None) -> int:
    """Exact integer from an int or an integral finite float; bool, str, NaN, inf are rejected (ScenarioError)."""
    if isinstance(v, (float, np.floating)) and math.isfinite(v) and float(v).is_integer():
        v = int(v)
    if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
        raise ScenarioError(f"{what}: butun son kiritilishi kerak")
    v = int(v)
    if (lo is not None and v < lo) or (hi is not None and v > hi):
        raise ScenarioError(f"{what}: qiymat {lo if lo is not None else ''}..{hi if hi is not None else ''} oralig'ida bo'lishi kerak")
    return v


def validate(scn, P) -> dict:
    """Normalised scenario or ScenarioError (Uzbek message)."""
    if not isinstance(scn, dict):
        raise ScenarioError("Ssenariy lug'at (dict) bo'lishi kerak")
    purpose = scn.get("purpose")
    if purpose not in PURPOSES:
        raise ScenarioError(f"Noma'lum kredit maqsadi: {purpose!r}. Mumkin: {', '.join(PURPOSES)}")
    amount = _int(scn.get("amount_uzs"), "Kredit summasi (amount_uzs)", 1, MAX_AMOUNT)
    snap_day = _int(scn.get("snapshot_day"), "snapshot_day (kredit berilganidan baholash kunigacha kunlar)", 0, MAX_SNAPSHOT_DAY)
    region = scn.get("region")
    if region is not None and not (isinstance(region, str) and region in {r[0] for r in REGIONS}):
        raise ScenarioError(f"Noma'lum hudud: {region!r}")
    sup = scn.get("supplier")
    if not isinstance(sup, dict) or sup.get("mode") not in ("existing", "new"):
        raise ScenarioError("Ta'minotchi (supplier) 'mode': 'existing' yoki 'new' bo'lishi kerak")
    if sup["mode"] == "existing":
        sid = sup.get("supplier_id")
        if not isinstance(sid, str) or sid not in P.suppliers.index:
            raise ScenarioError(f"Noma'lum ta'minotchi: {sid if isinstance(sid, str) else repr(sid)}")
        supplier = dict(mode="existing", supplier_id=sid)
    else:
        oked = sup.get("oked")
        if not isinstance(oked, str) or not re.fullmatch(r"\d{2}\.\d{2}", oked):
            raise ScenarioError("Ta'minotchi OKED kodi 'NN.NN' ko'rinishida bo'lishi kerak (masalan 46.23)")
        supplier = dict(mode="new", oked=oked, registered_days_before=_int(
            sup.get("registered_days_before"), "registered_days_before", 0, 40_000))
    txs = scn.get("transactions")
    if not isinstance(txs, list):
        raise ScenarioError("transactions ro'yxat bo'lishi kerak")
    if len(txs) > MAX_TX:
        raise ScenarioError(f"Tranzaksiyalar soni {MAX_TX} tadan oshmasligi kerak (kiritildi: {len(txs)})")
    norm = []
    for i, t in enumerate(txs, 1):
        if not isinstance(t, dict):
            raise ScenarioError(f"{i}-tranzaksiya lug'at bo'lishi kerak")
        if t.get("type") not in TX_TYPES:
            raise ScenarioError(f"{i}-tranzaksiya: noma'lum tur {t.get('type')!r}. Mumkin: {', '.join(TX_TYPES)}")
        kind = t.get("counterparty_kind")
        if kind is not None and kind not in CP_KINDS:
            raise ScenarioError(f"{i}-tranzaksiya: noma'lum counterparty_kind {kind!r}")
        norm.append(dict(i=i, day=_int(t.get("day"), f"{i}-tranzaksiya kuni (day)", 0, snap_day),
                         type=t["type"], amount_uzs=_int(t.get("amount_uzs"), f"{i}-tranzaksiya summasi", 1, MAX_AMOUNT),
                         counterparty_kind=kind or DEFAULT_KIND[t["type"]]))
    # balance check: the loan arrives on day 0; inflows are applied before outflows of the same day
    balance = amount
    for t in sorted(norm, key=lambda t: (t["day"], t["type"] not in INFLOWS, t["i"])):
        if t["type"] in INFLOWS:
            balance += t["amount_uzs"]
        else:
            if t["amount_uzs"] > balance:
                raise ScenarioError(f"{t['i']}-tranzaksiya ({t['day']}-kun): hisobda mablag' yetarli emas "
                                    f"(kerak: {_som(t['amount_uzs'])}, mavjud: {_som(balance)})")
            balance -= t["amount_uzs"]
    return dict(purpose=purpose, amount_uzs=amount, snapshot_day=snap_day, region=region, supplier=supplier, transactions=norm)


def _branch(P, scn) -> str:
    if scn["supplier"]["mode"] == "existing":
        mode = P.loans.loc[P.loans["supplier_id"] == scn["supplier"]["supplier_id"], "branch_id"].mode()
        if len(mode):
            return str(mode.iloc[0])
    names = [r[0] for r in REGIONS]
    return f"BR-{names.index(scn['region']) * 2:02d}" if scn["region"] in names else "BR-00"


def _scenario_frames(P, scn: dict):
    """(loans, transactions, suppliers) rows of the scenario, dtype-compatible with the portfolio tables."""
    snap = P.loans["snapshot_date"].iloc[0]
    disb = (snap - scn["snapshot_day"] * DAY).as_unit("ns")
    sup = scn["supplier"]
    if sup["mode"] == "existing":
        sid, sup_rows = sup["supplier_id"], None
    else:
        sid = SCENARIO_SUPPLIER_ID
        sup_rows = pd.DataFrame([dict(supplier_id=sid, name="Yangi ta'minotchi (ssenariy)", oked_code=sup["oked"],
                                      region=scn["region"] or REGIONS[0][0],
                                      registered_at=(disb - sup["registered_days_before"] * DAY).as_unit("ns"),
                                      linked_employee_id=None)])
    loan = pd.DataFrame([dict(loan_id=SCENARIO_LOAN_ID, borrower_id=SCENARIO_BORROWER_ID, purpose=scn["purpose"],
                              amount_uzs=scn["amount_uzs"], disbursed_at=disb, snapshot_date=snap,
                              branch_id=_branch(P, scn), supplier_id=sid, invoice_amount_uzs=scn["amount_uzs"])])
    acct, rows = f"ACC-{SCENARIO_BORROWER_ID}", []

    def add(loan_id, account, day, typ, amount, cp, kind):
        rows.append(dict(txn_id=f"T-SCN-{len(rows):03d}", loan_id=loan_id, account_id=account,
                         ts=(disb + day * DAY).as_unit("ns"), type=typ, amount_uzs=int(amount),
                         counterparty_id=cp, counterparty_kind=kind))

    add(SCENARIO_LOAN_ID, acct, 0, "disbursement", scn["amount_uzs"], None, "none")
    for t in sorted(scn["transactions"], key=lambda t: (t["day"], t["i"])):
        if t["type"] == "supplier_payment":
            add(SCENARIO_LOAN_ID, acct, t["day"], t["type"], t["amount_uzs"], sid, "supplier")
            add(None, f"ACC-{sid}", t["day"], "income", t["amount_uzs"], SCENARIO_BORROWER_ID, "person")   # supplier ledger
        else:
            cp = None if t["counterparty_kind"] == "none" else f"{t['counterparty_kind'][:3].upper()}-SCN-{t['i']}"
            add(SCENARIO_LOAN_ID, acct, t["day"], t["type"], t["amount_uzs"], cp, t["counterparty_kind"])
    return loan, pd.DataFrame(rows), sup_rows, sid


LOAN_COLS = ["loan_id", "amount_uzs", "invoice_amount_uzs", "disbursed_at", "snapshot_date", "supplier_id", "branch_id", "purpose"]
TX_COLS = ["txn_id", "loan_id", "account_id", "ts", "type", "amount_uzs", "counterparty_id", "counterparty_kind"]


def _features(P, scn: dict, full: bool = False) -> pd.Series:
    """Feature row of the scenario loan. full=True runs on a complete copy of the demo portfolio (reference)."""
    loan, tx_new, sup_new, sid = _scenario_frames(P, scn)
    loans_all, sup_all = P.loans.reset_index(), P.suppliers.reset_index()
    if full:
        loans_ctx, tx_ctx, sup_ctx = loans_all, P.tx, sup_all
    elif sup_new is None:                                   # existing supplier: its loans, their transactions, its ledger
        loans_ctx = loans_all[loans_all["supplier_id"] == sid]
        tx_ctx = P.tx[P.tx["loan_id"].isin(loans_ctx["loan_id"]) | (P.tx["account_id"] == f"ACC-{sid}")]
        sup_ctx = sup_all[sup_all["supplier_id"] == sid]
    else:                                                   # new supplier: nothing else names it
        loans_ctx, tx_ctx, sup_ctx = loans_all.iloc[:0], P.tx.iloc[:0], sup_all.iloc[:0]
    loans = pd.concat([loans_ctx[LOAN_COLS], loan[LOAN_COLS]], ignore_index=True)
    loans["disbursed_at"] = loans["disbursed_at"].astype("datetime64[ns]")
    loans["snapshot_date"] = loans["snapshot_date"].astype("datetime64[ns]")
    tx = pd.concat([tx_ctx[TX_COLS].astype({"type": str, "counterparty_kind": str}), tx_new], ignore_index=True)
    tx["ts"] = tx["ts"].astype("datetime64[ns]")
    suppliers = sup_ctx if sup_new is None else pd.concat([sup_ctx, sup_new], ignore_index=True)
    F = build_features(loans, tx, suppliers, P.employees)
    return F.loc[SCENARIO_LOAN_ID]


def score_scenario(scenario: dict) -> dict:
    """Score a judge-built scenario live. Same shape as `score_loan` plus `percentile` (0-100, rank of this
    risk within the demo portfolio) and `features` (the computed feature values)."""
    from .service import get_portfolio
    P = get_portfolio()
    t0 = time.perf_counter()
    scn = validate(scenario, P)
    row = _features(P, scn)
    x = row.to_frame().T[P.model.features]
    risk = float(P.model.score(x)[0])
    contrib = P.model.contributions(x)[0]
    oked = scn["supplier"]["oked"] if scn["supplier"]["mode"] == "new" else P.suppliers.at[scn["supplier"]["supplier_id"], "oked_code"]
    reasons = build_reasons(P.model.features, x.iloc[0].to_numpy(float), contrib, {"supplier_oked": oked}, with_refs=False)
    pct = 100.0 * float(np.searchsorted(P.sorted_risk, risk, side="right")) / len(P.sorted_risk)
    return dict(risk=risk, color=color_of(risk, P.thresholds), reasons=reasons, model_version=MODEL_VERSION,
                latency_ms=round((time.perf_counter() - t0) * 1000, 3), percentile=round(pct, 1),
                features={k: float(v) for k, v in row[P.model.features].items()})


# ------------------------------------------------------------------ presets (inputs only; scores are always live)
def get_scenario_presets() -> list[dict]:
    from .service import get_portfolio
    P = get_portfolio()
    ring = _ring_supplier(P)
    return [
        dict(id="honest_purchase", title_uz="Ta'minotchidan oddiy xarid",
             description_uz="Kredit tushgach, summa ta'minotchiga o'tkaziladi; keyin oyiga bir marta kichik tushum.",
             scenario=dict(purpose="tikuv_dastgohi", amount_uzs=40_000_000, region=None, snapshot_day=150,
                           supplier=dict(mode="new", oked="28.94", registered_days_before=2000),
                           transactions=[dict(day=3, type="supplier_payment", amount_uzs=39_000_000)] +
                           [dict(day=d, type="income", amount_uzs=1_500_000, counterparty_kind="business") for d in (45, 75, 105, 135)])),
        dict(id="fast_cash_out", title_uz="Kredit tushishi bilan tez naqdlashtirish",
             description_uz="Birinchi kuni mablag'ning katta qismi naqd yechiladi, ta'minotchiga atigi kichik summa o'tkaziladi. "
                            "Naqd yechish kunini (day) o'zgartirib ko'ring.",
             scenario=dict(purpose="chorva", amount_uzs=50_000_000, region=None, snapshot_day=60,
                           supplier=dict(mode="new", oked="01.41", registered_days_before=900),
                           transactions=[dict(day=1, type="cash_withdrawal", amount_uzs=42_000_000),
                                         dict(day=5, type="supplier_payment", amount_uzs=5_000_000)])),
        dict(id="supplier_ring", title_uz="Ko'p qarz oluvchi, bitta ta'minotchi, naqd chiqarish",
             description_uz="Ko'p kredit bitta ta'minotchiga to'lanadi va ta'minotchi pulni tez naqdlashtiradi (demo portfeldagi "
                            f"mavjud ta'minotchi {ring['supplier_id']}). Bu kredit shu ta'minotchining yana bir qarz oluvchisi.",
             scenario=dict(purpose=ring["purpose"], amount_uzs=ring["amount"], region=None, snapshot_day=90,
                           supplier=dict(mode="existing", supplier_id=ring["supplier_id"]),
                           transactions=[dict(day=2, type="supplier_payment", amount_uzs=int(ring["amount"] * 0.92))])),
        dict(id="asset_resale", title_uz="Aktiv sotib olinadi, keyin bir martalik tushum va naqd chiqarish",
             description_uz="Ta'minotchiga to'lov o'tkaziladi; ikki oy o'tib katta bir martalik tushum keladi va u naqd yechiladi.",
             scenario=dict(purpose="chorva", amount_uzs=60_000_000, region=None, snapshot_day=120,
                           supplier=dict(mode="new", oked="01.41", registered_days_before=1200),
                           transactions=[dict(day=3, type="supplier_payment", amount_uzs=58_000_000),
                                         dict(day=60, type="p2p_in", amount_uzs=40_000_000, counterparty_kind="person"),
                                         dict(day=61, type="cash_withdrawal", amount_uzs=38_000_000)])),
    ]


def _ring_supplier(P) -> dict:
    """Existing demo supplier with many borrowers and a high downstream cash share (chosen from features only)."""
    F = P.features
    sup = P.loans["supplier_id"]
    g = pd.DataFrame({"supplier_id": sup, "deg": F["sup_in_degree"], "cash": F["sup_cash_share"]}).drop_duplicates("supplier_id")
    cand = g[g["deg"] >= 6].sort_values(["cash", "deg", "supplier_id"], ascending=[False, False, True])
    sid = str((cand if len(cand) else g.sort_values(["deg", "supplier_id"], ascending=[False, True])).iloc[0]["supplier_id"])
    mine = P.loans[P.loans["supplier_id"] == sid]
    return dict(supplier_id=sid, purpose=str(mine["purpose"].astype(str).mode().iloc[0]),
                amount=int(round(float(mine["amount_uzs"].median()), -6)) or 50_000_000)
