"""Scenario scoring: a judge builds a loan and its transactions, the frozen model v2 scores it live.

The scenario becomes one extra loan next to the demo portfolio's real context (never mutating the cached
portfolio): the same `build_features` and the same model / reason templates as for a portfolio loan.
For speed the context is restricted to what the scenario loan's features can see - the loans naming the same
supplier, their transactions and the supplier's ledger (every money-trail feature of a loan depends only on
its own transactions and on its supplier's loans / ledger); tests assert that this gives exactly the same features
as running on a full copy of the portfolio (`_features(..., full=True)`).

Assumptions (documented in docs/api_contract.md): without `invoice_amount_uzs` there is no invoice (invoice_gap uses its
-1 'no invoice' convention and no invoice-gap reason is shown); the borrower's
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
from .reasons_i18n import money

SCENARIO_LOAN_ID, SCENARIO_SUPPLIER_ID, SCENARIO_BORROWER_ID = "L-SCENARIO", "S-SCENARIO", "B-SCENARIO"
TX_TYPES = ("cash_withdrawal", "supplier_payment", "p2p_out", "p2p_in", "external_transfer", "income")
INFLOWS = {"p2p_in", "income"}
CP_KINDS = ("person", "business", "external", "supplier", "none")
DEFAULT_KIND = {"cash_withdrawal": "none", "supplier_payment": "supplier", "p2p_out": "person", "p2p_in": "person",
                "external_transfer": "external", "income": "business"}
MAX_TX, MAX_AMOUNT = 60, 10 ** 11
SNAPSHOT_DAY_MIN, SNAPSHOT_DAY_MAX = 2, 364       # the model was trained on loans 2..364 days old at the snapshot
MAX_SINGLE_X, MAX_INFLOW_X = 10, 20                # absurd inputs: one amount > 10 x loan, all inflows > 20 x loan
KEYS_TOP = ("purpose", "amount_uzs", "region", "snapshot_day", "supplier", "transactions", "invoice_amount_uzs")
KEYS_SUPPLIER = {"existing": ("mode", "supplier_id"), "new": ("mode", "oked", "registered_days_before")}
KEYS_TX = ("day", "type", "amount_uzs", "counterparty_kind")
# reasons about the supplier's own ledger are not shown for a NEW supplier (the scenario has no supplier ledger)
NEW_SUPPLIER_HIDDEN = frozenset({"sup_cash_share", "sup_exit_share", "sup_cash_lag_days", "sup_active_span_days"})
DAY = pd.Timedelta(days=1)


def _som(v) -> str:
    return f"{int(round(v)):,}".replace(",", " ") + " so'm"


def _e(uz: str, en: str, ru: str) -> ScenarioError:
    return ScenarioError(uz, en, ru)


def _int(v, what, lo=None, hi=None) -> int:
    """Exact integer from an int or an integral finite float; bool, str, NaN, inf are rejected (ScenarioError).
    `what` is the name of the value in (uz, en, ru)."""
    if isinstance(v, (float, np.floating)) and math.isfinite(v) and float(v).is_integer():
        v = int(v)
    if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
        raise _e(f"{what[0]}: butun son kiritilishi kerak", f"{what[1]}: a whole number is required", f"{what[2]}: требуется целое число")
    v = int(v)
    if (lo is not None and v < lo) or (hi is not None and v > hi):
        rng = f"{lo if lo is not None else ''}..{hi if hi is not None else ''}"
        raise _e(f"{what[0]}: qiymat {rng} oralig'ida bo'lishi kerak", f"{what[1]}: the value must be in the range {rng}",
                 f"{what[2]}: значение должно быть в диапазоне {rng}")
    return v


def _no_unknown(d: dict, allowed, where):
    """`where` is the place in (uz, en, ru)."""
    extra = [k for k in d if k not in allowed]
    if extra:
        keys, ok = ", ".join(sorted(map(str, extra))), ", ".join(allowed)
        raise _e(f"Noma'lum kalit(lar) {where[0]}: {keys}. Mumkin: {ok}", f"Unknown key(s) {where[1]}: {keys}. Allowed: {ok}",
                 f"Неизвестные ключи {where[2]}: {keys}. Допустимые: {ok}")


def _w(uz: str, en: str, ru: str) -> tuple:
    return (uz, en, ru)


def validate(scn, P) -> dict:
    """Normalised scenario or ScenarioError (message in uz / en / ru)."""
    if not isinstance(scn, dict):
        raise _e("Ssenariy lug'at (dict) bo'lishi kerak", "The scenario must be an object (dict)", "Сценарий должен быть объектом (dict)")
    _no_unknown(scn, KEYS_TOP, _w("ssenariyda", "in the scenario", "в сценарии"))
    purpose = scn.get("purpose")
    if purpose not in PURPOSES:
        ok = ", ".join(PURPOSES)
        raise _e(f"Noma'lum kredit maqsadi: {purpose!r}. Mumkin: {ok}", f"Unknown loan purpose: {purpose!r}. Allowed: {ok}",
                 f"Неизвестная цель кредита: {purpose!r}. Допустимые: {ok}")
    amount = _int(scn.get("amount_uzs"), _w("Kredit summasi (amount_uzs)", "Loan amount (amount_uzs)", "Сумма кредита (amount_uzs)"), 1, MAX_AMOUNT)
    snap_day = _int(scn.get("snapshot_day"), _w("snapshot_day (kredit berilganidan baholash kunigacha kunlar)",
                                               "snapshot_day (days from disbursement to the evaluation date)",
                                               "snapshot_day (дней от выдачи кредита до даты оценки)"))
    if not SNAPSHOT_DAY_MIN <= snap_day <= SNAPSHOT_DAY_MAX:
        raise _e(f"snapshot_day {SNAPSHOT_DAY_MIN}..{SNAPSHOT_DAY_MAX} kun oralig'ida bo'lishi kerak: model bir yilgacha "
                 "(2-364 kunlik) berilgan kreditlar bo'yicha o'qitilgan",
                 f"snapshot_day must be between {SNAPSHOT_DAY_MIN} and {SNAPSHOT_DAY_MAX} days: the model was trained on loans "
                 "issued up to a year earlier (2-364 days)",
                 f"snapshot_day должен быть в диапазоне {SNAPSHOT_DAY_MIN}..{SNAPSHOT_DAY_MAX} дней: модель обучена на кредитах, "
                 "выданных до года назад (2-364 дня)")
    invoice = scn.get("invoice_amount_uzs")
    if invoice is not None:
        invoice = _int(invoice, _w("Hisob-faktura summasi (invoice_amount_uzs)", "Invoice amount (invoice_amount_uzs)",
                                   "Сумма счёта-фактуры (invoice_amount_uzs)"), 1, MAX_AMOUNT)
        if invoice > MAX_SINGLE_X * amount:
            raise _e(f"Hisob-faktura summasi kredit summasidan {MAX_SINGLE_X} baravardan katta bo'lishi mumkin emas",
                     f"The invoice amount cannot be more than {MAX_SINGLE_X} times the loan amount",
                     f"Сумма счёта-фактуры не может превышать сумму кредита более чем в {MAX_SINGLE_X} раз")
    region = scn.get("region")
    if region is not None and not (isinstance(region, str) and region in {r[0] for r in REGIONS}):
        raise _e(f"Noma'lum hudud: {region!r}", f"Unknown region: {region!r}", f"Неизвестный регион: {region!r}")
    sup = scn.get("supplier")
    if not isinstance(sup, dict) or sup.get("mode") not in ("existing", "new"):
        raise _e("Ta'minotchi (supplier) 'mode': 'existing' yoki 'new' bo'lishi kerak", "The supplier 'mode' must be 'existing' or 'new'",
                 "'mode' поставщика (supplier) должен быть 'existing' или 'new'")
    _no_unknown(sup, KEYS_SUPPLIER[sup["mode"]], _w("ta'minotchida", "in the supplier", "у поставщика"))
    if sup["mode"] == "existing":
        sid = sup.get("supplier_id")
        if not isinstance(sid, str) or sid not in P.suppliers.index:
            shown = sid if isinstance(sid, str) else repr(sid)
            raise _e(f"Noma'lum ta'minotchi: {shown}", f"Unknown supplier: {shown}", f"Неизвестный поставщик: {shown}")
        supplier = dict(mode="existing", supplier_id=sid)
    else:
        oked = sup.get("oked")
        if not isinstance(oked, str) or not re.fullmatch(r"[0-9]{2}\.[0-9]{2}", oked):
            raise _e("Ta'minotchi OKED kodi 'NN.NN' ko'rinishida bo'lishi kerak (masalan 46.23)",
                     "The supplier OKED code must look like 'NN.NN' (for example 46.23)",
                     "Код ОКЭД поставщика должен иметь вид 'NN.NN' (например, 46.23)")
        reg = "registered_days_before"
        supplier = dict(mode="new", oked=oked, registered_days_before=_int(sup.get(reg), _w(reg, reg, reg), 0, 40_000))
    txs = scn.get("transactions")
    if not isinstance(txs, list):
        raise _e("transactions ro'yxat bo'lishi kerak", "transactions must be a list", "transactions должен быть списком")
    if len(txs) > MAX_TX:
        raise _e(f"Tranzaksiyalar soni {MAX_TX} tadan oshmasligi kerak (kiritildi: {len(txs)})",
                 f"The number of transactions must not exceed {MAX_TX} (entered: {len(txs)})",
                 f"Количество транзакций не должно превышать {MAX_TX} (введено: {len(txs)})")
    norm = []
    for i, t in enumerate(txs, 1):
        if not isinstance(t, dict):
            raise _e(f"{i}-tranzaksiya lug'at bo'lishi kerak", f"Transaction {i} must be an object (dict)",
                     f"Транзакция {i} должна быть объектом (dict)")
        _no_unknown(t, KEYS_TX, _w(f"{i}-tranzaksiyada", f"in transaction {i}", f"в транзакции {i}"))
        if t.get("type") not in TX_TYPES:
            ok = ", ".join(TX_TYPES)
            raise _e(f"{i}-tranzaksiya: noma'lum tur {t.get('type')!r}. Mumkin: {ok}",
                     f"Transaction {i}: unknown type {t.get('type')!r}. Allowed: {ok}",
                     f"Транзакция {i}: неизвестный тип {t.get('type')!r}. Допустимые: {ok}")
        kind = t.get("counterparty_kind")
        if kind is not None and kind not in CP_KINDS:
            raise _e(f"{i}-tranzaksiya: noma'lum counterparty_kind {kind!r}", f"Transaction {i}: unknown counterparty_kind {kind!r}",
                     f"Транзакция {i}: неизвестный counterparty_kind {kind!r}")
        norm.append(dict(i=i, day=_int(t.get("day"), _w(f"{i}-tranzaksiya kuni (day)", f"Transaction {i} day (day)", f"День транзакции {i} (day)"), 0, snap_day),
                         type=t["type"], amount_uzs=_int(t.get("amount_uzs"), _w(f"{i}-tranzaksiya summasi", f"Transaction {i} amount",
                                                                               f"Сумма транзакции {i}"), 1, MAX_AMOUNT),
                         counterparty_kind=kind or DEFAULT_KIND[t["type"]]))
    for t in norm:
        if t["amount_uzs"] > MAX_SINGLE_X * amount:
            a = [money(t["amount_uzs"], lang) for lang in ("uz", "en", "ru")]
            raise _e(f"{t['i']}-tranzaksiya summasi ({a[0]}) kredit summasidan {MAX_SINGLE_X} baravardan katta: ssenariy real emas",
                     f"Transaction {t['i']} amount ({a[1]}) is more than {MAX_SINGLE_X} times the loan amount: the scenario is not realistic",
                     f"Сумма транзакции {t['i']} ({a[2]}) более чем в {MAX_SINGLE_X} раз больше суммы кредита: сценарий нереалистичен")
    inflow = sum(t["amount_uzs"] for t in norm if t["type"] in INFLOWS)
    if inflow > MAX_INFLOW_X * amount:
        a = [money(inflow, lang) for lang in ("uz", "en", "ru")]
        raise _e(f"Barcha kirimlar yig'indisi ({a[0]}) kredit summasidan {MAX_INFLOW_X} baravardan katta: ssenariy real emas",
                 f"The total of all inflows ({a[1]}) is more than {MAX_INFLOW_X} times the loan amount: the scenario is not realistic",
                 f"Сумма всех поступлений ({a[2]}) более чем в {MAX_INFLOW_X} раз больше суммы кредита: сценарий нереалистичен")
    # balance check: the loan arrives on day 0; inflows are applied before outflows of the same day
    balance = amount
    for t in sorted(norm, key=lambda t: (t["day"], t["type"] not in INFLOWS, t["i"])):
        if t["type"] in INFLOWS:
            balance += t["amount_uzs"]
        else:
            if t["amount_uzs"] > balance:
                need = [money(t["amount_uzs"], lang) for lang in ("uz", "en", "ru")]
                have = [money(balance, lang) for lang in ("uz", "en", "ru")]
                raise _e(f"{t['i']}-tranzaksiya ({t['day']}-kun): hisobda mablag' yetarli emas (kerak: {need[0]}, mavjud: {have[0]})",
                         f"Transaction {t['i']} (day {t['day']}): insufficient funds in the account (needed: {need[1]}, available: {have[1]})",
                         f"Транзакция {t['i']} (день {t['day']}): на счёте недостаточно средств (нужно: {need[2]}, доступно: {have[2]})")
            balance -= t["amount_uzs"]
    return dict(purpose=purpose, amount_uzs=amount, snapshot_day=snap_day, region=region, supplier=supplier,
                transactions=norm, invoice_amount_uzs=invoice)


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
                              branch_id=_branch(P, scn), supplier_id=sid,
                              # 0 = no invoice: the feature builder then uses its training convention (invoice_gap = -1)
                              invoice_amount_uzs=scn["invoice_amount_uzs"] or 0)])
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
    hidden = set()
    if scn["invoice_amount_uzs"] is None:
        hidden.add("invoice_gap")                                    # no invoice was entered: no invoice-gap statement
    if scn["supplier"]["mode"] == "new":
        hidden |= NEW_SUPPLIER_HIDDEN                                 # no supplier ledger was entered
    reasons = build_reasons(P.model.features, x.iloc[0].to_numpy(float), contrib, {"supplier_oked": oked},
                            with_refs=False, exclude=hidden)
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
             title=dict(uz="Ta'minotchidan oddiy xarid", en="An ordinary purchase from a supplier", ru="Обычная покупка у поставщика"),
             description=dict(uz="Kredit tushgach, summa ta'minotchiga o'tkaziladi; keyin oyiga bir marta kichik tushum.",
                              en="After the loan arrives, the amount is paid to the supplier; then a small income payment once a month.",
                              ru="После поступления кредита сумма переводится поставщику; затем раз в месяц небольшое поступление дохода."),
             scenario=dict(purpose="tikuv_dastgohi", amount_uzs=40_000_000, invoice_amount_uzs=40_000_000, region=None, snapshot_day=150,
                           supplier=dict(mode="new", oked="28.94", registered_days_before=2000),
                           transactions=[dict(day=3, type="supplier_payment", amount_uzs=39_000_000)] +
                           [dict(day=d, type="income", amount_uzs=1_500_000, counterparty_kind="business") for d in (45, 75, 105, 135)])),
        dict(id="fast_cash_out", title_uz="Kredit tushishi bilan tez naqdlashtirish",
             description_uz="Birinchi kuni mablag'ning katta qismi naqd yechiladi, ta'minotchiga atigi kichik summa o'tkaziladi. "
                            "Naqd yechish kunini (day) o'zgartirib ko'ring.",
             title=dict(uz="Kredit tushishi bilan tez naqdlashtirish", en="Cashing out quickly after the loan arrives",
                        ru="Быстрое снятие наличных сразу после поступления кредита"),
             description=dict(uz="Birinchi kuni mablag'ning katta qismi naqd yechiladi, ta'minotchiga atigi kichik summa o'tkaziladi. "
                                 "Naqd yechish kunini (day) o'zgartirib ko'ring.",
                              en="On the first day a large part of the funds is withdrawn in cash and only a small amount goes to the supplier. "
                                 "Try changing the day of the cash withdrawal (day).",
                              ru="В первый день большая часть средств снимается наличными, поставщику переводится лишь небольшая сумма. "
                                 "Попробуйте изменить день снятия наличных (day)."),
             scenario=dict(purpose="chorva", amount_uzs=50_000_000, invoice_amount_uzs=50_000_000, region=None, snapshot_day=60,
                           supplier=dict(mode="new", oked="01.41", registered_days_before=900),
                           transactions=[dict(day=1, type="cash_withdrawal", amount_uzs=42_000_000),
                                         dict(day=5, type="supplier_payment", amount_uzs=5_000_000)])),
        dict(id="supplier_ring", title_uz="Ko'p qarz oluvchi, bitta ta'minotchi, naqd chiqarish",
             description_uz="Ko'p kredit bitta ta'minotchiga to'lanadi va ta'minotchi pulni tez naqdlashtiradi (demo portfeldagi "
                            f"mavjud ta'minotchi {ring['supplier_id']}). Bu kredit shu ta'minotchining yana bir qarz oluvchisi.",
             title=dict(uz="Ko'p qarz oluvchi, bitta ta'minotchi, naqd chiqarish", en="Many borrowers, one supplier, cash-out",
                        ru="Много заёмщиков, один поставщик, снятие наличных"),
             description=dict(uz="Ko'p kredit bitta ta'minotchiga to'lanadi va ta'minotchi pulni tez naqdlashtiradi (demo portfeldagi "
                                 f"mavjud ta'minotchi {ring['supplier_id']}). Bu kredit shu ta'minotchining yana bir qarz oluvchisi.",
                              en="Many loans are paid to one supplier and the supplier quickly turns the money into cash (existing supplier "
                                 f"{ring['supplier_id']} of the demo portfolio). This loan is one more borrower of that supplier.",
                              ru="Многие кредиты переводятся одному поставщику, и поставщик быстро снимает деньги наличными (существующий "
                                 f"поставщик {ring['supplier_id']} демо-портфеля). Этот кредит — ещё один заёмщик этого поставщика."),
             scenario=dict(purpose=ring["purpose"], amount_uzs=ring["amount"], invoice_amount_uzs=ring["amount"], region=None, snapshot_day=90,
                           supplier=dict(mode="existing", supplier_id=ring["supplier_id"]),
                           transactions=[dict(day=2, type="supplier_payment", amount_uzs=int(ring["amount"] * 0.92))])),
        dict(id="asset_resale", title_uz="Aktiv sotib olinadi, keyin bir martalik tushum va naqd chiqarish",
             description_uz="Ta'minotchiga to'lov o'tkaziladi; ikki oy o'tib katta bir martalik tushum keladi va u naqd yechiladi.",
             title=dict(uz="Aktiv sotib olinadi, keyin bir martalik tushum va naqd chiqarish",
                        en="The asset is bought, then a one-off inflow and a cash withdrawal",
                        ru="Актив куплен, затем разовое поступление и снятие наличных"),
             description=dict(uz="Ta'minotchiga to'lov o'tkaziladi; ikki oy o'tib katta bir martalik tushum keladi va u naqd yechiladi.",
                              en="A payment goes to the supplier; two months later a large one-off inflow arrives and is withdrawn in cash.",
                              ru="Поставщику переводится платёж; через два месяца приходит крупное разовое поступление, и оно снимается наличными."),
             scenario=dict(purpose="chorva", amount_uzs=60_000_000, invoice_amount_uzs=60_000_000, region=None, snapshot_day=120,
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
