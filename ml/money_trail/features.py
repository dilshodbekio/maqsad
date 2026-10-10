"""Money-trail features: one row per loan, computed as of the portfolio's snapshot_date.

Single source of truth for training, serving (ml/api.py) and ml/generator/check_leakage.py.
Only transactions, supplier / employee tables and the loan's own money fields are used:
NO photo / geo columns and NO labels. Missing events get an explicit sentinel (400 days = "never").
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ml.generator.config import MATCHING_OKED

DAY = pd.Timedelta(days=1)
NEVER = 400.0

FEATURE_COLUMNS = [
    # cash timing / shares
    "first_cash_days", "cash_share_d1", "cash_share_d7", "cash_share_d30", "cash_share_all",
    # supplier payment, invoice gap
    "sup_pay_share", "first_sup_pay_days", "invoice_gap",
    # money leaving the bank / P2P
    "leave_share_30", "p2p_out_count", "p2p_in_count",
    # one-off inbound and what follows it
    "big_inbound_share", "big_inbound_day", "big_inbound_person", "post_inbound_exit",
    "income_count", "income_after_big_count", "days_since_income", "income_stopped_after_inbound",
    # supplier graph
    "sup_in_degree", "sup_cash_share", "sup_exit_share", "sup_cash_lag_days",
    "sup_active_span_days", "sup_age_at_first_payment_days",
    # bank-employee link and declared supplier OKED
    "sup_employee_link", "emp_branch_match", "oked_mismatch",
]


def build_features(loans: pd.DataFrame, transactions: pd.DataFrame, suppliers: pd.DataFrame,
                   employees: pd.DataFrame) -> pd.DataFrame:
    L = loans.set_index("loan_id")
    sup, emp = suppliers.set_index("supplier_id"), employees.set_index("employee_id")
    snap = L["snapshot_date"].max()
    tx = transactions[transactions["ts"] <= snap]
    amt, ix = L["amount_uzs"], L.index
    F = pd.DataFrame(index=ix)

    bt = tx[tx.loan_id.notna()].copy()
    bt["type"] = bt["type"].astype(str)
    bt["day"] = (bt["ts"] - bt["loan_id"].map(L["disbursed_at"])) / DAY

    def share(mask, upto=None):
        d = bt[mask] if upto is None else bt[mask & (bt["day"] <= upto)]
        return d.groupby("loan_id")["amount_uzs"].sum().reindex(ix).fillna(0) / amt

    # ---- cash timing / shares
    cash = bt["type"] == "cash_withdrawal"
    F["first_cash_days"] = bt[cash].groupby("loan_id")["day"].min().reindex(ix).fillna(NEVER)
    for k in (1, 7, 30):
        F[f"cash_share_d{k}"] = share(cash, k)
    F["cash_share_all"] = share(cash)

    # ---- supplier payment, invoice gap
    sp = bt["type"] == "supplier_payment"
    F["sup_pay_share"] = share(sp)
    F["first_sup_pay_days"] = bt[sp].groupby("loan_id")["day"].min().reindex(ix).fillna(NEVER)
    F["invoice_gap"] = 1 - share(sp) * amt / L["invoice_amount_uzs"]

    # ---- leaving the bank / P2P
    leave = bt["type"].isin(["cash_withdrawal", "external_transfer", "p2p_out"])
    F["leave_share_30"] = share(leave, 30)
    for typ in ("p2p_out", "p2p_in"):
        F[f"{typ}_count"] = bt[bt["type"] == typ].groupby("loan_id").size().reindex(ix).fillna(0)

    # ---- biggest one-off inbound, exits right after it, income regularity
    inb = bt[bt["type"].isin(["p2p_in", "income"])]
    big = inb.sort_values("amount_uzs", kind="stable").groupby("loan_id").tail(1).set_index("loan_id")
    F["big_inbound_share"] = (big["amount_uzs"] / amt).reindex(ix).fillna(0)
    F["big_inbound_day"] = big["day"].reindex(ix).fillna(-1)
    F["big_inbound_person"] = (big["counterparty_kind"].astype(str) == "person").astype(float).reindex(ix).fillna(0)
    out = bt[cash | (bt["type"] == "external_transfer")].join(big[["day", "amount_uzs"]].add_prefix("b_"), on="loan_id")
    out = out[(out["day"] >= out["b_day"]) & (out["day"] <= out["b_day"] + 5)]
    post = out.groupby("loan_id")["amount_uzs"].sum().reindex(ix).fillna(0)
    F["post_inbound_exit"] = (post / big["amount_uzs"].reindex(ix)).fillna(0)
    inc = bt[bt["type"] == "income"]
    F["income_count"] = inc.groupby("loan_id").size().reindex(ix).fillna(0)
    after = inc.join(big[["day"]].add_prefix("b_"), on="loan_id")
    F["income_after_big_count"] = (after["day"] > after["b_day"]).groupby(after["loan_id"]).sum().reindex(ix).fillna(0)
    last_inc = inc.groupby("loan_id")["ts"].max().reindex(ix)
    F["days_since_income"] = ((snap - last_inc) / DAY).fillna(NEVER)
    big_ts = L["disbursed_at"] + pd.to_timedelta(F["big_inbound_day"].clip(lower=0), unit="D")
    F["income_stopped_after_inbound"] = ((last_inc.isna() | (last_inc < big_ts))
                                         & (F["big_inbound_share"] > 0.1)).astype(float)

    # ---- supplier graph (supplier accounts have loan_id = null)
    sid = L["supplier_id"]
    F["sup_in_degree"] = sid.map(sid.value_counts())
    pay = tx[tx["type"].astype(str) == "supplier_payment"]
    received = pay.groupby("counterparty_id")["amount_uzs"].sum()
    first_rec = pay.groupby("counterparty_id")["ts"].min()
    sa = tx[tx.loan_id.isna()].copy()
    sa["supplier_id"] = sa["account_id"].str.replace("ACC-", "", regex=False)
    sa["type"] = sa["type"].astype(str)
    span = (sa.groupby("supplier_id")["ts"].max() - sa.groupby("supplier_id")["ts"].min()) / DAY
    sa = sa[sa["type"] != "income"]                     # ledger inflows are not exits
    s_cash = sa[sa["type"] == "cash_withdrawal"].groupby("supplier_id")["amount_uzs"].sum()
    s_exit = sa.groupby("supplier_id")["amount_uzs"].sum()
    first_cash = sa[sa["type"] == "cash_withdrawal"].groupby("supplier_id")["ts"].min()
    rec = sid.map(received)
    F["sup_cash_share"] = (sid.map(s_cash).fillna(0) / rec).fillna(0).clip(upper=1.5)
    F["sup_exit_share"] = (sid.map(s_exit).fillna(0) / rec).fillna(0).clip(upper=1.5)
    F["sup_cash_lag_days"] = ((sid.map(first_cash) - sid.map(first_rec)) / DAY).fillna(NEVER)
    F["sup_active_span_days"] = sid.map(span).fillna(0)
    F["sup_age_at_first_payment_days"] = ((sid.map(first_rec) - sid.map(sup["registered_at"])) / DAY).fillna(-1)

    # ---- bank-employee link, OKED vs purpose
    linked = sid.map(sup["linked_employee_id"])
    F["sup_employee_link"] = linked.notna().astype(float)
    F["emp_branch_match"] = (linked.map(emp["branch_id"]) == L["branch_id"]).astype(float)
    oked = sid.map(sup["oked_code"])
    F["oked_mismatch"] = [float(o not in MATCHING_OKED[pu]) for o, pu in zip(oked, L["purpose"].astype(str))]

    return F[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0)
