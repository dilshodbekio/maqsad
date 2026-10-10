"""Leakage / trivial-separability check for a generated split.

    python -m ml.generator.check_leakage --split train

For every typology (honest loans as the negative class) it reports single-feature
AUC for a set of money-trail / graph / geo / photo features, flags any >= 0.90, and
scores the simple rule "cash >= 50% of the loan within day 0-1" on recall@20%.
Features use only data up to the snapshot. Labels are used only for scoring.
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .build import OUT_ROOT
from .config import MATCHING_OKED

DAY = pd.Timedelta(days=1)
FLAG_AUC = 0.90
BUDGET = 0.20


def load(split: str, root: Path = OUT_ROOT) -> dict:
    d = root / split
    return {n: pd.read_parquet(d / f"{n}.parquet")
            for n in ("loans", "transactions", "suppliers", "employees", "labels")}


def features(t: dict) -> pd.DataFrame:
    loans = t["loans"].set_index("loan_id")
    tx, sup = t["transactions"], t["suppliers"].set_index("supplier_id")
    snap = loans["snapshot_date"].iloc[0]
    amt = loans["amount_uzs"]
    F = pd.DataFrame(index=loans.index)

    bt = tx[tx.loan_id.notna()].copy()
    bt["type"] = bt["type"].astype(str)
    bt["day"] = (bt["ts"] - bt["loan_id"].map(loans["disbursed_at"])) / DAY

    def share(mask, upto=None):
        d = bt[mask] if upto is None else bt[mask & (bt["day"] <= upto)]
        return (d.groupby("loan_id")["amount_uzs"].sum().reindex(loans.index).fillna(0) / amt)

    cash = bt["type"] == "cash_withdrawal"
    F["first_cash_days"] = bt[cash].groupby("loan_id")["day"].min().reindex(loans.index).fillna(400)
    for k in (1, 7, 30):
        F[f"cash_share_d{k}"] = share(cash, k)
    F["cash_share_all"] = share(cash)
    sp = bt["type"] == "supplier_payment"
    F["sup_pay_share"] = share(sp)
    F["sup_pay_flag"] = (F["sup_pay_share"] > 0).astype(float)
    F["first_sup_pay_days"] = bt[sp].groupby("loan_id")["day"].min().reindex(loans.index).fillna(400)
    paid = share(sp) * amt
    F["invoice_gap"] = 1 - paid / loans["invoice_amount_uzs"]
    leave = bt["type"].isin(["cash_withdrawal", "external_transfer", "p2p_out"])
    F["leave_share_30"] = share(leave, 30)
    F["p2p_out_count"] = bt[bt["type"] == "p2p_out"].groupby("loan_id").size().reindex(loans.index).fillna(0)
    F["p2p_in_count"] = bt[bt["type"] == "p2p_in"].groupby("loan_id").size().reindex(loans.index).fillna(0)

    inb = bt[bt["type"].isin(["p2p_in", "income"])]
    big = inb.sort_values("amount_uzs").groupby("loan_id").tail(1).set_index("loan_id")
    F["big_inbound_share"] = (big["amount_uzs"] / amt).reindex(loans.index).fillna(0)
    F["big_inbound_day"] = big["day"].reindex(loans.index).fillna(-1)
    F["big_inbound_person"] = (big["counterparty_kind"].astype(str) == "person").astype(float).reindex(loans.index).fillna(0)
    # cash leaving within 5 days after the biggest inbound, relative to it
    m = bt[cash | (bt["type"] == "external_transfer")].join(big[["day", "amount_uzs"]].add_prefix("b_"), on="loan_id")
    m = m[(m["day"] >= m["b_day"]) & (m["day"] <= m["b_day"] + 5)]
    post = m.groupby("loan_id")["amount_uzs"].sum().reindex(loans.index).fillna(0)
    F["post_inbound_exit"] = (post / (big["amount_uzs"].reindex(loans.index))).fillna(0)
    inc = bt[bt["type"] == "income"]
    F["income_count"] = inc.groupby("loan_id").size().reindex(loans.index).fillna(0)
    last_inc = inc.groupby("loan_id")["ts"].max().reindex(loans.index)
    F["days_since_income"] = ((snap - last_inc) / DAY).fillna(400)
    F["income_stopped_after_inbound"] = (
        ((last_inc.isna()) | (last_inc < loans["disbursed_at"] + pd.to_timedelta(F["big_inbound_day"].clip(lower=0), unit="D")))
        & (F["big_inbound_share"] > 0.1)).astype(float)

    # --- supplier / graph features
    sid = loans["supplier_id"]
    F["sup_in_degree"] = sid.map(sid.value_counts())
    received = tx[(tx["type"].astype(str) == "supplier_payment")].groupby("counterparty_id")["amount_uzs"].sum()
    acct = tx[tx.loan_id.isna()].copy()
    acct["supplier_id"] = acct["account_id"].str.replace("ACC-", "", regex=False)
    acct["type"] = acct["type"].astype(str)
    s_cash = acct[acct["type"] == "cash_withdrawal"].groupby("supplier_id")["amount_uzs"].sum()
    s_exit = acct.groupby("supplier_id")["amount_uzs"].sum()
    rec = sid.map(received)
    F["sup_cash_share"] = (sid.map(s_cash).fillna(0) / rec).fillna(0).clip(upper=1.5)
    F["sup_exit_share"] = (sid.map(s_exit).fillna(0) / rec).fillna(0).clip(upper=1.5)
    first_rec = tx[(tx["type"].astype(str) == "supplier_payment")].groupby("counterparty_id")["ts"].min()
    first_cash = acct[acct["type"] == "cash_withdrawal"].groupby("supplier_id")["ts"].min()
    F["sup_cash_lag_days"] = ((sid.map(first_cash) - sid.map(first_rec)) / DAY).fillna(400)
    F["sup_in_deg_x_cash"] = F["sup_in_degree"] * F["sup_cash_share"]   # combination feature
    F["sup_employee_link"] = sid.map(sup["linked_employee_id"]).notna().astype(float)
    oked = sid.map(sup["oked_code"])
    F["oked_mismatch"] = [float(o not in MATCHING_OKED[pu]) for o, pu in zip(oked, loans["purpose"].astype(str))]
    F["sup_recent"] = ((loans["disbursed_at"] - sid.map(sup["registered_at"])) / DAY < 365).astype(float)

    # --- geo / photo (inputs of ml/photo, included to check for leakage there too)
    dkm = 111.0 * np.hypot(loans["photo_lat"] - loans["borrower_lat"],
                           (loans["photo_lon"] - loans["borrower_lon"]) * np.cos(np.radians(loans["borrower_lat"])))
    F["photo_far_km"] = dkm
    F["photo_dup_count"] = loans["photo_id"].map(loans["photo_id"].value_counts()) - 1
    F["photo_gap_days"] = (loans["photo_captured_at"] - loans["disbursed_at"]) / DAY
    return F.replace([np.inf, -np.inf], np.nan).fillna(0)


def recall_at_budget(score: np.ndarray, y: np.ndarray, budget=BUDGET) -> float:
    """Recall in the top `budget` share; ties broken at random (expected value)."""
    n, B = len(y), budget * len(y)
    order = np.argsort(-score, kind="stable")
    s, yy = score[order], y[order]
    thr = s[int(B) - 1]
    above = s > thr
    tied = s == thr
    tp = yy[above].sum() + (B - above.sum()) * yy[tied].mean()
    return float(tp / y.sum())


def rule_score(F: pd.DataFrame) -> np.ndarray:
    return (F["cash_share_d1"] >= 0.5).astype(float).to_numpy()


def report(split: str) -> str:
    t = load(split)
    F, lab = features(t), t["labels"].set_index("loan_id")
    lab = lab.reindex(F.index)
    out = io.StringIO()
    w = lambda s="": out.write(s + "\n")
    w(f"=== LEAKAGE CHECK: {split} ({len(F)} loans, {int(lab.is_fraud.sum())} fraud, "
      f"{(lab.is_fraud & ~lab.fraud_visible).sum()} fraud events not yet visible at snapshot) ===")
    honest = ~lab.is_fraud
    w(f"\nSingle-feature AUC vs honest loans (sign-agnostic: max(AUC, 1-AUC); direction shown). "
      f"Flag threshold {FLAG_AUC}. Only fraud whose event is visible before the snapshot is used.")
    flagged, summary = [], []
    for typ in ("T1", "T2", "T3", "T4", "ALL"):
        pos = (lab.is_fraud & lab.fraud_visible) if typ == "ALL" else (lab.typology == typ) & lab.fraud_visible
        mask = pos | honest
        y = pos[mask].astype(int).to_numpy()
        rows = []
        for c in F.columns:
            x = F.loc[mask, c].to_numpy()
            if np.ptp(x) == 0:
                continue
            a = roc_auc_score(y, x)
            rows.append((max(a, 1 - a), "+" if a >= 0.5 else "-", c))
        rows.sort(reverse=True)
        summary.append((typ, int(pos.sum()), rows[0], dict((c, a) for a, _, c in rows).get('oked_mismatch')))
        w(f"\n[{typ}] n_fraud={int(pos.sum())}  top features:")
        for a, d, c in rows[:8]:
            flag = "  <-- LEAK? (>= %.2f)" % FLAG_AUC if a >= FLAG_AUC else ""
            w(f"   {c:<30s} AUC={a:.3f} ({d}){flag}")
            if a >= FLAG_AUC and typ != "ALL":
                flagged.append((typ, c, a))
    w(f"\nFeatures with single-feature AUC >= {FLAG_AUC}: "
      + (", ".join(f"{t}:{c}({a:.3f})" for t, c, a in flagged) if flagged else "none"))

    w("\nSUMMARY (best single-feature AUC per typology; visible fraud vs honest)")
    for typ, n, (a, d, c), oked_a in summary:
        extra = f"  oked_mismatch AUC={oked_a:.3f}" if typ == "T4" else ""
        w(f"   {typ:<3s} n={n:4d}  best={c} AUC={a:.3f}{extra}")

    # recall@20% of each single feature on all fraud (visible or not), best direction
    y_all = lab.is_fraud.to_numpy().astype(int)
    w(f"\nrecall@{int(BUDGET*100)}% of the best single features (all fraud incl. not-yet-visible; "
      f"random baseline = {BUDGET:.2f}):")
    res = []
    for c in F.columns:
        x = F[c].to_numpy()
        if np.ptp(x) == 0:
            continue
        a = roc_auc_score(y_all, x)
        res.append((recall_at_budget(x if a >= 0.5 else -x, y_all), c))
    for r, c in sorted(res, reverse=True)[:6]:
        w(f"   {c:<30s} recall@20%={r:.3f}")

    # simple rule
    rs = rule_score(F)
    flagged_mask = rs > 0
    w("\nSimple rule: '>= 50% of the loan withdrawn as cash within day 0-1 -> fraud'")
    w(f"   flagged {flagged_mask.sum()} loans ({flagged_mask.mean():.1%} of portfolio); "
      f"precision={y_all[flagged_mask].mean():.3f}; recall (flag only)={y_all[flagged_mask].sum() / y_all.sum():.3f}")
    w(f"   recall@20% budget (ties random)={recall_at_budget(rs, y_all):.3f}  "
      f"(random selection = {BUDGET:.2f})")
    w("   rule hit rate per group:")
    for typ in ("T1", "T2", "T3", "T4"):
        m = lab.typology == typ
        for stl in (False, True):
            mm = m & (lab.stealth == stl)
            if typ in ("T1", "T2") and mm.sum():
                w(f"      {typ} {'stealth' if stl else 'plain  '} n={int(mm.sum()):4d}  flagged={flagged_mask[mm.to_numpy()].mean():.1%}")
        if typ in ("T3", "T4"):
            w(f"      {typ}         n={int(m.sum()):4d}  flagged={flagged_mask[m.to_numpy()].mean():.1%}")
    for prof, g in lab[honest].groupby("profile"):
        w(f"      honest/{prof:<17s} n={len(g):4d}  flagged={flagged_mask[lab.index.get_indexer(g.index)].mean():.1%}")
    return out.getvalue()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test", "all"], default="all")
    args = ap.parse_args(argv)
    for s in (["train", "test"] if args.split == "all" else [args.split]):
        text = report(s)
        print(text)
        (OUT_ROOT / s / "leakage_report.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
