"""Leakage / trivial-separability check for a generated split.

    python -m ml.generator.check_leakage --split train

Sections of the report (honest loans are the negative class everywhere, visible fraud the positive):
  A. money-trail features        - engineered, single-feature AUC per typology (+ recall@20%)
  B. raw columns of all tables   - every column of loans / transactions / suppliers / employees plus
                                   derived ones (loan age, amount vs purpose, employee branch == loan
                                   branch, ...), single-feature AUC per typology
  C. depth-2 decision tree       - per typology on the raw columns of B (catches 2-column combinations)
  D. photo module inputs         - photo / geo columns; NOT used by the money-trail model
Anything with AUC >= 0.90 is flagged, except the intended T1 cash features. Features use only data up
to the snapshot; labels are used only for scoring.
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.tree import DecisionTreeClassifier, export_text

from .build import OUT_ROOT
from ml.money_trail.features import build_features
from ml.money_trail.metrics import BUDGET, recall_at_budget

from .config import MATCHING_OKED

DAY = pd.Timedelta(days=1)
FLAG_AUC = 0.90
TYPOLOGIES = ("T1", "T2", "T3", "T4")

# photo / geo inputs: evaluated separately, never used by the money-trail model
PHOTO_ENGINEERED = ["photo_far_km", "photo_dup_count", "photo_gap_days"]
PHOTO_RAW = ["photo_lat", "photo_lon", "borrower_lat", "borrower_lon", "photo_id_num",
             "photo_age_days", "photo_gap_days", "photo_gap_neg", "photo_far_km",
             "photo_coord_dup_count", "borrower_coord_dup_count", "photo_id_dup_count"]
# the features a T1 (fast cash-out) detector is SUPPOSED to separate on - not flagged for T1:
# fast cash-out, and "no supplier payment" (T1's definition; invoice_gap is its complement), and the
# timing of the first outflow
INTENDED_T1 = {"first_cash_days", "cash_share_d1", "cash_share_d7", "cash_share_d30",
               "cash_share_all", "leave_share_30", "n_cash_withdrawal", "share_cash_withdrawal",
               "sup_pay_share", "invoice_gap", "share_supplier_payment", "first_tx_day"}


def load(split: str, root: Path = OUT_ROOT) -> dict:
    d = root / split
    return {n: pd.read_parquet(d / f"{n}.parquet")
            for n in ("loans", "transactions", "suppliers", "employees", "labels")}


def _id_num(s: pd.Series) -> pd.Series:
    return s.astype(str).str.extract(r"(\d+)$")[0].astype(float)


def features(t: dict) -> pd.DataFrame:
    """Money-trail features (single source: ml.money_trail.features) + the photo module inputs."""
    F = build_features(t["loans"], t["transactions"], t["suppliers"], t["employees"])
    loans = t["loans"].set_index("loan_id")
    # --- geo / photo (inputs of ml/photo; reported in a separate section)
    dkm = 111.0 * np.hypot(loans["photo_lat"] - loans["borrower_lat"],
                           (loans["photo_lon"] - loans["borrower_lon"]) * np.cos(np.radians(loans["borrower_lat"])))
    F["photo_far_km"] = dkm
    F["photo_dup_count"] = loans["photo_id"].map(loans["photo_id"].value_counts()) - 1
    F["photo_gap_days"] = (loans["photo_captured_at"] - loans["disbursed_at"]) / DAY
    return F.replace([np.inf, -np.inf], np.nan).fillna(0)


def raw_features(t: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every column of loans / transactions / suppliers / employees as a per-loan numeric column
    (numeric columns as they are, ids -> trailing number, timestamps -> age in days, transactions
    aggregated per loan) plus derived columns. Returns (numeric, categorical)."""
    loans, tx = t["loans"], t["transactions"]
    sup, emp = t["suppliers"].set_index("supplier_id"), t["employees"].set_index("employee_id")
    L = loans.set_index("loan_id")
    snap = L["snapshot_date"].iloc[0]
    sid, amt = L["supplier_id"], L["amount_uzs"]
    n, c = {}, {}

    # ---- loans (raw)
    n["loan_row_pos"] = pd.Series(np.arange(len(L)), index=L.index)
    n["loan_id_num"] = _id_num(L.index.to_series())
    n["borrower_id_num"] = _id_num(L["borrower_id"])
    n["supplier_id_num"] = _id_num(sid)
    n["photo_id_num"] = _id_num(L["photo_id"])
    n["branch_num"] = _id_num(L["branch_id"])
    n["amount_uzs"], n["invoice_amount_uzs"] = amt, L["invoice_amount_uzs"]
    n["term_months"] = L["term_months"].astype(float)
    for col in ("borrower_lat", "borrower_lon", "photo_lat", "photo_lon"):
        n[col] = L[col].astype(float)
    n["loan_age_days"] = (snap - L["disbursed_at"]) / DAY
    n["photo_age_days"] = (snap - L["photo_captured_at"]) / DAY
    c["borrower_name_masked"], c["purpose"] = L["borrower_name_masked"], L["purpose"]
    c["region"], c["branch_id"] = L["region"], L["branch_id"]

    # ---- loans (derived)
    n["age_over_term"] = n["loan_age_days"] / (L["term_months"] * 30.4375)
    n["photo_gap_days"] = (L["photo_captured_at"] - L["disbursed_at"]) / DAY
    n["photo_gap_neg"] = (n["photo_gap_days"] < 0).astype(float)
    n["photo_far_km"] = 111.0 * np.hypot(L["photo_lat"] - L["borrower_lat"],
                                         (L["photo_lon"] - L["borrower_lon"]) * np.cos(np.radians(L["borrower_lat"])))
    pair = L["photo_lat"].astype(str) + "|" + L["photo_lon"].astype(str)
    n["photo_coord_dup_count"] = pair.map(pair.value_counts()) - 1
    bpair = L["borrower_lat"].astype(str) + "|" + L["borrower_lon"].astype(str)
    n["borrower_coord_dup_count"] = bpair.map(bpair.value_counts()) - 1
    n["photo_id_dup_count"] = L["photo_id"].map(L["photo_id"].value_counts()) - 1
    med = amt.groupby(L["purpose"].astype(str)).transform("median")
    n["amount_vs_purpose"] = np.log(amt / med)
    n["invoice_ratio"] = L["invoice_amount_uzs"] / amt

    # ---- suppliers + employees (raw + derived)
    n["sup_in_degree"] = sid.map(sid.value_counts())
    n["sup_age_at_disb_days"] = (L["disbursed_at"] - sid.map(sup["registered_at"])) / DAY
    n["sup_age_at_snapshot_days"] = (snap - sid.map(sup["registered_at"])) / DAY
    c["sup_oked_code"], c["sup_region"] = sid.map(sup["oked_code"]), sid.map(sup["region"])
    c["sup_name"] = sid.map(sup["name"])
    linked = sid.map(sup["linked_employee_id"])
    n["sup_employee_link"] = linked.notna().astype(float)
    n["emp_id_num"] = _id_num(linked.astype("object").where(linked.notna(), None)).fillna(-1)
    n["emp_branch_match"] = (linked.map(emp["branch_id"]) == L["branch_id"]).astype(float)
    c["emp_role"] = linked.map(emp["role"]).astype("object").fillna("none")
    n["sup_region_match"] = (sid.map(sup["region"]).astype(str) == L["region"].astype(str)).astype(float)
    n["oked_mismatch"] = [float(o not in MATCHING_OKED[pu])
                          for o, pu in zip(c["sup_oked_code"], L["purpose"].astype(str))]

    # ---- transactions of the borrower account, aggregated per loan
    bt = tx[tx.loan_id.notna()].copy()
    bt["type"] = bt["type"].astype(str)
    bt["row_pos"] = bt.index
    bt["day"] = (bt["ts"] - bt["loan_id"].map(L["disbursed_at"])) / DAY
    g = bt.groupby("loan_id")
    ix = L.index
    n["n_tx"] = g.size().reindex(ix).fillna(0)
    n["n_tx_types"] = g["type"].nunique().reindex(ix).fillna(0)
    for typ in ("disbursement", "cash_withdrawal", "supplier_payment", "p2p_out", "p2p_in",
                "external_transfer", "income"):
        d = bt[bt["type"] == typ].groupby("loan_id")["amount_uzs"]
        n[f"n_{typ}"] = d.size().reindex(ix).fillna(0)
        n[f"share_{typ}"] = (d.sum().reindex(ix).fillna(0) / amt)
    first = bt[bt["type"] != "disbursement"].groupby("loan_id")["day"]
    n["first_tx_day"] = first.min().reindex(ix).fillna(400)
    n["last_tx_day"] = g["day"].max().reindex(ix).fillna(0)
    n["tx_amount_mean_ratio"] = g["amount_uzs"].mean().reindex(ix).fillna(0) / amt
    n["tx_amount_max_ratio"] = g["amount_uzs"].max().reindex(ix).fillna(0) / amt
    n["share_round_100k"] = (bt["amount_uzs"] % 100_000 == 0).groupby(bt["loan_id"]).mean().reindex(ix).fillna(0)
    n["share_round_1m"] = (bt["amount_uzs"] % 1_000_000 == 0).groupby(bt["loan_id"]).mean().reindex(ix).fillna(0)
    n["tx_hour_mean"] = bt["ts"].dt.hour.groupby(bt["loan_id"]).mean().reindex(ix).fillna(0)
    n["tx_weekend_share"] = (bt["ts"].dt.dayofweek >= 5).groupby(bt["loan_id"]).mean().reindex(ix).fillna(0)
    n["tx_row_pos_mean"] = g["row_pos"].mean().reindex(ix).fillna(0)
    n["txn_id_num_mean"] = _id_num(bt["txn_id"]).groupby(bt["loan_id"]).mean().reindex(ix).fillna(0)
    cp = bt[bt["counterparty_id"].notna() & (bt["counterparty_kind"].astype(str) != "supplier")]
    n["n_counterparties"] = cp.groupby("loan_id")["counterparty_id"].nunique().reindex(ix).fillna(0)
    shared = cp["counterparty_id"].map(cp.groupby("counterparty_id")["loan_id"].nunique()) > 1
    n["n_shared_counterparty_tx"] = shared.groupby(cp["loan_id"]).sum().reindex(ix).fillna(0)
    # running balance of the borrower account (inflows first at equal timestamps)
    bt["signed"] = np.where(bt["type"].isin(["disbursement", "p2p_in", "income"]), bt["amount_uzs"], -bt["amount_uzs"])
    bt["is_out"] = bt["signed"] < 0
    bt = bt.sort_values(["loan_id", "ts", "is_out"], kind="stable")
    bal = bt.groupby("loan_id")["signed"].cumsum()
    n["min_balance_ratio"] = (bal.groupby(bt["loan_id"]).min().reindex(ix).fillna(0) / amt)
    n["final_balance_ratio"] = (bal.groupby(bt["loan_id"]).last().reindex(ix).fillna(0) / amt)

    # ---- supplier account ledger (loan_id is null)
    sa = tx[tx.loan_id.isna()].copy()
    sa["supplier_id"] = sa["account_id"].str.replace("ACC-", "", regex=False)
    sa["type"] = sa["type"].astype(str)
    out = sa[sa["type"] != "income"]
    n["sup_n_outflows"] = sid.map(out.groupby("supplier_id").size()).fillna(0)
    n["sup_n_cash_withdrawals"] = sid.map(out[out["type"] == "cash_withdrawal"].groupby("supplier_id").size()).fillna(0)
    n["sup_n_inflows"] = sid.map(sa[sa["type"] == "income"].groupby("supplier_id").size()).fillna(0)
    n["sup_tx_ts_span_days"] = sid.map((sa.groupby("supplier_id")["ts"].max() - sa.groupby("supplier_id")["ts"].min()) / DAY).fillna(0)

    N = pd.DataFrame(n).replace([np.inf, -np.inf], np.nan).fillna(0).astype(float)
    C = pd.DataFrame({k: v.astype(str) for k, v in c.items()})
    return N, C


def oof_target_encode(cat: pd.Series, y: np.ndarray, folds, m: float = 10.0) -> np.ndarray:
    """Out-of-fold smoothed target mean of a categorical column (single-feature AUC of a nominal)."""
    c, out = cat.to_numpy(), np.zeros(len(y))
    for tr, va in folds:
        prior = y[tr].mean()
        s = pd.DataFrame({"c": c[tr], "y": y[tr]}).groupby("c")["y"].agg(["sum", "count"])
        enc = (s["sum"] + m * prior) / (s["count"] + m)
        out[va] = pd.Series(c[va]).map(enc).fillna(prior).to_numpy()
    return out


def sym_auc(y, x):
    a = roc_auc_score(y, x)
    return max(a, 1 - a), ("+" if a >= 0.5 else "-")


def rule_score(F: pd.DataFrame) -> np.ndarray:
    return (F["cash_share_d1"] >= 0.5).astype(float).to_numpy()


def _rank(F: pd.DataFrame, y: np.ndarray, cols):
    rows = []
    for col in cols:
        x = F[col].to_numpy()
        if np.ptp(x) == 0:
            continue
        a, d = sym_auc(y, x)
        rows.append((a, d, col))
    return sorted(rows, reverse=True)


def report(split: str) -> str:
    t = load(split)
    F, lab = features(t), t["labels"].set_index("loan_id")
    lab = lab.reindex(F.index)
    N, C = raw_features(t)
    N, C = N.reindex(F.index), C.reindex(F.index)
    L_sup = t["loans"].set_index("loan_id")["supplier_id"].reindex(F.index)
    out = io.StringIO()
    w = lambda s="": out.write(s + "\n")
    w(f"=== LEAKAGE CHECK: {split} ({len(F)} loans, {int(lab.is_fraud.sum())} fraud, "
      f"{(lab.is_fraud & ~lab.fraud_visible).sum()} fraud events not yet visible at snapshot) ===")
    w(f"AUC = sign-agnostic max(AUC, 1-AUC) of one feature, typology vs honest loans; only fraud whose event "
      f"is visible before the snapshot is positive. Flag threshold {FLAG_AUC}; intended T1 cash features are not flagged.")
    honest = ~lab.is_fraud
    money_cols = [c for c in F.columns if c not in PHOTO_ENGINEERED]
    raw_num_money = [c for c in N.columns if c not in PHOTO_RAW]
    raw_num_photo = [c for c in N.columns if c in PHOTO_RAW]
    flags = []                                    # (section, typ, name, auc)
    best = {}                                     # typ -> dict of summary numbers

    def mask_y(typ):
        pos = (lab.is_fraud & lab.fraud_visible) if typ == "ALL" else (lab.typology == typ) & lab.fraud_visible
        mask = (pos | honest).to_numpy()
        return mask, pos.to_numpy()[mask].astype(int), int(pos.sum())

    # ------------------------------------------------------------------ A
    w("\n" + "=" * 78 + "\nA. MONEY-TRAIL FEATURES (engineered, single-feature AUC)\n" + "=" * 78)
    for typ in (*TYPOLOGIES, "ALL"):
        mask, y, npos = mask_y(typ)
        rows = _rank(F[mask].reset_index(drop=True), y, money_cols)
        best.setdefault(typ, {})["money"] = rows[0]
        w(f"\n[{typ}] n_fraud={npos}  top features:")
        for a, d, c in rows[:8]:
            bad = a >= FLAG_AUC and not (typ == "T1" and c in INTENDED_T1) and typ != "ALL"
            if bad:
                flags.append(("A money-trail", typ, c, a))
            w(f"   {c:<30s} AUC={a:.3f} ({d})" + ("  <-- FLAG" if bad else
              "  (intended T1 cash feature)" if a >= FLAG_AUC and typ == "T1" and c in INTENDED_T1 else ""))
        if typ == "T4":
            oa = dict((c, a) for a, _, c in rows).get("oked_mismatch")
            if oa is not None:
                w(f"   (oked_mismatch AUC={oa:.3f})")
        if typ == "T2":
            for name in ("emp_branch_match", "sup_employee_link"):
                x = F.loc[mask, name].to_numpy()
                prec = y[x > 0].mean() if (x > 0).any() else float("nan")
                w(f"   {name}: flagged loans={int((x > 0).sum())}  precision={prec:.3f}  recall={y[x > 0].sum() / y.sum():.3f}")

    # ------------------------------------------------------------------ B
    w("\n" + "=" * 78 + "\nB. RAW COLUMNS of loans / transactions / suppliers / employees + derived "
      "(single-feature AUC; categoricals = out-of-fold target encoding)\n" + "=" * 78)
    folds_cache = {}
    enc_cache = {}
    for typ in TYPOLOGIES:
        mask, y, npos = mask_y(typ)
        Nm = N[raw_num_money][mask].reset_index(drop=True)
        # folds are grouped by supplier (a ring supplier must not sit in both train and validation folds,
        # otherwise supplier name / OKED encodings "learn" the label through the supplier identity)
        groups = L_sup[mask].to_numpy()
        folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(np.zeros(len(y)), y, groups))
        folds_cache[typ] = folds
        TE = pd.DataFrame({k: oof_target_encode(C.loc[mask, k].reset_index(drop=True), y, folds)
                           for k in C.columns})
        enc_cache[typ] = TE
        rows = _rank(pd.concat([Nm, TE.add_prefix("te_")], axis=1), y, list(Nm.columns) + ["te_" + k for k in TE.columns])
        best[typ]["raw"] = rows[0]
        w(f"\n[{typ}] n_fraud={npos}  top raw/derived columns:")
        for a, d, c in rows[:8]:
            bad = a >= FLAG_AUC and not (typ == "T1" and c in INTENDED_T1)
            if bad:
                flags.append(("B raw", typ, c, a))
            w(f"   {c:<30s} AUC={a:.3f} ({d})" + ("  <-- FLAG" if bad else
              "  (intended T1 cash feature)" if a >= FLAG_AUC else ""))

    # ------------------------------------------------------------------ C
    w("\n" + "=" * 78 + "\nC. DEPTH-2 DECISION TREE on the raw/derived columns of B "
      "(5-fold CV AUC; tree fitted on all rows is printed)\n" + "=" * 78)
    for typ in TYPOLOGIES:
        mask, y, npos = mask_y(typ)
        X = pd.concat([N[raw_num_money][mask].reset_index(drop=True),
                       enc_cache[typ].add_prefix("te_")], axis=1)
        tree = DecisionTreeClassifier(max_depth=2, min_samples_leaf=20, random_state=0)
        p = cross_val_predict(tree, X, y, cv=folds_cache[typ], method="predict_proba")[:, 1]
        auc = roc_auc_score(y, p)
        tree.fit(X, y)
        used = {X.columns[i] for i in tree.tree_.feature if i >= 0}
        best[typ]["tree"] = (auc, sorted(used))
        bad = auc >= FLAG_AUC and not (typ == "T1" and used <= INTENDED_T1)
        if bad:
            flags.append(("C tree", typ, "+".join(sorted(used)), auc))
        w(f"\n[{typ}] CV AUC={auc:.3f}  columns used: {', '.join(sorted(used)) or '-'}" + ("  <-- FLAG" if bad else ""))
        w("   " + export_text(tree, feature_names=list(X.columns), decimals=2).replace("\n", "\n   "))

    # ------------------------------------------------------------------ D
    w("\n" + "=" * 78 + "\nD. PHOTO MODULE INPUTS - not used by the money-trail model "
      "(photo / geo columns; evaluated separately, combined with the money-trail score later)\n" + "=" * 78)
    Pcols = sorted(set(PHOTO_ENGINEERED) | set(raw_num_photo))
    P = pd.concat([F[PHOTO_ENGINEERED], N[[c for c in raw_num_photo if c not in PHOTO_ENGINEERED]]], axis=1)
    for typ in (*TYPOLOGIES, "ALL"):
        mask, y, npos = mask_y(typ)
        rows = _rank(P[mask].reset_index(drop=True), y, list(P.columns))
        best.setdefault(typ, {})["photo"] = rows[0]
        w(f"\n[{typ}] n_fraud={npos}  top photo/geo columns:")
        for a, d, c in rows[:6]:
            bad = a >= FLAG_AUC and typ != "ALL"
            if bad:
                flags.append(("D photo", typ, c, a))
            w(f"   {c:<30s} AUC={a:.3f} ({d})" + ("  <-- FLAG" if bad else ""))

    w("\n" + "-" * 78 + f"\nFLAGGED (AUC >= {FLAG_AUC}, intended T1 cash features excluded): "
      + ("none" if not flags else ""))
    for sec, typ, c, a in flags:
        w(f"   [{sec}] {typ}: {c} AUC={a:.3f}")

    # ------------------------------------------------------------------ summary + recall + rule
    w("\n" + "=" * 78 + "\nSUMMARY per typology (visible fraud vs honest)\n" + "=" * 78)
    for typ in TYPOLOGIES:
        a, d, c = best[typ]["money"]
        ra, rd, rc = best[typ]["raw"]
        ta, used = best[typ]["tree"]
        pa, pd_, pc = best[typ]["photo"]
        w(f"   {typ}: best money-trail feature {c} AUC={a:.3f} | best raw column {rc} AUC={ra:.3f} | "
          f"depth-2 tree CV AUC={ta:.3f} | [photo module] {pc} AUC={pa:.3f}")

    y_all = lab.is_fraud.to_numpy().astype(int)
    w(f"\nrecall@{int(BUDGET*100)}% of the best single MONEY-TRAIL features (all fraud incl. not-yet-visible; "
      f"random = {BUDGET:.2f}):")
    for title, cols in (("money-trail", money_cols),
                        ("photo module inputs - not used by money-trail model", PHOTO_ENGINEERED)):
        res = []
        for c in cols:
            x = F[c].to_numpy()
            if np.ptp(x) == 0:
                continue
            a = roc_auc_score(y_all, x)
            res.append((recall_at_budget(x if a >= 0.5 else -x, y_all), c))
        w(f"  [{title}]")
        for r, c in sorted(res, reverse=True)[:5 if cols is money_cols else 3]:
            w(f"   {c:<30s} recall@20%={r:.3f}")

    rs = rule_score(F)
    flagged_mask = rs > 0
    w("\nSimple rule: '>= 50% of the loan withdrawn as cash within day 0-1 -> fraud'")
    w(f"   flagged {flagged_mask.sum()} loans ({flagged_mask.mean():.1%} of portfolio); "
      f"precision={y_all[flagged_mask].mean():.3f}; recall (flag only)={y_all[flagged_mask].sum() / y_all.sum():.3f}")
    w(f"   recall@20% budget (ties random)={recall_at_budget(rs, y_all):.3f}  "
      f"(random selection = {BUDGET:.2f})")
    w("   rule hit rate per group:")
    for typ in TYPOLOGIES:
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
