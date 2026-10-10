"""Tests for ml.generator (small portfolios, no dependency on data/generated/)."""
from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.generator.build import build
from ml.generator.config import MATCHING_OKED, TEST, TRAIN

N = 500
FEATURE_TABLES = ("loans", "transactions", "suppliers", "employees")
LABEL_TOKENS = {"fraud", "typology", "profile", "stealth", "ring", "label", "labels",
                "visible"}
SAMPLE = Path(__file__).resolve().parents[2] / "data" / "sample" / "sample_loan.json"


def _tokens(name) -> set:
    return set(str(name).lower().replace("-", "_").split("_"))


@pytest.fixture(scope="module")
def train():
    return build(dataclasses.replace(TRAIN, n_loans=N))


@pytest.fixture(scope="module")
def test():
    return build(dataclasses.replace(TEST, n_loans=N))


# 1. determinism
def test_same_seed_identical(train):
    again = build(dataclasses.replace(TRAIN, n_loans=N))
    for name in (*FEATURE_TABLES, "labels", "photo_manifest"):
        pd.testing.assert_frame_equal(train[name], again[name])


def test_different_seed_differs(train):
    other = build(dataclasses.replace(TRAIN, n_loans=N, seed=TRAIN.seed + 1))
    assert not train["loans"].equals(other["loans"])
    assert not train["transactions"].equals(other["transactions"])
    assert not train["suppliers"].equals(other["suppliers"])


# 2. no label columns
@pytest.mark.parametrize("table", FEATURE_TABLES)
def test_no_label_columns(train, test, table):
    for t in (train, test):
        bad = [c for c in t[table].columns if _tokens(c) & LABEL_TOKENS]
        assert not bad, f"label-like columns in {table}: {bad}"


def test_no_label_values_in_categoricals(train):
    forbidden = {"T1", "T2", "T3", "T4", "livestock_seller", "cash_heavy"}
    for table in FEATURE_TABLES:
        df = train[table]
        for c in df.columns:
            if df[c].dtype == object or str(df[c].dtype) == "category":
                vals = set(map(str, df[c].dropna().unique()))
                assert not vals & forbidden, (table, c)


# 3. snapshot
@pytest.mark.parametrize("fx", ["train", "test"])
def test_snapshot_respected(request, fx):
    t = request.getfixturevalue(fx)
    snap = pd.Timestamp((TRAIN if fx == "train" else TEST).snapshot_date)
    assert (t["loans"].snapshot_date == snap).all()
    assert t["transactions"].ts.max() <= snap
    assert t["loans"].disbursed_at.max() <= snap
    assert t["loans"].photo_captured_at.max() <= snap


# 4. one disbursement per loan
@pytest.mark.parametrize("fx", ["train", "test"])
def test_one_disbursement_per_loan(request, fx):
    t = request.getfixturevalue(fx)
    loans, tx = t["loans"], t["transactions"]
    d = tx[tx["type"].astype(str) == "disbursement"]
    assert len(d) == len(loans)
    assert d.loan_id.is_unique
    assert set(d.loan_id) == set(loans.loan_id)
    m = d.merge(loans, on="loan_id", suffixes=("_t", "_l"))
    assert (m.ts == m.disbursed_at).all()
    assert (m.amount_uzs_t == m.amount_uzs_l).all()


# 5. unique ids and foreign keys
@pytest.mark.parametrize("fx", ["train", "test"])
def test_unique_ids_and_fks(request, fx):
    t = request.getfixturevalue(fx)
    loans, tx, sup, emp = t["loans"], t["transactions"], t["suppliers"], t["employees"]
    assert loans.loan_id.is_unique and loans.borrower_id.is_unique
    assert tx.txn_id.is_unique
    assert sup.supplier_id.is_unique and emp.employee_id.is_unique
    assert loans.supplier_id.isin(sup.supplier_id).all()
    assert tx.loan_id.dropna().isin(loans.loan_id).all()
    assert sup.linked_employee_id.dropna().isin(emp.employee_id).all()
    assert loans.branch_id.isin(emp.branch_id).all()
    lt = tx[tx.loan_id.notna()].merge(loans[["loan_id", "borrower_id"]], on="loan_id")
    assert (lt.account_id == "ACC-" + lt.borrower_id).all()
    assert set(sup.supplier_id) == set(loans.supplier_id)
    assert set(t["labels"].loan_id) == set(loans.loan_id)
    assert t["photo_manifest"].photo_id.isin(loans.photo_id).all()


# 6. disjoint pools
def test_train_test_disjoint(train, test):
    for table, col in (("suppliers", "supplier_id"), ("loans", "loan_id"),
                       ("loans", "borrower_id"), ("employees", "employee_id")):
        assert not set(train[table][col]) & set(test[table][col]), (table, col)
    assert not set(train["transactions"].txn_id) & set(test["transactions"].txn_id)
    assert not set(train["loans"].photo_id) & set(test["loans"].photo_id)


# 7. fraud rate and typologies
@pytest.mark.parametrize("fx,params", [("train", TRAIN), ("test", TEST)])
def test_fraud_rate_and_typologies(request, fx, params):
    t = request.getfixturevalue(fx)
    lab = t["labels"]
    rate = lab.is_fraud.mean()
    assert 0.08 <= rate <= 0.15
    assert abs(rate - params.fraud_rate) < 0.01
    assert set(lab.typology.dropna()) == {"T1", "T2", "T3", "T4"}
    assert lab.typology.isna().sum() == (~lab.is_fraud).sum()


@pytest.mark.parametrize("fx", ["train", "test"])
def test_t3_in_chorva_and_tikuv(request, fx):
    t = request.getfixturevalue(fx)
    m = t["labels"].merge(t["loans"][["loan_id", "purpose"]], on="loan_id")
    t3 = set(m[m.typology == "T3"].purpose.astype(str))
    assert {"chorva", "tikuv_dastgohi"} <= t3


# 8. honest OKED mismatch
@pytest.mark.parametrize("fx,params", [("train", TRAIN), ("test", TEST)])
def test_honest_oked_mismatch_rate(request, fx, params):
    t = request.getfixturevalue(fx)
    m = (t["labels"].merge(t["loans"][["loan_id", "purpose", "supplier_id"]], on="loan_id")
         .merge(t["suppliers"][["supplier_id", "oked_code"]], on="supplier_id"))
    h = m[~m.is_fraud]
    mism = np.array([o not in MATCHING_OKED[str(p)] for o, p in zip(h.oked_code, h.purpose)])
    rate = mism.mean()
    assert rate > 0
    assert 0.04 <= rate <= 0.11, rate
    assert abs(rate - params.honest_mismatch) < 0.035


# 9. amounts
@pytest.mark.parametrize("fx", ["train", "test"])
def test_amounts_int64_positive(request, fx):
    t = request.getfixturevalue(fx)
    for df, col in ((t["loans"], "amount_uzs"), (t["loans"], "invoice_amount_uzs"),
                    (t["transactions"], "amount_uzs")):
        assert df[col].dtype == np.int64, (col, df[col].dtype)
        assert (df[col] > 0).all()


def _min_balances(tx: pd.DataFrame) -> pd.Series:
    """Replay every account (borrowers and suppliers) in time order, inflows first at equal ts."""
    tx = tx.copy()
    inflow = tx["type"].astype(str).isin(["disbursement", "p2p_in", "income"])
    tx["signed"] = np.where(inflow, tx.amount_uzs, -tx.amount_uzs)
    tx["is_out"] = ~inflow
    tx = tx.sort_values(["account_id", "ts", "is_out"], kind="stable")
    return tx.groupby("account_id").signed.cumsum().groupby(tx.account_id).min()


@pytest.mark.parametrize("fx", ["train", "test"])
def test_no_account_goes_negative(request, fx):
    t = request.getfixturevalue(fx)
    tx = t["transactions"]
    mins = _min_balances(tx)
    assert (mins >= 0).all(), f"{int((mins < 0).sum())} accounts negative, min {int(mins.min())} UZS"
    # supplier accounts are part of the ledger and have an inflow for every settled supplier payment
    sup_acc = "ACC-" + t["suppliers"].supplier_id   # suppliers never paid have no account rows
    paid = tx[tx["type"].astype(str) == "supplier_payment"].groupby("counterparty_id").amount_uzs.sum()
    got = tx[tx.account_id.isin(sup_acc) & (tx["type"].astype(str) == "income")].assign(
        sid=lambda d: d.account_id.str[len("ACC-"):]).groupby("sid").amount_uzs.sum()
    assert (paid.sort_index() == got.reindex(paid.index).sort_index()).all()


# ring loans are drawn from the same date range as everyone else
@pytest.mark.parametrize("fx", ["train", "test"])
def test_loan_age_within_range_and_term(request, fx):
    loans = request.getfixturevalue(fx)["loans"]
    age = (loans.snapshot_date - loans.disbursed_at) / pd.Timedelta(days=1)
    assert age.max() <= 365 and age.min() >= 2
    assert (age <= loans.term_months * 30.4375).all()


@pytest.mark.parametrize("fx", ["train", "test"])
def test_ring_loans_not_older_than_others(request, fx):
    t = request.getfixturevalue(fx)
    m = t["labels"].merge(t["loans"], on="loan_id")
    age = (m.snapshot_date - m.disbursed_at) / pd.Timedelta(days=1)
    ring = m.ring_id.notna()
    assert ring.any()
    assert age[ring].max() <= age[~ring].max() + 1


# reused photo: only the image is shared; time and place belong to the loan
@pytest.mark.parametrize("fx", ["train", "test"])
def test_photo_reuse_keeps_own_time_and_place(request, fx):
    t = request.getfixturevalue(fx)
    loans, lab = t["loans"], t["labels"]
    assert (loans.photo_captured_at >= loans.disbursed_at).all()
    dup = loans[loans.photo_id.duplicated(keep=False)]
    assert len(dup) > 0 and lab.photo_reused.any()
    same_place = dup.groupby("photo_id").apply(
        lambda g: g[["photo_lat", "photo_lon"]].drop_duplicates().shape[0] == 1, include_groups=False)
    assert same_place.mean() < 0.2
    same_time = dup.groupby("photo_id").photo_captured_at.nunique() == 1
    assert same_time.mean() < 0.2


@pytest.fixture(scope="module")
def big_train():
    return build(TRAIN)    # full 5000 loans (~1 s) for statistical checks


def test_t2_amount_matches_purpose(big_train):
    from ml.generator.config import MEDIAN_AMOUNT
    m = big_train["labels"].merge(big_train["loans"], on="loan_id")
    dev = np.abs(np.log(m.amount_uzs / m.purpose.astype(str).map(MEDIAN_AMOUNT)))
    t2, honest = dev[m.typology == "T2"], dev[~m.is_fraud]
    assert len(t2) > 100
    assert (t2 > 0.9).mean() < 0.035 and (honest > 0.9).mean() < 0.035


def test_t4_supplier_in_degree_not_a_fingerprint(big_train):
    m = big_train["labels"].merge(big_train["loans"], on="loan_id")
    deg = m.supplier_id.map(m.supplier_id.value_counts())
    t4, honest = deg[m.typology == "T4"], deg[~m.is_fraud]
    assert 0.7 < t4.median() / honest.median() < 1.4
    assert 0.7 < t4.mean() / honest.mean() < 1.4


def test_honest_employee_links_point_to_borrower_branch(big_train):
    m = (big_train["labels"].merge(big_train["loans"], on="loan_id")
         .merge(big_train["suppliers"][["supplier_id", "linked_employee_id"]], on="supplier_id")
         .merge(big_train["employees"], left_on="linked_employee_id", right_on="employee_id",
                suffixes=("", "_emp")))
    h = m[m.typology.isna()]
    by_sup = h.groupby("supplier_id").apply(
        lambda g: g.branch_id_emp.iloc[0] in set(g.branch_id), include_groups=False)
    assert len(by_sup) >= 10
    assert 0.5 <= by_sup.mean() <= 0.95


# 10. sample JSON
def _walk_keys(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield f"{path}/{k}", k
            yield from _walk_keys(v, f"{path}/{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from _walk_keys(v, f"{path}[{i}]")


def test_sample_json_no_labels():
    data = json.loads(SAMPLE.read_text(encoding="utf-8"))
    assert data["loans"], "sample has no loans"
    bad = [p for p, k in _walk_keys(data) if _tokens(k) & LABEL_TOKENS]
    assert not bad, bad


def test_sample_json_structure():
    data = json.loads(SAMPLE.read_text(encoding="utf-8"))
    for rec in data["loans"]:
        assert {"loan", "transactions", "supplier", "photos"} <= set(rec)
        assert rec["loan"]["loan_id"]


# 11. runtime
@pytest.mark.slow
def test_runtime_5000_loans():
    t0 = time.perf_counter()
    out = build(dataclasses.replace(TRAIN, n_loans=5000))
    assert time.perf_counter() - t0 < 60
    assert len(out["loans"]) == 5000
