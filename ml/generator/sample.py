"""Write data/sample/sample_loan.json: one honest + one fraud loan for the backend.

    python -m ml.generator.sample

No labels / typology / generator-internal fields are included. Loans are picked
deterministically (first match by loan_id) from the train split.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .build import OUT_ROOT
from .check_leakage import load

SAMPLE_PATH = Path(__file__).resolve().parents[2] / "data" / "sample" / "sample_loan.json"


def _jsonable(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.to_json(orient="records", date_format="iso", date_unit="s"))


def loan_record(t: dict, loan_id: str) -> dict:
    loans, tx, sup = t["loans"], t["transactions"], t["suppliers"]
    loan = loans[loans.loan_id == loan_id].iloc[0]
    s = sup[sup.supplier_id == loan.supplier_id].iloc[0]
    in_degree = int((loans.supplier_id == loan.supplier_id).sum())
    acct = f"ACC-{loan.borrower_id}"
    txs = tx[tx.loan_id == loan_id].drop(columns=["loan_id"])
    sup_tx = tx[tx.account_id == f"ACC-{loan.supplier_id}"].drop(columns=["loan_id"])
    return {
        "loan": json.loads(loan.drop(["photo_id", "photo_lat", "photo_lon", "photo_captured_at"])
                           .to_frame().T.to_json(orient="records", date_format="iso", date_unit="s"))[0],
        "account_id": acct,
        "transactions": _jsonable(txs),
        "supplier": {**json.loads(s.to_frame().T.to_json(orient="records", date_format="iso", date_unit="s"))[0],
                     "in_degree": in_degree},
        "supplier_account_transactions": _jsonable(sup_tx),
        "photos": [{
            "photo_id": loan.photo_id, "lat": float(loan.photo_lat), "lon": float(loan.photo_lon),
            "captured_at": loan.photo_captured_at.isoformat(),
        }],
    }


def main():
    t = load("train")
    lab = t["labels"].set_index("loan_id")
    tx = t["transactions"]
    n_tx = tx[tx.loan_id.notna()].groupby("loan_id").size()
    inc = tx[tx["type"].astype(str) == "income"].groupby("loan_id").size()
    has_supplier_pay = set(tx[tx["type"].astype(str) == "supplier_payment"].loan_id)
    honest = lab[(lab.profile == "normal") & lab.index.isin(has_supplier_pay)
                 & lab.index.isin(inc[inc >= 2].index)].index
    fraud = lab[(lab.typology == "T1") & ~lab.stealth & lab.fraud_visible].index
    # pick loans with a readable, not-too-long history
    pick = lambda idx: sorted(i for i in idx if 5 <= n_tx.get(i, 0) <= 12)[0]
    out = {
        "_note": "Synthetic data (seed 42, train split). Two example loans: A and B. "
                 "No labels included; risk/reasons are produced by ml.api.",
        "snapshot_date": str(t["loans"].snapshot_date.iloc[0].date()),
        "loans": [loan_record(t, pick(honest)), loan_record(t, pick(fraud))],
    }
    SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SAMPLE_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {SAMPLE_PATH} ({SAMPLE_PATH.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
