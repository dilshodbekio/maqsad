"""Write data/sample/sample_loan.json: two example loans for the backend.

    python -m ml.generator.sample

No labels / typology / generator-internal fields are included. One loan is drawn at random from the
honest loans and one from the visible-fraud loans of the train split (both with a readable history of
5-12 transactions; fixed seed), and the pair is shuffled so the file order carries no information.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .check_leakage import load

SEED = 20260401
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
    rng = np.random.default_rng(SEED)
    readable = set(n_tx[(n_tx >= 5) & (n_tx <= 12)].index)
    honest = sorted(set(lab[~lab.is_fraud].index) & readable)
    fraud = sorted(set(lab[lab.is_fraud & lab.fraud_visible].index) & readable)
    picks = [honest[rng.integers(len(honest))], fraud[rng.integers(len(fraud))]]
    rng.shuffle(picks)
    out = {
        "_note": "Synthetic data (train split, seed 42). Risk and reasons are produced by ml.api.",
        "snapshot_date": str(t["loans"].snapshot_date.iloc[0].date()),
        "loans": [loan_record(t, i) for i in picks],
    }
    SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SAMPLE_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {SAMPLE_PATH} ({SAMPLE_PATH.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
