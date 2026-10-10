"""Serving side of the money-trail model (used by ml/api.py).

The deployed service has no data/generated/ (gitignored). On first use the ML package generates the
demo portfolio in memory - the DEMO split config (seed 7), deterministic, ~1 s - computes the
money-trail features once and scores every loan with the trained model from ml/artifacts/. Labels from
the generator are dropped immediately and never reach the API. Nothing here is hard-coded: every risk,
colour and reason comes from the model and the features.
"""
from __future__ import annotations

import threading
import time
from functools import cached_property

import numpy as np
import pandas as pd

from ml.errors import LoanNotFoundError
from ml.generator.build import build
from ml.generator.config import MATCHING_OKED, SPLITS

from .features import build_features
from .colors import color_of, rank_thresholds
from .model import MODEL_VERSION, MoneyTrailModel, load_model
from .reasons import NO_REASON_UZ, TEMPLATES, build_evidence, build_reasons

DEMO_SPLIT = "demo"
LOAN_FIELDS = ["loan_id", "borrower_id", "borrower_name_masked", "purpose", "amount_uzs", "term_months",
               "disbursed_at", "snapshot_date", "region", "branch_id", "supplier_id", "invoice_amount_uzs"]
TXN_FIELDS = ["txn_id", "ts", "type", "amount_uzs", "counterparty_id", "counterparty_kind"]


def _py(v):
    """numpy / pandas scalar -> JSON-friendly Python value."""
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return None
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    return v.item() if hasattr(v, "item") else v


class Portfolio:
    def __init__(self):
        t0 = time.perf_counter()
        tables = build(SPLITS[DEMO_SPLIT])
        tables.pop("labels")                                       # evaluation-only; never served
        self.loans = tables["loans"].set_index("loan_id")
        self.suppliers = tables["suppliers"].set_index("supplier_id")
        self.tx = tables["transactions"]
        self.model: MoneyTrailModel = load_model()
        self.features = build_features(tables["loans"], self.tx, tables["suppliers"], tables["employees"])
        self.risk = pd.Series(self.model.score(self.features), index=self.features.index)
        contrib = pd.DataFrame(self.model.contributions(self.features), index=self.features.index,
                               columns=self.model.features)
        self._contrib = contrib
        self.thresholds = rank_thresholds(self.risk.to_numpy())      # colours by queue rank on THIS portfolio
        self.init_s = time.perf_counter() - t0
        self._bt = self.tx[self.tx.loan_id.notna()]
        self._queue = self._make_queue()

    # ------------------------------------------------------------------ helpers
    def _check(self, loan_id: str):
        if not isinstance(loan_id, str) or loan_id not in self.loans.index:
            raise LoanNotFoundError(loan_id if isinstance(loan_id, str) else repr(loan_id))

    def _oked(self, loan_id: str) -> str:
        return self.suppliers.at[self.loans.at[loan_id, "supplier_id"], "oked_code"]

    def _reasons(self, loan_id: str, with_refs: bool) -> list[dict]:
        ctx = self._context(loan_id) if with_refs else {"supplier_oked": self._oked(loan_id)}
        return build_reasons(self.model.features, self.features.loc[loan_id, self.model.features].to_numpy(float),
                             self._contrib.loc[loan_id].to_numpy(float), ctx, with_refs=with_refs)

    def _context(self, loan_id: str) -> dict:
        loan, F = self.loans.loc[loan_id], self.features.loc[loan_id]
        sup = self.suppliers.loc[loan.supplier_id]
        txs = self._bt[self._bt.loan_id == loan_id].copy()
        txs["day"] = (txs["ts"] - loan.disbursed_at) / pd.Timedelta(days=1)
        inb = txs[txs["type"].astype(str).isin(["p2p_in", "income"])]
        txs["type"] = txs["type"].astype(str)
        big = inb.sort_values("amount_uzs", kind="stable").tail(1)
        invoice = dict(supplier_name=sup["name"], supplier_oked=sup["oked_code"],
                       oked_matches_purpose=bool(sup["oked_code"] in MATCHING_OKED[str(loan.purpose)]),
                       invoice_amount_uzs=int(loan.invoice_amount_uzs))
        graph = dict(supplier_name=sup["name"], in_degree=int(F["sup_in_degree"]),
                     cash_share=float(F["sup_cash_share"]), exit_share=float(F["sup_exit_share"]),
                     active_span_days=float(F["sup_active_span_days"]), employee_link=bool(F["sup_employee_link"]))
        return dict(loan_id=loan_id, supplier_id=loan.supplier_id, supplier_oked=sup["oked_code"], txs=txs,
                    big_inbound_txn=None if big.empty else big["txn_id"].iloc[0], invoice=invoice, graph=graph)

    def _make_queue(self) -> list[dict]:
        order = self.risk.sort_values(ascending=False, kind="stable").index
        l, feats = self.loans.loc[order], self.model.features
        X = self.features.loc[order, feats].to_numpy(float)
        C = self._contrib.loc[order].to_numpy(float)
        oked = l["supplier_id"].map(self.suppliers["oked_code"]).to_numpy()
        risk = self.risk.loc[order].to_numpy(float)
        cols = {c: l[c].to_numpy() for c in ("borrower_name_masked", "amount_uzs")}
        purpose, region = l["purpose"].astype(str).to_numpy(), l["region"].astype(str).to_numpy()
        top = C.argmax(axis=1)                      # strongest risk-raising feature = reasons[0]
        out = []
        for i, lid in enumerate(order):
            j = top[i]
            ctx = {"supplier_oked": oked[i]}
            if C[i, j] <= 1e-9:
                text = NO_REASON_UZ
            elif feats[j] == "income_rate_per_month" and X[i, j] < 0:      # not a shown fact: take the full path
                rs = build_reasons(feats, X[i], C[i], ctx, with_refs=False)
                text = rs[0]["text_uz"] if rs else NO_REASON_UZ
            else:
                text = TEMPLATES[feats[j]](X[i, j], ctx)
            out.append(dict(
                loan_id=lid, borrower_name_masked=cols["borrower_name_masked"][i], purpose=purpose[i],
                amount_uzs=int(cols["amount_uzs"][i]), region=region[i], risk=float(risk[i]),
                color=color_of(float(risk[i]), self.thresholds),
                top_reason=text))
        return out

    # ------------------------------------------------------------------ public
    def queue(self) -> list[dict]:
        return [dict(q) for q in self._queue]

    def score(self, loan_id: str) -> dict:
        """Live scoring of one loan from its cached feature row: model inference + reasons."""
        self._check(loan_id)
        t0 = time.perf_counter()
        x = self.features.loc[[loan_id]]
        risk = float(self.model.score(x)[0])
        contrib = self.model.contributions(x)[0]
        reasons = build_reasons(self.model.features, x.iloc[0].to_numpy(float), contrib,
                                {"supplier_oked": self._oked(loan_id)}, with_refs=False)
        return dict(risk=risk, color=color_of(risk, self.thresholds), reasons=reasons,
                    model_version=MODEL_VERSION, latency_ms=round((time.perf_counter() - t0) * 1000, 3))

    def _photos(self, loan_id: str) -> list[dict]:
        """Portfolio photos of the loan (subset of loans only) with capture time / place and duplicates."""
        from ml.photo.index import get_index
        row = self.loans.loc[loan_id]
        return [dict(photo_id=p["photo_id"], path=p["path"], purpose=p["purpose"],
                     captured_at=_py(row["photo_captured_at"]), lat=float(row["photo_lat"]),
                     lon=float(row["photo_lon"]), duplicate_of=p["duplicate_of"], source="portfolio")
                for p in get_index().photos_of(loan_id)]

    def loan(self, loan_id: str) -> dict:
        self._check(loan_id)
        ctx = self._context(loan_id)
        risk = float(self.risk[loan_id])
        reasons = self._reasons(loan_id, with_refs=True)
        refs = [r for item in reasons for r in item["evidence_refs"]]
        txs = ctx["txs"].sort_values(["ts", "txn_id"])
        return dict(
            loan={"loan_id": loan_id, **{k: _py(self.loans.at[loan_id, k]) for k in LOAN_FIELDS[1:]}},
            transactions=[{k: _py(r[k]) for k in TXN_FIELDS} for _, r in txs.iterrows()],
            photos=self._photos(loan_id), invoice=ctx["invoice"],
            risk=risk, color=color_of(risk, self.thresholds), reasons=reasons,
            evidence=build_evidence(refs, ctx), model_version=MODEL_VERSION)


_lock = threading.Lock()
_portfolio: Portfolio | None = None


def get_portfolio() -> Portfolio:
    global _portfolio
    if _portfolio is None:
        with _lock:
            if _portfolio is None:
                _portfolio = Portfolio()
    return _portfolio
