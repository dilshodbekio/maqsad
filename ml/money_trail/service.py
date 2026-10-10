"""Serving side of the money-trail model (used by ml/api.py).

The deployed service has no data/generated/ (gitignored). On first use the ML package generates the
demo portfolio in memory - the DEMO split config (seed 7), deterministic, ~1 s - computes the
money-trail features once and scores every loan with the trained model from ml/artifacts/. Labels from
the generator are dropped immediately and never reach the API. Nothing here is hard-coded: every risk,
colour and reason comes from the model and the features.
"""
from __future__ import annotations

import copy
import threading
import time
from functools import cached_property

import numpy as np
import pandas as pd

from ml import fusion
from ml.errors import LoanNotFoundError
from ml.sandbox import validate_sandbox_id
from ml.generator.build import build
from ml.generator.config import MATCHING_OKED, SPLITS

from .features import build_features
from .colors import color_of, rank_thresholds
from .model import MODEL_VERSION, MoneyTrailModel, load_model
from .reasons import NO_REASON_UZ, TEMPLATES, build_evidence, build_reasons, is_skipped

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
        self.employees = tables["employees"]
        self.model: MoneyTrailModel = load_model()
        self.features = build_features(tables["loans"], self.tx, tables["suppliers"], tables["employees"])
        self.risk = pd.Series(self.model.score(self.features), index=self.features.index)
        contrib = pd.DataFrame(self.model.contributions(self.features), index=self.features.index,
                               columns=self.model.features)
        self._contrib = contrib
        self._base_reason = {}
        self.thresholds = rank_thresholds(self.risk.to_numpy())      # colours by queue rank on THIS portfolio
        self.sorted_risk = np.sort(self.risk.to_numpy())
        self.init_s = time.perf_counter() - t0
        self._bt = self.tx[self.tx.loan_id.notna()]
        self._signals = self._portfolio_signals()          # photo signals of the demo portfolio's own photos
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
            elif is_skipped(feats[j], float(X[i, j])):                     # not a shown fact: take the full path
                rs = build_reasons(feats, X[i], C[i], ctx, with_refs=False)
                text = rs[0]["text_uz"] if rs else NO_REASON_UZ
            else:
                text = TEMPLATES[feats[j]](X[i, j], ctx)
            base = color_of(float(risk[i]), self.thresholds)
            item = dict(
                loan_id=lid, borrower_name_masked=cols["borrower_name_masked"][i], purpose=purpose[i],
                amount_uzs=int(cols["amount_uzs"][i]), region=region[i], risk=float(risk[i]),
                color=base, top_reason=text, base_color=base, priority=fusion.priority(base, risk[i]), escalations=[])
            self._base_reason[lid] = text
            if lid in self._signals:
                self._apply(item, self._signals[lid])
            out.append(item)
        out.sort(key=lambda q: -q["priority"])               # stable: ties keep the money-trail order
        return out

    def _apply(self, item: dict, signals: list[dict]) -> dict:
        """Fusion policy on a queue item (in place): colour only goes up, risk is untouched."""
        res = fusion.apply_policy(item["base_color"], signals)
        item["color"], item["escalations"] = res["color"], res["escalations"]
        item["priority"] = fusion.priority(res["color"], item["risk"])
        base = self._base_reason[item["loan_id"]]
        ordered = fusion.order_reasons([dict(text_uz=e["text_uz"]) for e in res["escalations"]],
                                       [] if base == NO_REASON_UZ else [dict(text_uz=base)], res["escalations"])
        item["top_reason"] = ordered[0]["text_uz"] if ordered else NO_REASON_UZ     # == get_loan().reasons[0]
        return item

    def _portfolio_signals(self) -> dict:
        """Fusion signals of the demo portfolio's own photos (duplicates in the index, distance, capture time)."""
        from ml.photo import config as photo_config
        from ml.photo.checker import km_between
        from ml.photo.index import get_index
        idx, out = get_index(), {}
        for lid in idx.by_loan:
            if lid not in self.loans.index:
                continue
            row = self.loans.loc[lid]
            km = km_between(float(row["photo_lat"]), float(row["photo_lon"]), float(row["borrower_lat"]), float(row["borrower_lon"]))
            time_ok = bool(row["disbursed_at"] <= row["photo_captured_at"] <= row["snapshot_date"])
            for p in idx.photos_of(lid):
                sig = fusion.photo_signals(ref=f"photo:{p['photo_id']}", photo_label=p["path"], duplicates=p["duplicate_of"],
                                           geo_ok=km <= photo_config.GEO_MAX_KM, distance_km=km, time_ok=time_ok,
                                           asset_match=p.get("asset_match"), uploaded=False, purpose=str(row["purpose"]))
                if sig:
                    out.setdefault(lid, []).extend(sig)
        return out

    # ------------------------------------------------------------------ public
    def fused_counts(self) -> dict:
        """How the default queue's colours relate to the 10% / 10% rank rule: the photo policy can only add to it."""
        q = self._queue
        n = lambda f: sum(1 for x in q if f(x))
        return dict(n_red_total=n(lambda x: x["color"] == "red"), n_yellow_total=n(lambda x: x["color"] == "yellow"),
                    n_escalated_to_red=n(lambda x: x["color"] == "red" and x["base_color"] != "red"),
                    n_escalated_to_yellow=n(lambda x: x["color"] == "yellow" and x["base_color"] == "green"))

    def queue(self, sandbox_id=None) -> list[dict]:
        """Queue sorted by priority (colour rank, then risk). With a sandbox_id the escalations from that sandbox's
        uploaded photos are included; other sandboxes and the default view are unaffected."""
        items = copy.deepcopy(self._queue)          # callers may edit what they get; the cache must stay intact
        if validate_sandbox_id(sandbox_id) is None:
            return items
        from ml.photo.checker import sandbox_signals
        extra = sandbox_signals(sandbox_id)
        if not extra:
            return items
        pos = {q["loan_id"]: i for i, q in enumerate(items)}
        for lid, sig in extra.items():
            if lid in pos:
                self._apply(items[pos[lid]], self._signals.get(lid, []) + sig)
        items.sort(key=lambda q: -q["priority"])
        return items

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
                     lon=float(row["photo_lon"]), duplicate_of=p["duplicate_of"], asset_match=p.get("asset_match"),
                     source="portfolio")
                for p in get_index().photos_of(loan_id)]

    def loan(self, loan_id: str, sandbox_id=None) -> dict:
        validate_sandbox_id(sandbox_id)
        self._check(loan_id)
        ctx = self._context(loan_id)
        risk = float(self.risk[loan_id])
        reasons = self._reasons(loan_id, with_refs=True)
        refs = [r for item in reasons for r in item["evidence_refs"]]
        txs = ctx["txs"].sort_values(["ts", "txn_id"])
        # fusion: photo escalations (portfolio photos + this sandbox's uploads) come first, as policy reasons
        signals = list(self._signals.get(loan_id, []))
        if sandbox_id is not None:
            from ml.photo.checker import sandbox_signals
            signals += sandbox_signals(sandbox_id).get(loan_id, [])
        base = color_of(risk, self.thresholds)
        fused = fusion.apply_policy(base, signals)
        policy_reasons = [dict(code=e["code"], text_uz=e["text_uz"], weight=None, evidence_refs=e["evidence_refs"],
                               source="fusion_policy") for e in fused["escalations"]]
        photo_evidence = {s["evidence_ref"]: s["evidence"] for s in signals}
        photo_items = copy.deepcopy([photo_evidence[r] for r in dict.fromkeys(r for e in fused["escalations"] for r in e["evidence_refs"])])
        return dict(
            loan={"loan_id": loan_id, **{k: _py(self.loans.at[loan_id, k]) for k in LOAN_FIELDS[1:]}},
            transactions=[{k: _py(r[k]) for k in TXN_FIELDS} for _, r in txs.iterrows()],
            photos=self._photos(loan_id), invoice=ctx["invoice"],
            risk=risk, color=fused["color"], base_color=base, priority=fusion.priority(fused["color"], risk),
            escalations=fused["escalations"],
            reasons=fusion.order_reasons(policy_reasons, reasons, fused["escalations"]),
            evidence=photo_items + build_evidence(refs, ctx), model_version=MODEL_VERSION)


_lock = threading.Lock()
_portfolio: Portfolio | None = None


def get_portfolio() -> Portfolio:
    global _portfolio
    if _portfolio is None:
        with _lock:
            if _portfolio is None:
                _portfolio = Portfolio()
    return _portfolio
