"""Per-loan transaction sequences for honest profiles and fraud typologies T1-T4.

All times are epoch seconds; delays are drawn in days. Every behaviour parameter is
sampled from a distribution (no fixed thresholds), and fraud variants deliberately
overlap with honest ones - see ml/CLAUDE.md anti-leakage rules.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DAY = 86400
MIN_DELAY = 0.01  # days (~15 min)


@dataclass
class Loan:
    idx: int
    loan_id: str
    account: str
    t0: int            # disbursement time
    amount: int
    invoice: int
    purpose: str
    supplier_idx: int
    supplier_id: str
    stealth: bool = False


INFLOW_TYPES = frozenset({"disbursement", "p2p_in", "income"})


@dataclass
class Emitter:
    snap_s: int
    rows: list = field(default_factory=list)
    counter: int = 0

    def add(self, loan_id, account, ts, typ, amount, cp=None, kind="none"):
        amount = int(round(amount, -3))
        ts = int(ts)
        if amount <= 0 or ts > self.snap_s:
            return
        self.rows.append((loan_id, account, ts, typ, amount, cp, kind))

    def person(self, rng):
        return f"P{rng.integers(1_000_000, 9_999_999)}"

    def business(self, rng):
        return f"BZ{rng.integers(100_000, 999_999)}"

    def external(self, rng):
        return f"X{rng.integers(100_000, 999_999)}"


def cap_outflows(rows):
    """Replay rows per account in time order and cap every outflow at the available balance.

    Row = (loan_id, account_id, ts, type, amount, counterparty_id, counterparty_kind).
    Inflows are disbursement / p2p_in / income; everything else is an outflow. An outflow that
    exceeds the balance is reduced to it (floored to 1000 UZS); one that cannot be paid at all is
    dropped. At equal timestamps inflows are applied first. No account can end up negative.
    Returns (rows, n_reduced, n_dropped).
    """
    order = sorted(range(len(rows)), key=lambda i: (rows[i][1], rows[i][2],
                                                    rows[i][3] not in INFLOW_TYPES, i))
    out, bal, acct, reduced, dropped = [], 0, None, 0, 0
    for i in order:
        r = rows[i]
        if r[1] != acct:
            acct, bal = r[1], 0
        if r[3] in INFLOW_TYPES:
            bal += r[4]
            out.append(r)
            continue
        amt = min(r[4], bal) // 1000 * 1000
        if amt <= 0:
            dropped += 1
            continue
        reduced += amt < r[4]
        bal -= amt
        out.append(r[:4] + (amt,) + r[5:])
    return out, reduced, dropped


def _chunks(rng, total, k):
    if k <= 1:
        return [total]
    return list(total * rng.dirichlet(np.ones(k) * 2.0))


def disburse(em, L):
    em.add(L.loan_id, L.account, L.t0, "disbursement", L.amount)


def supplier_payment(em, rng, L, frac, mean_delay, max_delay=25.0):
    """Pay `frac` of the invoice to the declared supplier, in 1-2 installments."""
    pay = min(L.invoice * frac, L.amount)
    k = 1 if rng.random() < 0.8 else 2
    d = min(max(MIN_DELAY, rng.exponential(mean_delay)), max_delay)
    for part in _chunks(rng, pay, k):
        ts = L.t0 + d * DAY
        em.add(L.loan_id, L.account, ts, "supplier_payment", part, L.supplier_id, "supplier")
        d += rng.exponential(2.0)
    return pay


def cash_out(em, rng, L, total, mean_delay, k, start=0.0, hi=None, typ="cash_withdrawal"):
    d = start + max(MIN_DELAY, rng.exponential(mean_delay))
    for part in _chunks(rng, total, k):
        dd = d if hi is None else min(d, start + hi)
        cp, kind = (None, "none") if typ == "cash_withdrawal" else (em.external(rng), "external")
        em.add(L.loan_id, L.account, L.t0 + dd * DAY, typ, part, cp, kind)
        d += rng.exponential(mean_delay * 0.5)


def noise_p2p(em, rng, L, lam_out, lam_in):
    for _ in range(rng.poisson(lam_out)):
        d = rng.uniform(1, 200)
        em.add(L.loan_id, L.account, L.t0 + d * DAY, "p2p_out",
               L.amount * rng.uniform(0.005, 0.03), em.person(rng), "person")
    for _ in range(rng.poisson(lam_in)):
        d = rng.uniform(1, 200)
        em.add(L.loan_id, L.account, L.t0 + d * DAY, "p2p_in",
               L.amount * rng.uniform(0.005, 0.03), em.person(rng), "person")


def regular_income(em, rng, L, start_day, stop_day=None, p_month=0.85, scale=(0.015, 0.06)):
    d = start_day
    while L.t0 + d * DAY <= em.snap_s and d < 36 * 30:
        if stop_day is not None and d > stop_day:
            break
        if rng.random() < p_month:
            amt = L.amount * rng.uniform(*scale) * rng.lognormal(0, 0.3)
            em.add(L.loan_id, L.account, L.t0 + d * DAY, "income", amt, em.business(rng), "business")
        d += 30 + rng.normal(0, 3)


def one_off_inbound(em, rng, L):
    """Occasional large one-off inflow (relative, big order, refund) that honest loans also have."""
    d = rng.uniform(20, 250)
    amt = L.amount * rng.uniform(0.08, 0.85)
    ts = L.t0 + d * DAY
    if rng.random() < 0.5:
        em.add(L.loan_id, L.account, ts, "p2p_in", amt, em.person(rng), "person")
    else:
        em.add(L.loan_id, L.account, ts, "income", amt, em.business(rng), "business")
    if rng.random() < 0.55:
        cash_out(em, rng, L, amt * rng.uniform(0.3, 1.0), mean_delay=3.0, start=d,
                 k=int(rng.integers(1, 3)))


def _base(em, rng, L, p_partial, p_cash, income_prob, income_stop=None, p_cash_purchase=0.07):
    """Honest-looking core: supplier payment, some leftover cash, small noise, monthly income.

    With probability p_cash_purchase the asset is bought for cash at a market (no supplier
    payment, slower cash-out than T1) - a hard negative for T1.
    """
    disburse(em, L)
    if rng.random() < p_cash_purchase:
        cash_out(em, rng, L, L.amount * rng.uniform(0.7, 0.95), mean_delay=rng.uniform(4, 15),
                 k=int(rng.integers(1, 4)), hi=60)
    else:
        frac = rng.uniform(0.5, 0.92) if rng.random() < p_partial else rng.uniform(0.93, 1.0)
        paid = supplier_payment(em, rng, L, frac, mean_delay=3.0)
        left = L.amount - paid
        if left > 0.03 * L.amount and rng.random() < p_cash:
            cash_out(em, rng, L, left * rng.uniform(0.3, 1.0), mean_delay=7.0,
                     k=int(rng.integers(1, 4)), hi=40)
    noise_p2p(em, rng, L, 1.0, 0.8)
    if rng.random() < 0.40:
        one_off_inbound(em, rng, L)
    if rng.random() < income_prob:
        regular_income(em, rng, L, rng.uniform(20, 90), stop_day=income_stop)


# ---------------------------------------------------------------- honest profiles
def honest_normal(em, rng, p, L):
    _base(em, rng, L, p_partial=0.15, p_cash=0.55, income_prob=0.9)


def honest_livestock_seller(em, rng, p, L):
    sale_day = rng.uniform(30, 150)
    stop = sale_day if rng.random() < 0.15 else None  # honest sellers mostly keep earning
    _base(em, rng, L, p_partial=0.15, p_cash=0.55, income_prob=0.9, income_stop=stop)
    amt = L.amount * rng.uniform(0.15, 0.60)
    ts = L.t0 + sale_day * DAY
    if rng.random() < 0.30:   # buyer is a person
        em.add(L.loan_id, L.account, ts, "p2p_in", amt, em.person(rng), "person")
    else:                     # buyer is a business / market
        em.add(L.loan_id, L.account, ts, "income", amt, em.business(rng), "business")
    typ = "cash_withdrawal" if rng.random() < 0.85 else "external_transfer"
    cash_out(em, rng, L, amt * rng.uniform(0.4, 1.0), mean_delay=3.0,
             k=int(rng.integers(1, 3)), start=sale_day, typ=typ)


def honest_cash_heavy(em, rng, p, L):
    disburse(em, L)
    paid = supplier_payment(em, rng, L, rng.uniform(0.40, 0.65), mean_delay=3.0)
    cash = min(L.amount * rng.uniform(0.25, 0.55), L.amount - paid)
    cash_out(em, rng, L, cash, mean_delay=1.5, k=int(rng.integers(1, 4)), hi=10)
    for m in range(1, 9):  # cash payroll
        if rng.random() < 0.8:
            em.add(L.loan_id, L.account, L.t0 + (30 * m + rng.normal(0, 3)) * DAY,
                   "cash_withdrawal", L.amount * rng.uniform(0.02, 0.06))
    noise_p2p(em, rng, L, 2.0, 1.0)
    regular_income(em, rng, L, rng.uniform(10, 40), p_month=0.9, scale=(0.03, 0.09))


# ---------------------------------------------------------------- fraud typologies
def t1_fast_cash_out(em, rng, p, L):
    disburse(em, L)
    small_pay_prob = 0.9 if L.stealth else 0.2
    paid = 0.0
    if rng.random() < small_pay_prob:
        frac = rng.uniform(0.05, 0.25) * L.amount / L.invoice
        paid = supplier_payment(em, rng, L, frac, mean_delay=4.0)
    share = rng.beta(6, 2) if rng.random() < 0.7 else rng.beta(3, 3)
    cash_total = min(L.amount * share, L.amount - paid)
    if L.stealth:  # drip over 5-10 days
        window = rng.uniform(5, 10)
        k = int(rng.integers(2, 5))
        for part, d in zip(_chunks(rng, cash_total, k), np.sort(rng.uniform(0.5, window, k))):
            em.add(L.loan_id, L.account, L.t0 + d * DAY, "cash_withdrawal", part)
    else:
        cash_out(em, rng, L, cash_total, mean_delay=p.t1_delay_mean_days, k=int(rng.integers(1, 4)))
    rem = L.amount - paid - cash_total
    if rem > 0.03 * L.amount and rng.random() < 0.7:
        out = rem * rng.uniform(0.2, 0.9)
        if rng.random() < 0.6:
            em.add(L.loan_id, L.account, L.t0 + (1 + rng.exponential(3.0)) * DAY, "p2p_out",
                   out, em.person(rng), "person")
        else:
            em.add(L.loan_id, L.account, L.t0 + (1 + rng.exponential(3.0)) * DAY,
                   "external_transfer", out, em.external(rng), "external")
    noise_p2p(em, rng, L, 0.6, 0.4)
    if rng.random() < 0.25:
        regular_income(em, rng, L, rng.uniform(30, 120), p_month=0.6)


def t2_ring_member(em, rng, p, L):
    disburse(em, L)
    paid = supplier_payment(em, rng, L, rng.uniform(0.6, 1.0), mean_delay=2.0, max_delay=14.0)
    left = L.amount - paid
    if left > 0.03 * L.amount and rng.random() < 0.5:
        cash_out(em, rng, L, left * rng.uniform(0.3, 1.0), mean_delay=7.0,
                 k=int(rng.integers(1, 3)), hi=40)
    noise_p2p(em, rng, L, 1.0, 0.8)
    if rng.random() < 0.5:
        regular_income(em, rng, L, rng.uniform(20, 90), p_month=0.7)


def t3_asset_resale(em, rng, p, L):
    """Returns the resale timestamp (epoch s) for the visibility label."""
    resale_day = rng.uniform(20, p.t3_max_day)
    stop = resale_day if rng.random() < 0.8 else None  # regular income stops after resale
    _base(em, rng, L, p_partial=0.15, p_cash=0.5, income_prob=0.6, income_stop=stop,
          p_cash_purchase=0.0)
    amt = L.invoice * (rng.uniform(0.15, 0.35) if rng.random() < 0.35 else rng.uniform(0.35, 0.9))
    ts = L.t0 + resale_day * DAY
    if rng.random() < 0.65:   # buyer is a person
        em.add(L.loan_id, L.account, ts, "p2p_in", amt, em.person(rng), "person")
    else:
        em.add(L.loan_id, L.account, ts, "income", amt, em.business(rng), "business")
    typ = "cash_withdrawal" if rng.random() < 0.6 else "external_transfer"
    cash_out(em, rng, L, amt * rng.uniform(0.5, 1.0),
             mean_delay=10.0 if rng.random() < 0.25 else 1.5,
             k=int(rng.integers(1, 3)), start=resale_day, typ=typ)
    return int(ts)


def t4_fake_invoice(em, rng, p, L):
    # Money trail is ordinary; weak extra signals (larger invoice-payment gap, a bit more cash).
    _base(em, rng, L, p_partial=0.45, p_cash=0.65, income_prob=0.85, p_cash_purchase=0.03)


# ---------------------------------------------------------------- supplier-side outflows
def supplier_outflows(em, rng, receipts, account, spec):
    """Cash-out / transfers from a supplier account after it received loan money.

    `receipts` = [(ts, amount)] of supplier payments that actually settled (after balance capping).
    """
    for ts, amt in receipts:
        share = float(np.clip(spec["cash_prop"] * rng.uniform(0.7, 1.3), 0, 1))
        total = amt * share
        if total <= 0:
            continue
        if spec["ring"]:
            d = rng.lognormal(np.log(spec["median_delay"]), 0.8)
        else:
            d = min(rng.exponential(spec.get('mean_delay', 12.0)), 90.0)
        d = max(d, MIN_DELAY)
        for part in _chunks(rng, total, int(rng.integers(1, 3))):
            if rng.random() < spec["cash_type_share"]:
                em.add(None, account, ts + d * DAY, "cash_withdrawal", part)
            else:
                em.add(None, account, ts + d * DAY, "external_transfer", part,
                       em.external(rng), "external")
            d += rng.exponential(1.0)
