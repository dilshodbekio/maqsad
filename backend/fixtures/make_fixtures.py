"""Generate STUB-ONLY synthetic fixtures for backend/frontend development.

!!! THESE FIXTURES ARE FOR STUB MODE ONLY (ml_mode == "stub") !!!
- They are NOT the ml/ generator, NOT a model, and must never be shown as model output.
  In the demo the real portfolio, scores and reasons come from `ml.api` (Dilshodbek).
- Every record is synthetic (seeded random); no real people, banks, suppliers or transactions.
- risk / color / reasons are computed by a transparent hand-written RULE over the synthetic
  transactions (`stub_rule_score`), only so the UI has varied data to render.
  Color thresholds 0.33 / 0.66 are a stub-only assumption; real thresholds come from ml.
- OKED codes are NACE Rev.2-style codes chosen for illustration; the purpose <-> OKED mapping
  is a stub assumption. Amount ranges are synthetic assumptions, not bank statistics.
- Inner shapes follow docs/api_contract_changes.md #5 (temporary until the ml generator's JSON).

Usage:  python backend/fixtures/make_fixtures.py [--seed 42] [--n 40]
Output: backend/fixtures/loans.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
from datetime import datetime, timedelta
from pathlib import Path

OUT_PATH = Path(__file__).resolve().parent / "loans.json"

PURPOSES = ["tikuv_dastgohi", "chorva", "issiqxona", "parranda", "asalarichilik"]
# purpose -> matching OKED codes (illustrative, NACE Rev.2-style)
PURPOSE_OKED = {
    "tikuv_dastgohi": ["46.64"],  # wholesale of textile/sewing machinery
    "chorva": ["46.23"],          # wholesale of live animals
    "issiqxona": ["46.73"],       # wholesale of construction materials
    "parranda": ["46.23", "01.47"],
    "asalarichilik": ["01.49"],   # raising of other animals (incl. bees)
}
GENERIC_OKED = "46.90"  # non-specialised wholesale (used legitimately too)
MISMATCH_OKED = ["56.10", "68.20", "47.71"]  # restaurants, real estate, clothing retail
# synthetic amount ranges, mln UZS (assumption)
AMOUNT_MLN = {
    "tikuv_dastgohi": (20, 80),
    "chorva": (40, 150),
    "issiqxona": (50, 150),
    "parranda": (25, 90),
    "asalarichilik": (15, 60),
}
REGIONS = [
    "Toshkent", "Samarqand", "Farg'ona", "Andijon", "Namangan", "Buxoro", "Qashqadaryo",
    "Surxondaryo", "Xorazm", "Navoiy", "Jizzax", "Sirdaryo", "Qoraqalpog'iston",
]
INITIALS = "ABDFGHIJKLMNOQRSTUXYZ"
# profile -> count for n=40 (scaled for other n)
PROFILE_SHARE = [
    ("honest", 18), ("honest_suspicious", 6), ("T1_cash_out", 5),
    ("T2_supplier_ring", 5), ("T3_asset_resale", 3), ("T4_fake_invoice", 3),
]
LOW, HIGH = 0.33, 0.66  # stub-only color thresholds


def fmt_uzs(x: int) -> str:
    return f"{x:,}".replace(",", " ")


def make_suppliers(rng: random.Random) -> dict:
    suppliers = {}
    n = 0

    def add(oked, purpose, kind, cash_out, employee_link=False):
        nonlocal n
        n += 1
        sid = f"S{n:03d}"
        suppliers[sid] = {
            "supplier_id": sid,
            "supplier_name": f"Ta'minotchi {sid} MChJ",
            "oked_code": oked,
            "purpose": purpose,
            "kind": kind,
            "cash_out_share": round(rng.uniform(*cash_out), 2),
            "employee_link": employee_link,
        }

    for p in PURPOSES:
        for _ in range(3):
            add(rng.choice(PURPOSE_OKED[p]), p, "legit", (0.03, 0.2))
        add(GENERIC_OKED, p, "generic", (0.15, 0.4))
        add(rng.choice(MISMATCH_OKED), p, "mismatch", (0.3, 0.7))
    add(GENERIC_OKED, None, "ring", (0.7, 0.92), employee_link=True)
    return suppliers


def pick(suppliers: dict, rng: random.Random, purpose: str, kind: str) -> dict:
    pool = [s for s in suppliers.values() if s["kind"] == kind and s["purpose"] in (purpose, None)]
    return rng.choice(pool)


def make_loan(i: int, profile: str, purpose: str, suppliers: dict, rng: random.Random):
    loan_id = f"L{i:04d}"
    lo, hi = AMOUNT_MLN[purpose]
    amount = rng.randint(lo, hi) * 1_000_000
    t0 = datetime(2026, 1, 5, 9) + timedelta(days=rng.randint(0, 230), minutes=rng.randint(0, 480))

    if profile == "T2_supplier_ring":
        sup = pick(suppliers, rng, purpose, "ring")
    elif profile == "T4_fake_invoice":
        sup = pick(suppliers, rng, purpose, "mismatch")
    elif profile == "honest_suspicious" and rng.random() < 0.5:
        sup = pick(suppliers, rng, purpose, "generic")
    else:
        sup = pick(suppliers, rng, purpose, "legit")

    txns = []

    def txn(day: float, ttype: str, amt: float, counterparty: str):
        txns.append({
            "txn_id": f"{loan_id}-T{len(txns) + 1:02d}",
            "ts": (t0 + timedelta(days=day, minutes=rng.randint(5, 300))).isoformat(timespec="seconds"),
            "type": ttype,
            "amount_uzs": int(round(amt, -3)),
            "counterparty": counterparty,
        })

    txn(0, "disbursement", amount, "MKBANK-kredit")
    if profile == "honest":
        txn(rng.uniform(0.5, 6), "supplier_payment", amount * rng.uniform(0.75, 0.97), sup["supplier_id"])
        if rng.random() < 0.6:
            txn(rng.uniform(4, 25), "cash_withdrawal", amount * rng.uniform(0.02, 0.12), "kassa")
        for _ in range(rng.randint(1, 3)):
            txn(rng.uniform(20, 60), "income", amount * rng.uniform(0.03, 0.1), f"P2P-{rng.randint(1000, 9999)}")
    elif profile == "honest_suspicious":
        txn(rng.uniform(0.5, 5), "supplier_payment", amount * rng.uniform(0.45, 0.7), sup["supplier_id"])
        txn(rng.uniform(2, 8), "cash_withdrawal", amount * rng.uniform(0.2, 0.4), "kassa")
        if rng.random() < 0.5:  # e.g. farmer legally sells part of livestock
            txn(rng.uniform(30, 60), "income", amount * rng.uniform(0.2, 0.35), f"P2P-{rng.randint(1000, 9999)}")
    elif profile == "T1_cash_out":
        txn(rng.uniform(0, 1.5), "cash_withdrawal", amount * rng.uniform(0.55, 0.95), "kassa")
        if rng.random() < 0.6:
            txn(rng.uniform(1, 4), "p2p_out", amount * rng.uniform(0.03, 0.15), f"P2P-{rng.randint(1000, 9999)}")
    elif profile == "T2_supplier_ring":
        txn(rng.uniform(0, 2), "supplier_payment", amount * rng.uniform(0.8, 1.0), sup["supplier_id"])
    elif profile == "T3_asset_resale":
        txn(rng.uniform(1, 5), "supplier_payment", amount * rng.uniform(0.75, 0.95), sup["supplier_id"])
        resale_day = rng.uniform(15, 40)
        resale = amount * rng.uniform(0.45, 0.8)
        txn(resale_day, "income", resale, f"P2P-{rng.randint(1000, 9999)}")
        txn(resale_day + rng.uniform(0, 2), "cash_withdrawal", resale * rng.uniform(0.7, 0.95), "kassa")
    elif profile == "T4_fake_invoice":
        txn(rng.uniform(0.5, 4), "supplier_payment", amount * rng.uniform(0.8, 0.98), sup["supplier_id"])
        if rng.random() < 0.5:
            txn(rng.uniform(3, 15), "external_transfer", amount * rng.uniform(0.02, 0.1), "EXT-BANK")
    txns.sort(key=lambda t: t["ts"])

    loan = {
        "loan_id": loan_id,
        "borrower_name_masked": f"{rng.choice(INITIALS)}*** {rng.choice(INITIALS)}***",
        "purpose": purpose,
        "amount_uzs": amount,
        "disbursed_at": t0.date().isoformat(),
        "region": rng.choice(REGIONS),
        "branch_id": f"B{rng.randint(1, 12):02d}",
        "supplier_id": sup["supplier_id"],
    }
    return loan, txns, sup


def stub_rule_score(loan: dict, txns: list, sup: dict, in_degree: int) -> tuple:
    """Transparent STUB rule (not a model): weighted flags -> risk in [0, 1] + reasons."""
    amount = loan["amount_uzs"]
    t0 = datetime.fromisoformat(txns[0]["ts"])

    def share(types, max_days):
        s = sum(t["amount_uzs"] for t in txns if t["type"] in types
                and (datetime.fromisoformat(t["ts"]) - t0).total_seconds() <= max_days * 86400)
        return s / amount

    cash_2d = share({"cash_withdrawal"}, 2)
    cash_30d = share({"cash_withdrawal"}, 30)
    supplier_share = share({"supplier_payment"}, 365)
    payments_after = [t for t in txns if t["type"] == "supplier_payment"]
    big_income = [t for t in txns if t["type"] == "income" and t["amount_uzs"] >= 0.4 * amount
                  and payments_after and t["ts"] > payments_after[0]["ts"]]
    oked_ok = sup["oked_code"] in PURPOSE_OKED[loan["purpose"]]

    reasons = []
    if cash_2d >= 0.2:
        reasons.append(("CASH_FAST", 0.9 * cash_2d,
                        f"Kredit tushgandan keyin 2 kun ichida summaning {round(cash_2d * 100)}% i naqd yechilgan"))
    elif cash_30d >= 0.02:
        reasons.append(("CASH_30D", 0.5 * cash_30d,
                        f"30 kun ichida summaning {round(cash_30d * 100)}% i naqd yechilgan"))
    if supplier_share < 0.5:
        reasons.append(("LOW_SUPPLIER_PAYMENT", 0.4 * (1 - supplier_share),
                        f"Ta'minotchiga summaning faqat {round(supplier_share * 100)}% i to'langan"))
    if payments_after and not oked_ok:
        w = 0.15 if sup["oked_code"] == GENERIC_OKED else 0.45
        reasons.append(("OKED_MISMATCH", w,
                        f"Ta'minotchi faoliyat turi (OKED {sup['oked_code']}) kredit maqsadiga mos emas"))
    if payments_after and in_degree >= 4 and sup["cash_out_share"] >= 0.5:
        reasons.append(("SUPPLIER_RING", 0.5,
                        f"Ta'minotchiga {in_degree} ta qarz oluvchi pul o'tkazgan, u pulning "
                        f"{round(sup['cash_out_share'] * 100)}% ini naqdlashtirgan"))
    if payments_after and sup["employee_link"]:
        reasons.append(("EMPLOYEE_LINK", 0.25, "Ta'minotchi bank xodimi bilan bog'liq"))
    if big_income:
        p = round(big_income[0]["amount_uzs"] / amount * 100)
        reasons.append(("ASSET_RESALE", 0.6,
                        f"Ta'minotchiga to'lovdan keyin kredit summasining {p}% iga teng bir martalik kirim bo'lgan"))

    total = sum(w for _, w, _ in reasons)
    risk = round(1 - math.exp(-1.4 * total), 3)
    color = "red" if risk >= HIGH else "yellow" if risk >= LOW else "green"
    reasons = sorted(reasons, key=lambda r: -r[1])
    out = [{"code": c, "text_uz": t, "weight": round(w, 3)} for c, w, t in reasons]
    if not out:
        out = [{"code": "NO_FLAGS", "text_uz": "Stub qoidalari bo'yicha sezilarli xavf belgisi topilmadi", "weight": 0.0}]
    return risk, color, out


TXN_LABEL_UZ = {
    "disbursement": "kredit tushdi",
    "cash_withdrawal": "naqd yechish",
    "supplier_payment": "ta'minotchiga to'lov",
    "p2p_out": "P2P o'tkazma (chiquvchi)",
    "p2p_in": "P2P o'tkazma (kiruvchi)",
    "external_transfer": "boshqa bankka o'tkazma",
    "income": "kirim",
}


def build(seed: int, n: int) -> dict:
    rng = random.Random(seed)
    suppliers = make_suppliers(rng)
    scale = n / sum(c for _, c in PROFILE_SHARE)
    profiles = [p for p, c in PROFILE_SHARE for _ in range(max(1, round(c * scale)))]
    profiles = (profiles + ["honest"] * n)[:n]
    rng.shuffle(profiles)
    purposes = [PURPOSES[i % len(PURPOSES)] for i in range(n)]
    rng.shuffle(purposes)

    raw = [make_loan(i + 1, profiles[i], purposes[i], suppliers, rng) for i in range(n)]
    in_degree: dict = {}
    for loan, txns, sup in raw:
        if any(t["type"] == "supplier_payment" for t in txns):
            in_degree[sup["supplier_id"]] = in_degree.get(sup["supplier_id"], 0) + 1

    loans = []
    for loan, txns, sup in raw:
        deg = in_degree.get(sup["supplier_id"], 0)
        risk, color, reasons = stub_rule_score(loan, txns, sup, deg)
        paid = sum(t["amount_uzs"] for t in txns if t["type"] == "supplier_payment")
        invoice = None
        if paid:
            invoice = {
                "supplier_id": sup["supplier_id"],
                "supplier_name": sup["supplier_name"],
                "oked_code": sup["oked_code"],
                "oked_matches_purpose": sup["oked_code"] in PURPOSE_OKED[loan["purpose"]],
                "amount_uzs": paid,
            }
        evidence = [{
            "ref": f"txn:{t['txn_id']}",
            "kind": "txn",
            "summary_uz": f"{t['ts'][:10]}: {TXN_LABEL_UZ[t['type']]} {fmt_uzs(t['amount_uzs'])} so'm ({t['counterparty']})",
            "data": t,
        } for t in txns]
        if invoice:
            evidence.append({
                "ref": f"invoice:{loan['loan_id']}",
                "kind": "invoice",
                "summary_uz": f"Faktura: {invoice['supplier_name']}, OKED {invoice['oked_code']}, "
                              f"{fmt_uzs(paid)} so'm",
                "data": invoice,
            })
            evidence.append({
                "ref": f"graph:{sup['supplier_id']}",
                "kind": "graph",
                "summary_uz": f"Ta'minotchi {sup['supplier_id']}: {deg} ta qarz oluvchi, "
                              f"naqdlashtirish ulushi {round(sup['cash_out_share'] * 100)}%",
                "data": {"supplier_id": sup["supplier_id"], "in_degree": deg,
                         "cash_out_share": sup["cash_out_share"], "employee_link": sup["employee_link"]},
            })
        loans.append({
            "loan": loan,
            "transactions": txns,
            "photos": [],
            "invoice": invoice,
            "evidence": evidence,
            "risk": risk,
            "color": color,
            "reasons": reasons,
        })

    return {
        "_comment": "STUB ONLY. Synthetic fixtures generated by backend/fixtures/make_fixtures.py for "
                    "development when ml.api is not importable. Not model output; risk/reasons come from a "
                    "hand-written stub rule. Do not edit by hand - re-run the script.",
        "seed": seed,
        "n_loans": n,
        "stub_color_thresholds": [LOW, HIGH],
        "loans": loans,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args()
    data = build(args.seed, args.n)
    args.out.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    colors = {c: sum(1 for l in data["loans"] if l["color"] == c) for c in ("green", "yellow", "red")}
    purposes = {p: sum(1 for l in data["loans"] if l["loan"]["purpose"] == p) for p in PURPOSES}
    print(f"wrote {args.out} ({args.n} loans, seed={args.seed}) colors={colors} purposes={purposes}")


if __name__ == "__main__":
    main()
