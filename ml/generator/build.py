"""Synthetic portfolio builder.

    python -m ml.generator.build --split train        # seed 42, 5000 loans
    python -m ml.generator.build --split test         # seed 1337, 5000 loans
    python -m ml.generator.build --split all

Writes data/generated/<split>/{loans,transactions,suppliers,employees,labels,
photo_manifest}.parquet + meta.json. Generated data is gitignored.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import flows
from .config import (BRANCHES_PER_REGION, EMPLOYEE_ROLES, EMPLOYEES_PER_BRANCH,
                     GENERIC_OKED, MATCHING_OKED, MEDIAN_AMOUNT, PURPOSE_WEIGHTS,
                     PURPOSES, REGIONS, SPLITS, UNRELATED_OKED, Params)
from .flows import DAY, Emitter, Loan

OUT_ROOT = Path(__file__).resolve().parents[2] / "data" / "generated"

FIRST = ("Baraka", "Zarafshon", "Nur", "Oltin", "Sharq", "Yangi", "Orzu", "Bahor",
         "Dala", "Hosil", "Vodiy", "Mehr", "Sapfar", "Navbahor")
MID = ("Agro", "Servis", "Savdo", "Invest", "Texno", "Plast", "Ta'minot", "Trade")
SUFFIX = ("MChJ", "XK", "YaTT")
GIVEN = tuple("ABDFGHIJKLMNOQRSTUXYZ")


def _plan_loans(p: Params, rng):
    """Per-loan attributes before any supplier / transaction logic."""
    N = p.n_loans
    n_fraud = int(round(N * p.fraud_rate))
    mix = np.array(p.typology_mix) / sum(p.typology_mix)
    counts = np.floor(mix * n_fraud).astype(int)
    counts[0] += n_fraud - counts.sum()
    kind = np.array(["honest"] * (N - n_fraud)
                    + sum(([f"T{i + 1}"] * c for i, c in enumerate(counts)), []), dtype=object)
    rng.shuffle(kind)

    pw = np.array(PURPOSE_WEIGHTS) / sum(PURPOSE_WEIGHTS)
    purpose = rng.choice(PURPOSES, size=N, p=pw).astype(object)
    # T3 (asset resale) hits chorva 40%, tikuv 30%, the rest 30%.
    t3 = np.where(kind == "T3")[0]
    t3_w = np.array([0.30, 0.40, 0.10, 0.10, 0.10])
    purpose[t3] = rng.choice(PURPOSES, size=len(t3), p=t3_w)

    rw = np.array(p.region_weights, dtype=float)
    region = rng.choice(len(REGIONS), size=N, p=rw / rw.sum())
    branch = region * BRANCHES_PER_REGION + rng.integers(0, BRANCHES_PER_REGION, N)

    med = np.array([MEDIAN_AMOUNT[x] for x in purpose], dtype=float) * p.amount_scale
    amount = np.maximum(5_000_000, np.round(med * rng.lognormal(0, 0.35, N), -5)).astype(np.int64)
    invoice = np.round(amount * rng.uniform(0.90, 1.05, N), -5).astype(np.int64)
    disb = (_snap_s(p) - rng.uniform(2, 365, N) * DAY).astype(np.int64)

    stealth = np.zeros(N, dtype=bool)
    t1 = kind == "T1"
    stealth[t1] = rng.random(t1.sum()) < p.p_stealth

    profile = np.array(["none"] * N, dtype=object)
    seller_p = {"chorva": 0.12, "parranda": 0.07}
    for i in np.where(kind == "honest")[0]:
        u = rng.random()
        ps = seller_p.get(purpose[i], 0.0) * p.seller_scale
        if u < ps:
            profile[i] = "livestock_seller"
        elif u < ps + p.cash_heavy_share:
            profile[i] = "cash_heavy"
        elif u < ps + p.cash_heavy_share + p.honest_mismatch:
            profile[i] = "oked_mismatch"
        else:
            profile[i] = "normal"
    return dict(kind=kind, purpose=purpose, region=region, branch=branch, amount=amount,
                invoice=invoice, disb=disb, stealth=stealth, profile=profile)


def _snap_s(p: Params) -> int:
    return int(pd.Timestamp(p.snapshot_date).timestamp())


def _plan_rings(p, rng, pl):
    """Partition T2 loans into rings; override purpose/region/branch/disbursement."""
    idx = list(np.where(pl["kind"] == "T2")[0])
    rng.shuffle(idx)
    snap_s = _snap_s(p)
    rings, pos = [], 0
    while pos < len(idx):
        stealth = rng.random() < p.p_stealth
        size = int(rng.integers(2, 4)) if stealth else 4 + int(rng.poisson(p.ring_extra_mean))
        if len(idx) - pos - size < 2:
            size = len(idx) - pos
        members = idx[pos:pos + size]
        pos += size
        primary = pl["purpose"][members[0]]
        region = pl["region"][members[0]]
        branch = pl["branch"][members[0]]
        window = rng.uniform(30, 150) * DAY
        start = snap_s - rng.uniform(2, 365) * DAY - window
        for m in members:
            if rng.random() < 0.8:
                pl["purpose"][m] = primary
            if rng.random() < 0.8:
                pl["region"][m], pl["branch"][m] = region, branch
            pl["disb"][m] = int(min(snap_s - 2 * DAY, start + rng.uniform(0, window)))
            pl["stealth"][m] = stealth
        rings.append(dict(members=members, stealth=stealth, primary=primary, branch=branch))
    return rings


def _new_supplier(rng, group, oked, ring=False, stealth=False):
    mean_delay = 12.0
    if ring:
        cash_prop = rng.beta(2.5, 3) if stealth else (rng.beta(4.5, 2.5) if rng.random() < 0.7 else rng.beta(2, 4))
    elif rng.random() < 0.22:                       # some honest suppliers are cash-heavy
        cash_prop = rng.beta(5, 3)
        mean_delay = 5.0
    elif group == "honest":
        cash_prop = rng.beta(2, 7)
    elif group == "t4":
        cash_prop = rng.beta(2.5, 5)
    else:                                            # generic / unrelated
        cash_prop = rng.beta(2, 5)
    recent = {"t4": 0.30, "ring": 0.15}.get(group, 0.08)
    return dict(group=group, oked=oked, ring=ring, stealth=stealth, cash_prop=cash_prop,
                median_delay=6.0 if stealth else 2.0, mean_delay=mean_delay, cash_type_share=rng.uniform(0.3, 0.95) if ring else 0.55,
                recent=rng.random() < recent, employee=None)


def _plan_suppliers(p, rng, pl, rings):
    N = p.n_loans
    kind, purpose, profile = pl["kind"], pl["purpose"], pl["profile"]
    sup: list[dict] = []
    sup_of = np.full(N, -1, dtype=np.int64)

    def pool_draw(members, n_sup, group, okeds, sigma, weights_uniform=False):
        if len(members) == 0:
            return
        base = len(sup)
        for _ in range(n_sup):
            sup.append(_new_supplier(rng, group, okeds(rng)))
        w = np.ones(n_sup) if weights_uniform else rng.lognormal(0, sigma, n_sup)
        sup_of[members] = base + rng.choice(n_sup, size=len(members), p=w / w.sum())

    # honest-looking suppliers with an OKED that matches the purpose (used by honest, T1, T3)
    for k in PURPOSES:
        m = np.where((purpose == k) & np.isin(kind, ["honest", "T1", "T3"])
                     & (profile != "oked_mismatch"))[0]
        pool_draw(m, max(3, round(len(m) / 4.0)), "honest",
                  lambda r, k=k: str(r.choice(MATCHING_OKED[k])), 1.0)
    # honest OKED-mismatch loans: generic-trade (65%) or unrelated (35%) suppliers
    mm = np.where(profile == "oked_mismatch")[0]
    generic = mm[rng.random(len(mm)) < 0.65]
    unrel = np.setdiff1d(mm, generic)
    pool_draw(generic, max(3, round(len(generic) / 3)), "generic",
              lambda r: str(r.choice(GENERIC_OKED)), 0.8)
    pool_draw(unrel, max(3, round(len(unrel) / 3)), "unrelated",
              lambda r: str(r.choice(UNRELATED_OKED)), 0.8)
    # T4: dedicated pool, OKED never matches the purpose (unrelated 70% / generic 30%)
    t4 = np.where(kind == "T4")[0]
    pool_draw(t4, max(3, round(len(t4) / 2.2)), "t4",
              lambda r: str(r.choice(UNRELATED_OKED if r.random() < 0.7 else GENERIC_OKED)),
              0, weights_uniform=True)
    # T2: one supplier per ring
    for ring_id, ring in enumerate(rings):
        u = rng.random()
        if u < 0.65:
            oked = str(rng.choice(MATCHING_OKED[ring["primary"]]))
        elif u < 0.90:
            oked = str(rng.choice(GENERIC_OKED))
        else:
            oked = str(rng.choice(UNRELATED_OKED))
        s = _new_supplier(rng, "ring", oked, ring=True, stealth=ring["stealth"])
        s["ring_id"] = ring_id
        s["ring_branch"] = ring["branch"]
        sup.append(s)
        sup_of[ring["members"]] = len(sup) - 1
    assert (sup_of >= 0).all()
    return sup, sup_of


def _ids(prefix, n, off, rng, width=6):
    perm = rng.permutation(n)
    return np.array([f"{prefix}{off + int(x):0{width}d}" for x in perm], dtype=object)


def build(p: Params) -> dict:
    t_start = time.perf_counter()
    rng = np.random.default_rng(p.seed)
    snap_s = _snap_s(p)
    N = p.n_loans

    pl = _plan_loans(p, rng)
    rings = _plan_rings(p, rng, pl)
    sup, sup_of = _plan_suppliers(p, rng, pl, rings)

    # drop suppliers nobody uses, then reindex
    used = np.unique(sup_of)
    remap = {int(o): i for i, o in enumerate(used)}
    sup = [sup[o] for o in used]
    sup_of = np.array([remap[int(x)] for x in sup_of])
    S = len(sup)

    # ---- employees and ids (random permutations: row order carries no label information)
    n_branch = len(REGIONS) * BRANCHES_PER_REGION
    emp_branch = np.repeat(np.arange(n_branch), EMPLOYEES_PER_BRANCH)
    emp_ids = _ids("E", len(emp_branch), p.id_offset // 100, rng, 5)
    loan_ids = _ids("L", N, p.id_offset, rng)
    borrower_ids = _ids("B", N, p.id_offset, rng)
    sup_ids = _ids("S", S, p.id_offset, rng)

    for j, s in enumerate(sup):
        if s["ring"]:
            if rng.random() < p.ring_employee_link:
                same = np.where(emp_branch == s["ring_branch"])[0]
                s["employee"] = int(rng.choice(same if rng.random() < 0.8 else np.arange(len(emp_branch))))
        elif rng.random() < p.honest_employee_link:
            s["employee"] = int(rng.integers(0, len(emp_branch)))

    # ---- supplier registration (relative to the supplier's first loan)
    first_disb = np.full(S, np.iinfo(np.int64).max)
    np.minimum.at(first_disb, sup_of, pl["disb"])
    age_days = np.array([rng.uniform(10, 330) if s["recent"] else 365 + rng.exponential(5 * 365)
                         for s in sup])
    registered = first_disb - (age_days * DAY).astype(np.int64)

    # ---- supplier region: usually the region of its borrowers
    sup_region = np.empty(S, dtype=np.int64)
    rw = np.array(p.region_weights, dtype=float)
    reg_by_sup = pd.Series(pl["region"]).groupby(sup_of).agg(lambda x: x.mode().iloc[0])
    for j in range(S):
        sup_region[j] = reg_by_sup[j] if rng.random() < 0.8 else rng.choice(len(REGIONS), p=rw / rw.sum())

    # ---- transactions
    em = Emitter(snap_s)
    accounts = np.array([f"ACC-{b}" for b in borrower_ids], dtype=object)
    t3_event = {}
    for i in range(N):
        L = Loan(i, loan_ids[i], accounts[i], int(pl["disb"][i]), int(pl["amount"][i]),
                 int(pl["invoice"][i]), pl["purpose"][i], int(sup_of[i]), sup_ids[sup_of[i]],
                 bool(pl["stealth"][i]))
        k = pl["kind"][i]
        if k == "honest":
            fn = {"livestock_seller": flows.honest_livestock_seller,
                  "cash_heavy": flows.honest_cash_heavy}.get(pl["profile"][i], flows.honest_normal)
            fn(em, rng, p, L)
        elif k == "T1":
            flows.t1_fast_cash_out(em, rng, p, L)
        elif k == "T2":
            flows.t2_ring_member(em, rng, p, L)
        elif k == "T3":
            t3_event[i] = flows.t3_asset_resale(em, rng, p, L)
        else:
            flows.t4_fake_invoice(em, rng, p, L)
    for j, s in enumerate(sup):
        flows.supplier_outflows(em, rng, j, sup_ids[j], f"ACC-{sup_ids[j]}", s)

    tx = pd.DataFrame(em.rows, columns=["loan_id", "account_id", "ts", "type", "amount_uzs",
                                        "counterparty_id", "counterparty_kind"])
    tx = tx.sort_values(["ts", "account_id"], kind="stable").reset_index(drop=True)
    tx.insert(0, "txn_id", [f"T{p.id_offset * 100 + i:09d}" for i in range(len(tx))])
    tx["ts"] = pd.to_datetime(tx["ts"], unit="s").astype("datetime64[ns]")
    for c in ("type", "counterparty_kind"):
        tx[c] = tx[c].astype("category")

    # ---- geo + photos
    reg_lat = np.array([r[1] for r in REGIONS])
    reg_lon = np.array([r[2] for r in REGIONS])
    b_lat = reg_lat[pl["region"]] + rng.normal(0, 0.25, N)
    b_lon = reg_lon[pl["region"]] + rng.normal(0, 0.25, N)
    fraud = pl["kind"] != "honest"
    far = rng.random(N) < np.where(fraud, p.photo_far_fraud, p.photo_far_honest)
    ang = rng.uniform(0, 2 * np.pi, N)
    rad = np.where(far, rng.uniform(0.1, 1.5, N), np.abs(rng.normal(0, 0.002, N)))
    ph_lat = b_lat + rad * np.sin(ang)
    ph_lon = b_lon + rad * np.cos(ang)
    cap = pl["disb"] + (rng.uniform(1, 20, N) * DAY).astype(np.int64)
    cap = np.minimum(cap, snap_s - (rng.uniform(1, 48, N) * 3600).astype(np.int64))
    photo_idx = np.arange(N)
    by_purpose = {k: np.where(pl["purpose"] == k)[0] for k in PURPOSES}
    ring_of = {m: r for r in rings for m in r["members"]}
    reused = rng.random(N) < np.where(fraud, p.photo_reuse_fraud, p.photo_reuse_honest)
    src_lat, src_lon, src_cap = ph_lat.copy(), ph_lon.copy(), cap.copy()
    for i in np.where(reused)[0]:
        r = ring_of.get(i)
        if r is not None and len(r["members"]) > 1 and rng.random() < 0.6:
            cand = [m for m in r["members"] if m != i]
            j = int(rng.choice(cand))
        else:
            pool = by_purpose[pl["purpose"][i]] if rng.random() < 0.7 else np.arange(N)
            j = int(rng.choice(pool))
            if j == i:
                j = (j + 1) % N
        photo_idx[i] = j
        ph_lat[i], ph_lon[i], cap[i] = src_lat[j], src_lon[j], src_cap[j]
    photo_ids = _ids("P", N, p.id_offset, rng)
    photo_id = photo_ids[photo_idx]

    # ---- frames
    name = [f"{rng.choice(GIVEN)}*** {rng.choice(GIVEN)}***" for _ in range(N)]
    loans = pd.DataFrame({
        "loan_id": loan_ids, "borrower_id": borrower_ids, "borrower_name_masked": name,
        "purpose": pd.Categorical(pl["purpose"], categories=PURPOSES),
        "amount_uzs": pl["amount"], "term_months": rng.choice([12, 24, 36], N, p=[0.3, 0.5, 0.2]).astype("int8"),
        "disbursed_at": pd.to_datetime(pl["disb"], unit="s").astype("datetime64[ns]"),
        "snapshot_date": pd.Timestamp(p.snapshot_date),
        "region": pd.Categorical([REGIONS[r][0] for r in pl["region"]]),
        "branch_id": [f"BR-{b:02d}" for b in pl["branch"]],
        "supplier_id": sup_ids[sup_of], "invoice_amount_uzs": pl["invoice"],
        "borrower_lat": b_lat.astype("float32"), "borrower_lon": b_lon.astype("float32"),
        "photo_id": photo_id, "photo_lat": ph_lat.astype("float32"), "photo_lon": ph_lon.astype("float32"),
        "photo_captured_at": pd.to_datetime(cap, unit="s").astype("datetime64[ns]"),
    }).sort_values("loan_id").reset_index(drop=True)

    sup_names = [f"{rng.choice(FIRST)} {rng.choice(MID)} {rng.choice(SUFFIX)}" for _ in range(S)]
    suppliers = pd.DataFrame({
        "supplier_id": sup_ids, "name": sup_names, "oked_code": [s["oked"] for s in sup],
        "region": pd.Categorical([REGIONS[r][0] for r in sup_region]),
        "registered_at": pd.to_datetime(registered, unit="s").astype("datetime64[ns]"),
        "linked_employee_id": [emp_ids[s["employee"]] if s["employee"] is not None else None for s in sup],
    }).sort_values("supplier_id").reset_index(drop=True)

    employees = pd.DataFrame({
        "employee_id": emp_ids, "branch_id": [f"BR-{b:02d}" for b in emp_branch],
        "role": rng.choice(EMPLOYEE_ROLES, len(emp_ids)),
    }).sort_values("employee_id").reset_index(drop=True)

    # ---- labels (kept out of the feature tables)
    loan_tx_types = tx[tx.loan_id.notna()].assign(type=lambda d: d["type"].astype(str)).groupby("loan_id")["type"].agg(set)
    has = lambda lid, t: t in loan_tx_types.get(lid, ())
    visible = np.zeros(N, dtype=bool)
    for i in np.where(fraud)[0]:
        k, lid = pl["kind"][i], loan_ids[i]
        visible[i] = (has(lid, "cash_withdrawal") if k == "T1" else
                      has(lid, "supplier_payment") if k == "T2" else
                      t3_event.get(i, snap_s + 1) <= snap_s if k == "T3" else True)
    ring_id = np.full(N, None, dtype=object)
    for r_id, r in enumerate(rings):
        for m in r["members"]:
            ring_id[m] = f"R{r_id:04d}"
    dist_km = 111.0 * np.hypot(ph_lat - b_lat, (ph_lon - b_lon) * np.cos(np.radians(b_lat)))
    labels = pd.DataFrame({
        "loan_id": loan_ids, "is_fraud": fraud,
        "typology": np.where(fraud, pl["kind"], None), "stealth": pl["stealth"],
        "profile": pl["profile"], "ring_id": ring_id, "fraud_visible": visible,
        "photo_reused": reused, "photo_far": dist_km > 5.0,
    }).sort_values("loan_id").reset_index(drop=True)

    manifest = (pd.DataFrame({"photo_id": photo_id, "asset_class": pl["purpose"][photo_idx]})
                .drop_duplicates("photo_id").sort_values("photo_id").reset_index(drop=True))
    manifest["image_file"] = None   # filled when real openly licensed images are assigned
    manifest["license"] = None

    runtime = time.perf_counter() - t_start
    meta = dict(
        params=dataclasses.asdict(p), runtime_s=round(runtime, 2),
        rows=dict(loans=len(loans), transactions=len(tx), suppliers=len(suppliers),
                  employees=len(employees), photo_manifest=len(manifest)),
        fraud_rate=round(float(fraud.mean()), 4),
        typology_counts={t: int((pl["kind"] == t).sum()) for t in ("T1", "T2", "T3", "T4")},
        fraud_not_yet_visible=int((fraud & ~visible).sum()),
        honest_profiles={k: int((pl["profile"] == k).sum()) for k in
                         ("normal", "livestock_seller", "cash_heavy", "oked_mismatch")},
        rings=len(rings), ring_sizes=[len(r["members"]) for r in rings],
    )
    return dict(loans=loans, transactions=tx, suppliers=suppliers, employees=employees,
                labels=labels, photo_manifest=manifest, meta=meta)


def write(tables: dict, split: str, out_root: Path = OUT_ROOT) -> Path:
    out = out_root / split
    out.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        if name == "meta":
            (out / "meta.json").write_text(json.dumps(df, indent=2, default=str))
        else:
            df.to_parquet(out / f"{name}.parquet", index=False)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test", "all"], default="all")
    ap.add_argument("--n", type=int, default=None, help="override number of loans")
    ap.add_argument("--seed", type=int, default=None, help="override seed")
    args = ap.parse_args(argv)
    for name in (["train", "test"] if args.split == "all" else [args.split]):
        p = SPLITS[name]
        over = {k: v for k, v in (("n_loans", args.n), ("seed", args.seed)) if v is not None}
        p = dataclasses.replace(p, **over)
        tables = build(p)
        out = write(tables, name)
        m = tables["meta"]
        print(f"[{name}] seed={p.seed} runtime={m['runtime_s']}s rows={m['rows']} "
              f"fraud_rate={m['fraud_rate']} -> {out}")


if __name__ == "__main__":
    main()
