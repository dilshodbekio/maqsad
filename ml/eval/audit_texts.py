"""Audit of every user-facing Uzbek string the ML layer produces (what judges read on screen).

    .venv/Scripts/python -m ml.eval.audit_texts --tag before|after

Collects: reasons, escalations, top_reason and evidence summaries of EVERY loan of the demo portfolio (default view); the 4 scenario
presets and 30 random valid scenarios (reasons); the messages of the invalid scenarios (ScenarioError); `asset_match.label_uz` and the
resulting reasons / escalations of `check_photo` on 20 committed images in a fresh sandbox. Flags: "None", "nan", "inf", braces,
"0 foiz", "atigi 0", impossible negative numbers, > 100% shown as percent, English words, duplicated reasons, label / matches_purpose
contradictions, empty or very short strings, doubled spaces, unbalanced brackets, lowercase start. Writes ml/artifacts/audit_texts_<tag>.json and docs/audit_texts.md.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ml import api
from ml.errors import ScenarioError
from ml.generator.config import PURPOSES
from ml.photo import asset, config
from ml.photo.index import get_index

ROOT = Path(__file__).resolve().parents[2]
ENGLISH = {"the", "and", "with", "loan", "payment", "cash", "supplier", "invoice", "photo", "asset", "score", "risk", "green", "yellow",
           "red", "true", "false", "null", "error", "none", "unknown", "match", "matches", "mismatch", "income", "transfer", "withdrawal",
           "balance", "not", "found", "invalid", "high", "low", "days", "day", "of", "for", "to", "is", "was", "no", "yes"}
NO_REASON = "Sezilarli xavf belgisi aniqlanmadi"


def flags_of(text: str, kind: str) -> list[str]:
    """Problems of one user-facing string. kind: 'reason' | 'message'."""
    f = []
    if not isinstance(text, str) or not text.strip():
        return ["empty string"]
    t = text
    if re.search(r"\bNone\b|\bnull\b|\bNaN\b|\bnan\b|\bNone\b", t):
        f.append("None / nan")
    if re.search(r"\binf\b|\bInfinity\b", t, re.I):
        f.append("inf")
    if "{" in t or "}" in t:
        f.append("braces")
    if re.search(r"(?<![\d.,])0 foiz|\b0\.0 kun|(?<![\d.,])0 ta\b", t):
        f.append("0 foiz / 0 ta / 0.0 kun")
    if re.search(r"atigi 0(?![.,\d])", t):
        f.append("atigi 0")
    if re.search(r"(?<![\d\w])-\d", t):
        f.append("negative number")
    if any(int(m) > 100 for m in re.findall(r"(\d{3,}) foiz(?! va undan ko'p)", t)):
        f.append("> 100% shown as percent")
    # field names / values of the API (in quotes, brackets or after "Mumkin:") are identifiers, not English prose
    prose = re.sub(r"`[^`]*`|\([^)]*\)|'[a-z_]+'|Mumkin:.*$", " ", t)
    words = re.findall(r"(?<![\w`_.'])[A-Za-z]+(?![\w`_'])", prose)
    eng = sorted({w.lower() for w in words if w.lower() in ENGLISH})
    if eng:
        f.append("English: " + ",".join(eng))
    if "  " in t:
        f.append("double space")
    if t.count("(") != t.count(")"):
        f.append("unbalanced brackets")
    if t.strip() != t or (kind == "reason" and len(t) < 8):
        f.append("too short / padded")
    if kind == "reason" and t[0].isalpha() and t[0].islower():
        f.append("lowercase start")
    if re.search(r"foiz\s+(i|ini|iga)\b|%", t):
        f.append("percent suffix style")
    return f


def collect_portfolio() -> list[tuple[str, str, str, str]]:
    """(source, kind, text, context) for every loan of the default queue."""
    out = []
    q = api.get_queue()
    for item in q:
        out.append(("queue.top_reason", "reason", item["top_reason"], item["loan_id"]))
        for e in item["escalations"]:
            out.append(("queue.escalation", "reason", e["text_uz"], item["loan_id"]))
    for item in q:
        d = api.get_loan(item["loan_id"])
        reasons = [r["text_uz"] for r in d["reasons"]]
        for r in d["reasons"]:
            out.append(("loan.reason", "reason", r["text_uz"], item["loan_id"]))
        for e in d["evidence"]:
            out.append(("loan.evidence", "message", e["summary_uz"], item["loan_id"]))
        if len(reasons) != len(set(reasons)):
            out.append(("loan.duplicate_reason", "reason", "DUPLICATE: " + " | ".join(reasons), item["loan_id"]))
        codes = [r["code"] for r in d["reasons"]]
        if len(codes) != len(set(codes)):
            out.append(("loan.duplicate_code", "reason", "DUPLICATE CODE: " + ",".join(codes), item["loan_id"]))
        expected = reasons[0] if reasons else NO_REASON
        if item["top_reason"] != expected:
            out.append(("queue.top_reason!=reasons[0]", "reason", f"{item['top_reason']} != {expected}", item["loan_id"]))
    return out


def random_scenario(rng: random.Random, P) -> dict:
    amount = rng.randint(5, 200) * 1_000_000
    snap = rng.choice([2, 3, 5, 30, 90, 200, 364])
    sup = (dict(mode="existing", supplier_id=rng.choice(list(P.suppliers.index))) if rng.random() < 0.5 else
           dict(mode="new", oked=f"{rng.randint(1, 99):02d}.{rng.randint(0, 99):02d}", registered_days_before=rng.choice([0, 10, 500, 5000])))
    budget, txs = amount, []
    for _ in range(rng.randint(0, 12)):
        typ = rng.choice(["cash_withdrawal", "supplier_payment", "p2p_out", "p2p_in", "external_transfer", "income"])
        a = rng.randint(1, max(1, amount // 10))
        if typ not in ("p2p_in", "income"):
            if a > budget:
                continue
            budget -= a
        txs.append(dict(day=rng.randint(0, snap), type=typ, amount_uzs=a))
    scn = dict(purpose=rng.choice(PURPOSES), amount_uzs=amount, snapshot_day=snap, supplier=sup, transactions=txs)
    if rng.random() < 0.7:
        scn["invoice_amount_uzs"] = amount
    return scn


def collect_scenarios() -> list[tuple[str, str, str, str]]:
    from ml.money_trail.service import get_portfolio
    P = get_portfolio()
    out = []
    scns = [("preset:" + p["id"], p["scenario"]) for p in api.get_scenario_presets()]
    rng = random.Random(2026)
    scns += [(f"random:{i}", random_scenario(rng, P)) for i in range(30)]
    for name, scn in scns:
        r = api.score_scenario(scn)
        texts = [x["text_uz"] for x in r["reasons"]]
        for t in texts:
            out.append(("scenario.reason", "reason", t, name))
        if len(texts) != len(set(texts)):
            out.append(("scenario.duplicate_reason", "reason", "DUPLICATE: " + " | ".join(texts), name))
    for p in api.get_scenario_presets():
        out.append(("preset.title", "message", p["title_uz"], p["id"]))
        out.append(("preset.description", "message", p["description_uz"], p["id"]))
    base = dict(purpose="chorva", amount_uzs=10_000_000, snapshot_day=30, supplier=dict(mode="new", oked="01.41", registered_days_before=100),
                transactions=[dict(day=1, type="cash_withdrawal", amount_uzs=1_000_000)])
    mutations = [
        lambda s: s.update(purpose="x"), lambda s: s.update(amount_uzs=0), lambda s: s.update(amount_uzs="10"), lambda s: s.update(amount_uzs=float("nan")),
        lambda s: s.update(snapshot_day=1), lambda s: s.update(snapshot_day=365), lambda s: s.update(region="Mars"), lambda s: s.update(foo=1),
        lambda s: s["supplier"].update(mode="x"), lambda s: s["supplier"].update(oked="abc"), lambda s: s["supplier"].update(registered_days_before=-1),
        lambda s: s["supplier"].update(bar=1), lambda s: s.update(supplier=dict(mode="existing", supplier_id="S-nope")),
        lambda s: s["transactions"][0].update(type="teleport"), lambda s: s["transactions"][0].update(amount_uzs=0),
        lambda s: s["transactions"][0].update(day=31), lambda s: s["transactions"][0].update(counterparty_kind="alien"),
        lambda s: s["transactions"][0].update(note="x"), lambda s: s["transactions"][0].update(amount_uzs=10_000_001),
        lambda s: s["transactions"][0].update(amount_uzs=10 ** 9), lambda s: s.update(transactions=[dict(day=1, type="income", amount_uzs=1000)] * 61),
        lambda s: s.update(transactions="none"), lambda s: s.update(invoice_amount_uzs=10 ** 9),
        lambda s: s.update(transactions=[dict(day=d, type="income", amount_uzs=10 * 10_000_000) for d in (1, 2)] + [dict(day=3, type="p2p_in", amount_uzs=1)]),
        lambda s: s.update(transactions=[dict(day=1, type="cash_withdrawal", amount_uzs=6_000_000), dict(day=2, type="supplier_payment", amount_uzs=5_000_000)]),
    ]
    for k, mut in enumerate(mutations):
        s = copy.deepcopy(base)
        mut(s)
        try:
            api.score_scenario(s)
            out.append(("scenario.error", "message", "NO ERROR RAISED for invalid scenario %d" % k, f"mutation:{k}"))
        except ScenarioError as e:
            out.append(("scenario.error", "message", str(e), f"mutation:{k}"))
    return out


def collect_photos() -> tuple[list[tuple[str, str, str, str]], list[dict]]:
    from ml.money_trail.service import get_portfolio
    P = get_portfolio()
    idx = get_index()
    rows = list(csv.DictReader(open(config.ASSETS / "manifest.csv", encoding="utf-8")))
    pick = [r for p in PURPOSES for r in [x for x in rows if x["purpose"] == p][:2]] + [x for x in rows if x["purpose"] == "other"][:6] \
        + [x for x in rows if x["purpose"] == "faces_test"][:4]
    free = {p: [l for l in P.loans.index if str(P.loans.at[l, "purpose"]) == p and l not in idx.by_loan] for p in PURPOSES}
    sb, out, contradictions = "audit-fresh", [], []
    api.reset_sandbox(sb)
    for k, r in enumerate(pick):
        target = PURPOSES[(PURPOSES.index(r["purpose"]) + (k % 2)) % 5] if r["purpose"] in PURPOSES else PURPOSES[k % 5]   # matching or not
        loan = free[target][k]
        res = api.check_photo(loan, (config.ASSETS / r["file"]).read_bytes(), sandbox_id=sb)
        am = res["asset_match"]
        if am is None:
            out.append(("photo.asset_match", "message", "asset_match is None", r["file"]))
            continue
        out.append(("photo.label_uz", "message", am["label_uz"], r["file"]))
        d = api.get_loan(loan, sb)
        for e in d["escalations"]:
            out.append(("photo.escalation", "reason", e["text_uz"], f"{r['file']}->{loan}"))
        for x in d["reasons"]:
            out.append(("photo.reason", "reason", x["text_uz"], f"{r['file']}->{loan}"))
        # label / matches_purpose / text contradictions
        mism = [e["text_uz"] for e in d["escalations"] if e["code"] == "PHOTO_ASSET_MISMATCH"]
        bad = None
        if am["matches_purpose"] and (am["label"] != target or mism):
            bad = "matches_purpose True but label != purpose or a mismatch reason exists"
        if not am["matches_purpose"] and am["label"] == target and any("mos aktiv aniqlanmadi" in t for t in mism):
            bad = "label equals the purpose but the text says 'no matching asset'"
        if not am["matches_purpose"] and not mism:
            bad = "matches_purpose False but no PHOTO_ASSET_MISMATCH reason"
        if bad:
            contradictions.append(dict(file=r["file"], loan=loan, purpose=target, asset=am, problem=bad, texts=mism))
    api.reset_sandbox(sb)
    return out, contradictions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="after")
    a = ap.parse_args()
    api.warm_up()
    items = collect_portfolio() + collect_scenarios()
    photo_items, contradictions = collect_photos()
    items += photo_items
    flagged = defaultdict(list)
    pre = {"loan.duplicate_reason", "loan.duplicate_code", "scenario.duplicate_reason", "queue.top_reason!=reasons[0]"}
    for source, kind, text, ctx in items:
        if source in pre or text.startswith("NO ERROR RAISED") or text == "asset_match is None":
            flagged[source if source in pre else text.split(" for ")[0]].append((text, ctx, source))
            continue
        for fl in flags_of(text, kind):
            flagged[fl].append((text, ctx, source))
    for c in contradictions:
        flagged["label/matches_purpose contradiction"].append((c["problem"], c["file"], "photo"))
    counts = Counter({k: len(v) for k, v in flagged.items()})
    uniq = {k: sorted({t[0] for t in v})[:6] for k, v in flagged.items()}
    reasons = sorted({t for s, kd, t, c in items if kd == "reason" and s in ("loan.reason", "scenario.reason", "photo.reason", "queue.top_reason")})
    sample = random.Random(1).sample(reasons, 10)
    res = dict(tag=a.tag, n_strings=len(items), n_unique=len({t for _, _, t, _ in items}), counts=dict(counts), examples=uniq,
               contradictions=contradictions, sample_reasons=sample,
               sources=dict(Counter(s for s, _, _, _ in items)))
    (ROOT / "ml" / "artifacts" / f"audit_texts_{a.tag}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8", newline="\n")
    render()
    print(json.dumps({k: res[k] for k in ("n_strings", "n_unique", "counts")}, indent=1, ensure_ascii=False))


def render():
    art = ROOT / "ml" / "artifacts"
    b = json.loads((art / "audit_texts_before.json").read_text(encoding="utf-8")) if (art / "audit_texts_before.json").exists() else None
    af = json.loads((art / "audit_texts_after.json").read_text(encoding="utf-8")) if (art / "audit_texts_after.json").exists() else None
    L = ["# Text audit (user-facing Uzbek strings)", "",
         "Produced by `python -m ml.eval.audit_texts --tag before|after`. Every reason, escalation, `top_reason`, evidence summary, scenario reason, "
         "`ScenarioError` message and asset label of: all loans of the demo portfolio (default view), the 4 scenario presets, 30 random valid scenarios, "
         "the invalid scenarios and `check_photo` on 20 committed images in a fresh sandbox is checked for: None / nan / inf, braces, '0 foiz', 'atigi 0', "
         "impossible negative numbers, > 100% shown as percent, English words, duplicated reasons, `top_reason` != `reasons[0]`, label / matches_purpose "
         "contradictions, empty or very short strings, double spaces, unbalanced brackets, lowercase start, percent-suffix style.", ""]
    L += ["**How to read the table.** 'before' = this checker run on the templates as they were before this round; 'after' = the final templates. "
          "The checker itself was corrected after its first run (it had counted the Uzbek word \"to'lanmagan\" as English 'to' 3 984 times and treated "
          "field names in brackets as English); the numbers below are from the corrected checker for both columns.", ""]
    if b or af:
        keys = sorted(set((b or {}).get("counts", {})) | set((af or {}).get("counts", {})))
        L += ["| Check | before | after |", "|---|---|---|"]
        for k in keys:
            L.append(f"| {k} | {(b or {}).get('counts', {}).get(k, 0)} | {(af or {}).get('counts', {}).get(k, 0)} |")
        L.append(f"| **strings checked** | {(b or {}).get('n_strings', '-')} | {(af or {}).get('n_strings', '-')} |")
        L.append(f"| unique strings | {(b or {}).get('n_unique', '-')} | {(af or {}).get('n_unique', '-')} |")
        L.append("")
        if af and not af["counts"]:
            L += ["**After the fixes the audit finds nothing to flag.**", ""]
    if af:
        L += ["## Ten random reasons from the final run", ""] + [f"- {t}" for t in af["sample_reasons"]] + [""]
        if af["examples"]:
            L += ["## Remaining flags (first examples)", ""] + [f"- **{k}**: " + " / ".join(v[:3]) for k, v in af["examples"].items()] + [""]
    (ROOT / "docs" / "audit_texts.md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
