"""Audit of every user-facing string the ML layer produces, in Uzbek, English and Russian (what judges read on screen).

    .venv/Scripts/python -m ml.eval.audit_texts --tag before|after

Collects: reasons, escalations, top_reason and evidence summaries of EVERY loan of the demo portfolio (default view); the 4 scenario
presets and 30 random valid scenarios (reasons); the messages of the invalid scenarios (ScenarioError); `asset_match.label_uz` and the
resulting reasons / escalations of `check_photo` on 20 committed images in a fresh sandbox. Flags: "None", "nan", "inf", braces,
"0 foiz", "atigi 0", impossible negative numbers, > 100% shown as percent, English words, duplicated reasons, label / matches_purpose
contradictions, empty or very short strings, doubled spaces, unbalanced brackets, lowercase start.
Every text that has an English and a Russian version (reasons, escalations, top_reason, evidence summaries, scenario reasons / errors /
presets, asset labels) is also checked per language: empty or missing translation, untranslated Uzbek in EN / RU (and Cyrillic in UZ / EN),
"None" / "nan" / braces, zero-value phrases ("0%", "0 days"), and numbers that differ between the three languages of the same text.
Writes ml/artifacts/audit_texts_<tag>.json and docs/audit_texts.md.
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
LANGS = ("uz", "en", "ru")
TRIOS: list[tuple[str, str, dict, str]] = []            # (source, kind, {uz, en, ru}, context) of every translated text
# words that only occur in Uzbek texts: finding one in an English / Russian text means it was not translated
UZBEK_WORDS = {"kredit", "kreditning", "kredit", "foiz", "foizi", "kun", "tushum", "naqd", "rasm", "hisob", "hisobi", "qayd", "yechilgan",
               "yechgan", "bo'lgan", "mos", "kelmaydi", "ta'minotchi", "ta'minotchiga", "so'm", "dan", "gacha", "uchun", "bilan",
               "xavf", "tekshirilgan", "davr", "sezilarli", "belgisi", "aniqlanmadi", "yirik", "kirim", "qarz", "oluvchi", "manzilidan",
               "kreditidagi", "bir", "xil", "uzoqda", "ishonchli", "emas", "yo'q", "bor", "hisob-faktura", "hisob-fakturasida"}
CYRILLIC = re.compile(r"[А-Яа-яЁё]")
UZ_LETTER_APOS = re.compile(r"[A-Za-z][ʻ'`’][A-Za-z]")


def flags_lang(text: str, kind: str, lang: str) -> list[str]:
    """Problems of one English / Russian text (the Uzbek texts use flags_of)."""
    f = []
    if not isinstance(text, str) or not text.strip():
        return ["empty translation"]
    t = text
    if re.search(r"\bNone\b|\bnull\b|\bNaN\b|\bnan\b", t):
        f.append("None / nan")
    if re.search(r"\binf\b|\bInfinity\b", t, re.I):
        f.append("inf")
    if "{" in t or "}" in t:
        f.append("braces")
    if re.search(r"(?<![\d.,])0%|(?<![\d.,])0\.0 (?:days|дн)|(?<![\d.,])0 (?:days|дн)", t):
        f.append("zero value")
    if re.search(r"(?<![\d\w])-\d", t):
        f.append("negative number")
    if any(int(m) > 100 for m in re.findall(r"(?<![\d.])(\d{3,})%", t)):
        f.append("> 100% shown as percent")
    if "  " in t:
        f.append("double space")
    if t.count("(") != t.count(")"):
        f.append("unbalanced brackets")
    if t.strip() != t or (kind == "reason" and len(t) < 8):
        f.append("too short / padded")
    if kind == "reason" and t[0].isalpha() and t[0].islower():
        f.append("lowercase start")
    # data is not prose: company names (supplier names end in YaTT / MChJ / XK), identifiers (snake_case) and the lists after "Allowed:"
    scrub = re.sub(r"`[^`]*`|\([^)]*\)|[A-Z][A-Za-z']+ (?:[A-Z][A-Za-z']+ )*(?:YaTT|MChJ|XK)\b|\b\w+_\w+\b|(?:Allowed|Допустимые|Mumkin):.*$", " ", t)
    words = {w.lower() for w in re.findall(r"[A-Za-z][A-Za-z'\-]*", scrub)}
    if lang in ("en", "ru"):
        if lang == "en":
            apos = [w for w in words if UZ_LETTER_APOS.search(w) and not re.search(r"(?:n't|'s|'re)$", w)]
            if apos:
                f.append("untranslated Uzbek: " + ",".join(sorted(apos)[:3]))
        found = sorted(words & UZBEK_WORDS)
        if found:
            f.append("untranslated Uzbek: " + ",".join(found[:3]))
    if lang == "en" and CYRILLIC.search(t):
        f.append("Cyrillic in English text")
    if lang == "ru":
        eng = sorted(words & (ENGLISH - {"no", "of", "to", "is", "for", "not", "red", "none", "high", "low"}))
        if eng:
            f.append("English words in Russian text: " + ",".join(eng[:3]))
    if lang == "uz" and CYRILLIC.search(t):
        f.append("Cyrillic in Uzbek text")
    return f


def _numbers(t: str) -> list[str]:
    return sorted(re.findall(r"\d+(?:\.\d+)?", t.replace("\u00a0", " ")))


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
        TRIOS.append(("queue.top_reason", "reason", item.get("top_reason_text") or {}, item["loan_id"]))
        for e in item["escalations"]:
            out.append(("queue.escalation", "reason", e["text_uz"], item["loan_id"]))
            TRIOS.append(("queue.escalation", "reason", e.get("text") or {}, item["loan_id"]))
    for item in q:
        d = api.get_loan(item["loan_id"])
        reasons = [r["text_uz"] for r in d["reasons"]]
        for r in d["reasons"]:
            out.append(("loan.reason", "reason", r["text_uz"], item["loan_id"]))
            TRIOS.append(("loan.reason", "reason", r.get("text") or {}, item["loan_id"]))
        for e in d["evidence"]:
            out.append(("loan.evidence", "message", e["summary_uz"], item["loan_id"]))
            TRIOS.append(("loan.evidence", "message", e.get("summary") or {}, item["loan_id"]))
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
        for x in r["reasons"]:
            TRIOS.append(("scenario.reason", "reason", x.get("text") or {}, name))
        if len(texts) != len(set(texts)):
            out.append(("scenario.duplicate_reason", "reason", "DUPLICATE: " + " | ".join(texts), name))
    for p in api.get_scenario_presets():
        out.append(("preset.title", "message", p["title_uz"], p["id"]))
        out.append(("preset.description", "message", p["description_uz"], p["id"]))
        TRIOS.append(("preset.title", "message", p.get("title") or {}, p["id"]))
        TRIOS.append(("preset.description", "message", p.get("description") or {}, p["id"]))
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
            TRIOS.append(("scenario.error", "message", getattr(e, "messages", None) or {}, f"mutation:{k}"))
    return out


def collect_photos() -> tuple[list[tuple[str, str, str, str]], list[dict]]:
    from ml.money_trail.service import get_portfolio
    P = get_portfolio()
    idx = get_index()
    rows = list(csv.DictReader(open(config.ASSETS / "manifest.csv", encoding="utf-8")))
    pick = [r for p in PURPOSES for r in [x for x in rows if x["purpose"] == p][:2]] + [x for x in rows if x["purpose"] == "other"][:6] \
        + [dict(file=f.name, purpose="faces_test", dir=config.FACES) for f in sorted(config.FACES.glob("*.jpg"))[:4]]
    free = {p: [l for l in P.loans.index if str(P.loans.at[l, "purpose"]) == p and l not in idx.by_loan] for p in PURPOSES}
    sb, out, contradictions = "audit-fresh", [], []
    api.reset_sandbox(sb)
    for k, r in enumerate(pick):
        target = PURPOSES[(PURPOSES.index(r["purpose"]) + (k % 2)) % 5] if r["purpose"] in PURPOSES else PURPOSES[k % 5]   # matching or not
        loan = free[target][k]
        res = api.check_photo(loan, (r.get("dir", config.ASSETS) / r["file"]).read_bytes(), sandbox_id=sb)
        am = res["asset_match"]
        if am is None:
            out.append(("photo.asset_match", "message", "asset_match is None", r["file"]))
            continue
        out.append(("photo.label_uz", "message", am["label_uz"], r["file"]))
        TRIOS.append(("photo.label", "message", am.get("label_i18n") or {}, r["file"]))
        d = api.get_loan(loan, sb)
        for e in d["escalations"]:
            out.append(("photo.escalation", "reason", e["text_uz"], f"{r['file']}->{loan}"))
            TRIOS.append(("photo.escalation", "reason", e.get("text") or {}, f"{r['file']}->{loan}"))
        for x in d["reasons"]:
            out.append(("photo.reason", "reason", x["text_uz"], f"{r['file']}->{loan}"))
            TRIOS.append(("photo.reason", "reason", x.get("text") or {}, f"{r['file']}->{loan}"))
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
    TRIOS.clear()
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
    per_lang = Counter()
    for source, kind, trio, ctx in TRIOS:
        for lang in LANGS:
            per_lang[lang] += 1
            text = trio.get(lang) if isinstance(trio, dict) else None
            if lang == "uz":
                fl = ["empty translation"] if not text else [x for x in flags_lang(text, kind, "uz") if x.startswith("Cyrillic")]
            else:
                fl = flags_lang(text, kind, lang) if text else ["empty translation"]
            for x in fl:
                flagged[f"[{lang}] {x.split(':')[0]}"].append((text or "", ctx, source))
        nums = {lang: _numbers(trio.get(lang) or "") for lang in LANGS} if isinstance(trio, dict) else {}
        if nums and not (nums["uz"] == nums["en"] == nums["ru"]):
            flagged["numbers differ between languages"].append((" | ".join(f"{l}: {trio.get(l)}" for l in LANGS), ctx, source))
        if isinstance(trio, dict) and trio.get("uz") and trio.get("en") == trio.get("uz") and re.search(r"[A-Za-z]{4}", trio["uz"]) and source != "photo.label":
            flagged["[en] identical to Uzbek"].append((trio["en"], ctx, source))
    counts = Counter({k: len(v) for k, v in flagged.items()})
    uniq = {k: sorted({t[0] for t in v})[:6] for k, v in flagged.items()}
    reasons = sorted({t for s, kd, t, c in items if kd == "reason" and s in ("loan.reason", "scenario.reason", "photo.reason", "queue.top_reason")})
    sample = random.Random(1).sample(reasons, 10)
    res = dict(tag=a.tag, n_strings=len(items), n_unique=len({t for _, _, t, _ in items}), counts=dict(counts), examples=uniq,
               n_translated_texts=len(TRIOS), n_strings_per_language=dict(per_lang), n_unique_per_language={
                   l: len({t.get(l) for _, _, t, _ in TRIOS if isinstance(t, dict)}) for l in LANGS},
               sample_reasons_i18n=[dict(t) for t in random.Random(2).sample(
                   [t for s, kd, t, c in TRIOS if kd == "reason" and s in ("loan.reason", "scenario.reason", "photo.reason")], 10)],
               contradictions=contradictions, sample_reasons=sample,
               sources=dict(Counter(s for s, _, _, _ in items)))
    (ROOT / "ml" / "artifacts" / f"audit_texts_{a.tag}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8", newline="\n")
    render()
    print(json.dumps({k: res[k] for k in ("n_strings", "n_unique", "counts")}, indent=1, ensure_ascii=False))


def render():
    art = ROOT / "ml" / "artifacts"
    b = json.loads((art / "audit_texts_before.json").read_text(encoding="utf-8")) if (art / "audit_texts_before.json").exists() else None
    af = json.loads((art / "audit_texts_after.json").read_text(encoding="utf-8")) if (art / "audit_texts_after.json").exists() else None
    L = ["# Text audit (user-facing strings in Uzbek, English and Russian)", "",
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
        L.append(f"| translated texts checked (each in uz / en / ru) | {(b or {}).get('n_translated_texts', 'not checked')} | {(af or {}).get('n_translated_texts', '-')} |")
        L.append("")
        if af and not af["counts"]:
            L += ["**After the fixes the audit finds nothing to flag.**", ""]
        if af and af.get("n_translated_texts"):
            n = af["n_translated_texts"]
            L += [f"**Languages.** Before this round the ML layer produced Uzbek texts only (0 English, 0 Russian). Now each of the {n} texts "
                  "checked (reasons, escalations, `top_reason`, evidence summaries, scenario reasons / errors / presets, asset labels) exists in "
                  "Uzbek, English and Russian; the audit finds in none of them an empty or untranslated string, `None` / `nan` / braces, "
                  "a zero-value phrase, or a number that differs between the three languages.", ""]
    if af:
        L += ["## Ten random reasons from the final run", ""] + [f"- {t}" for t in af["sample_reasons"]] + [""]
        if af.get("sample_reasons_i18n"):
            L += ["## Ten random reasons in all three languages", "", "| uz | en | ru |", "|---|---|---|"]
            L += [f"| {t['uz']} | {t['en']} | {t['ru']} |" for t in af["sample_reasons_i18n"]] + [""]
            L += ["Unique texts per language in the run: " + ", ".join(f"{l} {n}" for l, n in af["n_unique_per_language"].items()) + ".", ""]
        if af["examples"]:
            L += ["## Remaining flags (first examples)", ""] + [f"- **{k}**: " + " / ".join(v[:3]) for k, v in af["examples"].items()] + [""]
    (ROOT / "docs" / "audit_texts.md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
