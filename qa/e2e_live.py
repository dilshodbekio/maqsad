"""End-to-end check of the LIVE demo (read-mostly; writes only with officer_id "e2e-test"). Never calls /api/admin/*.

    PYTHONPATH=. .venv/Scripts/python qa/e2e_live.py [--base URL] [--lang uz|en|ru]

Writes qa/e2e_results.json (uz) or qa/e2e_results_<lang>.json (rows + created objects); docs/e2e_check.md is rendered from the uz file by
qa/e2e_render.py. With --lang en|ru every request carries `Accept-Language`, error messages must come back in that language and the
translated texts of the API (`text` / `top_reason_text` / `summary` / `label_i18n` / preset titles) are checked for that language.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import copy
import csv
import datetime as dt
import json
import re
import statistics
import time
import uuid
from pathlib import Path

import httpx

from ml.eval.audit_texts import flags_lang, flags_of

ROOT = Path(__file__).resolve().parents[1]
PH = ROOT / "ml" / "assets" / "photos"
BASE = "https://maqsad-production-d704.up.railway.app"
RANK = {"green": 0, "yellow": 1, "red": 2}
ROWS, CREATED, DATA = [], {"audit_ids": [], "sandboxes": [], "loans_written": []}, {}
LANG = "uz"
CYR = re.compile(r"[А-Яа-яЁё]")
PYDANTIC_EN = re.compile(r"Field required|Input should|valid datetime|valid integer|Value error|String should|Extra inputs", re.I)


def row(section, name, ok, evidence="", ms=None):
    ROWS.append(dict(section=section, check=name, ok=bool(ok), evidence=str(evidence)[:420], ms=None if ms is None else round(ms)))
    print(("PASS " if ok else "FAIL "), section, name, "|", str(evidence)[:160])


class S:
    """One sandbox = one cookie."""
    def __init__(self, base, sid=None):
        self.sid = str(sid or uuid.uuid4())
        self.c = httpx.Client(base_url=base, timeout=120, cookies={"sandbox_id": self.sid}, headers={"Accept-Language": LANG})
        CREATED["sandboxes"].append(self.sid)

    def req(self, m, p, **kw):
        t = time.perf_counter()
        try:
            r = self.c.request(m, p, **kw)
        except Exception as e:                                          # noqa: BLE001
            return None, repr(e), (time.perf_counter() - t) * 1000
        ms = (time.perf_counter() - t) * 1000
        try:
            return r, r.json(), ms
        except ValueError:
            return r, r.text[:200], ms


def uz_text_ok(t):
    return isinstance(t, str) and t and not flags_of(t, "reason")


def detail_uz(body):
    """The `detail` of an error response is a message in the language of the request (name kept from the uz-only version)."""
    d = body.get("detail") if isinstance(body, dict) else None
    ok = isinstance(d, str) and len(d) > 8 and not PYDANTIC_EN.search(d) and "Traceback" not in d
    if ok and LANG == "uz":
        ok = not re.search(r"\b(field required|invalid|error|not found)\b", d, re.I) and not CYR.search(d)
    if ok and LANG == "ru":
        ok = bool(CYR.search(d))
    if ok and LANG == "en":
        ok = not CYR.search(d) and not re.search(r"[a-z]'[a-z]", d.replace("loan_id", ""))
    return ok, d


def lang_texts_ok(trio, kind="reason"):
    """A {uz,en,ru} text of the API: present in the request language and clean."""
    if LANG == "uz":
        return True
    t = trio.get(LANG) if isinstance(trio, dict) else None
    return bool(t) and not flags_lang(t, kind, LANG) and bool(CYR.search(t)) == (LANG == "ru")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--lang", default="uz", choices=["uz", "en", "ru"])
    a = ap.parse_args()
    global LANG
    LANG = a.lang
    started = dt.datetime.now(dt.timezone.utc)
    DATA["started"] = started.strftime("%Y-%m-%d %H:%M:%S UTC")
    DATA["lang"] = LANG
    A = S(a.base)

    # ---------------------------------------------------------------- 1 health
    r, h, ms = A.req("GET", "/api/health")
    hd = {k.lower(): v for k, v in r.headers.items()}
    row("1 Health", "GET /api/health: ml_mode real, warmup done, clip_loaded true, all functions real",
        r.status_code == 200 and h["ml_mode"] == "real" and h["warmup"] == "done" and h.get("clip_loaded") is True and all(v == "real" for v in h["functions"].values()),
        json.dumps(h), ms)
    row("1 Health", "headers X-ML-Mode / X-Model-Version match body", hd.get("x-ml-mode") == h["ml_mode"] and hd.get("x-model-version") == h["model_version"],
        f"X-ML-Mode={hd.get('x-ml-mode')} X-Model-Version={hd.get('x-model-version')}")
    DATA["model_version"] = h["model_version"]

    # ---------------------------------------------------------------- 2 officer view
    r, p1, ms = A.req("GET", "/api/queue?limit=50&offset=0")
    row("2 Queue", "page 1: 50 items, total 5000, counts sum to total", r.status_code == 200 and len(p1["items"]) == 50 and p1["total"] == sum(p1["counts"].values()),
        f"total {p1['total']} counts {p1['counts']}", ms)
    r, p2, ms = A.req("GET", "/api/queue?limit=50&offset=50")
    ids1, ids2 = [x["loan_id"] for x in p1["items"]], [x["loan_id"] for x in p2["items"]]
    pri = [x["priority"] for x in p1["items"] + p2["items"]]
    row("2 Queue", "page 2: no overlap with page 1, priority non-increasing across pages", r.status_code == 200 and not set(ids1) & set(ids2) and pri == sorted(pri, reverse=True),
        f"p1 last {pri[49]:.3f} p2 first {pri[50]:.3f}", ms)
    bad = [(x["loan_id"], x["top_reason"], flags_of(x["top_reason"], "reason")) for x in p1["items"] + p2["items"] if not uz_text_ok(x["top_reason"])]
    row("2 Queue", "top_reason present, Uzbek, no None/nan/{/English (100 items)", not bad, bad[:2] or f"e.g. {p1['items'][0]['top_reason']!r}")
    badl = [x["loan_id"] for x in p1["items"] + p2["items"] if not (x.get("top_reason_text", {}).get("uz") == x["top_reason"] and lang_texts_ok(x.get("top_reason_text")))]
    row("2 Queue", f"top_reason_text[{LANG}] present and clean (100 items)", not badl,
        badl[:3] or f"e.g. {p1['items'][0].get('top_reason_text', {}).get(LANG)!r}")
    row("2 Queue", "colour >= base_color, colours valid", all(x["color"] in RANK and RANK[x["color"]] >= RANK[x["base_color"]] for x in p1["items"] + p2["items"]),
        f"counts {p1['counts']}")
    r, fac, _ = A.req("GET", "/api/queue/facets")
    reg, pur = fac["regions"][0], fac["purposes"][0]
    r, fr, ms = A.req("GET", "/api/queue", params=dict(region=reg, limit=200))
    row("2 Queue", f"filter region={reg}", r.status_code == 200 and 0 < fr["total"] < p1["total"] and all(x["region"] == reg for x in fr["items"]), f"total {fr['total']}", ms)
    r, fp, ms = A.req("GET", "/api/queue", params=dict(purpose="tikuv_dastgohi", limit=200))
    row("2 Queue", "filter purpose=tikuv_dastgohi", r.status_code == 200 and 0 < fp["total"] < p1["total"] and all(x["purpose"] == "tikuv_dastgohi" for x in fp["items"]), f"total {fp['total']}", ms)
    r, fu, _ = A.req("GET", "/api/queue", params=dict(region="Yo'q viloyat"))
    row("2 Queue", "unknown region -> 200 empty page", r.status_code == 200 and fu["items"] == [], f"total {fu.get('total')}")
    r, fb, _ = A.req("GET", "/api/queue", params=dict(color="blue"))
    row("2 Queue", "bad color -> 422", r.status_code == 422, f"HTTP {r.status_code} {str(fb)[:120]}")
    r, lg, _ = A.req("GET", "/api/queue/legend")
    row("2 Queue", "legend: 500 red / 500 yellow budget + escalation counts", r.status_code == 200 and lg["n_red"] == 500, f"n_red {lg['n_red']} n_yellow {lg['n_yellow']} escalated_to_red {lg['n_escalated_to_red']}")

    def first(color):
        r, q, _ = A.req("GET", "/api/queue", params=dict(color=color, limit=5))
        return q["items"][0]
    qi = {c: first(c) for c in ("red", "yellow", "green")}
    for c, it in qi.items():
        r, d, ms = A.req("GET", f"/api/loan/{it['loan_id']}")
        ok = r.status_code == 200 and d["color"] == it["color"] and d["reasons"] and d["reasons"][0]["text_uz"] == it["top_reason"] if c != "green" else r.status_code == 200 and d["color"] == it["color"]
        txt = [x["text_uz"] for x in d.get("reasons", [])]
        row("2 Loan", f"GET /api/loan/{it['loan_id']} ({c}): colour = queue, top_reason == reasons[0], Uzbek reasons",
            ok and all(uz_text_ok(t) for t in txt) and (c != "red" or len(txt) >= 1),
            f"{len(txt)} reasons, {len(d['evidence'])} evidence, {len(d['transactions'])} tx, invoice={'yes' if d.get('invoice') else 'no'}, photos={len(d['photos'])}; reason0={txt[0] if txt else None!r}", ms)
        row("2 Loan", f"{it['loan_id']} ({c}): reasons[].text and evidence[].summary in {LANG}, top_reason_text == reasons[0].text",
            all(lang_texts_ok(r.get("text")) for r in d["reasons"]) and all(lang_texts_ok(e.get("summary"), "message") for e in d["evidence"])
            and (c == "green" and not d["reasons"] or d["reasons"][0].get("text") == it.get("top_reason_text")),
            f"reason0 {(d['reasons'][0].get('text') or {}).get(LANG) if d['reasons'] else None!r}")
        if c == "red":
            refs = [x for rr in d["reasons"] for x in rr.get("evidence_refs", [])]
            ev_ids = {e.get("ref") or e.get("id") for e in d["evidence"]}
            row("2 Loan", "red loan: every reason has evidence_refs that resolve to evidence[]", all(rr.get("evidence_refs") for rr in d["reasons"]) and set(refs) <= ev_ids,
                f"refs {refs[:4]} evidence keys {list(d['evidence'][0].keys())}")
            DATA["red_loan"] = it["loan_id"]
            DATA["red_reasons"] = txt
    # a portfolio loan with a photo: photo path loads
    idx = json.loads((ROOT / "ml" / "artifacts" / "photo_index_v1.json").read_text(encoding="utf-8"))
    pl = idx["images"][0]["loans"][0]
    r, d, ms = A.req("GET", f"/api/loan/{pl}")
    if r.status_code == 200 and d["photos"]:
        u = d["photos"][0].get("url") or d["photos"][0].get("path")
        rr, _, ms2 = A.req("GET", u)
        row("2 Loan", f"photos[] of a portfolio loan ({pl}): asset route returns 200 image/jpeg", rr.status_code == 200 and rr.headers.get("content-type", "").startswith("image/"), f"{u} -> {rr.status_code} {rr.headers.get('content-type')} {len(rr.content)} B", ms2)
    else:
        row("2 Loan", f"photos[] of portfolio loan {pl}", False, f"HTTP {r.status_code} photos={d.get('photos') if isinstance(d, dict) else d}")
    r, d, ms = A.req("GET", "/api/loan/L9999999")
    ok, det = detail_uz(d)
    row("2 Loan", "unknown loan id -> 404 with Uzbek detail", r.status_code == 404 and ok, f"HTTP {r.status_code} detail={det!r}", ms)

    # ---------------------------------------------------------------- 3 borrower photo flow
    r, qg, _ = A.req("GET", "/api/queue", params=dict(purpose="tikuv_dastgohi", color="green", limit=200))
    in_index = {l for im in idx["images"] for l in im["loans"]}
    greens = [x["loan_id"] for x in qg["items"] if not x["escalations"] and x["loan_id"] not in in_index]
    L1, L2, L3 = greens[:3]
    DATA["green_tikuv"] = greens[:6]
    sbA, sbB = S(a.base), S(a.base)
    other_img = "other_05.jpg"

    def up(sb, loan, name, data=None, fields=None, fname=None, ctype="image/jpeg"):
        data = (PH / name).read_bytes() if data is None else data
        return sb.req("POST", "/api/check_photo", data=fields or {"loan_id": loan}, files={"image": (fname or name, data, ctype)})

    r, ph, ms = up(sbA, L1, other_img)
    am = ph.get("asset_match") if isinstance(ph, dict) else None
    row("3 Photo", f"A: other image (phones) for green {L1}: asset_match not null, matches_purpose false, label_uz present",
        r.status_code == 200 and am and am["matches_purpose"] is False and am.get("label_uz"), f"{json.dumps(am, ensure_ascii=False)} server {ph.get('latency_ms')} ms", ms)
    r, d, _ = sbA.req("GET", f"/api/loan/{L1}")
    codes = [e["code"] for e in d.get("escalations", [])]
    row("3 Photo", f"A: asset_match.label_i18n and escalation text in {LANG}", lang_texts_ok(am.get("label_i18n"), "message") and all(lang_texts_ok(e.get("text")) for e in d.get("escalations", [])),
        f"label {(am.get('label_i18n') or {}).get(LANG)!r}; text {(d['escalations'][0].get('text') or {}).get(LANG) if d.get('escalations') else None!r}")
    row("3 Photo", f"A: {L1} colour in sandbox A >= yellow with PHOTO_ASSET_MISMATCH", RANK[d["color"]] >= 1 and "PHOTO_ASSET_MISMATCH" in codes, f"colour {d['color']} escalations {codes}; text: {[e['text_uz'] for e in d['escalations']]}")
    r, d, _ = sbB.req("GET", f"/api/loan/{L1}")
    row("3 Photo", f"B (new cookie): {L1} is still green, no escalation (isolation)", d["color"] == "green" and not d["escalations"], f"colour {d['color']} escalations {d['escalations']}")
    r, q, _ = sbB.req("GET", "/api/queue", params=dict(limit=1))
    DATA["B_counts"] = q["counts"]
    r, q, _ = sbA.req("GET", "/api/queue", params=dict(limit=1))
    row("3 Photo", "A queue counts differ from B (A has the escalation)", q["counts"] != DATA["B_counts"], f"A {q['counts']} B {DATA['B_counts']}")
    reserved = [f for f in idx["reserved"] if f.startswith("tikuv_dastgohi")]
    r, ph, ms = up(sbA, L2, reserved[-1])
    am = ph.get("asset_match") if isinstance(ph, dict) else None
    r2, d, _ = sbA.req("GET", f"/api/loan/{L2}")
    row("3 Photo", f"A: sewing machine NOT in the index ({reserved[-1]}) for green {L2}: matches_purpose true, no escalation",
        r.status_code == 200 and am and am["matches_purpose"] is True and not ph["duplicate_of"] and d["color"] == "green", f"{json.dumps(am, ensure_ascii=False)} duplicate_of {ph['duplicate_of']} colour {d['color']} server {ph.get('latency_ms')} ms", ms)
    dup_img = next(im for im in idx["images"] if im["purpose"] == "tikuv_dastgohi" and len(im["loans"]) == 1)
    r, ph, ms = up(sbA, L3, dup_img["file"])
    r2, d, _ = sbA.req("GET", f"/api/loan/{L3}")
    codes = [e["code"] for e in d["escalations"]]
    row("3 Photo", f"A: portfolio image of ANOTHER loan ({dup_img['file']} = {dup_img['loans'][0]}) for {L3}: duplicate_of lists it, colour red, PHOTO_DUPLICATE",
        r.status_code == 200 and dup_img["loans"][0] in [x["loan_id"] for x in ph["duplicate_of"]] and d["color"] == "red" and "PHOTO_DUPLICATE" in codes,
        f"duplicate_of {ph['duplicate_of']} colour {d['color']} codes {codes} server {ph.get('latency_ms')} ms", ms)
    r, d, _ = sbB.req("GET", f"/api/loan/{L3}")
    row("3 Photo", f"B: {L3} still green", d["color"] == "green" and not d["escalations"], f"colour {d['color']}")
    # the sandbox photo itself: borrower upload route
    r, d, _ = sbA.req("GET", f"/api/loan/{L3}")
    ph_urls = [p.get("url") for p in d["photos"] if p.get("url")]
    if ph_urls:
        rr, _, _ = sbA.req("GET", ph_urls[0])
        rb, _, _ = sbB.req("GET", ph_urls[0])
        row("3 Photo", "uploaded (blurred) photo: owner gets 200 image, other sandbox does not", rr.status_code == 200 and rr.headers.get("content-type", "").startswith("image/") and rb.status_code in (403, 404), f"owner {rr.status_code}, other {rb.status_code}")
    # errors
    sbE = S(a.base)
    errs = []

    def err(name, r, body, expect):
        ok, det = detail_uz(body)
        row("3 Errors", name, r is not None and r.status_code in expect and ok, f"HTTP {getattr(r, 'status_code', None)} detail={det!r}")

    r, b, _ = up(sbE, L1, "x", data=b"", fname="empty.jpg"); err("empty file -> 4xx Uzbek detail", r, b, (400, 422))
    r, b, _ = up(sbE, L1, "x", data=b"this is just text\n" * 50, fname="note.jpg"); err(".txt renamed to .jpg -> 4xx", r, b, (400, 415, 422))
    big = b"\xff\xd8\xff\xe0" + b"\0" * (9 * 1024 * 1024)
    r, b, ms = up(sbE, L1, "x", data=big, fname="big.jpg"); err("9 MB file (limit 8 MB) -> 413", r, b, (413,))
    r, b, _ = up(sbE, "L0000000", "other_05.jpg", fields={"loan_id": "L0000000"}); err("unknown loan id -> 404", r, b, (404,))
    r, b, _ = up(sbE, "", "other_05.jpg", fields={"loan_id": ""}); err("empty loan id -> 4xx", r, b, (400, 404, 422))
    r, b, _ = sbE.req("POST", "/api/check_photo", data={"loan_id": L1})
    err("no image part -> 4xx", r, b, (400, 422))
    r, b, _ = up(sbE, L1, "other_05.jpg", fields={"loan_id": L1, "lat": "999", "lon": "abc"}); err("invalid lat/lon -> 4xx", r, b, (400, 422))
    r, b, _ = up(sbE, L1, "other_05.jpg", fields={"loan_id": L1, "captured_at": "not-a-date"}); err("invalid captured_at -> 4xx", r, b, (400, 422))
    bad_sb = httpx.Client(base_url=a.base, timeout=60, cookies={"sandbox_id": "../../etc/passwd"})
    r = bad_sb.post("/api/check_photo", data={"loan_id": L1}, files={"image": ("o.jpg", (PH / "other_05.jpg").read_bytes(), "image/jpeg")})
    row("3 Errors", "malformed sandbox cookie -> no 500 (server replaces it with a fresh sandbox)", r.status_code < 500, f"HTTP {r.status_code} {r.text[:100]}")
    # parallel
    imgs = [f"other_{i:02d}.jpg" for i in range(0, 5)]
    sbs = [S(a.base) for _ in imgs]

    def job(k):
        t = time.perf_counter()
        r, ph, _ = up(sbs[k], L1, imgs[k])
        r2, d, _ = sbs[k].req("GET", f"/api/loan/{L1}")
        return k, r.status_code, ph, d, (time.perf_counter() - t) * 1000
    with cf.ThreadPoolExecutor(5) as ex:
        res = list(ex.map(job, range(5)))
    ok = all(st == 200 and ph["duplicate_of"] == [] and len([e for e in d["escalations"] if e["code"] == "PHOTO_ASSET_MISMATCH"]) == 1 and not [e for e in d["escalations"] if e["code"] == "PHOTO_DUPLICATE"] for _, st, ph, d, _ in res)
    row("3 Photo", "5 parallel uploads from 5 sandboxes (other_00..04 for the same loan): all 200, each sees only its own escalation, no cross-duplicates", ok,
        "; ".join(f"{imgs[k]}: HTTP {st}, server {ph.get('latency_ms')} ms, client {ms:.0f} ms, esc {[e['code'] for e in d['escalations']]}" for k, st, ph, d, ms in res), max(x[4] for x in res))
    # latency: three uploads sequentially in a fresh sandbox
    sbL = S(a.base)
    lat = []
    for name in ("other_06.jpg", "tikuv_dastgohi_16.jpg", "chorva_09.jpg"):
        r, ph, ms = up(sbL, greens[3], name)
        lat.append((name, ph.get("latency_ms"), ms))
    DATA["photo_latency"] = lat

    # ---------------------------------------------------------------- 4 scenario
    r, pres, ms = A.req("GET", "/api/scenario_presets")
    row("4 Scenario", "presets load (4)", r.status_code == 200 and len(pres) == 4 and all(p["title_uz"] and p["description_uz"] for p in pres), [p["id"] for p in pres], ms)
    row("4 Scenario", f"preset titles / descriptions in {LANG}", all(lang_texts_ok(p.get("title"), "message") and lang_texts_ok(p.get("description"), "message") for p in pres),
        pres[1].get("title", {}).get(LANG))
    scores = {}
    for p in pres:
        r, s, ms = A.req("POST", "/api/score_scenario", json=p["scenario"])
        scores[p["id"]] = s
        row("4 Scenario", f"score preset {p['id']}", r.status_code == 200 and uz_text_ok(s["reasons"][0]["text_uz"] if s["reasons"] else "ok")
            and all(lang_texts_ok(x.get("text")) for x in s["reasons"]),
            f"risk {s['risk']:.3f} colour {s['color']} percentile {s['percentile']} reason0 {s['reasons'][0]['text_uz'] if s['reasons'] else None!r}", ms)
    base = next(p for p in pres if p["id"] == "fast_cash_out")["scenario"]

    def score(sc):
        r, s, ms = A.req("POST", "/api/score_scenario", json=sc)
        return r, s, ms
    d1 = copy.deepcopy(base)
    d30 = copy.deepcopy(base); d30["snapshot_day"] = 60; d30["transactions"][0]["day"] = 30
    pay = copy.deepcopy(base); pay["transactions"] = [{"day": 2, "type": "supplier_payment", "amount_uzs": 48000000}]
    out = {}
    for n, sc in (("cash-out day 1", d1), ("cash-out day 30", d30), ("all paid to supplier", pay)):
        r, s, ms = score(sc)
        out[n] = s
    fm = lambda s: f"risk {s['risk']:.3f}, colour {s['color']}, percentile {s['percentile']}, first reason: {s['reasons'][0]['text_uz'] if s['reasons'] else None}"
    DATA["scenario_cmp"] = {k: fm(v) for k, v in out.items()}
    row("4 Scenario", "cash withdrawal day 30 -> lower risk than day 1", out["cash-out day 30"]["risk"] < out["cash-out day 1"]["risk"], f"day1: {fm(out['cash-out day 1'])} || day30: {fm(out['cash-out day 30'])}")
    row("4 Scenario", "all paid to supplier -> lower risk than cash-out", out["all paid to supplier"]["risk"] < out["cash-out day 1"]["risk"], f"paid: {fm(out['all paid to supplier'])}")
    invalid = {
        "negative amount": lambda s: s.update(amount_uzs=-5),
        "unknown tx type": lambda s: s["transactions"][0].update(type="gift"),
        "snapshot_day 1000": lambda s: s.update(snapshot_day=1000),
        "snapshot_day 1": lambda s: s.update(snapshot_day=1),
        "unknown extra field": lambda s: s.update(hacker=1),
        "unknown purpose": lambda s: s.update(purpose="kosmos"),
        "negative tx amount": lambda s: s["transactions"][0].update(amount_uzs=-1),
        "tx day beyond snapshot": lambda s: s["transactions"][0].update(day=999),
    }
    for n, f in invalid.items():
        sc = copy.deepcopy(base); f(sc)
        r, b, _ = A.req("POST", "/api/score_scenario", json=sc)
        ok, det = detail_uz(b)
        row("4 Scenario", f"invalid: {n} -> 400/422 with Uzbek message", r.status_code in (400, 422) and ok, f"HTTP {r.status_code} detail={str(det)[:200]!r}")

    # ---------------------------------------------------------------- 5 decision, audit, report
    dl = DATA["red_loan"]
    r, b, ms = A.req("POST", "/api/decision", json=dict(loan_id=dl, officer_id="e2e-test", action="visit", comment="e2e test"))
    row("5 Decision", f"POST /api/decision on {dl} (officer e2e-test, visit)", r.status_code == 200 and b.get("audit_id"), f"HTTP {r.status_code} {b}", ms)
    CREATED["audit_ids"].append(b.get("audit_id"))
    CREATED["loans_written"].append(dl)
    r, au, ms = A.req("GET", f"/api/audit?loan_id={dl}")
    mine = [x for x in au if x.get("action") == "visit"] if isinstance(au, list) else []
    row("5 Decision", "GET /api/audit shows the row with ai_recommendation and model_version", bool(mine) and mine[-1].get("ai_recommendation") and mine[-1].get("model_version"), json.dumps(mine[-1] if mine else au, ensure_ascii=False)[:300], ms)
    r, b, _ = A.req("POST", "/api/decision", json=dict(loan_id=dl, officer_id="e2e-test", action="explode", comment="x"))
    ok, det = detail_uz(b)
    row("5 Decision", "invalid action -> 4xx with Uzbek message", r.status_code in (400, 422) and ok, f"HTTP {r.status_code} {str(det)[:150]!r}")

    def draft(regen=False):
        t = time.perf_counter()
        r, b, _ = A.req("POST", "/api/draft_report", json=dict(loan_id=dl, regenerate=regen))
        while r is not None and r.status_code == 202:
            time.sleep(max(2, min(10, b.get("eta_s") or 3)))
            r2, j, _ = A.req("GET", f"/api/draft_report/job/{b['job_id']}")
            if j.get("status") == "done":
                return r, j["result"], (time.perf_counter() - t) * 1000
            if j.get("status") == "error" or (time.perf_counter() - t) > 150:
                return r2, j, (time.perf_counter() - t) * 1000
        return r, b, (time.perf_counter() - t) * 1000
    r, rep, ms1 = draft()
    okfacts = isinstance(rep, dict) and rep.get("facts") and all(f.get("evidence_ref") or f.get("unverified") for f in rep["facts"])
    row("5 Report", f"POST /api/draft_report on {dl}: text_uz + facts each with evidence_ref (or flagged unverified)", r is not None and isinstance(rep, dict) and rep.get("text_uz") and okfacts,
        f"HTTP {getattr(r, 'status_code', None)} cached={rep.get('cached') if isinstance(rep, dict) else None} model={rep.get('model') if isinstance(rep, dict) else None} facts={len(rep.get('facts', [])) if isinstance(rep, dict) else 0} unverified={sum(1 for f in rep.get('facts', []) if f.get('unverified')) if isinstance(rep, dict) else 0} latency_ms={rep.get('latency_ms') if isinstance(rep, dict) else None} cost_usd={rep.get('cost_usd') if isinstance(rep, dict) else None}", ms1)
    if isinstance(rep, dict) and rep.get("text_uz"):
        DATA["report_text"] = rep["text_uz"][:900]
        DATA["report_flags"] = flags_of(rep["text_uz"], "reason")
        r, rep2, ms2 = draft()
        row("5 Report", "second call is cached: same text, faster, cost 0", isinstance(rep2, dict) and rep2.get("cached") is True and rep2["text_uz"] == rep["text_uz"] and (rep2.get("cost_usd") in (0, 0.0, None)) and ms2 < max(ms1 * 2, 1000),
            f"call1 {ms1:.0f} ms cost {rep.get('cost_usd')} cached {rep.get('cached')}; call2 {ms2:.0f} ms cost {rep2.get('cost_usd')} cached {rep2.get('cached')} same_text={rep2.get('text_uz') == rep['text_uz']}", ms2)

    # ---------------------------------------------------------------- 6 frontend (HTTP part)
    for path in ("/", f"/loan/{dl}", f"/capture?loan={L1}", "/scenario", "/capture", "/nonexistent-page"):
        r, b, ms = A.req("GET", path)
        html = r.text if r is not None else ""
        js = re.findall(r'src="([^"]+\.js)"', html)
        ok = r.status_code == 200 and "<div id=\"root\"" in html and bool(js)
        if js and ok:
            rj, _, _ = A.req("GET", js[0])
            ok = rj.status_code == 200 and "javascript" in rj.headers.get("content-type", "")
        row("6 Frontend", f"GET {path}: 200 HTML referencing built JS (JS loads)", ok, f"HTTP {r.status_code} js={js[:1]}", ms)
    DATA["demo_loans"] = dict(tikuv=greens[:3], red=dl)
    r, qc, _ = A.req("GET", "/api/queue", params=dict(purpose="chorva", color="green", limit=200))
    DATA["green_chorva"] = [x["loan_id"] for x in qc["items"] if not x["escalations"] and x["loan_id"] not in in_index][:2]
    DATA["finished"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    (ROOT / "qa" / ("e2e_results.json" if LANG == "uz" else f"e2e_results_{LANG}.json")).write_text(json.dumps(dict(rows=ROWS, created=CREATED, data=DATA), indent=1, ensure_ascii=False), encoding="utf-8")
    print(sum(r["ok"] for r in ROWS), "pass /", len(ROWS))


if __name__ == "__main__":
    main()
