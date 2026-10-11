"""Headless screenshots of the frontend (mobile 390x844 + desktop 1366x850) in one language, plus a language-switch test.

    .venv/Scripts/python qa/e2e_screens.py [--base URL] [--lang uz|en|ru|all]

For every page and size: console errors, HTTP >= 400, horizontal scroll, "None / nan / undefined" in the text, `<html lang>`, and that the
visible text is in the selected language (no Uzbek UI words in EN / RU, no Cyrillic in UZ / EN, Cyrillic present in RU).
The language is set the way a user does: by the choice remembered in localStorage (`maqsad:lang`). Screens: docs/e2e_screens/<lang>/.
"""
import argparse
import json
import re
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://maqsad-production-d704.up.railway.app"
res = json.loads((ROOT / "qa" / "e2e_results.json").read_text(encoding="utf-8"))
red, green = res["data"]["red_loan"], res["data"]["green_tikuv"][0]
PAGES = {"queue": "/", "loan_red": f"/loan/{red}", "capture": f"/capture?loan={green}", "scenario": "/scenario"}
SIZES = {"mobile": (390, 844), "desktop": (1366, 850)}
CYR = re.compile(r"[А-Яа-яЁё]")
UZ_UI = re.compile(r"Xavf bahosi|Navbat|Kredit |Qayta urinish|Hammasi|Hudud|Qaror|Dalolatnoma|Hisob-faktura|Pul izi|Rasm |Ssenariy|yuklanmoqda|so'm\b", re.I)
EN_UI = re.compile(r"\b(Risk score|Risk queue|Loan|Back to the queue|Money trail|Invoice|Decision|Audit log|Photos|Region|Purpose)\b")


def switcher_test(b, base):
    """Click UZ -> EN -> RU -> UZ on the queue page without a reload; the choice survives a reload."""
    ctx = b.new_context(viewport={"width": 390, "height": 844})
    pg = ctx.new_page()
    pg.goto(base + "/", wait_until="networkidle", timeout=90000)
    checks = []
    for target in ("en", "ru", "uz"):
        pg.get_by_role("button", name=target.upper(), exact=True).click()
        pg.wait_for_timeout(500)
        t = pg.inner_text("body")
        checks.append((target, pg.evaluate("document.documentElement.lang") == target,
                       {"en": "Risk queue" in t, "ru": "Очередь рисков" in t, "uz": "Xavf navbati" in t}[target],
                       pg.evaluate("localStorage.getItem('maqsad:lang')") == target,
                       pg.evaluate("document.querySelector('.lang-btn.active').textContent") == target.upper()))
    pg.get_by_role("button", name="RU", exact=True).click()
    pg.reload(wait_until="networkidle")
    pg.wait_for_timeout(1500)
    checks.append(("reload keeps RU", pg.evaluate("document.documentElement.lang") == "ru", "Очередь рисков" in pg.inner_text("body"), True, True))
    ctx.close()
    ok = all(all(c[1:]) for c in checks)
    print("PASS" if ok else "FAIL", "language switcher (UZ -> EN -> RU -> UZ, reload keeps the choice)", [c for c in checks if not all(c[1:])] or "")
    return dict(lang="all", page="switcher", size="mobile", ok=ok, checks=[[str(x) for x in c] for c in checks])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--lang", default="all", choices=["uz", "en", "ru", "all"])
    a = ap.parse_args()
    langs = ["uz", "en", "ru"] if a.lang == "all" else [a.lang]
    rows = []
    with sync_playwright() as p:
        b = p.chromium.launch()
        for lang in langs:
            out = ROOT / "docs" / "e2e_screens" / lang
            out.mkdir(parents=True, exist_ok=True)
            for sz, (w, h) in SIZES.items():
                for name, path in PAGES.items():
                    ctx = b.new_context(viewport={"width": w, "height": h}, device_scale_factor=1)
                    ctx.add_init_script(f"try {{ localStorage.setItem('maqsad:lang', '{lang}'); }} catch (e) {{}}")
                    pg = ctx.new_page()
                    errs, failed = [], []
                    pg.on("console", lambda m: errs.append(m.text[:200]) if m.type == "error" else None)
                    pg.on("pageerror", lambda e: errs.append("pageerror: " + str(e)[:200]))
                    pg.on("response", lambda r: failed.append(f"{r.status} {r.url[-60:]}") if r.status >= 400 else None)
                    pg.goto(a.base + path, wait_until="networkidle", timeout=90000)
                    pg.wait_for_timeout(2500)
                    sw, iw = pg.evaluate("[document.documentElement.scrollWidth, window.innerWidth]")
                    text = pg.inner_text("body")
                    html_lang = pg.evaluate("document.documentElement.lang")
                    bad = re.findall(r"\b(None|nan|undefined|NaN|\[object Object\])\b|\{\{|\}\}", text)
                    wrong = []
                    if html_lang != lang:
                        wrong.append(f"html lang={html_lang}")
                    if lang == "ru" and not CYR.search(text):
                        wrong.append("no Cyrillic")
                    if lang in ("uz", "en") and CYR.search(text):
                        wrong.append("Cyrillic: " + CYR.search(text).group())
                    if lang in ("en", "ru") and UZ_UI.search(text):
                        wrong.append("Uzbek UI text: " + UZ_UI.search(text).group())
                    if lang == "uz" and EN_UI.search(text):
                        wrong.append("English UI text: " + EN_UI.search(text).group())
                    pg.screenshot(path=str(out / f"{name}_{sz}.png"), full_page=True)
                    row = dict(lang=lang, page=name, path=path, size=sz, console_errors=errs[:5], http_errors=failed[:5], hscroll=sw > iw + 1,
                               scroll_w=sw, inner_w=iw, bad_text=bad[:5], wrong_language=wrong, chars=len(text))
                    rows.append(row)
                    ok = not (errs or failed or row["hscroll"] or bad or wrong)
                    print("PASS" if ok else "FAIL", lang, name, sz, {k: v for k, v in row.items() if v and k in ("console_errors", "http_errors", "bad_text", "wrong_language", "hscroll")} or "")
                    ctx.close()
        rows.append(switcher_test(b, a.base))
        b.close()
    suffix = "" if a.lang == "uz" else f"_{a.lang}"
    (ROOT / "qa" / f"e2e_screens{suffix}.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
