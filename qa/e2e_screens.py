"""Headless screenshots of the live frontend (mobile 390x844 + desktop 1366x850): console errors, horizontal scroll, bad-text scan.
    .venv/Scripts/python qa/e2e_screens.py
"""
import json
import re
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://maqsad-production-d704.up.railway.app"
res = json.loads((ROOT / "qa" / "e2e_results.json").read_text(encoding="utf-8"))
red, green = res["data"]["red_loan"], res["data"]["green_tikuv"][0]
PAGES = {"queue": "/", "loan_red": f"/loan/{red}", "capture": f"/capture?loan={green}", "scenario": "/scenario"}
SIZES = {"mobile": (390, 844), "desktop": (1366, 850)}
out = ROOT / "docs" / "e2e_screens"
out.mkdir(parents=True, exist_ok=True)
rows = []
with sync_playwright() as p:
    b = p.chromium.launch()
    for sz, (w, h) in SIZES.items():
        for name, path in PAGES.items():
            ctx = b.new_context(viewport={"width": w, "height": h}, device_scale_factor=1)
            pg = ctx.new_page()
            errs, failed = [], []
            pg.on("console", lambda m: errs.append(m.text[:200]) if m.type == "error" else None)
            pg.on("pageerror", lambda e: errs.append("pageerror: " + str(e)[:200]))
            pg.on("response", lambda r: failed.append(f"{r.status} {r.url[-60:]}") if r.status >= 400 else None)
            pg.goto(BASE + path, wait_until="networkidle", timeout=90000)
            pg.wait_for_timeout(2500)
            sw, iw = pg.evaluate("[document.documentElement.scrollWidth, window.innerWidth]")
            text = pg.inner_text("body")
            bad = re.findall(r"\b(None|nan|undefined|NaN|\[object Object\])\b|\{\{|\}\}", text)
            pg.screenshot(path=str(out / f"{name}_{sz}.png"), full_page=True)
            rows.append(dict(page=name, path=path, size=sz, console_errors=errs[:5], http_errors=failed[:5], scroll_w=sw, inner_w=iw, hscroll=sw > iw + 1, bad_text=bad[:5], chars=len(text)))
            print(rows[-1])
            ctx.close()
    b.close()
(ROOT / "qa" / "e2e_screens.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
