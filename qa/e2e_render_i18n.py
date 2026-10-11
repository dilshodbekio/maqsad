"""Render docs/e2e_check_i18n.md from qa/e2e_results*.json and qa/e2e_screens_all.json (three-language run)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LANGS = (("uz", "e2e_results.json"), ("en", "e2e_results_en.json"), ("ru", "e2e_results_ru.json"))
first = json.loads((ROOT / "qa" / "e2e_results.json").read_text(encoding="utf-8"))["data"]
out = ["# E2E check in three languages", "",
       f"Run: {first['started']} - {first['finished']} against **{__import__('os').environ.get('E2E_TARGET', 'a local server of branch `app/i18n-fixes` (real ML, CLIP loaded, no LLM key)')}**. Commands:", "",
       "```", "PYTHONPATH=. .venv/Scripts/python qa/e2e_live.py --lang uz|en|ru  [--base URL]", "PYTHONPATH=. .venv/Scripts/python qa/e2e_screens.py --lang all [--base URL]", "```", "",
       "## HTTP checks (`qa/e2e_live.py --lang ...`)", "", "| Language | passed | failed rows |", "|---|---|---|"]
for lang, f in LANGS:
    rows = json.loads((ROOT / "qa" / f).read_text(encoding="utf-8"))["rows"]
    bad = [r for r in rows if not r["ok"]]
    out.append(f"| {lang} | {len(rows) - len(bad)} of {len(rows)} | " + ("; ".join(f"{r['check'][:80]} ({r['evidence'][:60]})" for r in bad) or "-") + " |")
out += ["", "Each language run checks: error `detail` in the request language (empty file, text as .jpg, 9 MB file, unknown loan, invalid `captured_at`, "
        "invalid scenario fields - no English pydantic text), and `top_reason_text`, `reasons[].text`, `evidence[].summary`, `asset_match.label_i18n`, "
        "escalation texts, preset titles / descriptions and scenario reasons in that language, clean (`flags_lang` of `ml/eval/audit_texts.py`).", "",
        "## Pages (`qa/e2e_screens.py --lang all`, Chromium; mobile 390x844 and desktop 1366x850)", "",
        "| Language | pages x sizes | PASS | console errors | HTTP >= 400 | horizontal scroll | wrong-language text | None / nan |", "|---|---|---|---|---|---|---|---|"]
rows = json.loads((ROOT / "qa" / "e2e_screens_all.json").read_text(encoding="utf-8"))
for lang in ("uz", "en", "ru"):
    r = [x for x in rows if x.get("lang") == lang and "path" in x]
    ok = [x for x in r if not (x["console_errors"] or x["http_errors"] or x["hscroll"] or x["bad_text"] or x["wrong_language"])]
    out.append(f"| {lang} | {len(r)} | {len(ok)} | {sum(len(x['console_errors']) for x in r)} | {sum(len(x['http_errors']) for x in r)} | "
               f"{sum(x['hscroll'] for x in r)} | {sum(len(x['wrong_language']) for x in r)} | {sum(len(x['bad_text']) for x in r)} |")
sw = next(x for x in rows if x.get("page") == "switcher")
out += ["", f"Language switcher (UZ -> EN -> RU -> UZ without reload; `<html lang>`, visible text, localStorage and the active button follow; "
        f"the choice survives a reload): **{'PASS' if sw['ok'] else 'FAIL'}**.", "",
        "Screens: `docs/e2e_screens/<uz|en|ru>/<queue|loan_red|capture|scenario>_<mobile|desktop>.png`.", ""]
(ROOT / "docs" / "e2e_check_i18n.md").write_text("\n".join(out), encoding="utf-8", newline="\n")
print("\n".join(out))
