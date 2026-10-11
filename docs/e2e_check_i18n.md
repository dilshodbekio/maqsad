# E2E check in three languages

Run: 2026-10-11 04:30:44 UTC - 2026-10-11 04:31:21 UTC against **the LIVE service https://maqsad-production-d704.up.railway.app (deploy of PR #17: /api/health ml_mode real, clip_loaded true, top_reason_text present, UZ/EN/RU switcher in the header)**. Commands:

```
PYTHONPATH=. .venv/Scripts/python qa/e2e_live.py --lang uz|en|ru  [--base URL]
PYTHONPATH=. .venv/Scripts/python qa/e2e_screens.py --lang all [--base URL]
```

## HTTP checks (`qa/e2e_live.py --lang ...`)

| Language | passed | failed rows |
|---|---|---|
| uz | 67 of 67 | - |
| en | 67 of 67 | - |
| ru | 67 of 67 | - |

Each language run checks: error `detail` in the request language (empty file, text as .jpg, 9 MB file, unknown loan, invalid `captured_at`, invalid scenario fields - no English pydantic text), and `top_reason_text`, `reasons[].text`, `evidence[].summary`, `asset_match.label_i18n`, escalation texts, preset titles / descriptions and scenario reasons in that language, clean (`flags_lang` of `ml/eval/audit_texts.py`).

## Pages (`qa/e2e_screens.py --lang all`, Chromium; mobile 390x844 and desktop 1366x850)

| Language | pages x sizes | PASS | console errors | HTTP >= 400 | horizontal scroll | wrong-language text | None / nan |
|---|---|---|---|---|---|---|---|
| uz | 8 | 8 | 0 | 0 | 0 | 0 | 0 |
| en | 8 | 8 | 0 | 0 | 0 | 0 | 0 |
| ru | 8 | 8 | 0 | 0 | 0 | 0 | 0 |

## Fresh inspection report draft per language (live LLM path), red loan L1504044, new sandbox per language

| Language | call | status | cached | server latency | client time | cost_usd | tokens in/out | facts (unverified) |
|---|---|---|---|---|---|---|---|---|
| uz | 1: regenerate (LLM) | 200 | False | 3497.2 ms | 6979 ms | 0.003571 | 1351/682 | 6 (0) |
| uz | 2: same request | 200 | True | 14.7 ms | 200 ms | 0.0 | None/None | 6 (0) |
| en | 1: regenerate (LLM) | 200 | False | 8499.8 ms | 10436 ms | 0.003654 | 1187/737 | 6 (0) |
| en | 2: same request | 200 | True | 25.5 ms | 234 ms | 0.0 | None/None | 6 (0) |
| ru | 1: regenerate (LLM) | 200 | False | 3599.0 ms | 6596 ms | 0.003704 | 1239/740 | 6 (0) |
| ru | 2: same request | 200 | True | 10.6 ms | 121 ms | 0.0 | None/None | 6 (0) |

Total cost of the three fresh drafts: $0.010929. `lang` in each response equals the request language.


Language switcher (UZ -> EN -> RU -> UZ without reload; `<html lang>`, visible text, localStorage and the active button follow; the choice survives a reload): **PASS**.

Screens: `docs/e2e_screens/<uz|en|ru>/<queue|loan_red|capture|scenario>_<mobile|desktop>.png`.

## Bugs

No BLOCKER and no MAJOR bug found.

| Severity | Owner | Bug | One-line fix |
|---|---|---|---|
| MINOR | backend | The English draft starts with a shouting title copied from the prompt ("DRAFT INSPECTION REPORT FOR BANK LOAN-MONITORING OFFICER"). | Tell the LLM in `SYSTEM_PROMPT_EN` to start with a short sentence-case title; bump `PROMPT_VERSION`. |
| MINOR | backend | `DraftReport.text_uz` holds English / Russian text when `Accept-Language` is en / ru (name kept for the contract; `lang` tells the language). | Add `text` as an alias in a later contract version. |
| MINOR | qa | The check 'second draft call is faster' was flaky for two cached calls (111 ms vs 115 ms). | Fixed in `qa/e2e_live.py` (cached = `cached: true`, cost 0, not slower than 2x). |
