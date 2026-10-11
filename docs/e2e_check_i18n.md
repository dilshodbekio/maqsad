# E2E check in three languages

Run: 2026-10-11 00:55:20 UTC - 2026-10-11 00:55:44 UTC against **a local server of branch `app/i18n-fixes` (real ML, CLIP loaded, no LLM key)**. Commands:

```
PYTHONPATH=. .venv/Scripts/python qa/e2e_live.py --lang uz|en|ru  [--base URL]
PYTHONPATH=. .venv/Scripts/python qa/e2e_screens.py --lang all [--base URL]
```

## HTTP checks (`qa/e2e_live.py --lang ...`)

| Language | passed | failed rows |
|---|---|---|
| uz | 65 of 66 | POST /api/draft_report on L1504044: text_uz + facts each with evidence_ref (or f (HTTP 503 cached=None model=None facts=0 unverified=0 latency) |
| en | 65 of 66 | POST /api/draft_report on L1504044: text_uz + facts each with evidence_ref (or f (HTTP 503 cached=None model=None facts=0 unverified=0 latency) |
| ru | 65 of 66 | POST /api/draft_report on L1504044: text_uz + facts each with evidence_ref (or f (HTTP 503 cached=None model=None facts=0 unverified=0 latency) |

Each language run checks: error `detail` in the request language (empty file, text as .jpg, 9 MB file, unknown loan, invalid `captured_at`, invalid scenario fields - no English pydantic text), and `top_reason_text`, `reasons[].text`, `evidence[].summary`, `asset_match.label_i18n`, escalation texts, preset titles / descriptions and scenario reasons in that language, clean (`flags_lang` of `ml/eval/audit_texts.py`).

## Pages (`qa/e2e_screens.py --lang all`, Chromium; mobile 390x844 and desktop 1366x850)

| Language | pages x sizes | PASS | console errors | HTTP >= 400 | horizontal scroll | wrong-language text | None / nan |
|---|---|---|---|---|---|---|---|
| uz | 8 | 8 | 0 | 0 | 0 | 0 | 0 |
| en | 8 | 8 | 0 | 0 | 0 | 0 | 0 |
| ru | 8 | 8 | 0 | 0 | 0 | 0 | 0 |

Language switcher (UZ -> EN -> RU -> UZ without reload; `<html lang>`, visible text, localStorage and the active button follow; the choice survives a reload): **PASS**.

Screens: `docs/e2e_screens/<uz|en|ru>/<queue|loan_red|capture|scenario>_<mobile|desktop>.png`.
