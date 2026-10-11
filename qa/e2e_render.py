"""Render docs/e2e_check.md from qa/e2e_results.json + qa/e2e_screens.json (the bugs list is written by hand in BUGS below)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = json.loads((ROOT / "qa" / "e2e_results.json").read_text(encoding="utf-8"))
SC = json.loads((ROOT / "qa" / "e2e_screens.json").read_text(encoding="utf-8"))
D, rows = R["data"], R["rows"]
esc = lambda s: str(s).replace("|", "/").replace("\n", " ")

BUGS = [
    ("MAJOR", "frontend", "No demo / QR page exists: the frontend routes are only `/`, `/loan/{id}`, `/capture?loan={id}`, `/scenario` (see `frontend/src/App.tsx`). If the plan is a QR code for judges, nothing generates it; judges would have to type `/capture?loan=ID`.",
     "Add a `/demo` page that lists the demo loan ids with a QR code each pointing to `/capture?loan=ID` (or print the QR codes into the deck)."),
    ("MINOR", "backend", "Request-validation errors are Uzbek only in the prefix: `So'rov ma'lumotlari noto'g'ri - loan_id: Field required`, `captured_at: Input should be a valid datetime ...` (English pydantic text reaches the user). Same for the scenario enum message (`... 'parranda' or 'asalarichilik')`) and technical field names (`snapshot_day`, `amount_uzs`).",
     "Map pydantic error types (`missing`, `datetime_parsing`, `literal_error`) to Uzbek sentences and field names to Uzbek labels in the 400/422 handler."),
    ("MINOR", "ml", "The top reason of the #1 red loan (`Kredit berilganidan beri tushum qayd etilmagan`, L1504044) has `evidence_refs: []` (an absence-of-income reason has no event to cite); the UI says 'Alohida dalil havolasi yo'q'. A judge may read this as an unsupported claim. Also green loans show risk-raising wording as `top_reason` (e.g. 'Yirik kirim qayd etilmagan' on a green loan).",
     "Cite the loan window as evidence (e.g. a `window:` ref) for absence reasons; in the UI label green-loan reasons as the factors that raised the score most rather than as findings."),
    ("MINOR", "ml/backend", "The draft report could only be checked from the shared cache (both calls `cached: true`, cost 0, ~230 ms client time). A fresh generation (latency, cost_usd, rate-limit behaviour) was NOT tested to avoid LLM cost / quota.",
     "Before the demo, run one `regenerate: true` on a loan that is not pre-warmed and record latency and cost."),
]
out = ["# End-to-end check of the live demo", "",
       f"Live site: https://maqsad-production-d704.up.railway.app. Run {D['started']} - {D['finished']}, model_version `{D['model_version']}`. "
       "Scripts: `qa/e2e_live.py` (HTTP checks, writes `qa/e2e_results.json`), `qa/e2e_screens.py` (Playwright / Chromium screenshots), `qa/e2e_render.py` (this file). "
       "No app / ML code was changed. No `/api/admin/*` call was made. Everything that wrote data used `officer_id = e2e-test` or a throw-away sandbox (list at the end).", "",
       f"**{sum(r['ok'] for r in rows)} of {len(rows)} HTTP checks passed; the {sum(not r['ok'] for r in rows)} failures are explained under the table (1 is the missing evidence_refs of an absence reason, 2 are English fragments in 422 messages).**", "",
       "## 1. Results", "", "| # | Check | Result | Evidence | Latency |", "|---|---|---|---|---|"]
for r in rows:
    out.append(f"| {r['section']} | {esc(r['check'])} | {'PASS' if r['ok'] else '**FAIL**'} | {esc(r['evidence'])} | {str(r['ms']) + ' ms' if r['ms'] is not None else ''} |")
out += ["", "Notes on the failures: `red loan: every reason has evidence_refs ...` failed because the **first reason of L1504044 has an empty `evidence_refs`** (every ref that exists does resolve: `T...` ids are transactions in `transactions[]`, `invoice:` is the invoice; this is bug 3). "
        "`empty loan id` and `invalid captured_at` returned 422 as required, but with English pydantic fragments in `detail` (bug 2).", "",
        "## 2. Photo latency (three sequential uploads, fresh sandbox)", "", "| Image | server `latency_ms` | client round trip |", "|---|---|---|"]
for n, s, c in D["photo_latency"]:
    out.append(f"| {n} | {s} ms | {c:.0f} ms |")
out += ["", "Other uploads above: server 124-214 ms, client 570-940 ms (the first request after idle is the slowest).", "",
        "## 3. Scenario comparison (fast cash-out preset)", ""]
out += [f"- **{k}**: {v}" for k, v in D["scenario_cmp"].items()]
out += ["", "## 4. Frontend (headless Chromium, screenshots in `docs/e2e_screens/`)", "",
        "| Page | Viewport | Console errors | HTTP >= 400 | Horizontal scroll | 'None/nan/undefined' in text |", "|---|---|---|---|---|---|"]
for s in SC:
    out.append(f"| {s['path']} | {s['size']} | {len(s['console_errors'])} | {len(s['http_errors'])} | {'YES ' + str(s['scroll_w']) + ' > ' + str(s['inner_w']) if s['hscroll'] else 'no'} | {len(s['bad_text'])} |")
out += ["", "Screens: queue, loan_red, capture, scenario x mobile (390x844) / desktop (1366x850); e.g. `docs/e2e_screens/loan_red_mobile.png`. "
        "Unknown routes (`/nonexistent-page`) return the SPA shell (200).", "",
        "## 5. Bugs", "", "| Severity | Owner | Bug | Suggested fix |", "|---|---|---|---|"]
for sev, own, bug, fix in BUGS:
    out.append(f"| {sev} | {own} | {esc(bug)} | {esc(fix)} |")
out += ["", "No BLOCKER was found: queue, loan pages, photo flow (asset mismatch / match / duplicate / isolation / errors / parallel), scenario, decision + audit, draft report (cached) and all frontend pages worked on the live site.", "",
        "## 6. Demo helpers", "",
        "Green loans without photo escalation **in a fresh sandbox**, not in the portfolio photo index (a first upload of a new sewing-machine photo does not hit a duplicate).", "",
        f"- `tikuv_dastgohi` green: {', '.join('`' + x + '`' for x in D['green_tikuv'][:3])}",
        f"- `chorva` green: {', '.join('`' + x + '`' for x in D['green_chorva'])}",
        f"- red loan for the live walkthrough: `{D['red_loan']}` (rank 1 of the queue, 3 reasons with evidence, invoice block, transactions; reasons: " + "; ".join(D["red_reasons"]) + ")", "",
        "Photo test images: `ml/assets/photos/tikuv_dastgohi_16.jpg` (sewing machine, not in the index -> matches, no escalation), `other_05.jpg` (phones -> `PHOTO_ASSET_MISMATCH`, yellow), `tikuv_dastgohi_15.jpg` (in the index -> `PHOTO_DUPLICATE`, red).", "",
        "## 7. What this run created on the live service", "",
        f"- Audit rows: id {', '.join(str(x) for x in R['created']['audit_ids'])} (loan `{R['created']['loans_written'][0]}`, actor `e2e-test`, action `visit`, comment `e2e test`; visible only in that sandbox's view). The audit log is a hash chain, so the row cannot be removed.",
        f"- {len(R['created']['sandboxes'])} throw-away sandboxes (random UUID cookies; sandbox photos live in server memory only); ids in `qa/e2e_results.json`. Not reset (no admin call).",
        "- Draft report: no new LLM generation (both calls were served from the cache).", ""]
(ROOT / "docs" / "e2e_check.md").write_text("\n".join(out), encoding="utf-8", newline="\n")
print("ok", len(out))
