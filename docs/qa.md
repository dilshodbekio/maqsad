# Questions a judge may ask

Short answers. Each names the file or function the answer rests on. Questions about the models, the data and the metrics: see [`ml/README.md`](../ml/README.md) (the ML owner's answers, including limitations) and [`docs/metrics.md`](metrics.md).

## Product and decision

**Does the AI decide who gets punished or who gets a loan?**
No. The system orders loans by "check this first". The officer chooses approve / reject / visit; the AI recommendation is stored in a separate field next to the human action (`decision` in `backend/app/main.py`). The wording never says "guilty" (`SYSTEM_PROMPT_*` in `backend/app/llm.py`; reasons come from `ml/`).

**What are the three kinds of evidence?**
Money trail (`score_loan`, `ml/money_trail/`), photo (`check_photo`, `ml/photo/`) and invoice (supplier, OKED code, invoice-gap features). A photo can only raise a loan's colour, never lower it and never change `risk` (`ml/fusion.py`, `docs/api_contract.md` "Evidence fusion").

**Why does a loan have a colour?**
Colour is by rank in the served portfolio: top 10 % red, next 10 % yellow (the 20 % inspection budget), the rest green (`ml.api.get_color_thresholds`, `docs/api_contract.md` "Colour rule"). `risk` is a score for ranking, not a probability, and the UI never shows it as a percentage.

**What does the officer see as the reason?**
Plain-language reasons, each with references to real rows (transactions, invoice, supplier graph, photo, checked period) (`get_loan` in `ml/api.py`, `docs/api_contract.md` "Inner shapes").

## Data and privacy

**Is any real person or bank data used?**
No. The portfolio is generated (`ml/generator/`, method in `data/README.md` section 1) and photos are openly licensed (`ml/assets/photos/manifest.csv`). Names are masked synthetic names.

**What happens to an uploaded photo?**
The original is never written to disk. Faces are blurred, EXIF is dropped by re-encoding, and only the blurred JPEG and a perceptual hash are kept (`ml/photo/checker.py`, `check_photo` in `backend/app/main.py`). The multipart parser is configured not to spool uploads to a temp file (`create_app`), and the size limit is in `backend/app/limits.py`.

**Does it send data to an outside company?**
In the prototype exactly one call goes out: the draft report sends the evidence and reasons of one synthetic loan to the Gemini API (`draft_report` in `backend/app/llm.py`, `docs/deployment.md` section 2). In production the same code talks to a local LLM (vLLM / Ollama) by changing `LLM_BASE_URL` (`backend/app/config.py`). Everything else runs locally on CPU.

## LLM draft report

**How do you stop the LLM from making things up?**
The prompt contains only this loan's evidence and reasons. Then, in code and without an LLM, every fact is checked: the cited `evidence_ref` must exist and every number in the claim must appear in the evidence (`fact_check` in `backend/app/llm.py`). A fact that fails is shown with `unverified: true`, not hidden. The prose between the facts is not verified; the draft is labelled a draft.

**What does a draft cost and how long does it take?**
Measured: 3.9–6.0 s and 0.0025–0.0042 USD per report in three production log lines; the three-language live check gives similar values. A cached draft costs nothing and answers in milliseconds. The key is on the free tier, so the actual charge was $0. These are small samples: `docs/metrics_api.md`.

**What if the LLM is down or the quota is gone?**
The loan page works without it; the draft says it is not ready or the service is busy (`draft_report` in `backend/app/main.py`, `backend/app/draft_queue.py`). Drafts for the top loans can be pre-generated into the shared cache (`POST /api/admin/prewarm`). Limits: `docs/known_issues_app.md`.

## Security and audit

**Can the audit log be changed afterwards?**
Not through the application: SQLite triggers abort `UPDATE` and `DELETE`, and every row carries `hash = sha256(prev_hash + row)`, so an edit in the middle breaks the chain (`SCHEMA`, `row_hash`, `audit_verify` in `backend/app/db.py`; test in `backend/tests/test_report_decision_audit.py`). A full chain rewrite by a database administrator is not detected unless the head hash is stored elsewhere; this is said openly in `docs/deployment.md` section 4.

**Who is the officer in the audit?**
In the demo, a per-sandbox id such as `Xodim-XXXX`, set on the server; an `officer_id` in a request body is ignored (`decision`, `demo_officer_id` in `backend/app/main.py`). In production this should come from the bank's SSO (not built).

**Can a visitor abuse the open demo?**
Limits exist: rate limits per sandbox and per IP (`backend/app/ratelimit.py`), a regenerate limit of 1 per minute per sandbox and 3 per minute per IP, body-size limits counted on received bytes (`backend/app/limits.py`), and admin endpoints that need a token compared in constant time (`admin_check` in `backend/app/main.py`). The client IP is the rightmost `X-Forwarded-For` entry (`_client_ip`). Tests: `backend/tests/test_hardening.py`.

**Can one judge see another judge's uploads?**
No. Photos, decisions and queue effects are per sandbox cookie; the photo endpoint returns 404 for another sandbox (`photo` in `backend/app/main.py`, `backend/tests/test_session_audit_isolation.py`).

## Engineering

**Is anything in the demo hard-coded or mocked?**
No. With `REQUIRE_REAL_ML=1` (set in the `Dockerfile`) the service does not start without the real model, and `/api/health` reports `ml_mode` (`backend/app/ml_adapter.py`; the smoke test in `.github/workflows/ci.yml` requires `ml_mode == "real"`). The stub exists for development only, reads sample files and shows a visible STUB banner.

**What is tested?**
Backend: API behaviour including bad input (empty body, huge file, wrong type, unknown loan), the contract, the audit chain, isolation, the draft queue, i18n and the hardening limits (`backend/tests/`, run in CI). ML tests: `ml/tests/`. End-to-end checks against the live service in three languages: `qa/`, results in `docs/e2e_check.md` and `docs/e2e_check_i18n.md`.

**Can it scale to a whole bank?**
The prototype is one replica: the queue, rate limiter and sandbox state are in process memory (`docs/known_issues_app.md`). Cost per check is measured in `docs/cost.md` (a photo check is about 0.36 s of single-thread CPU); scaling out needs shared state, which is not built.

**Is any code from the qualifying round used?**
No (`README.md`).

**How is the live service deployed?**
Docker build from the repository on Railway, redeployed from `main` by GitHub Actions after the tests pass (`Dockerfile`, `railway.json`, `.github/workflows/deploy.yml`).

## Model and data questions

Detection quality, leakage checks, split design, the fraud typologies, the limits of the CLIP asset check and the face-blur measurement are answered in [`ml/README.md`](../ml/README.md) (sections 4 and 6), [`docs/metrics.md`](metrics.md) and [`data/README.md`](../data/README.md). All accuracy numbers are on synthetic data and say nothing about real accuracy, which a pilot on historical loans would measure.
