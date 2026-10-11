# Maqsad — WIUT Hackathon 2026 (FinTech track, partner: MKBANK)

AI system that remotely checks whether preferential microloans were used for their stated purpose,
using three kinds of evidence: **money trail, photo, invoice**. It builds a risk queue; a bank officer always makes the final decision.

**Full project brief: `docs/brief.md` — read it before doing anything non-trivial.**
The idea is fixed and agreed. Do not propose a different idea; only improve this one.

## Team and folder ownership
| Folder | Owner | Role |
|---|---|---|
| `ml/` | Dilshodbek | synthetic generator, money-trail model, photo analysis, evaluation |
| `backend/`, `frontend/` | Oybek | API, officer dashboard, borrower page, LLM report draft, audit log, deploy |
| `docs/` | Shohijahon | ROI, sources, deck, Q&A (except `docs/metrics.md` — written by Dilshodbek) |

Only edit files in your own folder. Shared files (`README.md`, `data/README.md`, `docs/api_contract.md`) change only by agreement.

## Hard rules (breaking these = disqualification)
- **Never invent numbers.** Every metric must come from code that actually ran. Unknown number → mark as "Taxmin" (assumption) or "needs measurement".
- **No hard-coded / mocked answers in the demo.** Mocks are allowed only during development.
- **No real personal or bank data.** Only synthetic data or openly licensed images.
- Every dataset, image source, model and library license goes into `data/README.md`. Synthetic data: generation method must be documented there.
- Production design must work **without external APIs** (local LLM via vLLM/Ollama). The prototype may call an external LLM only in `/draft_report` (and optional invoice extraction).
- The system never says someone is "guilty". It says "check this loan first, here is why".

## Integration (ml ↔ backend)
- Python 3.13. `ml/` is an importable package; backend imports it directly (single deployed service).
- Public interface lives in `ml/api.py` and must match `docs/api_contract.md`:
  - `score_loan(loan_id: str) -> dict`  → `{risk, color, reasons[{code, text_uz, weight}], model_version, latency_ms}`
  - `check_photo(loan_id: str, image_bytes: bytes, lat=None, lon=None, captured_at=None) -> dict`
  - `get_queue() -> list[dict]`, `get_loan(loan_id) -> dict`
- ML dependencies: `ml/requirements.txt`. Backend's requirements include it.
- Trained model artifacts: `ml/artifacts/` (commit only if small, < ~20 MB).

## Git workflow
- `main` must always work. Work on branches: `ml/...`, `app/...`, `docs/...`; merge via PR.
- `git pull` on `main` before starting work. Commit often (commit history proves work was done during the final).
- Never commit `.env`, API keys, large datasets (`data/generated/`, `*.parquet`).

## Language
- Code, comments, commit messages: English.
- All user-facing text (UI, reasons `text_uz`, reports): Uzbek (Latin).
