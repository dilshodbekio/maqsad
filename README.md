# Maqsad

Maqsad is an AI system that remotely checks whether preferential microloans were used for their stated purpose (money trail, photo, invoice) and builds a risk queue for the bank officer, who always makes the final decision.

WIUT Hackathon 2026, FinTech track, partner: MKBANK.

**Live demo:** https://maqsad-production-d704.up.railway.app

The system never says that someone is guilty. It says "check this loan first, and here is why".

## Demo flow

1. **Queue** (`/`): loans ranked by risk, coloured red / yellow / green, with filters.
2. **Loan** (`/loan/<id>`): reasons in plain language, the money trail, the invoice, photos and the evidence behind every reason.
3. **Photo check** (`/capture`): upload or take a photo for a loan. The server blurs faces, looks for duplicates, checks place and time, and compares the asset in the photo with the loan purpose. A photo can raise the loan's colour in the queue. Open `/capture` on a phone (for example via a QR code) to use the camera; the camera needs https.
4. **Scenario** (`/scenario`): enter a hypothetical loan and its transactions and see the live score. Nothing is stored. Also a good QR target.
5. **Draft report** (on the loan page): an LLM writes a draft inspection report from this loan's evidence only. Every fact is checked in code (the cited evidence must exist and every number must appear in it); facts that fail are shown as unverified, never hidden.
6. **Decision**: the officer chooses approve / reject / visit. The AI recommendation is stored next to it, separately.
7. **Audit**: every upload and decision is written to an append-only log with a hash chain.

Each visitor gets a private sandbox (cookie), so photos and decisions made by one judge are not visible to another.

## Architecture in short

One deployed service: a FastAPI app (`backend/`) serves the API under `/api/*` and the built React frontend (`frontend/dist`) at `/`.
The backend imports the `ml/` package directly, and only through `backend/app/ml_adapter.py`. The `ml/` package does the money-trail scoring (LightGBM), the photo checks (OpenCV, OpenCLIP) and the evidence-fusion policy. State (audit log, sandbox photos, draft cache) is kept in SQLite under `DATA_DIR`. The prototype calls an external LLM only for the draft report; production is designed to run without any external API (local LLM through vLLM or Ollama, only `LLM_BASE_URL` changes). Details and diagrams: [`docs/deployment.md`](docs/deployment.md).

```
backend/    FastAPI API, sandbox, rate limits, draft-report queue, audit log
frontend/   React + TypeScript + Vite (built to frontend/dist)
ml/         synthetic generator, money-trail model, photo checks, evaluation
data/       data sources, licences, generation method
docs/       brief, API contract, metrics, cost, QA
qa/         end-to-end scripts (Playwright) and their results
```

## Run locally

Python 3.13 and Node 22.

```bash
python -m venv .venv
.venv/Scripts/activate          # Linux/macOS: source .venv/bin/activate
pip install -r backend/requirements.txt -r ml/requirements.txt
cd frontend && npm ci && npm run build && cd ..
uvicorn backend.app.main:create_app --factory --port 8000
```

Open http://localhost:8000. The first start builds the demo portfolio in the background (`/api/health` shows `warmup`). Without CLIP weights the photo check still works but returns `asset_match: null`; download them once with `python -m ml.photo.asset --download`.
For frontend development run `npm run dev` in `frontend/` next to the backend.

### Docker

```bash
docker build -t maqsad .
docker run --rm -p 8000:8000 -e ADMIN_TOKEN=change-me maqsad
```

The image sets `REQUIRE_REAL_ML=1`: if `ml.api` cannot be imported the service refuses to start instead of silently falling back to a stub. CLIP weights are downloaded once at build time, never while the service runs. Deployment on Railway: [`docs/deployment.md`](docs/deployment.md).

## Environment variables

Names only; copy `backend/.env.example` to `.env` (never commit it). Settings are read in `backend/app/config.py`.

| Variable | Purpose |
|---|---|
| `PORT` | HTTP port (default 8000) |
| `DATA_DIR` | SQLite database and sandbox photos (mount a volume here) |
| `FRONTEND_DIST` | path of the built frontend (default `frontend/dist`) |
| `ML_ASSETS_DIR` | portfolio photos served at `/api/asset/` |
| `REQUIRE_REAL_ML` | `1` = refuse to start if `ml.api` cannot be imported |
| `ML_FORCE_STUB` | `1` = force the stub (tests only) |
| `ADMIN_TOKEN` | enables `/api/admin/*` (header `X-Admin-Token`) |
| `RATE_LIMIT_PER_MIN`, `RATE_LIMIT_IP_PER_MIN` | per-sandbox and per-IP request limits |
| `MAX_UPLOAD_MB` | photo size limit |
| `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY` | OpenAI-compatible LLM for the draft report |
| `LLM_TIMEOUT_S`, `LLM_REASONING_EFFORT`, `LLM_RPM` | LLM timeout, reasoning effort, calls per minute through the draft queue |
| `LLM_PRICE_IN_PER_1M`, `LLM_PRICE_OUT_PER_1M` | token prices for the logged `cost_usd` (empty = unknown, never guessed) |
| `MAQSAD_CLIP_CACHE` | directory of the CLIP weights (same at build and run time) |
| `MAQSAD_ASSET_CHECK` | `off` switches the CLIP asset check off |
| `MAQSAD_TORCH_THREADS` | torch thread count (1 matches the cost model) |
| `HF_HUB_OFFLINE` | `1` = never contact the model hub at run time |

## Tests

```bash
cd backend && python -m pytest          # API tests; also runs the real ml integration tests when ml dependencies are installed
python -m pytest ml/tests               # ml tests (from the repo root), see ml/README.md
cd frontend && npm run build            # type-checks (tsc) and builds
```

CI (`.github/workflows/ci.yml`) runs the backend tests and a Docker smoke test that requires the real model. End-to-end scripts against the live service are in `qa/` (they need Playwright, see `data/README.md`); their results are in `docs/e2e_check.md` and `docs/e2e_check_i18n.md`.

## Data and licences

All credit data is **synthetic**; photos are openly licensed (Wikimedia Commons, per-file licence in `ml/assets/photos/manifest.csv`). No real personal or bank data is used. Every dataset, image source, model and library with its licence is listed in [`data/README.md`](data/README.md).

## Qualifying round

No code from the qualifying (screening) round was used in this project.

## Team

- Dilshodbek Tolibjonov: AI / ML (`ml/`)
- Oybek Xushnazarov: application (`backend/`, `frontend/`, deploy)
- Shohijahon Shavkatov: business (`docs/`: ROI, sources, deck)

## More documentation

- [`ml/README.md`](ml/README.md): the ML layer, results, limitations, how to reproduce
- [`data/README.md`](data/README.md): data, assumptions, licences
- [`docs/metrics.md`](docs/metrics.md): ML metrics
- [`docs/metrics_api.md`](docs/metrics_api.md): API latency and LLM report cost
- [`docs/api_contract.md`](docs/api_contract.md): HTTP and Python interface
- [`docs/deployment.md`](docs/deployment.md): production design, prototype differences, audit verification
- [`docs/qa.md`](docs/qa.md): questions a judge may ask, with short answers
- [`docs/known_issues_app.md`](docs/known_issues_app.md): known limitations of the application
