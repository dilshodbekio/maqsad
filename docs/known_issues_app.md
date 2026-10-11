# Known issues and limits of the application (backend + frontend)

What a reader should know before trusting or load-testing the live demo. ML limits are in [`ml/README.md`](../ml/README.md) section 6 and [`docs/metrics.md`](metrics.md).

## LLM draft report

- **Gemini free-tier quota.** The prototype's LLM is on a free key. Google answered the development key with a limit of 5 requests per minute for `gemini-3.6-flash`, and a daily quota can run out (the provider then asks for very long waits, for example "retry in 1h49m"). The production key's limits were not read. Effect: many judges asking for drafts at once wait in a queue; when the quota is exhausted, drafts fail with a "service busy" message until it resets. The loan page itself keeps working.
- **Queue** (`backend/app/draft_queue.py`): one LLM call at a time, at most `LLM_RPM` (default 4) calls in any 60 s. A request that is not cached gets a `job_id` and polls it; the queue shows position and an estimated wait (an estimate, not a promise). More than 30 waiting user tasks → 503 "busy". A job older than 120 s reports "busy" (the task still finishes and fills the cache).
- **Jobs and the queue are in memory.** After a restart or redeploy every `job_id` is unknown (404); the client submits again and gets the cache or a new job.
- **Pre-warm is manual and Uzbek only.** `POST /api/admin/prewarm` fills the shared cache for the top-n loans (default 40) in Uzbek. Drafts in English and Russian are generated on demand, so the first requester waits longer. The shared cache is in SQLite under `DATA_DIR`; it survives restarts if the volume does.
- **Regenerate** costs a real LLM call: limited to 1 per minute per sandbox and 3 per minute per IP, and the result is kept in the caller's sandbox only.
- **The draft can be wrong.** Facts are checked in code (cited evidence exists, every number appears in the evidence), but the connecting prose is written by the LLM and is not verified. Facts that fail are shown as unverified. The officer must read the draft as a draft.
- **Cost figures are small samples** (n = 3 per source, `docs/metrics_api.md`), and on the free tier the actual charge was $0.
- No local model (vLLM / Ollama) was benchmarked; the production claim is that only `LLM_BASE_URL` changes, which was not tested against a local server.

## Sandbox, state and rate limits

- **Everything stateful is single-replica.** The rate limiter, the draft queue, the draft jobs and the ML sandbox state (photo effects on the queue and loans) are in process memory. Two replicas would behave inconsistently.
- **A restart loses in-memory state.** Rate-limit counters, draft jobs and the ML side's sandbox effects (queue colours raised by an uploaded photo) are gone and the queue returns to the default view. What survives depends on the volume mounted at `DATA_DIR` (SQLite: audit log, stored blurred photos, draft cache); without a volume the database is lost at redeploy. Whether the volume is attached in the live Railway project was not checked from this repository.
- **The sandbox is a cookie.** A visitor who clears cookies or opens a private window gets a new sandbox and no longer sees earlier uploads and decisions (they remain in the audit log). Photos of one sandbox are not visible to another.
- **Rate limits are in memory and per process.** `check_photo` and `draft_report`: 20 per minute per sandbox (`RATE_LIMIT_PER_MIN`) and 120 per minute per IP (`RATE_LIMIT_IP_PER_MIN`). Many visitors behind one NAT share the IP limit. The client IP is the rightmost `X-Forwarded-For` entry, which is correct only behind a proxy that appends the real peer address (Railway's does, as far as we know); behind a different proxy this needs checking.
- **Request sizes**: photo about 10.7 MB (image ≤ 8 MB), any other `POST` 64 KB; larger → 413. Whether a maximal scenario request (60 transactions) fits in 64 KB was not measured.
- Uploaded images are held in memory during the request (up to the limit); the original is never written to disk.

## Audit

- Append-only by SQLite triggers and a hash chain (`docs/deployment.md` section 4). The chain detects edits in the middle of the log, not a full rewrite of the chain, because the head hash is not anchored outside the database.
- There is no HTTP endpoint or UI to verify the chain; verification is `Database.audit_verify()`.
- In the prototype the actor is a demo id per sandbox, not a real officer identity.

## Frontend and devices

- **The camera needs https.** `getUserMedia` works only in a secure context (https or localhost), and the user must allow camera access. The live site is https; over plain http on a LAN address the camera is not available.
- Geolocation is optional and needs the user's permission; without it `geo_ok` is `null` and no distance check is made.
- The client clock is only compared with the server time; the server time is the primary timestamp.
- The UI is in three languages (uz, en, ru). Texts from the ML layer and the LLM come from templates in the code, not from a general translator, so an awkward wording in one language is possible.
- Automated checks use Chromium at 390x844 and 1366x850 (`qa/e2e_screens.py`); other browsers were not tested.

## Photo check (application side)

- `asset_match` is `null` when the CLIP weights are not available (`/api/health` shows `clip_loaded: false`); the asset-mismatch escalation is then never raised. The check is a flag for the officer, not proof (`docs/metrics.md`).
- The CLIP weights licence needs a legal review for production (`docs/api_contract.md`, Open points).

## Data

- All data is synthetic, so no accuracy on real loans is claimed anywhere; real accuracy is measured in a pilot.
- Names in the demo are masked synthetic names.
