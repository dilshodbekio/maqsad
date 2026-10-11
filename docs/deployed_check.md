# Deployed service check (https://maqsad-production-d704.up.railway.app)

Run at 2026-10-10 18:51:36 UTC by `python -m ml.eval.check_deployed`; model_version `money_trail_v2`. Read-only: only GET health / queue / legend / loan and POST check_photo in a separate smoke sandbox (cookie `sandbox_id=a7ff04f4-3498-59c3-8c36-4da4a179ce69`, derived from `ml-smoke-20261010-185136`; the backend requires a UUID); no admin / reset call.

| Check | Result | Detail |
|---|---|---|
| GET /api/health -> 200, ml_mode real, warmup done, all functions real | OK | HTTP 200; {"ml_mode": "real", "model_version": "money_trail_v2", "warmup": "done", "functions": {"get_queue": "real", "get_loan": "real", "score_loan": "real", "check_photo": "real", "reset_sandbox": "real", "get_color_thresholds": "real"}} |
| GET /api/queue first page: 50 items, sorted by priority (descending) | OK | HTTP 200; 50 items; first priorities [2.998, 2.997, 2.997] |
| queue colours valid, color >= base_color, counts add up to total | OK | counts {'red': 513, 'yellow': 501, 'green': 3986} total 5000 |
| the first page is the top of the queue (red loans first) | OK | first: L1504044 red risk 0.998 |
| top_reason of the first page: non-empty Uzbek text without flagged patterns | OK | e.g. 'Kredit berilganidan beri tushum qayd etilmagan' |
| GET /api/queue/legend: red + yellow budget reported with photo escalations | OK | HTTP 200; 500 red by risk + 13 escalated to red by the photo policy = 513 red; 501 yellow (500 by risk, 5 green loans raised to yellow, the rest moved to red) |
| GET /api/loan/L1504044: reasons[0] == queue top_reason, evidence present | OK | HTTP 200; 3 reasons, 5 evidence |
| POST /api/check_photo tikuv_dastgohi_12.jpg for L1504130 (new sandbox a7ff04f4-3498-59c3-8c36-4da4a179ce69): 200, no duplicate yet | OK | HTTP 200; {"asset_match": null, "duplicate_of": [], "geo_ok": null, "time_ok": null, "faces_blurred": false, "faces_count": 0, "latency_ms": 11.03} |
| same image again in the same sandbox for L1503050: duplicate_of contains L1504130 | OK | HTTP 200; duplicate_of=[{'loan_id': 'L1504130', 'similarity': 1.0}] |
| CLIP asset check is ACTIVE on the deployed service: asset_match present, sewing machine matches the sewing-machine loan | FAIL | asset_match=None  (null = the CLIP weights are not in the image / CLIP failed to load; check_photo degrades safely as designed, /api/health does not show clip_loaded) |
| GET /api/loan/L1504130 in the smoke sandbox now escalates (PHOTO_DUPLICATE) | OK | HTTP 200; color red; escalations ['PHOTO_DUPLICATE'] |
| another sandbox (fresh cookie) does NOT see the escalation | OK | HTTP 200 |

## Client-side latency (from the machine that ran the check, includes the network)

| Request | calls | min | p50 | p95 | max |
|---|---|---|---|---|---|
| GET /api/queue?limit=50 | 10 | 358 ms | 477 ms | 899 ms | 899 ms |
| GET /api/loan/L1504044 | 10 | 328 ms | 355 ms | 741 ms | 741 ms |
| POST /api/check_photo | 3 | 427 ms | 428 ms | 432 ms | 432 ms |

Latency includes the first request after idle time (cold paths) and the network; with 3 or 10 calls p95 is the slowest or second-slowest call.

## Failures (exact details)

- **CLIP asset check is ACTIVE on the deployed service: asset_match present, sewing machine matches the sewing-machine loan**: asset_match=None  (null = the CLIP weights are not in the image / CLIP failed to load; check_photo degrades safely as designed, /api/health does not show clip_loaded)
