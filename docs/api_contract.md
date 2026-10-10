# API contract (Dilshodbek ↔ Oybek)

Source: `docs/brief.md` §5. Any change requires agreement from both sides (via PR).
During development the backend may return mock responses; **in the demo only the real model runs**.

```
GET  /queue
  → [{loan_id, borrower_name_masked, purpose, amount_uzs, region, risk, color, top_reason}]
     sorted by risk, descending

GET  /loan/{loan_id}
  → {loan, transactions[], photos[], invoice?,
     risk, color, reasons:[{code, text_uz, weight}], evidence:[...], model_version}

POST /score_loan        {loan_id}
  → {risk: 0..1, color: "green|yellow|red", reasons[], model_version, latency_ms}

POST /check_photo       {loan_id, image (base64 or multipart), lat?, lon?, captured_at}
  → {asset_match: {label, score}, duplicate_of: [{loan_id, similarity}],
     geo_ok, time_ok, faces_blurred: bool, latency_ms}

POST /draft_report      {loan_id}
  → {text_uz, facts:[{claim, evidence_ref}], model, latency_ms, cost_usd}

POST /decision          {loan_id, officer_id, action: "approve|reject|visit", comment}
  → {audit_id}

GET  /audit?loan_id=...
  → [{ts, actor, action, ai_recommendation, model_version, inputs_hash}]
```

## Python interface (backend imports `ml/` directly)

| Endpoint | Backend calls | Owner |
|---|---|---|
| `GET /queue` | `ml.api.get_queue()` | ml |
| `GET /loan/{id}` | `ml.api.get_loan(loan_id)` | ml |
| `POST /score_loan` | `ml.api.score_loan(loan_id)` | ml |
| `POST /check_photo` | `ml.api.check_photo(loan_id, image_bytes, lat, lon, captured_at)` | ml |
| `POST /draft_report` | backend (LLM) | backend |
| `POST /decision`, `GET /audit` | backend (audit log) | backend |

## Open points (to agree)
- `color` thresholds (green/yellow/red): set by ml from the queue-budget logic; values recorded in `docs/metrics.md`.
- `evidence[]` item shape: proposed `{ref, kind: "txn|photo|invoice|graph", summary_uz, data}` — `ref` is what `/draft_report` cites as `evidence_ref`.
