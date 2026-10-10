"""Measure API latency against a deployed Maqsad (client-side wall time, includes the network).

    python backend/scripts/measure_api.py [--base URL] [--n 20] [--only queue,loan,photo,draft_cached,draft_fresh] [--out FILE.json]

Every number printed comes from real requests. Percentiles: nearest-rank on the sorted samples.
Pacing keeps the app's rate limit (20 requests/min per sandbox) and the free-tier LLM quota out of the picture;
requests that still fail are counted and reported, never dropped silently.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]


def pct(sorted_ms: list, p: float) -> float:
    return sorted_ms[max(0, math.ceil(p / 100 * len(sorted_ms)) - 1)]


def summarize(name: str, samples: list, errors: list) -> dict:
    s = sorted(samples)
    out = {"endpoint": name, "n_ok": len(s), "n_errors": len(errors), "errors": errors[:5]}
    if s:
        out.update(p50_ms=round(pct(s, 50), 1), p95_ms=round(pct(s, 95), 1), min_ms=round(s[0], 1),
                   max_ms=round(s[-1], 1), mean_ms=round(statistics.fmean(s), 1))
    return out


def timed(fn, pace_s: float, n: int):
    samples, errors, extra = [], [], []
    for i in range(n):
        t = time.perf_counter()
        try:
            r = fn(i)
            ms = (time.perf_counter() - t) * 1000
            if r.status_code == 200:
                samples.append(ms)
                extra.append(r.json() if r.headers.get("content-type", "").startswith("application/json") else None)
            else:
                errors.append(f"HTTP {r.status_code}: {r.text[:120]}")
        except Exception as e:  # noqa: BLE001
            errors.append(f"{type(e).__name__}: {e}"[:160])
        if pace_s and i < n - 1:
            time.sleep(pace_s)
    return samples, errors, extra


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="https://maqsad-production-d704.up.railway.app")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--only", default="queue,loan,photo,draft_cached,draft_fresh")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    want = set(a.only.split(","))
    c = httpx.Client(base_url=a.base, timeout=90, follow_redirects=True)
    health = c.get("/api/health").json()
    results = {"base": a.base, "measured_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "n_requested": a.n, "model_version": health.get("model_version"), "ml_mode": health.get("ml_mode"),
               "client": "python httpx, sequential, from a developer laptop (network included)", "endpoints": []}
    q = c.get("/api/queue", params={"limit": 50}).json()
    ids = [i["loan_id"] for i in q["items"]]

    if "queue" in want:
        s, e, _ = timed(lambda i: c.get("/api/queue", params={"limit": 50}), 0.3, a.n)
        results["endpoints"].append(summarize("GET /api/queue?limit=50", s, e))
    if "loan" in want:
        s, e, _ = timed(lambda i: c.get(f"/api/loan/{ids[i % len(ids)]}"), 0.3, a.n)
        results["endpoints"].append(summarize("GET /api/loan/{id} (rotating over 50 loans)", s, e))
    if "photo" in want:
        img = sorted((ROOT / "ml" / "assets" / "photos").glob("*.jpg"))[0]
        data = img.read_bytes()

        def photo(i):
            return c.post("/api/check_photo", data={"loan_id": ids[0]}, files={"image": (img.name, data, "image/jpeg")})

        s, e, extra = timed(photo, 3.2, a.n)  # < 20 requests / minute
        r = summarize(f"POST /api/check_photo ({img.name}, {len(data) // 1024} KB)", s, e)
        srv = sorted(x["latency_ms"] for x in extra if x)
        if srv:
            r.update(server_p50_ms=round(pct(srv, 50), 1), server_p95_ms=round(pct(srv, 95), 1))
        results["endpoints"].append(r)
    if "draft_cached" in want or "draft_fresh" in want:
        lid = ids[1]
        probe = c.post("/api/draft_report", json={"loan_id": lid})
        if probe.status_code == 200:
            d = probe.json()
            results["draft_sample"] = {k: d.get(k) for k in ("model", "latency_ms", "cost_usd", "cached", "input_tokens", "output_tokens")}
            results["draft_sample"].update(n_facts=len(d["facts"]), n_unverified=sum(f["unverified"] for f in d["facts"]))
        else:
            results["draft_probe_error"] = f"HTTP {probe.status_code}: {probe.text[:160]}"
        if "draft_cached" in want and probe.status_code == 200:
            s, e, extra = timed(lambda i: c.post("/api/draft_report", json={"loan_id": lid}), 0.5, a.n)
            r = summarize("POST /api/draft_report (cache hit)", s, e)
            r["cached_flags"] = sorted({bool(x.get("cached")) for x in extra if x})
            results["endpoints"].append(r)
        if "draft_fresh" in want:
            s, e, extra = timed(lambda i: c.post("/api/draft_report", json={"loan_id": ids[i % len(ids)], "regenerate": True}),
                                13.0, a.n)  # 13 s spacing: stays under a 5 requests/minute LLM quota
            r = summarize("POST /api/draft_report (regenerate=true, no cache)", s, e)
            srv = sorted(x["latency_ms"] for x in extra if x)
            if srv:
                r.update(server_p50_ms=round(pct(srv, 50), 1), server_p95_ms=round(pct(srv, 95), 1))
            costs = [x["cost_usd"] for x in extra if x and x.get("cost_usd") is not None]
            r["cost_usd_known_for"] = len(costs)
            toks = [(x["input_tokens"], x["output_tokens"]) for x in extra if x and x.get("input_tokens") is not None]
            if toks:
                r["input_tokens_median"] = statistics.median(t[0] for t in toks)
                r["output_tokens_median"] = statistics.median(t[1] for t in toks)
            results["endpoints"].append(r)

    text = json.dumps(results, indent=2, ensure_ascii=False)
    print(text)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
