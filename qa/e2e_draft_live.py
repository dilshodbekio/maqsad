"""One fresh LLM draft per language on one red loan (new sandbox each), then the same request again (cache). Writes qa/e2e_draft_live.json.

    PYTHONPATH=. .venv/Scripts/python qa/e2e_draft_live.py [--base URL] [--loan L1504044]

Costs a few cents (regenerate=true bypasses the cache once per language). Never calls admin / reset.
"""
import argparse
import json
import time
import uuid
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://maqsad-production-d704.up.railway.app"


def call(c, loan, regenerate):
    t = time.perf_counter()
    r = c.post("/api/draft_report", json={"loan_id": loan, "regenerate": regenerate})
    polls = 0
    while r.status_code == 202:
        time.sleep(3)
        polls += 1
        j = c.get(f"/api/draft_report/job/{r.json()['job_id']}").json()
        if j["status"] == "done":
            return 200, j["result"], (time.perf_counter() - t) * 1000, polls
        if j["status"] == "error" or time.perf_counter() - t > 240:
            return 500, j, (time.perf_counter() - t) * 1000, polls
    return r.status_code, r.json(), (time.perf_counter() - t) * 1000, polls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--loan", default="L1504044")
    a = ap.parse_args()
    out = []
    for lang in ("uz", "en", "ru"):
        sid = str(uuid.uuid4())
        c = httpx.Client(base_url=a.base, timeout=300, cookies={"sandbox_id": sid}, headers={"Accept-Language": lang})
        row = dict(lang=lang, sandbox=sid)
        for name, regen in (("fresh", True), ("again", False)):
            st, d, ms, polls = call(c, a.loan, regen)
            ok = st == 200
            row[name] = dict(status=st, client_ms=round(ms), polls=polls,
                             **({k: d.get(k) for k in ("lang", "cached", "latency_ms", "cost_usd", "model", "input_tokens", "output_tokens")} if ok else {"error": str(d)[:200]}),
                             facts=len(d.get("facts", [])) if ok else None, unverified=sum(f["unverified"] for f in d.get("facts", [])) if ok else None,
                             text=(d.get("text_uz") or "")[:700] if ok else None)
            print(lang, name, {k: v for k, v in row[name].items() if k != "text"})
        out.append(row)
    (ROOT / "qa" / "e2e_draft_live.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
