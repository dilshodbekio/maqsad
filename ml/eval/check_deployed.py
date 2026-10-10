"""Read-only live check of the deployed service -> docs/deployed_check.md.

    .venv/Scripts/python -m ml.eval.check_deployed [--base https://maqsad-production-d704.up.railway.app]

Calls only GET /api/health, /api/queue, /api/loan/{id} and POST /api/check_photo (one committed library image, in a NEW sandbox: the backend takes
the sandbox from a UUID cookie, so the sandbox id is derived from "ml-smoke-<timestamp>"). Never calls admin/reset or anything destructive; the
smoke sandbox is separate from every visitor's. If something fails, the exact request / response is written to the report; nothing is "fixed".
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from ml.eval.audit_texts import flags_of
from ml.photo import config

ROOT = Path(__file__).resolve().parents[2]
BASE = "https://maqsad-production-d704.up.railway.app"
RANK = {"green": 0, "yellow": 1, "red": 2}


class Client:
    def __init__(self, base, sandbox_uuid):
        self.base, self.cookie = base.rstrip("/"), f"sandbox_id={sandbox_uuid}"
        self.log = []

    def call(self, method, path, body=None, headers=None):
        h = {"Cookie": self.cookie, "User-Agent": "maqsad-ml-smoke/1.0", **(headers or {})}
        req = urllib.request.Request(self.base + path, data=body, method=method, headers=h)
        t = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                raw, code = r.read(), r.status
        except urllib.error.HTTPError as e:
            raw, code = e.read(), e.code
        except Exception as e:                                             # noqa: BLE001
            raw, code = repr(e).encode(), -1
        ms = (time.perf_counter() - t) * 1000
        self.log.append((method, path, code, ms))
        try:
            return code, json.loads(raw), ms
        except ValueError:
            return code, raw[:300].decode("utf-8", "replace"), ms


def multipart(fields: dict, file_name: str, data: bytes):
    b = "----mq" + uuid.uuid4().hex
    parts = []
    for k, v in fields.items():
        parts.append(f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    parts.append(f'--{b}\r\nContent-Disposition: form-data; name="image"; filename="{file_name}"\r\nContent-Type: image/jpeg\r\n\r\n'.encode()
                 + data + b"\r\n")
    parts.append(f"--{b}--\r\n".encode())
    return b"".join(parts), {"Content-Type": f"multipart/form-data; boundary={b}"}


def stats(ms):
    s = sorted(ms)
    return dict(n=len(s), min=s[0], p50=statistics.median(s), p95=s[min(len(s) - 1, int(0.95 * len(s)))], max=s[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE)
    a = ap.parse_args()
    started = dt.datetime.now(dt.timezone.utc)
    stamp = started.strftime("%Y%m%d-%H%M%S")
    sandbox = uuid.uuid5(uuid.NAMESPACE_URL, f"ml-smoke-{stamp}")
    c = Client(a.base, sandbox)
    rows, fails, lat = [], [], {}

    def check(name, ok, detail=""):
        rows.append((name, "OK" if ok else "FAIL", detail))
        if not ok:
            fails.append((name, detail))

    # ---- health
    code, h, ms = c.call("GET", "/api/health")
    check("GET /api/health -> 200, ml_mode real, warmup done, all functions real",
          code == 200 and h.get("ml_mode") == "real" and h.get("warmup") == "done" and all(v == "real" for v in h.get("functions", {}).values()),
          f"HTTP {code}; {json.dumps(h)[:300]}")
    model_version = h.get("model_version") if isinstance(h, dict) else None

    # ---- queue
    lat["queue"] = []
    for _ in range(10):
        code, q, ms = c.call("GET", "/api/queue?limit=50&offset=0")
        lat["queue"].append(ms)
    items = q["items"] if code == 200 else []
    pri = [x["priority"] for x in items]
    check("GET /api/queue first page: 50 items, sorted by priority (descending)", code == 200 and len(items) == 50 and pri == sorted(pri, reverse=True),
          f"HTTP {code}; {len(items)} items; first priorities {[round(p, 3) for p in pri[:3]]}")
    check("queue colours valid, color >= base_color, counts add up to total",
          code == 200 and all(x["color"] in RANK and RANK[x["color"]] >= RANK[x["base_color"]] for x in items)
          and sum(q["counts"].values()) == q["total"], f"counts {q.get('counts')} total {q.get('total')}")
    check("the first page is the top of the queue (red loans first)", code == 200 and items[0]["color"] == "red" and items[0]["risk"] > 0.5,
          f"first: {items[0]['loan_id']} {items[0]['color']} risk {items[0]['risk']:.3f}" if items else "")
    bad = [(x["loan_id"], x["top_reason"], flags_of(x["top_reason"], "reason")) for x in items if flags_of(x["top_reason"], "reason")]
    check("top_reason of the first page: non-empty Uzbek text without flagged patterns", not bad and all(x["top_reason"] for x in items),
          f"flagged: {bad[:3]}" if bad else f"e.g. {items[0]['top_reason']!r}" if items else "")
    code_l, leg, _ = c.call("GET", "/api/queue/legend")
    check("GET /api/queue/legend: red + yellow budget reported with photo escalations", code_l == 200 and leg.get("n_red") == 500,
          f"HTTP {code_l}; {leg.get('note') if isinstance(leg, dict) else leg}")

    # ---- loan page
    loan = items[0]["loan_id"] if items else "L1500001"
    lat["loan"] = []
    for _ in range(10):
        code, d, ms = c.call("GET", f"/api/loan/{loan}")
        lat["loan"].append(ms)
    ok = code == 200 and d.get("model_version") == model_version and d["reasons"] and d["reasons"][0]["text_uz"] == items[0]["top_reason"]
    check(f"GET /api/loan/{loan}: reasons[0] == queue top_reason, evidence present", ok and len(d.get("evidence", [])) > 0,
          f"HTTP {code}; {len(d.get('reasons', []))} reasons, {len(d.get('evidence', []))} evidence" if code == 200 else str(d)[:300])

    # ---- check_photo in the NEW sandbox (cookie): same image twice for two different loans -> duplicate
    index = json.loads((config.ARTIFACTS / "photo_index_v1.json").read_text(encoding="utf-8"))
    img_name = next(f for f in index["reserved"] if f.startswith("tikuv_dastgohi"))        # a sewing-machine image NOT in the portfolio index
    img = (config.ASSETS / img_name).read_bytes()
    code_p, page, _ = c.call("GET", "/api/queue?purpose=tikuv_dastgohi&color=green&limit=50")
    cand = [x["loan_id"] for x in page["items"] if not x["escalations"]][:3] if code_p == 200 else []
    lat["check_photo"], photo = [], []
    for k, ln in enumerate(cand[:3]):
        body, hdr = multipart({"loan_id": ln}, img_name, img)
        code, r, ms = c.call("POST", "/api/check_photo", body, hdr)
        lat["check_photo"].append(ms)
        photo.append((ln, code, r))
    first_ok = photo and photo[0][1] == 200 and photo[0][2].get("duplicate_of") == []
    check(f"POST /api/check_photo {img_name} for {cand[0] if cand else '?'} (new sandbox {sandbox}): 200, no duplicate yet", bool(first_ok),
          f"HTTP {photo[0][1]}; {json.dumps({k: v for k, v in photo[0][2].items() if k != 'blurred_image'})[:400]}" if photo else "no candidate loan")
    dup_ok = len(photo) > 1 and photo[1][1] == 200 and cand[0] in [x["loan_id"] for x in photo[1][2].get("duplicate_of", [])]
    check(f"same image again in the same sandbox for {cand[1] if len(cand) > 1 else '?'}: duplicate_of contains {cand[0] if cand else '?'}", dup_ok,
          f"HTTP {photo[1][1]}; duplicate_of={photo[1][2].get('duplicate_of')}" if len(photo) > 1 else "")
    am = photo[0][2].get("asset_match") if photo and isinstance(photo[0][2], dict) else None
    check("CLIP asset check is ACTIVE on the deployed service: asset_match present, sewing machine matches the sewing-machine loan",
          am is not None and am.get("matches_purpose") is True,
          f"asset_match={am}" + ("  (null = the CLIP weights are not in the image / CLIP failed to load; check_photo degrades safely as designed, "
                                 "/api/health does not show clip_loaded)" if am is None else ""))
    if cand:
        code, dl, _ = c.call("GET", f"/api/loan/{cand[0]}")
        esc = [e["code"] for e in dl.get("escalations", [])] if code == 200 else []
        check(f"GET /api/loan/{cand[0]} in the smoke sandbox now escalates (PHOTO_DUPLICATE)", "PHOTO_DUPLICATE" in esc,
              f"HTTP {code}; color {dl.get('color') if code == 200 else '?'}; escalations {esc}")
    # a different (fresh) cookie must not see this sandbox's escalation
    other = Client(a.base, uuid.uuid4())
    code, dl2, _ = other.call("GET", f"/api/loan/{cand[0]}") if cand else (0, {}, 0)
    check("another sandbox (fresh cookie) does NOT see the escalation", code == 200 and "PHOTO_DUPLICATE" not in [e["code"] for e in dl2.get("escalations", [])],
          f"HTTP {code}")

    # ---- report
    L = [f"# Deployed service check ({a.base})", "",
         f"Run at {started.strftime('%Y-%m-%d %H:%M:%S')} UTC by `python -m ml.eval.check_deployed`; model_version `{model_version}`. "
         "Read-only: only GET health / queue / legend / loan and POST check_photo in a separate smoke sandbox "
         f"(cookie `sandbox_id={sandbox}`, derived from `ml-smoke-{stamp}`; the backend requires a UUID); no admin / reset call.", "",
         "| Check | Result | Detail |", "|---|---|---|"]
    L += [f"| {n} | {r} | {d.replace('|', '/')[:300]} |" for n, r, d in rows]
    L += ["", "## Client-side latency (from the machine that ran the check, includes the network)", "", "| Request | calls | min | p50 | p95 | max |", "|---|---|---|---|---|---|"]
    for k, label in (("queue", "GET /api/queue?limit=50"), ("loan", f"GET /api/loan/{loan}"), ("check_photo", "POST /api/check_photo")):
        if lat.get(k):
            s = stats(lat[k])
            L.append(f"| {label} | {s['n']} | {s['min']:.0f} ms | {s['p50']:.0f} ms | {s['p95']:.0f} ms | {s['max']:.0f} ms |")
    L += ["", "Latency includes the first request after idle time (cold paths) and the network; with 3 or 10 calls p95 is the slowest or second-slowest call."]
    if fails:
        L += ["", "## Failures (exact details)", ""] + [f"- **{n}**: {d}" for n, d in fails]
    else:
        L += ["", "All checks passed."]
    (ROOT / "docs" / "deployed_check.md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
