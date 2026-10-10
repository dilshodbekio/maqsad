"""Measure CPU time per component and render docs/cost.md.

    .venv/Scripts/python -m ml.eval.measure_cost

CPU time = process CPU seconds (all threads) per call, measured warm on this machine; wall time is reported next to it.
Server prices are NOT invented: they are named parameters in PRICES with the URL and the date they were read; a price that
could not be confirmed on the provider's own page is marked as such. Every assumption is its own "Taxmin:" line.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from ml import api
from ml.photo import config, imageio

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "ml" / "artifacts" / "eval_cost.json"
DOC = ROOT / "docs" / "cost.md"
N = 100

# ---- public prices (read 2026-10-10). Nothing here is a guess; leave a value None if it cannot be sourced.
PRICES = dict(
    read_on="2026-10-10",
    railway=dict(url="https://docs.railway.com/reference/pricing/plans", cpu_usd_per_vcpu_month=20.0, cpu_usd_per_vcpu_minute=0.000463,
                 ram_usd_per_gb_month=10.0, ram_usd_per_gb_minute=0.000231, note="usage-based; the plan fee (Hobby $5 / Pro $20 per month) "
                 "includes usage up to the same amount (stated on the same page)"),
    dedicated=dict(name="Hetzner AX42 (AMD Ryzen 7 PRO 8700GE, 8 cores / 16 threads, 64 GB DDR5 ECC)",
                   spec_url="https://www.hetzner.com/dedicated-rootserver/ax42/",
                   price_eur_per_month=68.19, price_source="https://www.deskmodder.de/blog/?p=213645",
                   price_confirmed_on_official_page=False, cores=8,
                   note="the hardware is stated on Hetzner's page, but the price fields were empty when the page was read; 68.19 EUR/month "
                        "(excl. VAT, effective 2026-04-01) comes from a third-party news report and must be checked on hetzner.com"),
)


def cpu_wall(fn, n=N, warm=5):
    for _ in range(warm):
        fn()
    c0, w0 = time.process_time(), time.perf_counter()
    for _ in range(n):
        fn()
    return dict(cpu_ms=(time.process_time() - c0) / n * 1000, wall_ms=(time.perf_counter() - w0) / n * 1000, n=n)


def measure(single_thread: bool = False) -> dict:
    """CPU time per component. single_thread=True pins OpenCV / torch / OpenMP to one thread (what a 1-vCPU container does):
    with many threads the process CPU time is inflated by spinning worker threads and no longer means 'work done'."""
    if single_thread:
        cv2.setNumThreads(1)
        import torch
        torch.set_num_threads(1)
    t0c, t0w = time.process_time(), time.perf_counter()
    warm = api.warm_up()
    warm_cpu = time.process_time() - t0c
    q = api.get_queue()
    loans = [x["loan_id"] for x in q[:: len(q) // N]][:N]
    it = iter(range(10 ** 9))
    out = dict(warm_up=dict(cpu_s=warm_cpu, wall_s=time.perf_counter() - t0w, **warm))
    out["score_loan"] = cpu_wall(lambda: api.score_loan(loans[next(it) % len(loans)]))
    presets = api.get_scenario_presets()
    out["score_scenario"] = cpu_wall(lambda: api.score_scenario(presets[next(it) % 4]["scenario"]), n=40)
    rgb = imageio.decode((config.ASSETS / "faces_test_09.jpg").read_bytes())
    big = cv2.resize(rgb, None, fx=1600 / max(rgb.shape[:2]), fy=1600 / max(rgb.shape[:2]), interpolation=cv2.INTER_CUBIC)
    data = imageio.encode_jpeg(big)
    loan = loans[0]
    out["check_photo"] = cpu_wall(lambda: api.check_photo(loan, data), n=60)
    from ml.photo import hashing
    from ml.photo.asset import get_asset_model
    from ml.photo.checker import _detector
    from ml.photo.faces import blur_faces
    am = get_asset_model()
    parts = dict(decode_and_hash=cpu_wall(lambda: hashing.hash_image(imageio.decode(data)), n=40),
                 face_detect_and_blur=cpu_wall(lambda: blur_faces(_detector(), big), n=40),
                 clip_asset_check=cpu_wall(lambda: am.asset_match(big, "chorva"), n=40))
    out["check_photo_parts"] = parts
    import psutil
    out["rss_mb"] = psutil.Process().memory_info().rss / 2 ** 20
    out["machine"] = dict(cpu=platform.processor(), logical_cores=os.cpu_count(), python=sys.version.split()[0],
                          threads="1 thread (OpenCV, torch, OpenMP)" if single_thread else "all cores")
    return out


def usd_per_1000_railway(cpu_s: float) -> float:
    return cpu_s * 1000 * PRICES["railway"]["cpu_usd_per_vcpu_minute"] / 60


def render(multi: dict, m: dict) -> str:
    """m = single-thread measurement (the cost basis), multi = all cores (latency on this machine)."""
    P = PRICES
    R, D = P["railway"], P["dedicated"]
    cs = {k: m[k]["cpu_ms"] / 1000 for k in ("score_loan", "score_scenario", "check_photo")}
    L = []
    w = L.append
    w("# Cost per check")
    w("")
    w("All CPU times were **measured** by `python -m ml.eval.measure_cost` on the machine below (process CPU seconds, warm, "
      f"{N} calls per component); prices are public prices with their source and date. Numbers are for the prototype code on synthetic data. "
      "The cost basis is the **single-thread** measurement (OpenCV, torch and OpenMP pinned to one thread = one vCPU): with all cores the process "
      "CPU time is inflated by spinning worker threads (see section 1b) and does not mean work done.")
    w("")
    mach = m["machine"]
    w(f"Machine: {mach['cpu']}, {mach['logical_cores']} logical cores, Python {mach['python']}. Resident memory after warm-up: {m['rss_mb']:.0f} MB.")
    w("")
    w("## 1. Measured CPU time (single thread = one vCPU; the cost basis)")
    w("")
    w("| Component | CPU time per call | Wall time per call |")
    w("|---|---|---|")
    labels = [("score_loan", "Money-trail score (`score_loan`: model inference + reasons)"), ("score_scenario", "Scenario score (`score_scenario`)"),
              ("check_photo", "`check_photo` total (decode, hash, duplicates, face detect + blur, CLIP asset check, re-encode; 1600 px)")]
    for k, label in labels:
        w(f"| {label} | {m[k]['cpu_ms']:.0f} ms | {m[k]['wall_ms']:.0f} ms |")
    for k, label in (("decode_and_hash", "of which decode + perceptual hashes"), ("face_detect_and_blur", "of which face detection + blur"),
                     ("clip_asset_check", "of which CLIP asset check (embedding + probe)")):
        p = m["check_photo_parts"][k]
        w(f"| &nbsp;&nbsp;{label} | {p['cpu_ms']:.0f} ms | {p['wall_ms']:.0f} ms |")
    wu = m["warm_up"]
    w(f"| Warm-up once at start (`ml.api.warm_up()`: demo portfolio + features + scores + queue, face detector, CLIP) | {wu['cpu_s']:.0f} s | {wu['wall_s']:.0f} s |")
    w("")
    w("### 1b. The same on all cores of this machine (latency, not cost)")
    w("")
    w("| Component | CPU time (all threads) | Wall time |")
    w("|---|---|---|")
    for k, label in labels:
        w(f"| {label} | {multi[k]['cpu_ms']:.0f} ms | {multi[k]['wall_ms']:.0f} ms |")
    w(f"| Warm-up once | {multi['warm_up']['cpu_s']:.0f} s | {multi['warm_up']['wall_s']:.0f} s |")
    w("")
    w("Warm-up breakdown (wall): portfolio " + f"{wu['portfolio_s']} s, face detector {wu['face_detector_s']} s, CLIP {wu['clip_s']} s. "
      "Warm-up is paid once per process, not per check.")
    w("")
    w("## 2. Server prices used (public, read " + P["read_on"] + ")")
    w("")
    w(f"- **Railway** (usage-based): CPU ${R['cpu_usd_per_vcpu_month']:g} per vCPU per month (${R['cpu_usd_per_vcpu_minute']} per vCPU-minute), "
      f"memory ${R['ram_usd_per_gb_month']:g} per GB per month (${R['ram_usd_per_gb_minute']} per GB-minute). Source: {R['url']} (read {P['read_on']}). {R['note']}.")
    w(f"- **Dedicated CPU server, on-prem-like option**: {D['name']}. Hardware: {D['spec_url']}. Price: **{D['price_eur_per_month']} EUR per month** "
      f"(source: {D['price_source']}, read {P['read_on']}) - **not confirmed on the provider's page**: {D['note']}.")
    w("")
    w("## 3. Cost per 1 000 checks (variable CPU cost)")
    w("")
    rail = {k: usd_per_1000_railway(c) for k, c in cs.items()}
    core_hour_eur = D["price_eur_per_month"] / (30 * 24) / D["cores"]
    ded = {k: c * 1000 / 3600 * core_hour_eur for k, c in cs.items()}
    w("| Check | Railway (CPU only), USD per 1 000 | Dedicated server at 100% core utilisation, EUR per 1 000 |")
    w("|---|---|---|")
    for k, label in (("score_loan", "Money-trail score"), ("score_scenario", "Scenario score"), ("check_photo", "Photo check (incl. CLIP)")):
        w(f"| {label} | {rail[k]:.5f} | {ded[k]:.5f} |")
    w("")
    w(f"Formulas: Railway = CPU seconds x 1 000 x ${R['cpu_usd_per_vcpu_minute']} / 60. Dedicated = CPU seconds x 1 000 / 3 600 x "
      f"({D['price_eur_per_month']} EUR / 720 h / {D['cores']} cores) = CPU seconds x 1 000 x {core_hour_eur / 3600:.3e} EUR per core-second.")
    w("")
    w("## 4. Always-on cost (not per check)")
    w("")
    gb = m["rss_mb"] / 1024
    base_ram = gb * R["ram_usd_per_gb_month"]
    w(f"- Railway memory for the resident model + portfolio: {gb:.2f} GB x ${R['ram_usd_per_gb_month']:g} = **${base_ram:.2f} per month** (memory is billed while the service runs, whatever the number of checks).")
    w(f"- Dedicated server: **{D['price_eur_per_month']} EUR per month** flat (see the caveat above).")
    w("")
    w("## 5. LLM report (`/draft_report`)")
    w("")
    w("- Prototype: an external LLM is called only in `/draft_report`; the cost per report is **`LLM_COST_USD_PER_REPORT` = (parameter)** - take it from the backend's "
      "logged `cost_usd` (Oybek logs it); it is not estimated here.")
    w("- Production design: a local LLM (vLLM / Ollama) on the bank's own server, so cost = server time: **`LLM_SECONDS_PER_REPORT` = (parameter, to be measured with the chosen model)** "
      f"x the core-second price above (Railway ${R['cpu_usd_per_vcpu_minute'] / 60:.3e} per vCPU-second; dedicated {core_hour_eur / 3600:.3e} EUR per core-second; a GPU server needs its own price).")
    w("")
    w("## 6. Assumptions (for the ROI model)")
    w("")
    for t in (
        "Taxmin: one check = one call of the component in the table of section 1 on a warm process; the measured CPU time on this machine is representative of the server (a cloud vCPU may be slower or faster).",
        "Taxmin: the photo check uses a 1 600 px JPEG (the largest size the service accepts after downscaling); smaller photos cost less.",
        "Taxmin: Railway bills CPU for the time it is used; we count only the measured CPU seconds per check and the always-on memory separately, we do not count idle CPU, egress or storage.",
        "Taxmin: the dedicated-server column assumes 100% core utilisation; at utilisation u the cost per check is divided by u (for example at 30% it is 3.3 times higher).",
        "Taxmin: one month = 30 days = 720 hours.",
        "Taxmin: exchange rates and VAT are not applied; prices are quoted in their original currency.",
        "Taxmin: the money-trail features are computed once for the whole portfolio in a batch (warm-up); the per-check cost excludes it. A new loan needs only its own feature row.",
        "Taxmin: no cost for the bank officer's time, development, hosting of the dashboard or the audit log is included.",
    ):
        w(f"- {t}")
    w("")
    return "\n".join(L) + "\n"


def _run(single: bool) -> dict:
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    if single:
        env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", MAQSAD_TORCH_THREADS="1")
    out = subprocess.run([sys.executable, "-m", "ml.eval.measure_cost", "--worker"] + (["--single-thread"] if single else []),
                         capture_output=True, text=True, cwd=ROOT, env=env)
    return json.loads(out.stdout.strip().splitlines()[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--single-thread", action="store_true")
    a = ap.parse_args()
    if a.worker:                                  # one measurement in a fresh process; prints one JSON line
        print(json.dumps(measure(a.single_thread)))
        return
    single, multi = _run(True), _run(False)
    RES.write_text(json.dumps(dict(single_thread=single, all_cores=multi), indent=1), encoding="utf-8", newline="\n")
    DOC.write_text(render(multi, single), encoding="utf-8", newline="\n")
    print(DOC.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
