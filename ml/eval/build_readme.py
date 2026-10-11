"""Render ml/README.md. Every number comes from the stored evaluation results (ml/artifacts/eval_*.json), produced by the scripts in ml/eval.

    .venv/Scripts/python -m ml.eval.build_readme
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "ml" / "artifacts"
M = "Money-trail model (LightGBM)"
MET = "../docs/metrics.md"


def J(name):
    p = ART / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def ci(c):
    return f"[{c[0]:.3f}, {c[1]:.3f}]"


def rng(vals, f="{:.1f}"):
    return f"{f.format(min(vals))}-{f.format(max(vals))}"


def render() -> str:
    hold, dev, lat, photo, asset_ = J("eval_money_trail_holdout.json"), J("eval_money_trail_dev.json"), J("eval_latency.json"), J("eval_photo.json"), J("eval_asset.json")
    scen, cost, probe = J("eval_scenario.json"), J("eval_cost.json"), J("money_trail_v2.json")
    names = list(hold["recall"])
    rec, cl = hold["recall"], hold["ci"]["recall"]
    best = names[2]
    dup_v = photo["dup"]["variants"]["pd_crop"]
    faces = photo["faces"]
    sc = cost["single_thread"]
    z, p = asset_["zero_shot"], asset_["probe_cv"]
    run = hold["run"]
    n_img = len(list(csv.DictReader(open(ROOT / "ml" / "assets" / "photos" / "manifest.csv", encoding="utf-8"))))
    clip_s = rng([c["seconds"] for c in asset_["clip_load"]])
    L = []
    w = L.append
    w("# ml/ - AI / ML layer (owner: Dilshodbek)")
    w("")
    w("Public interface for the backend: `ml/api.py` (contract: [`docs/api_contract.md`](../docs/api_contract.md)). Metrics: [`docs/metrics.md`](../docs/metrics.md); "
      "cost: [`docs/cost.md`](../docs/cost.md); text audit: [`docs/audit_texts.md`](../docs/audit_texts.md). Python 3.13, `pip install -r ml/requirements.txt` "
      "(the first line adds the PyTorch **CPU** wheel index). All numbers below are on **synthetic data**; real accuracy is measured in a pilot on historical loans.")
    w("")
    w("## 1. What the ML layer does")
    w("")
    w("It checks, remotely, whether a preferential microloan was used for its stated purpose, from three kinds of evidence: the **money trail** "
      "(a LightGBM model on post-disbursement transactions, the declared supplier / invoice and the supplier's own flows), the **photo** of the asset "
      "(duplicate detection, face blur, geo / time checks, a CLIP check that the thing in the photo is the loan's asset) and the **invoice**. "
      "It builds a risk queue with plain-Uzbek reasons that point to evidence; it never says anyone is guilty and **a bank officer always decides**.")
    w("")
    w("## 2. Architecture")
    w("")
    w("```mermaid")
    w("flowchart LR")
    w("  TX[Transactions + invoice + supplier + employee link] --> F[Features as of snapshot, 29 money-trail features]")
    w("  F --> M[LightGBM v2 risk score 0..1]")
    w("  M --> R[Reasons: top-3 risk-raising features -> Uzbek text + evidence refs]")
    w("  M --> C[Colour by queue rank: top 10% red, next 10% yellow]")
    w("  SC[Scenario form: own loan + transactions] --> F")
    w("  IMG[Photo upload] --> EX[Decode, EXIF orientation, downscale 1600 px]")
    w("  EX --> H[pHash + dHash duplicate search: portfolio + same sandbox]")
    w("  EX --> FB[YuNet face detection + blur, re-encode: EXIF dropped, original never stored]")
    w("  EX --> GT[Geo distance + capture-time checks]")
    w("  EX --> CL[OpenCLIP ViT-B-32 + linear probe: asset matches purpose?]")
    w("  H --> FU[Fusion policy: PHOTO_DUPLICATE / FAR / TIME / ASSET_MISMATCH]")
    w("  GT --> FU")
    w("  CL --> FU")
    w("  C --> FU")
    w("  FU --> Q[Risk queue sorted by priority, escalations, reasons]")
    w("  R --> Q")
    w("  Q --> O[Bank officer reads reasons + evidence and DECIDES; audit log]")
    w("```")
    w("")
    w("The photo result never changes the money-trail `risk`; it can only raise the colour / priority (policy in [`ml/fusion.py`](fusion.py), no learned model).")
    w("")
    w("## 3. Components")
    w("")
    w("| Component | Method | Built by us vs. pre-trained | Key metric (source) | Latency |")
    w("|---|---|---|---|---|")
    w(f"| Synthetic portfolio generator (`ml/generator`) | rule-based simulation of 4 fraud typologies + honest-but-suspicious profiles | all ours | per-typology leakage check with a flag at AUC >= 0.90 (`ml/generator/check_leakage.py`; the T1 cash features are intended) | demo portfolio built in {rng([c['init_s'] for c in lat['cold']])} s |")
    w(f"| Money-trail model (`ml/money_trail`) | LightGBM on 29 features, supplier-grouped CV | features, labels, training, reasons: ours; LightGBM library | holdout recall@20% {rec[M]:.3f} (clustered CI {ci(cl[M]['clustered'])}), ROC-AUC {hold['roc']:.3f} ([metrics §2]({MET}#2-money-trail-v2-holdout--headline)) | `score_loan` p50 {rng([r['p50'] for r in lat['runs']])} ms |")
    w(f"| Reasons + evidence | LightGBM `pred_contrib` -> one template per feature (Uzbek, English, Russian: same facts and numbers), de-duplicated | all ours | text audit of {J('audit_texts_after.json')['n_translated_texts']} texts x 3 languages: {sum(J('audit_texts_after.json')['counts'].values())} flags ([`docs/audit_texts.md`](../docs/audit_texts.md)) | included above |")
    w(f"| Scenario scoring (`ml/money_trail/scenario.py`) | user-built loan scored live by the same model | all ours | features equal to a full portfolio copy (test); example in [metrics §5]({MET}#5-scenario-scoring-and-evidence-fusion) | p50 {rng([r['p50'] for r in scen['runs']], '{:.0f}')} ms |")
    w(f"| Photo duplicates (`ml/photo/hashing.py`) | pHash + dHash, whole image + central crop | hashing code ours (numpy / OpenCV) | held-out recall {pct(dup_v['pooled']['recall'])} at precision {pct(dup_v['pooled']['precision'])} over 8 transforms ([metrics §4]({MET}#duplicate-detection-under-transformations)) | part of `check_photo` |")
    w(f"| Face blur (`ml/photo/faces.py`) | YuNet detector, mosaic + blur, re-detect | detector pre-trained (OpenCV Zoo, MIT); pipeline ours | {faces['tp']} of {faces['truth']} hand-counted faces found (+{faces['fp']} false alarm), {faces['left_yunet']} detectable left after blur ([metrics §4]({MET}#face-blur-yunet-opencv-zoo-mit-haar-cascade-as-fallback)) | part of `check_photo` |")
    w(f"| Asset check (`ml/photo/asset.py`) | OpenCLIP ViT-B-32 embeddings + logistic-regression probe on our library | CLIP pre-trained (OpenAI weights); probe, library, threshold, evaluation ours | grouped CV accuracy {pct(p['acc']['mean'])} (probe) vs {pct(z['acc'])} (zero-shot) ([metrics §4]({MET}#asset-check-is-the-thing-in-the-photo-what-the-loan-was-for-openclip-vit-b-32-cpu)) | `check_photo` p50 {photo['lat']['p50']:.0f} ms, p95 {photo['lat']['p95']:.0f} ms (all parts, 1600 px) |")
    w(f"| Evidence fusion (`ml/fusion.py`) | documented policy dict, escalation only | all ours | no accuracy claimed (no labelled photo data) ([metrics §5]({MET}#fusion-policy-photo---queue)) | negligible |")
    w("")
    w("## 4. Headline results (synthetic data)")
    w("")
    w(f"**Money trail, holdout** (seed 2026, evaluated once at commit `{run['commit'][:10]}`, model sha256 `{run['model_sha256'][:12]}...`; [metrics §2]({MET}#recall20-inspection-budget)):")
    w("")
    w("| Method | recall@20% | 95% CI clustered by supplier |")
    w("|---|---|---|")
    for n in names[:3]:
        w(f"| {n} | {rec[n]:.3f} | {ci(cl[n]['clustered'])} |")
    w("| Random selection | 0.200 | |")
    w("")
    g = hold["groups"]
    gci = hold["ci"]["groups"]
    w("| Typology | recall@20% (model) | 95% CI | graded rule | best single feature |")
    w("|---|---|---|---|---|")
    for t in ("T1", "T2", "T3", "T4"):
        c = gci[t]
        w(f"| {t} | {pct(g[t][M])} | [{c[0] * 100:.1f}, {c[1] * 100:.1f}] | {pct(g[t][names[1]])} | {pct(g[t][names[2]])} |")
    w("")
    w(f"The honest false-positive rate inside the budget is {pct(g['honest/all'][M])}; honest look-alikes are flagged most "
      f"(`oked_mismatch` {pct(g['honest/oked_mismatch'][M])}, `livestock_seller` {pct(g['honest/livestock_seller'][M])}). "
      f"The best single feature beats the model on T3 ({pct(g['T3'][names[2]])} vs {pct(g['T3'][M])}); see [metrics §6]({MET}#6-known-failure-cases).")
    w("")
    w(f"**Photo duplicates** (deployed `pd_crop`, held-out half of the library; [metrics §4]({MET}#duplicate-detection-under-transformations)): "
      + ", ".join(f"{t} {pct(m['recall'], 0)}" for t, m in dup_v["per"].items() if t != "original") + f" recall, precision {pct(dup_v['pooled']['precision'])}; "
      "crops are the weak spot.")
    w("")
    w(f"**CLIP asset check** (author-grouped 5-fold CV, 5 seeds; [metrics §4]({MET}#asset-check-is-the-thing-in-the-photo-what-the-loan-was-for-openclip-vit-b-32-cpu)): "
      f"6-class accuracy {pct(p['acc']['mean'])} ({pct(p['acc']['min'])}-{pct(p['acc']['max'])}) for the probe vs {pct(z['acc'])} zero-shot; "
      f"'matches the loan purpose' F1 {pct(p['binary']['f1']['mean'])} (probe) vs {pct(z['binary']['f1']['mean'])} (zero-shot) (tiny library, 5 CV seeds; see metrics for the ranges).")
    w("")
    w("## 5. Data")
    w("")
    w("- **Synthetic portfolio** (generator, 5 000 loans per split; method and every assumption in [`data/README.md`](../data/README.md)). Synthetic because the project rules forbid real "
      "personal or bank data and no public dataset has loan purpose + supplier + invoice + post-disbursement flows (see \"Datasets considered\" in `data/README.md`).")
    w(f"- **Splits** ([metrics §1]({MET}#1-data--splits)): `train` (seed 42), `dev` (1337, looked at during development), `holdout` (2026, evaluated **once**, run recorded with model sha256 and commit hash; "
      "the evaluation script refuses to run it twice), `demo` (7, served by the app, never used for metrics). Disjoint ID spaces, hence disjoint suppliers.")
    w(f"- **Photos**: {n_img} openly licensed Wikimedia Commons images (5 asset purposes, `other` negatives, faces test) listed with author and licence in `ml/assets/photos/manifest.csv` and `data/README.md`.")
    w("")
    w("## 6. Honest limitations")
    w("")
    w(f"See [Known failure cases]({MET}#6-known-failure-cases). In short: everything is synthetic (rates, typologies and the generator's own structure are assumptions); T3 and T4 are the hard typologies "
      "and fraud that has not happened yet at the snapshot cannot be seen; young loans stay harder; the photo duplicate hash is not crop-proof; the CLIP library is tiny, contains a toy and a model "
      "labelled as assets and its weights licence needs legal review for production; fusion is a policy, not a measured model.")
    w("")
    w("## 7. Reproduce")
    w("")
    w("```")
    w(".venv/Scripts/python -m pip install -r ml/requirements.txt")
    w(".venv/Scripts/python -m ml.generator.build --split all          # train, dev, holdout, demo (in memory is enough; this writes data/generated/)")
    w(".venv/Scripts/python -m ml.generator.check_leakage --split all  # train + dev (never the holdout)")
    w(".venv/Scripts/python -m ml.money_trail.train                    # model v2, train split only")
    w(".venv/Scripts/python -m ml.eval.eval_money_trail --split dev")
    w(".venv/Scripts/python -m ml.eval.eval_money_trail --split holdout  # EXACTLY ONCE (refuses to run twice)")
    w(".venv/Scripts/python -m ml.eval.eval_money_trail --latency")
    w(".venv/Scripts/python -m ml.eval.eval_photo --write-config; python -m ml.photo.index; python -m ml.eval.eval_photo")
    w(".venv/Scripts/python -m ml.eval.eval_asset                      # CLIP: needs the weights (see section 8)")
    w(".venv/Scripts/python -m ml.eval.eval_scenario")
    w(".venv/Scripts/python -m ml.eval.measure_cost                    # docs/cost.md")
    w(".venv/Scripts/python -m ml.eval.audit_texts --tag after         # docs/audit_texts.md")
    w(".venv/Scripts/python -m ml.eval.build_metrics                   # docs/metrics.md from the stored results")
    w(".venv/Scripts/python -m ml.eval.build_readme                    # this file")
    w(".venv/Scripts/python -m pytest ml/tests -q                      # add -m slow for the slow tests")
    w("```")
    w("")
    w("## 8. Production design")
    w("")
    w("- **Runs fully on-prem; no external API in the ML path.** The money-trail model, the photo checks and CLIP run locally on CPU. The only external LLM call of the prototype is `/draft_report` in the backend; "
      "production uses a local LLM (vLLM / Ollama) for the report.")
    w("- **Privacy**: the original image is never written anywhere (decoded and hashed in memory); faces are blurred and EXIF is dropped by re-encoding; only the blurred JPEG and a perceptual hash are kept; "
      "sandbox photos live in memory only and are forgotten by `reset_sandbox`.")
    w("- **Cost / sizing**: measured CPU time and cost per 1 000 checks, with sources and assumptions, are in [`docs/cost.md`]"
      f"(../docs/cost.md) (photo check {sc['check_photo']['cpu_ms']:.0f} ms single-thread CPU; process start to a warm portfolio {rng([c['total_s'] for c in lat['cold']], '{:.0f}')} s; loading CLIP alone {clip_s} s).")
    w("")
    w("### Service start-up: call `ml.api.warm_up()` once")
    w("")
    w("`warm_up()` loads the demo portfolio (features, scores, queue), the face detector and CLIP, and returns the seconds each took plus `clip_loaded` / `clip_error` "
      f"(measured {rng([c['seconds'] for c in asset_['warm_up_total']])} s and {rng([c['rss_mb'] for c in asset_['warm_up_total']], '{:.0f}')} MB in total, [metrics §4]({MET}#asset-check-is-the-thing-in-the-photo-what-the-loan-was-for-openclip-vit-b-32-cpu)).")
    w("")
    w("### CLIP weights (asset check): pre-download at BUILD time, never at request time")
    w("")
    w("The weights (OpenAI ViT-B-32, 354 MB, sha256 checked by open_clip) are **not committed** and are **never downloaded while the service runs**: "
      "the service only reads `ViT-B-32.pt` from the cache directory. If the file is missing (or torch / the probe cannot be loaded) the failure is cached and logged **once** (logger `maqsad.asset`), "
      "`check_photo` keeps working and returns `asset_match: None` (no `PHOTO_ASSET_MISMATCH` escalation, no retry per upload, no blocking), and `warm_up()` reports `clip_loaded: false` with the reason in `clip_error`.")
    w("")
    w("Dockerfile (the only place that may download):")
    w("")
    w("```dockerfile")
    w("COPY . /app")
    w("WORKDIR /app")
    w("RUN pip install --no-cache-dir -r ml/requirements.txt")
    w("")
    w("# --- build time: pre-download the CLIP weights into the image (needs internet, runs once) ---")
    w("ENV MAQSAD_CLIP_CACHE=/opt/maqsad/clip")
    w("RUN python -m ml.photo.asset --download")
    w("")
    w("# --- runtime: no downloads; one thread per process matches the cost model in docs/cost.md ---")
    w("ENV HF_HUB_OFFLINE=1 MAQSAD_CLIP_CACHE=/opt/maqsad/clip \\")
    w("    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MAQSAD_TORCH_THREADS=1")
    w("```")
    w("")
    w("Check after building: `python -c \"from ml import api; print(api.warm_up())\"` must print `'clip_loaded': True`.")
    w("")
    w("| Environment variable | Meaning |")
    w("|---|---|")
    w("| `MAQSAD_CLIP_CACHE` | directory with `ViT-B-32.pt` (default `ml/.cache/clip`, git-ignored). Must be the same at build time and at runtime. |")
    w("| `HF_HUB_OFFLINE=1` | set by default by `ml/photo/asset.py`; open_clip never contacts the Hugging Face hub (the OpenAI weights come from OpenAI's public URL, only through `--download`). |")
    w("| `MAQSAD_ASSET_CHECK=off` | switch the asset check off (check_photo returns `asset_match: None`). |")
    w("| `MAQSAD_TORCH_THREADS`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS` | threads for torch / OpenMP. Without pinning, process CPU time per check is about 6 times higher than the single-thread figure used in `docs/cost.md`. |")
    w("")
    w("Licence note (needs legal review before production): see `data/README.md` and \"Known failure cases\" in `docs/metrics.md`.")
    w("")
    w("## 9. How it was built")
    w("")
    w("All code was written during the final with Claude Code as a coding assistant, using an orchestrator workflow with separate reviewer and tester subagents "
      "(the reviewer is read-only and reports leakage / honesty / contract problems; the tester writes tests and never changes implementation code). "
      "No code was reused from the qualifying round. Licences of every dataset, image, model and library: [`data/README.md`](../data/README.md).")
    w("")
    return "\n".join(L) + "\n"


def main():
    (ROOT / "ml" / "README.md").write_text(render(), encoding="utf-8", newline="\n")
    print("wrote ml/README.md")


if __name__ == "__main__":
    main()
