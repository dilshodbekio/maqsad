# ml/ - AI / ML layer (owner: Dilshodbek)

Public interface for the backend: `ml/api.py` (contract: `docs/api_contract.md`). Metrics: `docs/metrics.md`; cost: `docs/cost.md`.
Python 3.13, `pip install -r ml/requirements.txt` (the first line adds the PyTorch **CPU** wheel index).

## Service start-up: call `ml.api.warm_up()` once

`warm_up()` loads the demo portfolio (features, scores, queue), the face detector and CLIP, and returns the seconds each took plus
`clip_loaded` / `clip_error`. Measured on the development machine: ~16-25 s, ~1.4 GB memory (`docs/metrics.md`, section 4).

## CLIP weights (asset check): pre-download at BUILD time, never at request time

The weights (OpenAI ViT-B-32, 354 MB, sha256 checked by open_clip) are **not committed** and are **never downloaded while the service runs**:
the service only reads `ViT-B-32.pt` from the cache directory. If the file is missing (or torch / the probe cannot be loaded) the failure is
cached and logged **once** (logger `maqsad.asset`), `check_photo` keeps working and returns `asset_match: None` (no `PHOTO_ASSET_MISMATCH`
escalation, no retry per upload, no blocking), and `warm_up()` reports `clip_loaded: false` with the reason in `clip_error`.

Dockerfile (the only place that may download):

```dockerfile
COPY . /app
WORKDIR /app
RUN pip install --no-cache-dir -r ml/requirements.txt

# --- build time: pre-download the CLIP weights into the image (needs internet, runs once) ---
ENV MAQSAD_CLIP_CACHE=/opt/maqsad/clip
RUN python -m ml.photo.asset --download

# --- runtime: no downloads; one thread per process matches the cost model in docs/cost.md ---
ENV HF_HUB_OFFLINE=1 MAQSAD_CLIP_CACHE=/opt/maqsad/clip \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MAQSAD_TORCH_THREADS=1
```

Check after building: `python -c "from ml import api; print(api.warm_up())"` must print `'clip_loaded': True`.

| Environment variable | Meaning |
|---|---|
| `MAQSAD_CLIP_CACHE` | directory with `ViT-B-32.pt` (default `ml/.cache/clip`, git-ignored). Must be the same at build time and at runtime. |
| `HF_HUB_OFFLINE=1` | set by default by `ml/photo/asset.py`; open_clip never contacts the Hugging Face hub (the OpenAI weights come from OpenAI's public URL, only through `--download`). |
| `MAQSAD_ASSET_CHECK=off` | switch the asset check off (check_photo returns `asset_match: None`). |
| `MAQSAD_TORCH_THREADS`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS` | threads for torch / OpenMP. Without pinning, process CPU time per check is about 6 times higher than the single-thread figure used in `docs/cost.md`. |

Licence note (needs legal review before production): see `data/README.md` and "Known failure cases" in `docs/metrics.md`.
