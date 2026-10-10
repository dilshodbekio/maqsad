# syntax=docker/dockerfile:1
# Single service: FastAPI serves /api/* and the built frontend (frontend/dist) at /.

# ---- 1. frontend build ----
FROM node:22-slim AS frontend
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# ---- 2. ml requirements, if present (keeps the pip layer cached when only ml code changes) ----
FROM python:3.13-slim AS mlreq
COPY ml/ /src/ml/
RUN if [ -f /src/ml/requirements.txt ]; then cp /src/ml/requirements.txt /ml-requirements.txt; \
    else : > /ml-requirements.txt; fi

# ---- 3. runtime ----
FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000 \
    DATA_DIR=/data
WORKDIR /app

# libgomp1: OpenMP runtime needed by LightGBM wheels (not in -slim; without it `import lightgbm` raises OSError)
RUN apt-get update  && apt-get install -y --no-install-recommends libgomp1  && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /tmp/req/backend.txt
COPY --from=mlreq /ml-requirements.txt /tmp/req/ml.txt
# torch (if ml needs it) comes from the CPU-only index: no CUDA, much smaller image
RUN if grep -qiE '^[[:space:]]*torch' /tmp/req/ml.txt; then \
        EXTRA="--extra-index-url https://download.pytorch.org/whl/cpu"; else EXTRA=""; fi \
 && pip install $EXTRA -r /tmp/req/backend.txt -r /tmp/req/ml.txt

COPY backend/ backend/
COPY ml/ ml/
COPY --from=frontend /app/frontend/dist frontend/dist

# CLIP weights for the photo asset check (ml/README.md): downloaded ONCE here at build time, never while the service runs.
# Without them check_photo still works but returns asset_match: null.
ENV MAQSAD_CLIP_CACHE=/opt/maqsad/clip
RUN python -m ml.photo.asset --download
# runtime: no hub access, one thread per process (matches the cost model in docs/cost.md)
ENV HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MAQSAD_TORCH_THREADS=1

# Runs as root on purpose: Railway volumes (mounted at DATA_DIR) are root-owned.
RUN mkdir -p /data
EXPOSE 8000
CMD ["sh", "-c", "exec uvicorn backend.app.main:create_app --factory --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips '*'"]
