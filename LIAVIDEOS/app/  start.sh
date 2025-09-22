#!/usr/bin/env bash
set -euo pipefail

echo "[start] Checking GPU availability..."
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[start] nvidia-smi found:"
  nvidia-smi || true
  export NVIDIA_VISIBLE_DEVICES=all
else
  echo "[start] No GPU detected, running CPU mode."
fi

# small health check for ffmpeg
ffmpeg -version | head -n 1

# Launch API
echo "[start] Starting FastAPI on :8080"
exec uvicorn api:app --host 0.0.0.0 --port 8080
