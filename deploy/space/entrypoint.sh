#!/bin/bash
set -e

echo "[boot] pre-downloading models..."
python - <<'PY'
from huggingface_hub import snapshot_download
for repo in ("tarteel-ai/whisper-base-ar-quran",
             "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"):
    print(f"[boot] downloading {repo} ...")
    snapshot_download(repo)
print("[boot] models ready")
PY

echo "[boot] warming Silero VAD..."
python - <<'PY'
import torch
from silero_vad import load_silero_vad
load_silero_vad()
print("[boot] silero ready")
PY

echo "[boot] starting nginx..."
nginx

echo "[boot] starting FastAPI on :8000..."
uvicorn backend.main:app --host 127.0.0.1 --port 8000 &

echo "[boot] starting Streamlit on :8501..."
streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true &

wait
