---
title: Quran Captions
emoji: 🕌
colorFrom: green
colorTo: green
sdk: streamlit
sdk_version: 1.58.0
app_file: frontend/app.py
pinned: false
---

# Quran Caption App

Web app that adds Quranic captions to recitation videos — works with **any** reciter.

## How it works

```
Upload video/audio → ASR (tarteel-ai/whisper-base-ar-quran) → surah detection (fuzzy match vs 6,236 ayahs)
→ word timestamps → SRT/VTT → preview → optional MP4 burn-in (FFmpeg)
```

Canonical Quran text is the source of truth. ASR is only used for timing, never for caption text.

## Notes (hosted on Spaces)

- CPU-only; a 10-minute clip takes roughly an hour on the free tier. The page may look frozen while alignment runs — it is not.
- On first boot the app downloads ~2 GB of models into this instance's persistent disk; the first visit takes a few extra minutes.
- The FastAPI caption engine runs as a child process inside this Space.

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the Quran text (once)

The file `data/quran_uthmani.json` (6,236 ayahs, Uthmani script) is loaded from
[risan/quran-json](https://github.com/risan/quran-json). If missing, run:

```bash
python -c "import json,requests,os; r=requests.get('https://cdn.jsdelivr.net/gh/risan/quran-json@master/data/quran.json'); d=r.json(); out={f'{s}:{v[\"verse\"]}':v['text'] for s,vs in d.items() for v in vs}; json.dump(out,open('data/quran_uthmani.json','w',encoding='utf-8'),ensure_ascii=False,indent=2); print(len(out),'ayahs saved')"
```

### 3. Run the backend (FastAPI)

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Model weights download on first use (~150MB). Set `DEVICE=cuda` in `backend/config.py` to use a GPU.

### 4. Run the frontend (Streamlit)

```bash
streamlit run frontend/app.py
```

Open http://localhost:8501

## API

### POST /align
Upload audio/video → returns surah, ayah range, confidence, word timestamps, SRT, VTT.

### POST /render
Upload video + SRT → returns captioned MP4 (FFmpeg burn-in).

### GET /health
Health check.

## Config (`backend/config.py`)

| Var | Default | Description |
|-----|---------|-------------|
| `ASR_MODEL` | `tarteel-ai/whisper-base-ar-quran` | Whisper model for Quranic ASR |
| `DEVICE` | `cpu` | `cpu` or `cuda` |
| `CONFIDENCE_THRESHOLD` | `0.85` | Minimum detection confidence |

## Project structure

```
backend/
  main.py       FastAPI app (/align, /render, /health)
  aligner.py    ASR + word timestamps
  detector.py   Surah/ayah detection (fuzzy match)
  renderer.py   FFmpeg subtitle burn-in
  models.py     Pydantic schemas
  config.py     Settings
data/
  quran_uthmani.json   6236 ayahs (canonical text)
  quran_text.py        Quran data loading + normalization
frontend/
  app.py        Streamlit UI
```

## Roadmap

- [x] ASR with tarteel-ai model + word timestamps
- [x] Ayah detection via fuzzy matching + confidence scoring
- [x] SRT/VTT generation
- [x] MP4 burn-in via /render
- [x] Streamlit UI (upload, preview, download)
- [ ] WhisperX/ctc-forced-aligner for finer word alignment
- [ ] Multi-ayah window matching
- [ ] Manual timestamp correction
- [ ] Test with many reciters
