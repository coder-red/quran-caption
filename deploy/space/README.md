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

# Quran Captions

Upload a Quran recitation audio/video clip → surah detection → word-level forced alignment → captions (SRT/VTT) → MP4 with burned-in captions.

## How it works
1. Upload a clip (MP3/WAV/MP4)
2. ASR (tarteel-ai/whisper-base-ar-quran) transcribes speech
3. Fuzzy matching detects which surahs/ayahs are being recited
4. wav2vec2 CTC forced alignment produces per-word timestamps
5. Edit captions, preview, then render the MP4 with FFmpeg

## Notes
- CPU-only; a 10-minute clip takes roughly an hour on this free tier. The page may look frozen while alignment runs — it is not.
- On first boot the app downloads ~2 GB of models into this instance's persistent disk; the first visit takes a few extra minutes.
- The FastAPI caption engine runs as a child process inside this Space.
