@echo off
title Quran Caption - Launcher
cd /d "%~dp0"

echo Starting backend (ASR + captions) on http://localhost:8000 ...
start "Quran Caption - Backend" cmd /k "cd /d "%~dp0" && python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000"

echo Starting frontend on http://localhost:8501 ...
start "Quran Caption - Frontend" cmd /k "cd /d "%~dp0" && python -m streamlit run frontend/app.py --server.port 8501 --server.fileWatcherType none"

echo.
echo Both servers started. Open http://localhost:8501 in your browser.
echo Keep the two black windows open while you use the app.
echo To stop the app, close those two windows.
