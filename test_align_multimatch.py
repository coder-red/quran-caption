import sys, os, warnings, json
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

# Synthetic multi-surah clip: ikhlas + 2s silence + kawthar, 10s lead-in
a1, sr = sf.read("data/test_ikhlas_16k.wav")
a2, _ = sf.read("data/test_kawthar_16k.wav")
clip = np.concatenate([np.zeros(int(10 * sr)), a1, np.zeros(int(2 * sr)), a2, np.zeros(int(5 * sr))])
sf.write("data/test_multisurah.wav", clip, sr)

with open("data/test_multisurah.wav", "rb") as f:
    r = client.post("/align", files={"file": ("test.wav", f, "audio/wav")}, timeout=900)

print("status:", r.status_code)
data = r.json()
assert "error" not in data, data.get("error")
assert len(data["matches"]) >= 2, data["matches"]
first, second = data["matches"][0], data["matches"][1]
assert first["surah"]["id"] == 112 and second["surah"]["id"] == 108, data["matches"]
assert data["surah"]["id"] == 112
assert data["srt"], "empty srt"
ayahs = [s["ayah"] for s in data["segments"]]
assert 1 in ayahs and 4 in ayahs, ayahs
ikhlas_segs = [s for s in data["segments"] if s["surah"] == 112]
kawthar_segs = [s for s in data["segments"] if s["surah"] == 108]
assert ikhlas_segs and kawthar_segs, data["segments"]
ikhlas_end = max(s["end"] for s in ikhlas_segs)
kawthar_start = min(s["start"] for s in kawthar_segs)
assert ikhlas_end < kawthar_start, f"timeline overlap: ikhlas ends {ikhlas_end}, kawthar starts {kawthar_start}"
assert ikhlas_end < 40.0, f"ikhlas segments leak past 40s: {ikhlas_segs}"
print("matches:", [(m["surah"]["name_en"], m["start_ayah"], m["end_ayah"]) for m in data["matches"]])
print("segments:", [(s["ayah"], s["start"], s["end"]) for s in data["segments"]])
print("test_align_multimatch.py ALL PASSED")
