import sys, os, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

print("Routes:", [r.path for r in app.routes if hasattr(r, "path")])

r = client.get("/health")
print(f"/health -> {r.status_code}: {r.json()}")

tone_path = "data/test_tone.wav"
if not os.path.exists(tone_path):
    import numpy as np
    import soundfile as sf
    os.makedirs("data", exist_ok=True)
    sr = 16000
    t = np.linspace(0, 1, sr)
    tone = 0.3 * np.sin(2 * np.pi * 440 * t)
    silence = np.zeros(int(0.5 * sr))
    sf.write(tone_path, np.concatenate([tone, silence, tone]), sr)

with open(tone_path, "rb") as f:
    r = client.post("/align", files={"file": ("test.wav", f, "audio/wav")})
print(f"/align -> {r.status_code}")
try:
    data = r.json()
    if "error" in data:
        print(f"  Expected: no speech in test tone: {data['error'][:100]}")
    else:
        print(f"  Unexpected success: {data}")
except Exception as e:
    print(f"  parse error: {e}")

print("\nDone.")
