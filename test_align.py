import json
import os
import sys

import requests

API = "http://localhost:8000"
path = sys.argv[1] if len(sys.argv) > 1 else r"temp\tmp4azunr17.mp4"

with open(path, "rb") as f:
    resp = requests.post(
        f"{API}/align",
        files={"file": ("test.mp4", f, "video/mp4")},
        timeout=900,
    )

print("status", resp.status_code)
data = resp.json()
if "error" in data:
    print("ERROR:", data["error"])
    sys.exit(1)

print("surah:", data["surah"]["name_en"], "| ayahs:", data["start_ayah"], "-", data["end_ayah"],
      "| conf:", data["confidence"], "| segments:", len(data["segments"]))
for s in data["segments"][:12]:
    en = s["text_en"][:55].replace("\n", " ")
    print(f"  {s['ayah']:>3} {s['start']:7.2f}-{s['end']:7.2f}  {en}")

out = os.path.join("temp", "align_out.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False)
print(f"saved {out}")
