import json
import os
import sys

import requests

API = "http://localhost:8000"
path = sys.argv[1] if len(sys.argv) > 1 else "data/test_kawthar_16k.wav"

with open(path, "rb") as f:
    resp = requests.post(
        f"{API}/align",
        files={"file": (os.path.basename(path), f, "audio/wav")},
        timeout=900,
    )

print("status", resp.status_code)
data = resp.json()
if "error" in data:
    print("ERROR:", data["error"])
    sys.exit(1)

print("surah:", data["surah"]["name_en"], "| ayahs:", data["start_ayah"], "-", data["end_ayah"],
      "| conf:", data["confidence"], "| segments:", len(data["segments"]))
for s in data["segments"]:
    en = s["text_en"][:60].replace("\n", " ")
    print(f"  {s['ayah']:>3} {s['start']:7.2f}-{s['end']:7.2f}  {en}")

out = os.path.join("temp", "align_out.json")
os.makedirs("temp", exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False)
print("saved", out)

# ---- render test: tiny srt from segments
segments = data["segments"]


def fmt(sec):
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


srt_lines = []
for i, seg in enumerate(segments, 1):
    srt_lines.append(str(i))
    srt_lines.append(f"{fmt(seg['start'])} --> {fmt(seg['end'])}")
    srt_lines.append(seg["text_ar"])
    srt_lines.append("")
srt = "\n".join(srt_lines)

# need a video — reuse the alafasy mp3 wrapped into mp4 via ffmpeg
mp4 = os.path.join("temp", "test_input.mp4")
os.makedirs("temp", exist_ok=True)
os.system(
    f'"C:\\ffmpeg\\ffmpeg-master-latest-win64-gpl\\bin\\ffmpeg.exe" -y -i {path} '
    f'-f lavfi -i color=c=black:s=640x360:d=25 -shortest -c:v libx264 -preset ultrafast '
    f'-pix_fmt yuv420p -c:a aac {mp4} 2>nul'
)

with open(mp4, "rb") as f:
    r = requests.post(
        f"{API}/render",
        files={"file": ("test.mp4", f, "video/mp4")},
        data={"srt": srt, "font_size": "40", "font_family": "Segoe UI",
              "font_color": "#FFFFFF", "outline_color": "#000000", "outline_size": "2",
              "shadow_size": "0", "bold": "false", "box": "false", "position": "bottom",
              "width": "640", "height": "360"},
        timeout=180,
    )
print("render start:", r.status_code, r.text[:120])
job_id = r.json().get("job_id")
import time
for _ in range(120):
    st = requests.get(f"{API}/render/status/{job_id}", timeout=30).json()
    if st["status"] in ("done", "error"):
        print("render:", st["status"], st.get("error"), st.get("encoder"))
        break
    time.sleep(2)
else:
    print("render still running after 240s")

# media endpoints
media_id = "e2e_test"
with open(mp4, "rb") as f:
    mr = requests.post(f"{API}/media/{media_id}", files={"file": ("test.mp4", f, "video/mp4")}, timeout=300)
print("media upload:", mr.status_code, mr.text)
head = requests.head(f"{API}/media/{media_id}", timeout=30)
print("media head:", head.status_code, head.headers.get("content-length"))
rng = requests.get(f"{API}/media/{media_id}", headers={"Range": "bytes=0-1023"}, timeout=30)
print("media range:", rng.status_code, len(rng.content), rng.headers.get("content-range"))
