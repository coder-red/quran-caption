import sys, os, json, requests
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

CLIP = r"data\Voice from Heart Beautiful Quran Recitation by Sheikh Ahmed Mokhtar _ AWAZ.mp4"

with open(CLIP, "rb") as f:
    r = requests.post(
        "http://127.0.0.1:8000/align",
        files={"file": ("clip.mp4", f, "video/mp4")},
        timeout=7200,
    )

print("status", r.status_code)
data = r.json()
if "error" in data:
    print("ERROR:", data["error"])
    sys.exit(1)

print("surah:", data["surah"]["name_en"], "| ayahs:", data["start_ayah"], "-", data["end_ayah"],
      "| conf:", data["confidence"], "| segments:", len(data["segments"]))
print("matches:", [(m["surah"]["name_en"], m["start_ayah"], m["end_ayah"], m["confidence"]) for m in data.get("matches", [])])
for s in data["segments"][:25]:
    print(f"  {s['surah']}:{s['ayah']} {s['start']:7.2f}-{s['end']:7.2f} {s['text_en'][:55]}")

out = os.path.join("temp", "align_realclip.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False)
print("saved", out)
