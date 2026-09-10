import sys, json
sys.stdout.reconfigure(encoding="utf-8")
d = json.load(open(r"temp\align_realclip.json", encoding="utf-8"))
print("segments count:", len(d["segments"]))
for s in d["segments"]:
    print(f"  {s['surah']}:{s['ayah']} {s['start']:.2f}-{s['end']:.2f}")
print()
print("words count:", len(d.get("words", [])))
for w in d.get("words", [])[:25]:
    print("  ", w["text"], round(w["start"], 2), round(w["end"], 2))
