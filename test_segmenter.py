import sys, os, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from data.quran_text import load_ayah_texts
from backend.segmenter import build_segments, build_srt

texts = load_ayah_texts()

chunks = []
t = 0.0
for ayah in range(1, 8):
    words = texts[f"1:{ayah}"].split()
    mid = len(words) // 2
    for part in ([words[:mid], words[mid:]] if mid else [words]):
        if not part:
            continue
        dur = 2.5
        chunks.append({"text": " ".join(part), "start": round(t, 2), "end": round(t + dur, 2)})
        t += dur + 0.8

segs = build_segments(1, 1, 7, chunks)
print(f"{len(segs)} segments")
for s in segs:
    print(s["ayah"], f"{s['start']:.2f}-{s['end']:.2f}", s["text_en"][:60])
print("---SRT both---")
print(build_srt(segs, "both")[:400])
