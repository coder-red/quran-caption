import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from backend.detector import SurahDetector

d = json.load(open(r"temp\hashr_asr.json", encoding="utf-8"))
transcript = d["text"]
print("words:", len(transcript.split()))
det = SurahDetector()
det._build_index()
word_pos = det._word_pos
misses = []
for i, w in enumerate(transcript.split()[:350]):
    if w not in word_pos:
        misses.append(f"[{i}] '{w}'")
print(f"not-in-index ({len(misses)}):", " ".join(misses[:50]) if misses else "none")
matches = det.detect_all(transcript)
print(f"detect_all -> {len(matches)} matches:")
for m in matches:
    print(f"  surah {m.surah} ayahs {m.start_ayah}-{m.end_ayah} score {m.score:.3f} span {m.word_span}")
