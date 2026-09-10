import sys, os, json
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from backend.detector import SurahDetector
from data.quran_text import _normalize, _normalize_std

with open("temp/transcript.txt", encoding="utf-8") as f:
    transcript = f.read()

words = transcript.split()
print(f"transcript words: {len(words)}")

det = SurahDetector()
det._build_index()
word_pos = det._word_pos

misses = []
for i, w in enumerate(words):
    if w not in word_pos and i < 350:
        misses.append(f"  [{i}] '{w}'")

print(f"words 0-350 not in index ({len(misses)}):")
for m in misses:
    print(m)

matches = det.detect_all(transcript)
print(f"\ndetect_all -> {len(matches)} matches:")
for m in matches:
    print(f"  surah {m.surah} ayahs {m.start_ayah}-{m.end_ayah} score {m.score:.3f} span {m.word_span}")

json.dump([{"surah": m.surah, "start": m.start_ayah, "end": m.end_ayah,
            "score": m.score, "span": list(m.word_span)} for m in matches],
          open("temp/detect_fix_check.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
