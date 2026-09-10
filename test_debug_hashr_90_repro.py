import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from bisect import bisect_right
from backend.detector import SurahDetector
from backend.forced_aligner import ForcedAligner
from data.quran_text import load_ayah_texts, _normalize

stage = json.load(open(r"temp\hashr_asr.json", encoding="utf-8"))
transcript = stage["text"]
chunk_texts = stage["chunk_texts"]
chunk_bounds = stage["chunk_bounds"]
print("chunk_bounds:", chunk_bounds)
print("words in transcript:", len(transcript.split()))

cum = []
total = 0
for t in chunk_texts:
    total += len(t.split())
    cum.append(total)
print("cum:", cum)

matches = SurahDetector().detect_all(transcript)
print("matches:", [(m.surah, m.start_ayah, m.end_ayah, round(m.score,3), m.word_off, m.word_span) for m in matches])

def _match_bounds(m):
    if not m.word_span or not cum:
        return None
    i0, i1 = m.word_span
    c0 = min(bisect_right(cum, i0), len(chunk_bounds) - 1)
    c1 = min(bisect_right(cum, i1), len(chunk_bounds) - 1)
    return chunk_bounds[c0][0], chunk_bounds[c1][1]

texts_ar = load_ayah_texts()
m = matches[0]
start_ayah, end_ayah = min(m.start_ayah, m.end_ayah), max(m.start_ayah, m.end_ayah)
bounds = _match_bounds(m)
print("bounds:", bounds)

canon_words = []
word_off = getattr(m, "word_off", None) or {}
for ayah in range(start_ayah, end_ayah + 1):
    words = _normalize(texts_ar.get(f"{m.surah}:{ayah}", "")).split()
    first, last = word_off.get(ayah, (0, len(words)))
    canon_words.extend((ayah, w) for w in words[first:last])
print("canon words (%d):" % len(canon_words))

fa = ForcedAligner()
aligned = fa.align_words(r"temp\hashr_90s.wav", canon_words)
fa.unload()
if not aligned:
    print("ALIGN FAILED")
else:
    print("ALIGNED WORDS (90s clip, absolute times):")
    for w in aligned:
        if w["start"] > 42.0:
            print("  %7.2f %7.2f | %s" % (w["start"], w["end"], w["text"]))
