import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import soundfile as sf
from bisect import bisect_right
from backend.aligner import ASREngine
from backend.vad import get_speech_segments, merge_segments, chunk_for_whisper
from backend.detector import SurahDetector
from backend.forced_aligner import ForcedAligner
from backend.segmenter import segments_from_word_times
from data.quran_text import load_ayah_texts, _normalize

audio_path = r"temp\hashr_90s.wav"
dur = 89.977
segs = merge_segments(get_speech_segments(*sf.read(audio_path)))
print("segs:", [(round(s,2), round(e,2)) for s,e in segs])
wchunks = chunk_for_whisper(segs)
eng = ASREngine()
result = eng.transcribe_chunks(audio_path, wchunks)
eng.unload()
transcript = result.get("text", "").strip()
print("transcript words:", len(transcript.split()))

det = SurahDetector()
matches = det.detect_all(transcript)
print("matches:", [(m.surah, m.start_ayah, m.end_ayah, m.score, m.word_off) for m in matches])
if not matches:
    print("NO MATCHES"); sys.exit(1)

chunk_texts = result.get("chunk_texts", [])
chunk_bounds = result.get("chunk_bounds", [])
cum = []
total = 0
for t in chunk_texts:
    total += len(t.split())
    cum.append(total)

def _match_bounds(m):
    if not m.word_span or not cum:
        return None
    i0, i1 = m.word_span
    c0 = min(bisect_right(cum, i0), len(chunk_bounds) - 1)
    c1 = min(bisect_right(cum, i1), len(chunk_bounds) - 1)
    return chunk_bounds[c0][0], chunk_bounds[c1][1]

texts_ar = load_ayah_texts()
fa = ForcedAligner()
all_words = []
all_segments = []
matches_sorted = sorted(matches, key=lambda m: (m.word_span or (0, 0))[0])
for mi, m in enumerate(matches_sorted):
    start_ayah = min(m.start_ayah, m.end_ayah)
    end_ayah = max(m.start_ayah, m.end_ayah)
    bounds = _match_bounds(m)
    print(f"match {mi}: span {m.word_span} bounds {bounds}")
    sliced_path = audio_path
    offset = 0.0
    if bounds:
        if mi == 0:
            s = chunk_bounds[0][0] if chunk_bounds else max(0.0, bounds[0] - 1.0)
        else:
            s = bounds[0]
        s = max(0.0, s - 1.0)
        e = min(dur, bounds[1] + 1.0)
        if e - s >= 2.0 and e - s < dur - 2.0:
            sliced_path = eng.slice_audio(audio_path, s, e)
            offset = s
    print(f"  slice {round(s,2)}-{round(e,2)} -> {os.path.basename(sliced_path)} offset {offset}")

    canon_words = []
    if mi == 0:
        for kind in fa.detect_leading_openings(sliced_path):
            tag = {"bismillah": 0, "takbir": -1, "istiadha": -2}[kind]
            canon_raw = {
                "bismillah": texts_ar.get("1:1", ""),
                "takbir": "\u0661\u0671\u0644\u0644\u0651\u064e\u0647\u064f \u0623\u064e\u0643\u0652\u0628\u064e\u0631\u064f",
                "istiadha": "\u0623\u064e\u0639\u064f\u0648\u0630\u064f \u0628\u0650\u0671\u0644\u0644\u0651\u064e\u0647\u0650 \u0645\u0650\u0646\u064e \u0671\u0644\u0634\u0651\u064e\u064a\u0652\u0637\u064e\u0670\u0646\u0650 \u0671\u0644\u0631\u0651\u064e\u062c\u0650\u064a\u0645\u0650",
            }[kind]
            for word in _normalize(canon_raw).split():
                canon_words.append((tag, word))
    word_off = getattr(m, "word_off", None) or {}
    for ayah in range(start_ayah, end_ayah + 1):
        words = _normalize(texts_ar.get(f"{m.surah}:{ayah}", "")).split()
        first, last = word_off.get(ayah, (0, len(words)))
        canon_words.extend((ayah, w) for w in words[first:last])
    print(f"  canon words: {len(canon_words)}")

    aligned = fa.align_words(sliced_path, canon_words)
    print(f"  aligned: {None if aligned is None else len(aligned)}")
    if aligned:
        for w in aligned:
            w["start"] = round(w["start"] + offset, 3)
            w["end"] = round(w["end"] + offset, 3)
        all_words.extend(w for w in aligned if int(w["ayah"]) > 0)
        all_segments.extend(segments_from_word_times(
            surah=m.surah, start_ayah=start_ayah, end_ayah=end_ayah,
            word_times=aligned, word_off=word_off,
        ))
fa.unload()

print("TOTAL segments:", len(all_segments))
for s in all_segments:
    print(round(s["start"],2), round(s["end"],2), s["surah"], s["ayah"], "|", s["text_ar"][:45])
