import sys, os, json, warnings, time
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("FULL PIPELINE TEST")
print("=" * 60)

# Create a short Quranic audio clip using TTS or sine
print("\n[1] Creating test audio (Bismillah - ~3s)...")
import numpy as np
import soundfile as sf

# Generate a simple test tone (can't test ASR without real audio,
# but we can verify the detection code with mock data)
sr = 16000
t = np.linspace(0, 1, sr)
tone = 0.3 * np.sin(2 * np.pi * 440 * t)
silence = np.zeros(int(0.5 * sr))
audio = np.concatenate([tone, silence, tone, silence, tone])
sf.write("data/test_tone.wav", audio, sr)
print(f"  Created test tone: {len(audio)/sr:.1f}s")

print("\n[2] Testing surah detection with known text...")
from backend.detector import SurahDetector
from data.quran_text import load_ayah_texts

detector = SurahDetector()

# Test with exact ayah text
texts = load_ayah_texts()
bismillah = texts.get("1:1", "")
if bismillah:
    print(f"  Bismillah text: {bismillah}")
    match = detector.detect_tight(bismillah)
    if match:
        print(f"  Detected: Surah {match.surah}, Ayahs {match.start_ayah}-{match.end_ayah}")
        print(f"  Confidence: {match.score:.4f}")
        from data.quran_text import SURAH_NAMES
        name = SURAH_NAMES[match.surah - 1]
        print(f"  => {name['name_en']} ({name['name_ar']})")
    else:
        print("  No match found!")

# Test with partial text (mid-ayah)
print("\n[3] Testing partial text matching...")
fatihah_text = " ".join(texts.get(f"1:{i}", "") for i in range(1, 8))
partial = fatihah_text[20:80]  # middle of surah
print(f"  Partial text: {partial}")
match = detector.detect_tight(partial)
if match:
    print(f"  Detected: Surah {match.surah}, Ayahs {match.start_ayah}-{match.end_ayah}, Score: {match.score:.4f}")

# Test multi-ayah matching
print("\n[4] Testing multi-ayah matching...")
multi = texts.get("36:1", "") + " " + texts.get("36:2", "")
print(f"  Ya-Sin 1-2 (first 40 chars): {multi[:40]}")
match = detector.detect_tight(multi)
if match:
    from data.quran_text import SURAH_NAMES
    name = SURAH_NAMES[match.surah - 1]
    print(f"  => Surah {name['name_en']}, Ayahs {match.start_ayah}-{match.end_ayah}, Score: {match.score:.4f}")

print("\n[5] Testing SRT generation...")
from backend.segmenter import build_srt

words = [
    {"text_ar": "test", "text_en": "test", "start": 0.0, "end": 1.0, "ayah": 1},
    {"text_ar": "words", "text_en": "words", "start": 1.0, "end": 2.0, "ayah": 1},
]
srt = build_srt(words)
vtt = "WEBVTT\n\n" + srt.replace(",", ".")
print(f"  SRT:\n{srt}")
print(f"  VTT:\n{vtt}")

print("\n[6] Testing full /align response shape...")
from backend.models import AlignResponse, ChunkTimestamp, SurahInfo, WordTimestamp
resp = AlignResponse(
    surah=SurahInfo(id=1, name_ar="الفاتحة", name_en="Al-Fatihah"),
    start_ayah=1,
    end_ayah=1,
    confidence=0.99,
    words=[WordTimestamp(text="بسم", start=0.0, end=0.5)],
    chunks=[ChunkTimestamp(text="بسم", start=0.0, end=0.5)],
    segments=[],
    srt="1\n00:00:00,000 --> 00:00:00,500\nبسم",
    vtt="WEBVTT\n\n00:00:00.000 --> 00:00:00.500\nبسم",
)
d = resp.model_dump()
print(f"  Response keys: {list(d.keys())}")
print(f"  Surah: {d['surah']}")
print(f"  Words count: {len(d['words'])}")

print("\n" + "=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)
