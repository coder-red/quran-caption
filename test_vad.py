import sys, os, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import soundfile as sf
from backend.vad import get_speech_segments, merge_segments, chunk_for_whisper, speech_span

sr = 16000

# 1. Silence -> no segments
sil = np.zeros(3 * sr, dtype=np.float32)
assert get_speech_segments(sil) == [], "silence must produce no segments"

# 2. Real recitation -> at least one segment
audio, _ = sf.read("data/test_ikhlas_16k.wav")
segs = get_speech_segments(audio)
assert len(segs) >= 1, f"expected speech, got {segs}"

# 3. Speech with 1s gaps merges when max_gap=2.0
segs = [(0.0, 1.0), (2.5, 3.5)]
merged = merge_segments(segs, max_gap=2.0)
assert merged == [(0.0, 3.5)], f"expected merge, got {merged}"

# 4. merge_segments keeps distant segments separate
segs = [(0.0, 1.0), (5.0, 6.0)]
assert merge_segments(segs, max_gap=2.0) == [(0.0, 1.0), (5.0, 6.0)]

# 5. chunk_for_whisper never exceeds max_len; long segments get split
chunks = chunk_for_whisper([(0.0, 20.0), (22.0, 40.0), (45.0, 60.0)])
for s, e in chunks:
    assert e - s <= 30.0 + 1e-6, f"chunk too long: {s}-{e}"
assert chunks[0][0] == 0.0 and chunks[-1][1] == 60.0
assert len(chunks) == 3, f"expected 3 chunks, got {chunks}"

# 5b. A single continuous 75s speech segment gets split into 30s pieces
chunks = chunk_for_whisper([(0.0, 75.0)])
assert chunks == [(0.0, 30.0), (30.0, 60.0), (60.0, 75.0)], chunks

# 5c. Close segments merge while total stays <= max_len
chunks = chunk_for_whisper([(0.0, 10.0), (12.0, 20.0), (22.0, 29.0)])
assert chunks == [(0.0, 29.0)], chunks

# 6. speech_span pads and returns first-to-last
sp = speech_span([(10.0, 20.0)])
assert sp == (8.5, 21.5), f"expected padded span, got {sp}"

# 7. speech_span on empty -> None
assert speech_span([]) is None

# 8. energy fallback still works
from backend.vad import _energy_segments
fallback = _energy_segments(audio, sr)
assert fallback and fallback[0][0] < 8.0, f"energy fallback failed: {fallback}"

print("test_vad.py ALL PASSED")
