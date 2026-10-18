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

# 9. Concurrent get_speech_segments must not corrupt the heap.
# The torch-Silero branch drives one shared, lazily-loaded module-level
# model. /align runs each job in its own daemon thread with no other
# coordination, so two uploads used to hit that model simultaneously and
# kill the process with STATUS_HEAP_CORRUPTION (0xC0000374) — no Python
# traceback, just a dead engine. vad.py now holds _VAD_LOCK across the
# lazy load *and* the inference. This runs the check in a SUBPROCESS on
# purpose: the failure mode is a hard process abort, which would kill the
# test runner itself instead of failing an assertion.
import subprocess

_CONCURRENT = r"""
import threading, sys
import numpy as np, soundfile as sf
from backend.vad import get_speech_segments

audio, sr = sf.read("data/test_ikhlas_16k.wav", dtype="float32")
if audio.ndim > 1:
    audio = audio.mean(axis=1)
N, R = 4, 4
barrier = threading.Barrier(N)
out = {}

def worker(i):
    barrier.wait()
    try:
        for _ in range(R):
            out[i] = get_speech_segments(audio.copy(), sr)
    except BaseException as e:          # noqa: BLE001
        out[i] = ("EXC", repr(e))

ts = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(N)]
[t.start() for t in ts]
[t.join(timeout=180) for t in ts]
alive = [t for t in ts if t.is_alive()]
if alive:
    print("DEADLOCK"); sys.exit(3)
vals = [v for v in out.values()]
if len(out) != N:
    print("INCOMPLETE", sorted(out)); sys.exit(4)
if any(isinstance(v, tuple) for v in vals):
    print("EXCEPTION", vals); sys.exit(5)
# concurrency must not change the answer
if any(v != vals[0] for v in vals):
    print("DIVERGED"); sys.exit(6)
print("OK", len(vals[0]))
"""
_p = subprocess.run([sys.executable, "-c", _CONCURRENT], capture_output=True,
                    text=True, timeout=600)
_rc = _p.returncode
assert _rc == 0, (
    f"concurrent VAD broke the process (rc={_rc}, "
    f"0xC0000374={-1073740940}): {(_p.stdout or '').strip()} {(_p.stderr or '').strip()[-400:]}"
)

print("test_vad.py ALL PASSED")
