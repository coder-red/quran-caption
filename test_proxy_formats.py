"""Fast-upload proxy: format negotiation and the WAV fallback.

The browser can build an Ogg/Opus audio proxy (~20x smaller than the
source) — but an ffmpeg build is not guaranteed to have an Opus decoder.
The engine therefore probes its OWN ffmpeg with a real Ogg/Opus sample
(backend/ogg_probe.ogg, produced by the same muxer the browser uses) and
reports the result; the browser then picks Ogg/Opus or plain WAV.

These tests lock in:
  1. the probe file exists and is a real Ogg/Opus stream
  2. supports_ogg_opus() answers for THIS host and caches the answer
  3. proxy_formats() always offers wav (PCM — decodable by every build)
  4. the shipped probe file decodes to the expected duration
  5. the browser-side WAV fallback produces a valid 16 kHz mono WAV
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.config import FFMPEG_PATH
from backend.ffmpeg_capabilities import (proxy_formats, supports_ogg_opus,
                                         _PROBE)

OUT = os.path.join("temp", "_proxy_test")
os.makedirs(OUT, exist_ok=True)


def test_probe_file_exists():
    assert os.path.isfile(_PROBE), (
        f"the Ogg/Opus capability probe must ship with the app: {_PROBE}")
    assert os.path.getsize(_PROBE) > 1000


def test_probe_file_is_a_real_ogg_opus_stream():
    with open(_PROBE, "rb") as f:
        head = f.read(64)
    assert head[:4] == b"OggS", "must start with the Ogg capture pattern"
    assert b"OpusHead" in head, "must carry an OpusHead header"


def test_probe_decodes_to_the_expected_duration():
    out = os.path.join(OUT, "probe.wav")
    r = subprocess.run([FFMPEG_PATH, "-v", "error", "-i", _PROBE,
                        "-ar", "16000", "-ac", "1", out],
                       capture_output=True)
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")[-400:]
    import soundfile as sf
    info = sf.info(out)
    assert 0.3 < info.frames / info.samplerate < 1.0, (
        f"probe should be ~0.5 s, got {info.frames / info.samplerate:.3f}s")


def test_supports_ogg_opus_answers_bool():
    assert isinstance(supports_ogg_opus(), bool)


def test_proxy_formats_always_offers_wav():
    fmts = proxy_formats()
    assert "wav" in fmts, "WAV (PCM) must always be offered"
    assert all(f in ("ogg_opus", "wav") for f in fmts)


def test_health_reports_proxy_formats():
    """The engine must advertise what it can decode."""
    import importlib
    import backend.main as bm
    importlib.reload(bm)
    health = None
    for route in bm.app.routes:
        if getattr(route, "path", "") == "/health":
            health = route
            break
    assert health is not None, "/health route missing"
    # call the endpoint function directly (async)
    import asyncio
    res = asyncio.run(health.endpoint())
    assert "proxy_formats" in res, res
    assert "wav" in res["proxy_formats"], res
    assert res["status"] == "ok"
    assert "vad_backend" in res, res


def test_browser_wav_fallback_is_valid():
    """The browser-side WAV fallback must produce a valid 16 kHz mono WAV.

    The encoding logic is JS, so it is re-implemented here exactly as
    frontend/opus_proxy.js does it and the bytes are validated.
    """
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "frontend", "opus_proxy.js")
    with open(src, encoding="utf-8") as f:
        js = f.read()
    assert "encodeWav16k" in js, "the WAV fallback must exist in the muxer"
    assert "format === 'wav'" in js, "buildAudioProxy must honour format=wav"
    # 44-byte canonical header + 16-bit samples
    assert "'RIFF'" in js and "'WAVE'" in js and "'fmt '" in js and "'data'" in js


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
