"""Audio conversion reuse: no redundant ffmpeg transcode, no wrong deletes.

The align pipeline used to run ffmpeg twice on the same file (once to
extract 16 kHz mono audio, once more inside the ASR provider to re-encode
to 16 kHz mono). On a 512 MB Render instance that is a wasted full-file
transcode plus a wasted copy of the audio in RAM.

These tests lock in:
  1. an already-16 kHz-mono WAV is reused as-is (no ffmpeg, not owned)
  2. anything else is converted once and IS owned by the caller
  3. a provider that was handed a ready-made 16 kHz WAV does not delete it
  4. the pipeline deletes the temp WAV it extracted, but never the stored
     /media copy (which the render step still needs)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import soundfile as sf

from backend.audio_util import ensure_16k_mono, is_16k_mono_wav, extract_audio

TMP = os.path.join("temp", "_audio_reuse")
os.makedirs(TMP, exist_ok=True)


def _write(path, sr, channels, seconds=1.0):
    n = int(sr * seconds)
    data = np.zeros((n, channels) if channels > 1 else n, dtype="float32")
    sf.write(path, data, sr)
    return path


def test_16k_mono_wav_is_reused_without_conversion():
    p = _write(os.path.join(TMP, "ready.wav"), 16000, 1)
    assert is_16k_mono_wav(p), "16 kHz mono WAV should be recognised"
    out, owned = ensure_16k_mono(p)
    assert out == p, "must reuse the file, not create a new one"
    assert owned is False, "a reused file is NOT owned by the caller"


def test_non_16k_mono_is_converted_once_and_owned():
    p = _write(os.path.join(TMP, "stereo44.wav"), 44100, 2)
    assert not is_16k_mono_wav(p)
    out, owned = ensure_16k_mono(p)
    try:
        assert out != p, "a conversion must produce a new file"
        assert owned is True, "the caller owns (and must delete) the conversion"
        assert is_16k_mono_wav(out), "the result must be 16 kHz mono"
    finally:
        if os.path.exists(out):
            os.remove(out)


def test_provider_does_not_delete_a_borrowed_wav():
    """A provider handed a ready-made 16 kHz WAV must leave it in place —
    it is the pipeline's file, not the provider's."""
    from backend.providers.groq import GroqProvider

    p = _write(os.path.join(TMP, "borrowed.wav"), 16000, 1, seconds=1.0)
    prov = GroqProvider()
    out, owned = prov._ensure_16k_mono(p)
    assert out == p and owned is False
    assert os.path.exists(p), "the borrowed file must still exist"


def test_extract_audio_lands_in_temp_dir_and_is_16k_mono():
    from backend.config import TEMP_DIR

    src = _write(os.path.join(TMP, "src.wav"), 44100, 2, seconds=0.5)
    out, dur = extract_audio(src)
    try:
        assert is_16k_mono_wav(out), "extracted audio must be 16 kHz mono"
        managed = os.path.abspath(TEMP_DIR)
        assert os.path.abspath(out).startswith(managed), (
            f"extracted audio must live in the managed temp dir "
            f"({managed}), got {out}")
        assert dur > 0.4
    finally:
        if os.path.exists(out):
            os.remove(out)


def test_pipeline_deletes_its_own_temp_wav_but_not_the_media_copy():
    """End-to-end on the real align pipeline: the extracted WAV is cleaned
    up, the stored /media file survives for the render step."""
    from backend.config import TEMP_DIR
    from backend.main import _align_impl

    # A real recitation (a 200 Hz sine is not speech — the VAD rejects it).
    media = os.path.join(TMP, "media_keep.wav")
    real = "temp/probe30s.wav"
    if not os.path.exists(real):
        return  # no sample available
    import shutil

    shutil.copyfile(real, media)
    before = set(os.listdir(TEMP_DIR))
    try:
        resp = _align_impl(None, asr_backend="groq", media_path=media)
        data = resp.model_dump() if hasattr(resp, "model_dump") else resp
        assert not (isinstance(data, dict) and data.get("error")), data
        assert os.path.exists(media), "the stored /media copy must survive"
        after = set(os.listdir(TEMP_DIR))
        leaked = {f for f in after - before
                  if f.startswith("tmp") and f.endswith(".wav")}
        assert not leaked, f"pipeline leaked temp WAVs: {leaked}"
    finally:
        for f in os.listdir(TEMP_DIR):
            if f.startswith("tmp") and f.endswith(".wav"):
                try:
                    os.remove(os.path.join(TEMP_DIR, f))
                except OSError:
                    pass


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
