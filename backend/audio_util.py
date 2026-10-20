"""Audio conversion helpers shared by the align pipeline.

The align pipeline used to run ffmpeg on the same file twice (once to
extract 16 kHz mono audio for the VAD, once more inside the ASR provider
to re-encode to 16 kHz mono). On a 512 MB Render instance that is a wasted
full-file transcode and a wasted copy of the audio in RAM.

`ensure_16k_mono` is the single place that decides whether a conversion is
needed at all:

  * already a 16 kHz mono WAV  -> reuse the file, run NO ffmpeg
  * anything else              -> convert once, and the caller owns the
                                  result (so it can be cached/reused and
                                  deleted exactly once)
"""
from __future__ import annotations

import os
import subprocess
import tempfile

from backend.config import FFMPEG_PATH, TEMP_DIR

SAMPLE_RATE = 16000


def is_16k_mono_wav(path: str) -> bool:
    """True when `path` is a WAV file that is already 16 kHz mono PCM.

    Uses soundfile's header-only probe (no decoding of the payload), so it
    is cheap even for hour-long files.
    """
    if not path or not os.path.isfile(path):
        return False
    if not path.lower().endswith(".wav"):
        return False
    try:
        import soundfile as sf

        info = sf.info(path)
    except Exception:
        return False
    return (
        getattr(info, "samplerate", 0) == SAMPLE_RATE
        and getattr(info, "channels", 0) == 1
    )


def ensure_16k_mono(path: str, tmp_dir: str | None = None) -> tuple[str, bool]:
    """Return (wav_path, owned).

    `owned` is True when a NEW file was created and the caller must delete
    it; False when `path` was already suitable and is being reused as-is.
    """
    if is_16k_mono_wav(path):
        return path, False

    out_dir = tmp_dir or TEMP_DIR
    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError:
        out_dir = None
    fd, out = tempfile.mkstemp(suffix=".wav", dir=out_dir)
    os.close(fd)
    try:
        r = subprocess.run(
            [
                FFMPEG_PATH, "-y", "-i", path,
                "-ac", "1", "-ar", str(SAMPLE_RATE),
                "-acodec", "pcm_s16le", out,
            ],
            capture_output=True,
        )
    except Exception as exc:
        _silent_unlink(out)
        raise RuntimeError(f"ffmpeg conversion failed: {exc}") from exc
    if r.returncode != 0 or not os.path.exists(out) or os.path.getsize(out) == 0:
        msg = r.stderr.decode("utf-8", errors="replace")
        _silent_unlink(out)
        raise RuntimeError(_ffmpeg_error_msg("conversion", None, path,
                                             r.returncode, msg))
    return out, True


def _silent_unlink(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def extract_audio(video_path: str, sr: int = SAMPLE_RATE,
                  tmp_dir: str | None = None) -> tuple[str, float]:
    """Extract the audio track of a media file as mono PCM WAV.

    The result lives in TEMP_DIR (not the system temp dir) so the align
    pipeline can find, reuse and clean it up deterministically.
    """
    out_dir = tmp_dir or TEMP_DIR
    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError:
        out_dir = None
    fd, out_path = tempfile.mkstemp(suffix=".wav", dir=out_dir)
    os.close(fd)
    cmd = [
        FFMPEG_PATH, "-i", video_path, "-vn",
        "-acodec", "pcm_s16le", "-ar", str(sr), "-ac", "1", "-y", out_path,
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0 or not os.path.exists(out_path):
        msg = result.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(_ffmpeg_error_msg("extract_audio", cmd, video_path,
                                              result.returncode, msg))
    try:
        import soundfile as sf

        info = sf.info(out_path)
        duration = float(info.frames) / float(info.samplerate or sr)
    except Exception:
        duration = 0.0
    return out_path, duration


def _ffmpeg_error_msg(what: str, cmd, input_path: str, rc: int,
                       stderr: str) -> str:
    """A failure message that actually says what went wrong.

    The raw stderr of a failing ffmpeg is thousands of characters of banner
    with the useful line buried in it, and the tail often holds nothing but
    the build configuration. Report the input's size and magic bytes too —
    an empty or truncated upload is the usual real cause.
    """
    try:
        size = os.path.getsize(input_path)
    except OSError:
        size = -1
    try:
        with open(input_path, "rb") as f:
            head = f.read(8)
    except OSError:
        head = b""
    lines = [ln for ln in stderr.splitlines() if ln.strip()]
    # Drop the banner/config lines: they are noise that buries the error.
    interesting = [
        ln for ln in lines
        if not ln.startswith(("ffmpeg version", "built with", "configuration:",
                              "libav", "libsw", "libpost", "  "))
        and "enable-" not in ln
    ]
    tail = "\n".join(interesting[-8:]) if interesting else stderr[-400:]
    return (
        f"ffmpeg {what} failed rc={rc} for {os.path.basename(input_path)} "
        f"({size} bytes, magic={head!r}). ffmpeg: {tail.strip()[:800]}"
    )
