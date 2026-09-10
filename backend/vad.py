"""Speech activity detection for the caption pipeline.

Primary: Silero VAD (neural — robust to background music).
Fallback: RMS energy segmentation (clean-audio only).
Chunking follows the WhisperX cut-and-merge pattern: pack speech into
~30s chunks with boundaries on silence so Whisper never splits mid-ayah.
"""

import numpy as np

_MAX_CHUNK = 30.0
_TARGET_CHUNK = 30.0

# Cache the Silero model module-level: load_silero_vad() costs seconds
# (torch hub load); re-loading per get_speech_segments() call made a 60s
# clip pay ~14s of pure reload overhead. Single shared instance is enough
# (CPU-only, called sequentially in the align pipeline).
_VAD_MODEL = None


def _silero_available() -> bool:
    try:
        import silero_vad  # noqa: F401
        return True
    except Exception:
        return False


def get_speech_segments(
    audio: np.ndarray, sr: int = 16000, threshold: float = 0.5
) -> list[tuple[float, float]]:
    """Speech regions (start, end) in seconds. Silero VAD, energy fallback."""
    global _VAD_MODEL
    if _silero_available():
        try:
            import torch
            from silero_vad import get_speech_timestamps, load_silero_vad

            if _VAD_MODEL is None:
                _VAD_MODEL = load_silero_vad()
            ts = get_speech_timestamps(
                torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)),
                _VAD_MODEL,
                sampling_rate=sr,
                threshold=threshold,
            )
            return [(t["start"] / sr, t["end"] / sr) for t in ts]
        except Exception:
            pass
    return _energy_segments(audio, sr)


def _energy_segments(
    audio: np.ndarray, sr: int, hop_s: float = 0.05, min_sec: float = 4.0
) -> list[tuple[float, float]]:
    frame = max(1, int(hop_s * sr))
    n = len(audio) // frame
    if n == 0:
        return []
    rms = np.sqrt(np.mean(audio[: n * frame].reshape(-1, frame) ** 2, axis=1))
    thr = max(0.001, 0.05 * float(rms.max()))
    voiced = np.where(rms > thr)[0]
    if len(voiced) == 0:
        return []
    merged = []
    s0 = prev = voiced[0]
    for i in voiced[1:]:
        if i - prev > int(1.5 / hop_s):
            merged.append((s0 * hop_s, (prev + 1) * hop_s))
            s0 = i
        prev = i
    merged.append((s0 * hop_s, (prev + 1) * hop_s))
    return [(s, e) for s, e in merged if e - s >= min_sec]


def merge_segments(
    segments: list[tuple[float, float]], max_gap: float = 2.0
) -> list[tuple[float, float]]:
    """Merge segments separated by <= max_gap seconds (reciter pauses)."""
    if not segments:
        return []
    out = []
    cs, ce = segments[0]
    for s, e in segments[1:]:
        if s - ce <= max_gap:
            ce = max(ce, e)
        else:
            out.append((cs, ce))
            cs, ce = s, e
    out.append((cs, ce))
    return out


def chunk_for_whisper(
    segments: list[tuple[float, float]],
    target: float = _TARGET_CHUNK,
    max_len: float = _MAX_CHUNK,
) -> list[tuple[float, float]]:
    """Cut & merge speech into Whisper-sized chunks (<= max_len, ideally
    target). Long continuous segments are split at max_len boundaries."""
    if not segments:
        return []
    pieces = []
    for s, e in segments:
        while e - s > max_len:
            pieces.append((s, s + max_len))
            s += max_len
        if e > s:
            pieces.append((s, e))
    chunks = []
    cs, ce = pieces[0]
    for s, e in pieces[1:]:
        if e - cs <= max_len:
            ce = e
        else:
            chunks.append((cs, ce))
            cs, ce = s, e
    chunks.append((cs, ce))
    return chunks


def speech_span(
    segments: list[tuple[float, float]], pad: float = 1.5
) -> tuple[float, float] | None:
    """First-to-last segment with padding, for forced-alignment slicing."""
    if not segments:
        return None
    return (max(0.0, segments[0][0] - pad), segments[-1][1] + pad)


def silence_runs_from_segments(
    segments: list[tuple[float, float]], min_silence: float = 0.25
) -> list[tuple[float, float]]:
    """True inter-speech silence regions (start, end) in seconds, derived from
    UNMERGED VAD speech segments (silero keeps gaps >= 250ms, so consecutive
    segments already bracket every real pause).

    These runs are the ground truth for pause-based caption splitting:
    Whisper word timestamps drift 200-500ms and misplace silence onto words,
    so pause decisions must come from the audio, not from word-gap math.

    min_silence: only silences >= this duration are returned (shorter dips
    between tightly-joined words are articulation, not breaths)."""
    out = []
    for (_, e0), (s1, _) in zip(segments, segments[1:]):
        if s1 - e0 >= min_silence - 1e-6:
            out.append((e0, s1))
    return out


def _energy_speech_edges(
    audio: np.ndarray, sr: int, hop_s: float = 0.03
) -> list[tuple[float, float]]:
    """Fine-grained voiced regions (30ms hop) for the no-silero fallback."""
    frame = max(1, int(hop_s * sr))
    n = len(audio) // frame
    if n == 0:
        return []
    rms = np.sqrt(np.mean(audio[: n * frame].reshape(-1, frame) ** 2, axis=1))
    thr = max(0.001, 0.04 * float(rms.max()))
    voiced = rms > thr
    edges = []
    i = 0
    while i < n:
        if voiced[i]:
            s = i
            while i < n and voiced[i]:
                i += 1
            edges.append((s * hop_s, i * hop_s))
        else:
            i += 1
    return edges
