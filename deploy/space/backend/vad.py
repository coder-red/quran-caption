"""Speech activity detection for the caption pipeline.

Primary: Silero VAD (neural — robust to background music).
Fallback: RMS energy segmentation (clean-audio only).
Chunking follows the WhisperX cut-and-merge pattern: pack speech into
~30s chunks with boundaries on silence so Whisper never splits mid-ayah.
"""

import os
import threading

import numpy as np

_MAX_CHUNK = 30.0
_TARGET_CHUNK = 30.0
# Cache the Silero model module-level: load_silero_vad() costs seconds
# (torch hub load); re-loading per get_speech_segments() call made a 60s
# clip pay ~14s of pure reload overhead.
#
# The cache is shared, so BOTH the lazy load and the inference that follows
# must be serialised. The torch model was previously driven straight from
# whatever thread called in: /align runs each job in its own daemon thread
# (backend/main.py::_align_job_worker has no lock), so two concurrent
# uploads hit the same _VAD_MODEL at once and killed the whole process with
# STATUS_HEAP_CORRUPTION (0xC0000374, faulting module ntdll.dll, empty
# stderr) — reproduced deterministically by calling this function from two
# threads; single-threaded and the ONNX branch both survive. Hold the lock
# across load+inference, not just around the None check.
_VAD_MODEL = None
_VAD_LOCK = threading.Lock()

# Vendored Silero VAD weights (MIT, snakers4/silero-vad — same 2MB model the
# pip package ships as silero_vad.onnx, opset 16). Lets the neural VAD run
# through ONNX Runtime with numpy only: no torch, no downloads, works on
# weak PCs and the slim public deploy.
_VENDORED_ONNX = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "silero_vad.onnx")
_ONNX_SESSION = None


def _silero_available() -> bool:
    try:
        import silero_vad  # noqa: F401
        return True
    except Exception:
        return False


def vad_backend() -> str:
    """Which speech detector is ACTIVE right now.

    'torch' = the silero_vad pip package, 'onnx' = the vendored Silero
    weights through onnxruntime (bit-identical edges to torch, verified),
    'energy' = the dependency-free RMS fallback.

    This matters because the caption split points come from the detected
    pauses: the energy fallback finds different (coarser) pauses than
    Silero, so a clip analysed with 'energy' gets its captions cut at
    different places than the same clip analysed with 'onnx'. That is the
    "sync varies per clip" report — so the backend must never silently
    differ between hosts, and it must be reported.
    """
    if _silero_available():
        return "torch"
    try:
        if _onnx_session() is not None:
            return "onnx"
    except Exception:
        pass
    return "energy"


def get_speech_segments(
    audio: np.ndarray, sr: int = 16000, threshold: float = 0.5
) -> list[tuple[float, float]]:
    """Speech regions (start, end) in seconds. Backend order: torch Silero
    (when torch is installed) -> ONNX Silero (vendored 2MB weights, numpy
    only — same model, same edges) -> energy fallback (no deps)."""
    global _VAD_MODEL
    if _silero_available():
        try:
            import torch
            from silero_vad import get_speech_timestamps, load_silero_vad

            with _VAD_LOCK:
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
    onnx_segs = _onnx_segments(audio, sr, threshold)
    if onnx_segs is not None:
        return onnx_segs
    return _energy_segments(audio, sr)


def speech_probs(
    audio: np.ndarray, sr: int = 16000, hop_s: float = 0.05
) -> tuple[np.ndarray, str]:
    """Per-hop speech-probability curve for the sync inspector.

    Returns (curve, backend) where curve[i] is 0..1 over the hop starting at
    i*hop_s, and backend is 'onnx' (true Silero probability — the same signal
    the pause decisions were made from) or 'energy' (normalized RMS, an
    approximation reported as such).

    DIAGNOSTIC ONLY: no caption timing is derived from this. Deliberately
    avoids the torch model (the shared torch instance must only ever be
    driven under _VAD_LOCK by get_speech_segments — concurrent use crashed
    the process); ONNX is bit-identical on edges anyway."""
    x = np.ascontiguousarray(audio, dtype=np.float32).ravel()
    if x.size == 0:
        return np.zeros(0, dtype=np.float32), "energy"
    probs = _onnx_probs(x, sr)
    if probs is not None:
        return _bucket_max(probs, sr, hop_s), "onnx"
    rms = rms_frames(x, sr, hop_s)
    if rms.size == 0:
        return rms, "energy"
    ref = float(np.percentile(rms, 95))
    if ref <= 0:
        ref = float(rms.max()) or 1.0
    return np.clip(rms / max(ref, 1e-9), 0.0, 1.0).astype(np.float32), "energy"


def _bucket_max(probs: list[float], sr: int, hop_s: float) -> np.ndarray:
    """Collapse per-512-sample probs onto hop buckets using the max, so a
    short speech burst cannot vanish between two display columns."""
    hop = max(1, int(round(hop_s * sr)))
    n = int(np.ceil(len(probs) * _WIN / hop))
    out = np.zeros(n, dtype=np.float32)
    for j, p in enumerate(probs):
        b = (j * _WIN) // hop
        if b < n and p > out[b]:
            out[b] = p
    # 32ms model windows vs a 50ms display hop leave holes; carry the last
    # value forward so the curve is continuous (at most one hop of smear).
    for i in range(1, n):
        if out[i] == 0.0 and out[i - 1] > 0.0:
            out[i] = out[i - 1]
    return out


def rms_frames(
    audio: np.ndarray, sr: int = 16000, hop_s: float = 0.03
) -> np.ndarray:
    """RMS amplitude per hop (cumsum, one pass — cheap on hour-long clips)."""
    x = np.asarray(audio, dtype=np.float64).ravel()
    hop = max(1, int(round(hop_s * sr)))
    if x.size == 0:
        return np.zeros(0, dtype=np.float32)
    n = int(np.ceil(x.size / hop))
    edges = np.minimum(np.arange(n + 1, dtype=np.int64) * hop, x.size)
    c = np.concatenate([[0.0], np.cumsum(x * x)])
    total = c[edges[1:]] - c[edges[:-1]]
    return np.sqrt(total / np.maximum(1, edges[1:] - edges[:-1])).astype(
        np.float32)


def _onnx_session():
    """Cached ONNX Runtime session over the vendored Silero weights, or
    None (onnxruntime missing / file absent — caller falls through)."""
    global _ONNX_SESSION
    if _ONNX_SESSION is not None:
        return _ONNX_SESSION
    try:
        import onnxruntime as ort
    except Exception:
        return None
    if not os.path.isfile(_VENDORED_ONNX):
        return None
    try:
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        _ONNX_SESSION = ort.InferenceSession(
            _VENDORED_ONNX, providers=["CPUExecutionProvider"],
            sess_options=opts)
    except Exception:
        return None
    return _ONNX_SESSION


_WIN = 512


def _onnx_probs(audio: np.ndarray, sr: int = 16000) -> list[float] | None:
    """Raw Silero speech probability per 512-sample window (32ms at 16k).

    Feeds the model exactly like OnnxWrapper.audio_forward (512-sample
    windows, 64-sample carried context, carried RNN state). The segment
    builder and the sync inspector share this so the curve the user sees is
    literally the signal the pause decisions were made from."""
    if sr != 16000:
        return None
    session = _onnx_session()
    if session is None:
        return None
    try:
        import numpy as _np

        WIN, CTX = _WIN, 64
        x = _np.ascontiguousarray(audio, dtype=_np.float32).ravel()
        n = len(x)
        state = _np.zeros((2, 1, 128), dtype=_np.float32)
        context = _np.zeros(CTX, dtype=_np.float32)
        sr_arr = _np.array(sr, dtype=_np.int64)
        probs: list[float] = []
        for start in range(0, max(n, 1), WIN):
            chunk = x[start:start + WIN]
            if len(chunk) < WIN:
                chunk = _np.pad(chunk, (0, WIN - len(chunk)))
            inp = _np.concatenate([context, chunk])[None, :]
            out, state = session.run(None, {
                "input": inp, "state": state, "sr": sr_arr})
            probs.append(float(out.ravel()[0]))
            context = inp[0, -CTX:]
        return probs
    except Exception:
        return None


def _onnx_segments(
    audio: np.ndarray, sr: int = 16000, threshold: float = 0.5
) -> list[tuple[float, float]] | None:
    """Silero VAD segments through ONNX Runtime with numpy only (see
    _onnx_probs). Post-processing is a bit-for-bit port of
    get_speech_timestamps, so edges are identical to the torch path."""
    probs = _onnx_probs(audio, sr)
    if probs is None:
        return None
    n = len(np.ascontiguousarray(audio, dtype=np.float32).ravel())
    return _tstamps_from_probs(probs, n, sr, threshold)


def _tstamps_from_probs(
    probs: list[float], n_samples: int, sr: int = 16000,
    threshold: float = 0.5,
) -> list[tuple[float, float]]:
    """Line-port of silero_vad.get_speech_timestamps post-processing
    (MIT) with its defaults: neg_threshold=threshold-0.15, min_speech
    250ms, min_silence 100ms, pad 30ms, window 512. Operates on per-window
    speech probabilities; returns (start, end) seconds."""
    WIN = 512
    neg_threshold = max(threshold - 0.15, 0.01)
    min_speech = sr * 250 / 1000
    min_silence = sr * 100 / 1000
    pad = sr * 30 / 1000
    min_sil_at_max = sr * 98 / 1000

    triggered = False
    speeches: list[dict] = []
    current: dict = {}
    temp_end = 0
    prev_end = next_start = 0
    possible_ends: list[tuple[int, int]] = []

    for i, p in enumerate(probs):
        cur = WIN * i
        if p >= threshold and temp_end:
            if cur - temp_end > min_sil_at_max:
                possible_ends.append((temp_end, cur - temp_end))
            temp_end = 0
            if next_start < prev_end:
                next_start = cur
        if p >= threshold and not triggered:
            triggered = True
            current["start"] = cur
            continue
        if p < neg_threshold and triggered:
            if not temp_end:
                temp_end = cur
            if cur - temp_end < min_silence:
                continue
            current["end"] = temp_end
            if current["end"] - current["start"] > min_speech:
                speeches.append(current)
            current = {}
            prev_end = next_start = temp_end = 0
            triggered = False
            possible_ends = []
            continue

    if current and (n_samples - current["start"]) > min_speech:
        current["end"] = n_samples
        speeches.append(current)

    # 30ms padding with neighbor-aware splitting (verbatim port).
    for i, speech in enumerate(speeches):
        if i == 0:
            speech["start"] = int(max(0, speech["start"] - pad))
        if i != len(speeches) - 1:
            silence_duration = speeches[i + 1]["start"] - speech["end"]
            if silence_duration < 2 * pad:
                speech["end"] += int(silence_duration // 2)
                speeches[i + 1]["start"] = int(max(
                    0, speeches[i + 1]["start"] - silence_duration // 2))
            else:
                speech["end"] = int(min(n_samples, speech["end"] + pad))
                speeches[i + 1]["start"] = int(max(
                    0, speeches[i + 1]["start"] - pad))
        else:
            speech["end"] = int(min(n_samples, speech["end"] + pad))
    return [(s["start"] / sr, s["end"] / sr) for s in speeches]


def _energy_segments(
    audio: np.ndarray, sr: int, hop_s: float = 0.03
) -> list[tuple[float, float]]:
    """Zero-dependency fallback VAD (no torch/onnx): RMS energy with
    hysteresis (separate on/off thresholds) + hangover, so breath pauses
    survive as real gaps in the unmerged output and silence_runs stays
    usable. Coarse vs Silero (music fools it) but fine on clean recitation."""
    frame = max(1, int(hop_s * sr))
    n = len(audio) // frame
    if n == 0:
        return []
    rms = np.sqrt(np.mean(audio[: n * frame].reshape(-1, frame) ** 2, axis=1))
    peak = float(rms.max())
    if peak <= 0:
        return []
    thr_on = max(0.001, 0.10 * peak)
    thr_off = max(0.0005, 0.04 * peak)
    segs = []
    start = None
    hang = 0
    for i, v in enumerate(rms):
        if start is None:
            if v > thr_on:
                start = i
                hang = 0
        elif v > thr_off:
            hang = 0
        else:
            hang += 1
            if hang * hop_s >= 0.15:
                if (i - hang - start + 1) * hop_s >= 0.25:
                    segs.append((start * hop_s, (i - hang + 1) * hop_s))
                start = None
    if start is not None and (n - start) * hop_s >= 0.25:
        segs.append((start * hop_s, n * hop_s))
    return segs


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
