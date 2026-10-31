"""Audio-vs-caption audit for the sync inspector.

The caption pipeline decides *when* text appears from three pieces of
evidence: the Silero speech runs (measured breaths), the provider word
stamps, and the per-phrase bounds built on top of them. This module packages
that same evidence as one comparable timeline (a compact, JSON-safe payload
the browser can draw) plus a list of concrete defects ("this caption starts
0.9s after the voice", "the reciter speaks here and nothing is shown").

Nothing in this module changes caption timing. It is the instrument that
tells a human — and the next regression test — where the timing is wrong,
instead of "sync feels off on this clip".

Two deliberate choices:

* The reference for "voice is present" is the VAD, not an RMS threshold.
  With background music an energy gate never finds silence and invents
  pauses (measured on the Al-Hashr clip: 42 fake breaths).
* Caption edges are expected to sit *slightly wider* than the words that
  produced them (LEAD_IN 20ms, LEAD_OUT 60ms, MIN_DUR 0.5s, and madd tails
  held up to 3.5s over continuing voicing). Every check therefore reports a
  signed residual and only flags a breach of a human-visible threshold, or
  the inspector would paint working captions red.
"""

from __future__ import annotations

import numpy as np

from backend.segmenter import LEAD_IN, LEAD_OUT, SILENCE_SPLIT, VAD_EDGE_SNAP
from backend.vad import rms_frames, silence_runs_from_segments, speech_probs

# Silence runs shorter than SILENCE_SPLIT are articulation, not breaths: the
# product only ever splits a caption on a pause the reciter would hear, so
# flagging a caption that spans a 0.3s dip would paint working output red.
SILENCE_MIN = 0.25
# A stretch of voice with no caption must last this long to be worth showing
# (shorter than this is a word gap inside a phrase, not a missing caption).
EMPTY_SPEECH_MIN = 0.30
# Caption sitting over a measured breath: allow for the 30ms VAD pad on each
# speech run plus the 60ms lead-out before calling it a defect.
OVER_SILENCE_MIN = 0.15
# Word-boundary residuals. LEAD_IN means +20ms is normal; these thresholds
# are where a viewer can actually see the edge miss a word (the nearest
# word stamp is only a proxy — a misheard word is absent from the chain).
START_LATE_MAX = 0.35
START_EARLY_MAX = 0.50
# A cue this short is never displayed long enough to read (and the player
# hides `start == end` entirely — that cue is invisible, not just brief).
FLASH_MAX = 0.20

PEAK_HOP = 0.02      # waveform resolution before payload downsampling
PROB_HOP = 0.05      # speech-curve resolution (Silero window is 0.032s)


def downsample(arr: np.ndarray, n_out: int) -> np.ndarray:
    """Max-pool `arr` into n_out buckets (max keeps a short word visible
    when the whole clip is squeezed into a few thousand columns)."""
    n_out = max(1, int(n_out))
    n = len(arr)
    if n == 0:
        return np.zeros(n_out, dtype=arr.dtype)
    if n <= n_out:
        out = np.zeros(n_out, dtype=arr.dtype)
        out[:n] = arr
        return out
    edges = (np.arange(n_out + 1, dtype=np.int64) * n) // n_out
    edges[-1] = n
    if np.all(edges[1:] > edges[:-1]):
        return np.maximum.reduceat(arr, edges[:-1])
    return np.array([arr[edges[i]:max(edges[i] + 1, edges[i + 1])].max()
                     for i in range(n_out)])


def peaks_payload(rms: np.ndarray, n_out: int) -> tuple[list[int], float]:
    """Quantize the amplitude envelope to 0..127 ints plus its bucket size."""
    pooled = downsample(rms, n_out)
    ref = float(np.percentile(pooled, 99)) if pooled.size else 0.0
    if ref <= 0:
        ref = float(pooled.max()) if pooled.size else 0.0
    scale = 127.0 / ref if ref > 0 else 0.0
    return [int(min(127, max(0, round(float(v) * scale)))) for v in pooled], ref


def probs_payload(probs: np.ndarray, n_out: int) -> list[int]:
    pooled = downsample(probs, n_out)
    return [int(min(100, max(0, round(float(v) * 100)))) for v in pooled]


def measure_sync(segments, raw_segs):
    """capstart-vs-VAD-onset / capend-vs-VAD-offset residuals.

    segments: [{start, end}] caption phrases. raw_segs: [(s, e)] VAD speech.
    Returns dict(mean_start, std_start, mean_end, std_end, n) — the same
    metric as the 2026-09-19 Hashr verification (there: -0.08±0.19 /
    -0.05±0.22). Positive residual = caption edge lags voicing.
    """
    import statistics

    ds, de = [], []
    for seg in segments:
        onsets = [rs for rs, re in raw_segs if rs <= seg["start"] <= re]
        if not onsets:
            cands = [rs for rs, re in raw_segs
                     if abs(rs - seg["start"]) <= VAD_EDGE_SNAP]
            if not cands:
                continue
            onsets = [min(cands, key=lambda c: abs(c - seg["start"]))]
        else:
            onsets = [min(onsets)]
        offsets = [re for rs, re in raw_segs if rs <= seg["end"] <= re]
        if not offsets:
            cands = [re for rs, re in raw_segs
                     if abs(re - seg["end"]) <= VAD_EDGE_SNAP]
            if not cands:
                continue
            offsets = [min(cands, key=lambda c: abs(c - seg["end"]))]
        else:
            offsets = [max(offsets)]
        ds.append(round(seg["start"] - onsets[0], 3))
        de.append(round(seg["end"] - offsets[0], 3))
    if not ds:
        return {"n": 0}
    return {
        "n": len(ds),
        "mean_start": round(statistics.mean(ds), 3),
        "std_start": round(statistics.pstdev(ds), 3) if len(ds) > 1 else 0.0,
        "mean_end": round(statistics.mean(de), 3),
        "std_end": round(statistics.pstdev(de), 3) if len(de) > 1 else 0.0,
    }


def _merge(intervals: list[tuple[float, float]],
           min_len: float = 0.0) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for s, e in sorted(intervals):
        if e <= s:
            continue
        if out and s <= out[-1][1] + 1e-9:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out if e - s >= min_len]


def _subtract(spans: list[tuple[float, float]],
              cover: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Parts of `spans` not covered by any interval in `cover`."""
    out = []
    cov = _merge(cover)
    for s, e in _merge(spans):
        cur = s
        for cs, ce in cov:
            if ce <= cur or cs >= e:
                continue
            if cs > cur:
                out.append((cur, min(cs, e)))
            cur = max(cur, ce)
            if cur >= e:
                break
        if cur < e:
            out.append((cur, e))
    return [(s, e) for s, e in out if e > s]


def _nearest_word_start(words, t: float, limit: float = VAD_EDGE_SNAP):
    best = None
    for w in words or ():
        try:
            ws = float(w["start"])
        except Exception:
            continue
        d = t - ws
        if best is None or abs(d) < abs(best):
            best = d
        if abs(d) < 1e-9:
            break
    if best is None or abs(best) > limit:
        return None
    return best


def find_issues(segments, raw_segs, words=None, duration=None) -> list[dict]:
    """Concrete defects plus boundary clues. Each item is clickable in the
    UI: {"kind", "sev", "t0", "t1", "text", "seg"}.

    sev="warn" is an unambiguous defect (caption over a real breath, voice
    with no caption, an invisible/overlapping cue). sev="info" is a boundary
    that is merely *off the nearest word stamp*: the pipeline deliberately
    snaps an edge to the measured VAD onset/offset (up to VAD_EDGE_SNAP),
    so an offset from the word is a clue to look, not proof of a bug."""
    segs = [s for s in (segments or []) if "start" in s and "end" in s]
    issues: list[dict] = []
    silences = _merge(silence_runs_from_segments(
        raw_segs, min_silence=SILENCE_SPLIT), min_len=SILENCE_SPLIT)

    # 1. Caption over a measured breath (the player shows text while the
    #    reciter is silent — the classic "subtitles linger" complaint).
    for i, seg in enumerate(segs):
        for ss, se in silences:
            lo, hi = max(seg["start"], ss), min(seg["end"], se)
            if hi - lo <= OVER_SILENCE_MIN:
                continue
            issues.append({
                "kind": "over_silence", "sev": "warn",
                "t0": round(lo, 3), "t1": round(hi, 3), "seg": i,
                "text": f"caption shown over a {hi - lo:.2f}s breath",
            })

    # 2. Voice with no caption on screen.
    covered = [(s["start"] - LEAD_IN, s["end"] + LEAD_OUT) for s in segs]
    for s, e in _subtract(list(raw_segs), covered):
        if e - s < EMPTY_SPEECH_MIN:
            continue
        issues.append({
            "kind": "empty_speech", "sev": "warn",
            "t0": round(s, 3), "t1": round(e, 3), "seg": None,
            "text": f"{e - s:.2f}s of recitation with no caption",
        })

    # 3. Caption boundary that does not land on a word stamp (a clue, not a
    #    verdict). A caption that starts exactly where the reciter starts
    #    speaking is anchored to the best evidence there is — the measured
    #    onset — so it is never flagged, even when the first word stamp lags
    #    it by a second (measured on the Al-Hashr clip: 5 of 5 "late starts"
    #    were word-stamp lag, caption at the exact VAD onset).
    onsets = [s for s, _ in raw_segs]
    for i, seg in enumerate(segs):
        if any(abs(float(seg["start"]) - s) <= 0.10 for s in onsets):
            continue
        d = _nearest_word_start(words, float(seg["start"]))
        if d is None:
            continue
        if d > START_LATE_MAX:
            issues.append({
                "kind": "late_start", "sev": "info",
                "t0": round(float(seg["start"]), 3),
                "t1": round(float(seg["start"]) + d, 3), "seg": i,
                "text": f"caption starts {d:.2f}s after the nearest word stamp",
            })
        elif d < -START_EARLY_MAX:
            issues.append({
                "kind": "early_start", "sev": "info",
                "t0": round(float(seg["start"]), 3),
                "t1": round(float(seg["start"]) - d, 3), "seg": i,
                "text": f"caption starts {abs(d):.2f}s before the nearest word stamp",
            })

    # 4. Structural: invisible/flash cues and overlaps.
    for i, seg in enumerate(segs):
        dur = float(seg["end"]) - float(seg["start"])
        if dur <= 0:
            issues.append({
                "kind": "zero_duration", "sev": "warn",
                "t0": round(float(seg["start"]), 3),
                "t1": round(float(seg["end"]), 3), "seg": i,
                "text": "cue has no duration — the player never shows it",
            })
        elif dur < FLASH_MAX:
            issues.append({
                "kind": "flash_cue", "sev": "warn",
                "t0": round(float(seg["start"]), 3),
                "t1": round(float(seg["end"]), 3), "seg": i,
                "text": f"cue lasts only {dur:.2f}s",
            })
        if i + 1 < len(segs) and float(seg["end"]) > float(segs[i + 1]["start"]):
            issues.append({
                "kind": "overlap", "sev": "warn",
                "t0": round(float(segs[i + 1]["start"]), 3),
                "t1": round(float(seg["end"]), 3), "seg": i,
                "text": "two captions overlap",
            })

    issues.sort(key=lambda x: (x["t0"], x["kind"]))
    return issues


def audio_layers(audio: np.ndarray, sr: int,
                 max_points: int = 20000, hop_s: float = PROB_HOP,
                 speech_runs=None) -> dict:
    """The expensive half: VAD runs, amplitude envelope and speech curve.

    Depends only on the media file (not on the captions), so the API caches
    it per (path, mtime): re-inspecting after a caption edit costs nothing.
    Pass `speech_runs` to reuse runs an /align run already measured instead
    of paying for the VAD twice.
    """
    from backend.vad import get_speech_segments, vad_backend

    dur = round(len(audio) / sr, 3) if sr else 0.0
    n_out = max(200, min(int(max_points), int(dur / hop_s) or 1))
    eff_hop = round(dur / n_out, 4) if dur > 0 and n_out else hop_s

    raw = (list(speech_runs) if speech_runs is not None
           else get_speech_segments(audio, sr))
    sil = silence_runs_from_segments(raw, min_silence=SILENCE_MIN)

    rms = rms_frames(audio, sr, PEAK_HOP)
    curve, curve_backend = speech_probs(audio, sr, hop_s)

    peaks, peak_ref = peaks_payload(rms, n_out)
    probs = probs_payload(curve, n_out)
    return {
        "duration": dur,
        "hop": eff_hop,
        "peaks": peaks,
        "probs": probs,
        "peak_ref": round(peak_ref, 5),
        "curve_backend": curve_backend,
        "vad_backend": vad_backend(),
        "speech": [[round(s, 3), round(e, 3)] for s, e in raw],
        "silence": [[round(s, 3), round(e, 3)] for s, e in sil],
    }


def analyze(audio: np.ndarray, sr: int, segments, words=None,
            max_points: int = 20000, hop_s: float = PROB_HOP,
            layers: dict | None = None) -> dict:
    """Build the compact sync-inspector payload for one media file.

    Every lane shares one x-axis (`hop` = seconds per column). Pass `layers`
    (from audio_layers) to skip the analysis when only the captions changed.
    """
    base = layers if layers is not None else audio_layers(
        audio, sr, max_points=max_points, hop_s=hop_s)
    dur = base["duration"]
    raw = [tuple(r) for r in base["speech"]]

    issues = find_issues(segments, raw, words=words, duration=dur)
    # Headline numbers are derived from the flagged items only — the old
    # capstart-vs-VAD-onset residual is meaningless on continuous recitation
    # (one VAD run holds several phrases, so the residual grows with phrase
    # position, not with timing error). measure_sync() stays available for
    # offline audits of clips with isolated speech runs.
    by_kind: dict[str, list[dict]] = {}
    for it in issues:
        by_kind.setdefault(it["kind"], []).append(it)
    over_s = sum(i["t1"] - i["t0"] for i in by_kind.get("over_silence", []))
    empty_s = sum(i["t1"] - i["t0"] for i in by_kind.get("empty_speech", []))
    metrics = {
        "n_segments": len(segments or []),
        "n_speech_runs": len(raw),
        "speech_s": round(sum(e - s for s, e in raw), 3),
        "over_silence_s": round(over_s, 3),
        "empty_speech_s": round(empty_s, 3),
        "n_issues": len(issues),
        "n_warn": len([i for i in issues if i.get("sev") == "warn"]),
        "n_info": len([i for i in issues if i.get("sev") != "warn"]),
        "n_late": len(by_kind.get("late_start", [])),
        "n_early": len(by_kind.get("early_start", [])),
        "n_flash": len(by_kind.get("flash_cue", [])),
        "n_zero": len(by_kind.get("zero_duration", [])),
        "n_overlap": len(by_kind.get("overlap", [])),
    }
    out = dict(base)

    def _clamp(t) -> float:
        t = float(t)
        return max(0.0, min(t, dur)) if dur else max(0.0, t)

    def _span(d, key0="start", key1="end") -> list[float]:
        a, b = _clamp(d[key0]), _clamp(d[key1])
        return [round(a, 3), round(max(a, b), 3)]

    out.update({
        "words": [_span(w) for w in (words or [])
                  if "start" in w and "end" in w],
        "segs": [_span(s) for s in (segments or [])],
        "metrics": metrics,
        "issues": issues,
    })
    return out
