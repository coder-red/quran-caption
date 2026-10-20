"""Lightweight sync verification: acoustic edges + per-provider anchoring.

No models, no network — pure segmenter/provider math. Covers the three
lightweight sync fixes:

1. _vad_clamp_bounds containing-case snap (VAD_EDGE_SNAP=1.5):
   - misheard first word -> caption still starts at VAD onset
   - madd tail Whisper cut early -> caption holds to VAD offset
   - continuous recitation mid-segment -> NO oversnap (edge far away)
   - word-in-silence neighbor association stays strict (0.6)
2. _anchor_words_to_vad max_shift gate (ElevenLabs 0.25 vs Groq 0.6).
3. measure_sync(): capstart-vs-onset / capend-vs-offset harness used to
   score real clips (same metric as the 2026-09-19 Hashr verification).
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.segmenter import (
    MIN_DUR,
    MIN_GAP,
    VAD_CLAMP_TOL,
    VAD_EDGE_SNAP,
    _vad_clamp_bounds,
    apply_sync_padding,
)
from backend.providers.groq import _anchor_words_to_vad, MAX_ANCHOR_SHIFT
from backend.providers.elevenlabs import SCRIBE_ANCHOR_SHIFT


def test_containing_start_snaps_beyond_old_tol():
    # First MATCHED word starts 0.7s after the VAD onset (true first word was
    # misheard by ASR but voiced in the same segment). Old 0.6 tol kept the
    # late stamp -> caption appeared ~1 word late. New containing snap (1.5)
    # recovers the onset.
    assert VAD_EDGE_SNAP == 1.5
    raw = [(10.0, 15.0)]
    first = {"start": 10.7, "end": 11.1}
    last = {"start": 14.0, "end": 14.5}
    s, e = _vad_clamp_bounds(10.7, 14.5, first, last, raw)
    assert s == 10.0, s
    assert e == 15.0, e


def test_containing_end_holds_madd_tail():
    # Madd elongation: Whisper ends the word 1.2s before voicing stops.
    # Caption must hold to the VAD offset, not vanish mid-vowel.
    raw = [(10.0, 12.5)]
    first = {"start": 10.05, "end": 10.5}
    last = {"start": 10.9, "end": 11.3}
    s, e = _vad_clamp_bounds(10.05, 11.3, first, last, raw)
    assert s == 10.0, s
    assert e == 12.5, e


def test_continuous_recitation_no_oversnap():
    # One long breath, phrase sits mid-segment: VAD edges are far away
    # (previous/next ayahs' speech) -> stamp stands, no swallowing.
    raw = [(0.0, 30.0)]
    first = {"start": 12.0, "end": 12.4}
    last = {"start": 14.5, "end": 15.0}
    s, e = _vad_clamp_bounds(12.0, 15.0, first, last, raw)
    assert s == 12.0, s
    assert e == 15.0, e


def test_silence_neighbor_stays_strict():
    # Word stamped in a silence gap: neighbor onset 0.8s away (> 0.6) is a
    # different phrase's speech -> stamp stands.
    assert VAD_CLAMP_TOL == 0.6
    raw = [(10.0, 11.0), (12.0, 15.0)]
    first = {"start": 11.2, "end": 11.5}  # in the gap
    last = {"start": 11.2, "end": 11.5}
    s, e = _vad_clamp_bounds(11.2, 11.5, first, last, raw)
    assert s == 11.2, s
    assert e == 11.5, e


def test_anchor_scribe_cap_tighter_than_groq():
    # Same drift, different caps: Scribe keeps good stamps (0.25), Groq
    # corrects Whisper drift (0.6).
    assert SCRIBE_ANCHOR_SHIFT == 0.25
    assert MAX_ANCHOR_SHIFT == 0.6

    def words():
        return [
            {"text": "a", "start": 10.5, "end": 10.9},
            {"text": "b", "start": 11.0, "end": 14.5},
        ]

    raw = [(10.0, 15.0)]
    w_groq = words()
    _anchor_words_to_vad(w_groq, raw)  # default 0.6
    w_scribe = words()
    _anchor_words_to_vad(w_scribe, raw, max_shift=SCRIBE_ANCHOR_SHIFT)
    # Groq pulls the late first word harder toward the onset than Scribe.
    assert w_groq[0]["start"] < w_scribe[0]["start"]
    assert w_scribe[0]["start"] >= 10.5 - SCRIBE_ANCHOR_SHIFT - 1e-9


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


def test_measure_sync_perfect_alignment_scores_zero():
    segs = [{"start": 10.0, "end": 12.5}, {"start": 13.0, "end": 15.0}]
    raw = [(10.0, 12.5), (13.0, 15.0)]
    m = measure_sync(segs, raw)
    assert m["n"] == 2
    assert m["mean_start"] == 0.0
    assert m["mean_end"] == 0.0


def test_measure_sync_reports_late_start():
    segs = [{"start": 10.7, "end": 12.5}]  # 0.7 late (misheard head word)
    raw = [(10.0, 12.5)]
    m = measure_sync(segs, raw)
    assert m["mean_start"] == 0.7
    assert m["mean_end"] == 0.0


def _seg(s, e):
    return {"surah": 1, "ayah": 1, "start": s, "end": e,
            "text_ar": "x", "text_en": "y"}


def test_padding_min_dur_never_steals_next_onset():
    # Crowded phrases 0.05s apart (detector sliver upstream): the old code
    # let the first caption's 0.5s MIN_DUR punch past the next onset, so the
    # next caption appeared ~0.5s AFTER the voice and every caption after it
    # cascaded late. The next onset is sacred: the first caption yields.
    #
    # With onsets 0.05s apart there is NO displayable room for the first cue:
    # its cap (next onset - MIN_GAP) equals its own start, so any cue there
    # would be zero-length forever (`t < end` never true -- an invisible
    # caption that silently loses its text). The first phrase is therefore
    # MERGED into the next, which keeps its text and shows continuously.
    segs = [_seg(51.108, 51.13), _seg(51.158, 52.2)]
    out = apply_sync_padding([dict(s) for s in segs], [])
    assert len(out) == 1, out
    assert out[0]["start"] == 51.138, out  # raw start - lead-in, unpushed
    # nothing was lost and nothing was duplicated: identical text collapses
    assert out[0]["text_ar"] == "x", out
    for s in out:
        assert s["end"] > s["start"], out


def test_padding_crowded_with_silence_keeps_onsets():
    # Crowding + a real pause: onsets stay on the voice, ends stay out of
    # the pause, nothing overlaps, nothing collapses to zero. Old code
    # pushed the second caption 0.1s late and squeezed it to a 0.03s flash.
    segs = [_seg(10.0, 10.4), _seg(10.45, 10.5)]
    out = apply_sync_padding([dict(s) for s in segs], [(10.56, 11.5)])
    assert out[1]["start"] == 10.43, out
    assert out[0]["end"] <= out[1]["start"] - MIN_GAP + 1e-9, out
    assert out[0]["end"] <= 10.56 + 1e-9, out
    assert out[1]["end"] <= 10.56 + 1e-9, out
    for s in out:
        assert s["end"] > s["start"], out
        assert not (10.56 < s["end"] < 11.5), out


def test_padding_roomy_timing_unchanged():
    # No crowding, no silence: byte-identical to the long-verified behavior
    # (lead-in 0.02, lead-out 0.06, MIN_DUR fill).
    segs = [_seg(10.0, 10.4), _seg(12.0, 12.5)]
    out = apply_sync_padding([dict(s) for s in segs], [])
    assert [(s["start"], s["end"]) for s in out] == [
        (9.98, 10.48), (11.98, 12.56)], out
    assert MIN_DUR == 0.5


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
