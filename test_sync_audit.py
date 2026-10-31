"""Sync-inspector audit: payload shape + issue classification.

No models, no network: the VAD call is stubbed, so these run in milliseconds.
The point of the classifier is that a GOOD clip produces zero warnings — a
diagnostic that cries wolf is worse than none.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import backend.vad as vad
from backend.sync_audit import (
    EMPTY_SPEECH_MIN,
    FLASH_MAX,
    OVER_SILENCE_MIN,
    SILENCE_SPLIT,
    analyze,
    downsample,
    find_issues,
    peaks_payload,
    probs_payload,
)


def _seg(s, e, ayah=1):
    return {"surah": 1, "ayah": ayah, "start": s, "end": e,
            "text_ar": "x", "text_en": "x"}


def _kinds(issues, sev=None):
    return [i["kind"] for i in issues if sev is None or i["sev"] == sev]


def test_downsample_preserves_a_short_spike_and_length():
    arr = np.zeros(1000, dtype=np.float32)
    arr[500] = 1.0
    out = downsample(arr, 100)
    assert len(out) == 100
    assert out.max() == 1.0          # the spike survives the squeeze
    assert out.argmax() == 50        # ...in the right column
    assert len(downsample(arr, 5000)) == 5000   # upsampling pads


def test_payload_quantization_ranges():
    rms = np.linspace(0, 0.5, 400).astype(np.float32)
    peaks, ref = peaks_payload(rms, 100)
    assert len(peaks) == 100 and all(0 <= p <= 127 for p in peaks)
    assert peaks[-1] == 127 and ref > 0
    probs = probs_payload(np.linspace(0, 1, 400).astype(np.float32), 100)
    assert len(probs) == 100 and all(0 <= p <= 100 for p in probs)
    assert probs[-1] == 100


def test_good_clip_has_zero_warnings():
    """Captions that sit exactly on the voice, with word stamps on their
    edges, must not be flagged — this is the guard against a noisy tool."""
    runs = [(1.0, 5.0), (6.0, 10.0)]
    segs = [_seg(1.0, 5.0), _seg(6.0, 10.0)]
    words = [{"start": 1.0, "end": 1.5}, {"start": 6.0, "end": 6.5}]
    issues = find_issues(segs, runs, words=words)
    assert _kinds(issues, "warn") == []


def test_caption_over_a_real_breath_is_a_warning():
    runs = [(1.0, 3.0), (5.0, 7.0)]
    segs = [_seg(1.0, 5.5)]       # bleeds 0.5s into the 2s breath
    issues = find_issues(segs, runs)
    over = [i for i in issues if i["kind"] == "over_silence"]
    assert len(over) == 1 and over[0]["sev"] == "warn"
    assert abs(over[0]["t0"] - 3.0) < 1e-6 and abs(over[0]["t1"] - 5.0) < 1e-6


def test_short_articulation_dip_is_not_a_breath():
    """The pipeline only splits on pauses >= SILENCE_SPLIT; a caption
    spanning a 0.3s dip is working as designed."""
    assert 0.3 < SILENCE_SPLIT
    runs = [(1.0, 3.0), (3.3, 5.0)]
    segs = [_seg(1.0, 5.0)]
    assert "over_silence" not in _kinds(find_issues(segs, runs))


def test_uncaptioned_speech_is_a_warning_with_the_measured_gap():
    runs = [(1.0, 3.0), (5.0, 8.0)]
    segs = [_seg(1.0, 3.0)]
    empty = [i for i in find_issues(segs, runs) if i["kind"] == "empty_speech"]
    assert len(empty) == 1 and empty[0]["sev"] == "warn"
    assert abs(empty[0]["t1"] - empty[0]["t0"] - 3.0) < 1e-6


def test_tiny_gap_between_words_is_not_flagged():
    gap = EMPTY_SPEECH_MIN / 2
    runs = [(1.0, 3.0), (3.0 + gap, 3.0 + gap + 0.4)]
    segs = [_seg(1.0, 3.0), _seg(3.0 + gap, 3.0 + gap + 0.4)]
    assert "empty_speech" not in _kinds(find_issues(segs, runs))


def test_structural_defects_are_warnings():
    runs = [(1.0, 10.0)]
    segs = [_seg(1.0, 1.0), _seg(2.0, 2.0 + FLASH_MAX / 2), _seg(4.0, 8.0),
            _seg(7.5, 10.0)]
    kinds = _kinds(find_issues(segs, runs), "warn")
    assert "zero_duration" in kinds and "flash_cue" in kinds
    assert "overlap" in kinds


def test_boundary_offset_from_word_is_info_not_a_defect():
    """Mid-run phrase edges are the only place a word-stamp comparison is
    meaningful, and even then it is a clue, never a warning."""
    runs = [(1.0, 5.0)]
    segs = [_seg(1.0, 2.0), _seg(2.0, 5.0)]
    words = [{"start": 1.0, "end": 1.4}, {"start": 3.4, "end": 3.8}]
    issues = find_issues(segs, runs, words=words)
    infos = [i for i in issues if i["sev"] == "info"]
    assert infos, "a 0.9s boundary offset should be surfaced as a clue"
    assert _kinds(issues, "warn") == []


def test_caption_at_the_measured_onset_is_never_flagged():
    """Real Al-Hashr case: the caption starts exactly at the VAD onset while
    the first word stamp lags 0.85s. The caption is right, the stamp is
    late — flagging it would send someone chasing working code."""
    runs = [(146.018, 154.206)]
    segs = [_seg(146.018, 149.194), _seg(149.194, 154.206)]
    words = [{"start": 146.87, "end": 147.2}]
    assert find_issues(segs, runs, words=words) == []


def _stub_vad(runs):
    orig = vad.get_speech_segments
    vad.get_speech_segments = lambda audio, sr=16000, threshold=0.5: list(runs)
    return orig


def test_analyze_payload_is_json_safe_and_complete():
    sr = 16000
    audio = (0.02 * np.sin(
        2 * np.pi * 220 * np.arange(sr * 6) / sr)).astype(np.float32)
    audio[: sr] = 0.0                       # 1s of silence
    runs = [(1.0, 3.0), (4.0, 6.0)]
    segs = [_seg(1.0, 3.0), _seg(4.0, 6.0)]
    words = [{"start": 1.0, "end": 1.4}, {"start": 4.0, "end": 4.4}]
    orig = _stub_vad(runs)
    try:
        payload = analyze(audio, sr, segs, words, max_points=500)
    finally:
        vad.get_speech_segments = orig

    import json
    json.dumps(payload)                     # must be JSON-serializable
    for key in ("duration", "hop", "peaks", "probs", "speech", "silence",
                "words", "segs", "metrics", "issues", "vad_backend"):
        assert key in payload
    assert len(payload["peaks"]) == len(payload["probs"])
    assert abs(payload["duration"] - 6.0) < 1e-6
    assert abs(payload["hop"] * len(payload["peaks"]) - 6.0) < 0.05
    assert payload["metrics"]["n_segments"] == 2
    assert payload["metrics"]["n_warn"] == 0
    assert payload["curve_backend"] in ("onnx", "energy")


def test_analyze_survives_empty_input():
    orig = _stub_vad([])
    try:
        payload = analyze(np.zeros(0, dtype=np.float32), 16000, [], [])
    finally:
        vad.get_speech_segments = orig
    assert payload["duration"] == 0.0 and payload["issues"] == []
    assert len(payload["peaks"]) == len(payload["probs"])
