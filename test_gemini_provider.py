"""Tests for the Gemini ASR provider (mocked client — no API key needed).

Covers: no-key error, chunk planning for a 2h clip, per-chunk word offsets,
overlap dedupe, shrink-and-retry on unparseable responses, transcript joining.
"""
import json
import math

import numpy as np
import pytest

from backend.providers.gemini import GeminiProvider, MAX_REQUEST_SECONDS, OVERLAP_SECONDS


class FakeGenAI:
    """Minimal lookalike of google.genai.Client: models.generate_content()."""

    def __init__(self, responder):
        self.api_key = "test-key"
        self.responder = responder
        self.models = self
        self.calls = []

    def generate_content(self, model, contents, config=None):
        self.calls.append((model, contents, config))
        return type("Res", (), {"text": self.responder()})()


def make_provider(responder):
    p = GeminiProvider(api_key="test-key")
    p._client = FakeGenAI(responder)
    return p


def test_missing_api_key_raises():
    p = GeminiProvider(api_key=None)
    p.api_key = None  # ignore .env fallback (a real key may be present there)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        p._api()


def test_plan_chunks_tiles_whole_2h_clip():
    dur = 7575.0
    bounds = GeminiProvider._plan_chunks([(0.0, dur)], dur)
    assert bounds
    assert max(e - s for s, e in bounds) <= MAX_REQUEST_SECONDS + 2 * OVERLAP_SECONDS + 1e-6
    assert bounds[0][0] == pytest.approx(0.0)
    assert bounds[-1][1] == pytest.approx(dur)
    # 13 nominal 600s pieces -> 13 expanded bounds
    assert len(bounds) == math.ceil(dur / MAX_REQUEST_SECONDS)


def test_plan_chunks_expands_overlap():
    dur = 61.0
    bounds = GeminiProvider._plan_chunks([(0.0, 60.0)], dur)
    assert bounds[0] == (0.0, min(dur, 60.0 + OVERLAP_SECONDS))


def test_offsets_and_clean():
    p = make_provider(lambda: json.dumps({
        "words": [
            {"word": "بسم!", "start": 0.5, "end": 1.2},
            {"word": "الله", "start": 1.5, "end": 2.1},
        ]
    }))
    audio = np.zeros(16000 * 40, dtype=np.float32)
    words = p._call_or_shrink(audio, 0.0, 30.0)
    assert [w["text"] for w in words] == ["بسم", "الله"]
    assert words[0]["start"] == pytest.approx(0.5)


def test_shrink_on_unparseable_then_success():
    state = {"fail": True}

    def responder():
        if state["fail"]:
            state["fail"] = False
            return "not json at all"
        return json.dumps({"words": [{"word": "الرحمن", "start": 0.1, "end": 0.6}]})

    p = make_provider(responder)
    audio = np.zeros(16000 * 120, dtype=np.float32)
    words = p._call_or_shrink(audio, 0.0, 120.0)
    assert words and words[0]["text"] == "الرحمن"


def test_dedupe_overlaps_keeps_earlier_chunk():
    # chunk 0 really covered [0, 68); chunk 1's words before 68 are duplicates
    bounds = [(0.0, 68.0), (60.0, 128.0)]
    words = [
        {"text": "أ", "start": 55.0, "end": 56.0, "_chunk": 0},
        {"text": "ب", "start": 59.0, "end": 60.0, "_chunk": 0},
        {"text": "أ", "start": 66.5, "end": 67.5, "_chunk": 1},
        {"text": "ج", "start": 70.0, "end": 71.0, "_chunk": 1},
    ]
    out = GeminiProvider._dedupe_overlaps(words, bounds)
    assert [w["text"] for w in out] == ["أ", "ب", "ج"]
    assert all("_chunk" not in w for w in out)


def test_full_transcribe_with_slices(tmp_path):
    """End-to-end transcribe(): chunks planned from VAD slices, each called with
    the mocked client, words offset to absolute times, overlapped words
    deduped, transcript joined."""
    dur = 900.0
    audio = np.zeros(int(16000 * dur), dtype=np.float32)

    def responder():
        return json.dumps({
            "words": [{"word": f"كلم{i}", "start": 2.0 + i * 2.0, "end": 3.0 + i * 2.0}
                      for i in range(3)]
        })

    p = make_provider(responder)
    wav = tmp_path / "x.wav"
    import soundfile as sf
    sf.write(str(wav), audio, 16000)
    res = p.transcribe(str(wav), slices=[(0.0, 600.0), (600.0, 900.0)])
    assert res["words"]
    assert res["transcript"]
    assert all(0.0 <= w["start"] <= dur for w in res["words"])
    assert "كلم" in res["transcript"]
    # each bound produced exactly one API call (no shrink needed)
    assert len(p._client.calls) == 2