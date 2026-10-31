"""POST /media/{id}/sync: the audio half is measured once per file.

The inspector must never re-run the VAD for every caption edit, and it must
never lie about which media it read, so these tests pin: the response shape,
the layers cache, the 404 for an unknown id, and tolerance of captions that
run past the end of the audio (hand-edited segments do).

Audio: data/test_ikhlas_16k.wav (a real 21.7s recitation). A synthetic tone
does not work here — Silero classifies pure tones as non-speech, so there
would be nothing to caption, and the test would prove nothing.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi.testclient import TestClient

import pytest

import backend.main as main

ROOT = os.path.dirname(os.path.abspath(__file__))
AUDIO = os.path.join(ROOT, "data", "test_ikhlas_16k.wav")


def _client():
    return TestClient(main.app)


def _seg(a, b):
    return {"surah": 112, "ayah": 1, "start": a, "end": b,
            "text_ar": "قل هو الله أحد", "text_en": "Say, He is Allah, the One"}


def test_sync_endpoint_returns_layers_and_audit():
    mid = "test-sync-1"
    main.MEDIA_FILES[mid] = AUDIO
    try:
        body = {"segments": [_seg(0.0, 2.0)],
                "words": [{"text": "قل", "start": 0.1, "end": 0.6}],
                "max_points": 2000}
        r = _client().post(f"/media/{mid}/sync", json=body)
        assert r.status_code == 200, r.text
        d = r.json()
        for key in ("duration", "hop", "peaks", "probs", "speech", "silence",
                    "segs", "words", "metrics", "issues", "vad_backend",
                    "curve_backend"):
            assert key in d, key
        assert 21.0 < d["duration"] < 22.5
        assert len(d["peaks"]) == len(d["probs"]) > 0
        assert d["hop"] > 0
        assert d["metrics"]["n_segments"] == 1
        assert d["segs"] == [[0.0, 2.0]]
        # Only the first 2s are captioned, so the rest of the recitation is
        # uncovered and the audit has to say so out loud.
        assert d["metrics"]["empty_speech_s"] > 5.0
        assert any(i["kind"] == "empty_speech" for i in d["issues"])
    finally:
        main.MEDIA_FILES.pop(mid, None)


def test_second_call_reuses_the_cached_audio_layers():
    mid_a, mid_b = "test-sync-2a", "test-sync-2b"
    main.MEDIA_FILES[mid_a] = AUDIO
    main.MEDIA_FILES[mid_b] = AUDIO
    try:
        body = {"segments": [], "words": [], "max_points": 2000}
        c = _client()
        r1 = c.post(f"/media/{mid_a}/sync", json=body)
        assert r1.status_code == 200
        n_before = len(main._SYNC_LAYERS_CACHE)
        # Same file under another id: the key is the file (path + mtime),
        # not the id, so nothing is recomputed.
        r2 = c.post(f"/media/{mid_b}/sync", json=body)
        assert r2.status_code == 200
        assert len(main._SYNC_LAYERS_CACHE) == n_before
        # Same media, different captions: layers reused, audit recomputed.
        body2 = dict(body, segments=[_seg(0.0, 21.0)])
        r3 = c.post(f"/media/{mid_a}/sync", json=body2)
        assert r3.status_code == 200
        assert len(main._SYNC_LAYERS_CACHE) == n_before
        assert r1.json()["peaks"] == r3.json()["peaks"]
        assert r3.json()["segs"] == [[0.0, 21.0]]
    finally:
        main.MEDIA_FILES.pop(mid_a, None)
        main.MEDIA_FILES.pop(mid_b, None)


def test_unknown_media_is_a_404():
    r = _client().post("/media/does-not-exist/sync",
                       json={"segments": [], "words": []})
    assert r.status_code == 404


def test_captions_past_the_end_of_the_audio_are_clamped():
    """Hand-edited segments can outlive the file; the lanes stay on screen."""
    mid = "test-sync-3"
    main.MEDIA_FILES[mid] = AUDIO
    try:
        body = {"segments": [_seg(0.0, 999.0)],
                "words": [{"text": "x", "start": 0.0, "end": 999.0}],
                "max_points": 500}
        r = _client().post(f"/media/{mid}/sync", json=body)
        assert r.status_code == 200, r.text
        d = r.json()
        assert d["segs"][0][1] == pytest.approx(d["duration"], abs=0.01)
        assert d["words"][0][1] == pytest.approx(d["duration"], abs=0.01)
    finally:
        main.MEDIA_FILES.pop(mid, None)
