"""Captions rebuilt from a QUD³ alignment.

QUD tells us WHICH words were recited (ayah + word refs) and WHEN
(phoneme-level word times), so this path has no detector, no chain-hole
filling and no VAD snapping. The properties pinned here are exactly what the
Groq path got wrong on the 8-minute Al-Hashr clip: a repeated passage lost its
caption, and the words after a long madd went on screen ~1 s early.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.providers.qud import QudUnavailable
from backend.segmenter import (
    WORD_CUE_WORDS,
    _OPENER_RAW,
    _qud_retag_leading_opener,
    segments_from_qud_alignment,
)
from data.quran_text import _normalize, load_ayah_texts, load_english_texts


def _qseg(surah, ayah, w0, w1, t0=0.0, step=0.8, dur=0.6, conf=0.93, rel0=0.0):
    """One quran segment reciting canon words w0..w1 of (surah, ayah).

    Word times are relative to the segment's `time_from` (as QUD sends them);
    rel0 lets a second ayah be appended to the same segment."""
    words = []
    for i, wpos in enumerate(range(w0, w1 + 1)):
        words.append({
            "location": f"{surah}:{ayah}:{wpos}",
            "start": round(rel0 + i * step, 3),
            "end": round(rel0 + i * step + dur, 3),
        })
    return {
        "kind": "quran",
        "ref_from": f"{surah}:{ayah}:{w0}",
        "ref_to": f"{surah}:{ayah}:{w1}",
        "time_from": t0,
        "time_to": round(t0 + rel0 + (w1 - w0) * step + dur, 3),
        "confidence": conf,
        "words": words,
    }


def _joined(segments):
    return " ".join(s["text_ar"] for s in segments).split()


def test_full_ayah_one_phrase_complete_text_and_short_cues():
    canon = load_ayah_texts()["59:15"].split()
    segments, words = segments_from_qud_alignment([_qseg(59, 15, 1, 11)], [], [])
    assert segments, "expected captions"
    assert _joined(segments) == canon, _joined(segments)
    assert len(words) == 11
    for cue in segments:
        assert 1 <= len(cue["text_ar"].split()) <= WORD_CUE_WORDS
        assert cue["text_ar"].strip() and cue["text_en"].strip()
    for a, b in zip(segments, segments[1:]):
        assert a["end"] <= b["start"] + 0.001, (a, b)
    assert all(c["ayah"] == 15 for c in segments)


def test_mid_ayah_text_is_sliced_to_the_recited_words():
    canon = load_ayah_texts()["59:2"].split()
    segments, _ = segments_from_qud_alignment([_qseg(59, 2, 5, 8)], [], [])
    assert _joined(segments) == canon[4:8], _joined(segments)
    assert segments[0]["start"] < segments[-1]["start"]


def test_multi_ayah_segment_splits_per_ayah():
    q = _qseg(59, 1, 9, 11, t0=0.0)
    q2 = _qseg(59, 2, 1, 3, t0=0.0, rel0=2.4)
    q["words"].extend(q2["words"])
    q["ref_to"] = "59:2:3"
    q["time_to"] = q2["time_to"]
    segments, _ = segments_from_qud_alignment([q], [], [])
    assert {c["ayah"] for c in segments} == {1, 2}
    a1 = load_ayah_texts()["59:1"].split()
    a2 = load_ayah_texts()["59:2"].split()
    assert _joined([c for c in segments if c["ayah"] == 1]) == a1[8:11]
    assert _joined([c for c in segments if c["ayah"] == 2]) == a2[:3]


def test_repeat_is_captioned_twice_in_recitation_order():
    """The real Al-Hashr failure: 59:7:9-11 said twice. QUD returns the
    repeat as its own segment; both must caption, in order, with the
    repeated words — never the previous ayah's text."""
    canon = load_ayah_texts()["59:7"].split()
    first = _qseg(59, 7, 9, 11, t0=140.0)
    again = _qseg(59, 7, 9, 11, t0=146.0)
    segments, words = segments_from_qud_alignment([first, again], [], [])
    assert len(words) == 6
    stamps = sorted({round(w["start"], 1) for w in words})
    assert len(stamps) >= 5 and min(stamps) < 146.5 <= max(stamps)
    early = [c for c in segments if c["start"] < 145.0]
    late = [c for c in segments if c["start"] >= 145.0]
    assert early and late
    assert _joined(early) == canon[8:11]
    assert _joined(late) == canon[8:11]


def test_split_pieces_of_one_ayah_advance_through_its_translation():
    """QUD splits one ayah into several segments at a pause. Each piece used
    to show the whole clause its first word fell in, so consecutive cues
    repeated the same translation; the pieces must instead split the ayah's
    English between them, in order and complete."""
    en = load_english_texts()["59:2"].split()
    first = _qseg(59, 2, 1, 3, t0=10.0)
    rest = _qseg(59, 2, 4, 6, t0=13.0)
    segments, _ = segments_from_qud_alignment([first, rest], [], [])
    a = [c for c in segments if c["start"] < 12.9]
    b = [c for c in segments if c["start"] >= 12.9]
    assert a and b
    shown = " ".join(c["text_en"] for c in a + b).split()
    assert shown == en[:len(shown)], shown
    # the second piece adds words instead of repeating the first clause
    assert len(" ".join(c["text_en"] for c in b).split()) >= 4


def test_basmala_special_displays_1_1():
    canon = load_ayah_texts()["1:1"]
    qs = {
        "kind": "special", "special_type": "Basmala", "time_from": 0.3,
        "words": [{"location": f"0:0:{i}", "start": i * 0.9, "end": i * 0.9 + 0.7}
                  for i in range(1, 5)],
    }
    segments, _ = segments_from_qud_alignment([qs], [], [])
    assert {c["surah"] for c in segments} == {1}
    assert {c["ayah"] for c in segments} == {1}
    assert _joined(segments) == canon.split()


def test_istiadha_special_displays_1_0():
    qs = {
        "kind": "special", "special_type": "Isti'adha", "time_from": 0.0,
        "words": [{"location": f"0:1:{i}", "start": i * 0.9, "end": i * 0.9 + 0.7}
                  for i in range(1, 6)],
    }
    segments, _ = segments_from_qud_alignment([qs], [], [])
    assert {c["ayah"] for c in segments} == {0}
    assert _joined(segments) == _OPENER_RAW["istiadha"].split()


def test_istiadha_plus_basmala_splits_into_two_captions():
    words = []
    for i in range(1, 6):
        words.append({"location": f"0:1:{i}", "start": i * 0.9,
                      "end": i * 0.9 + 0.7})
    for i in range(1, 5):
        words.append({"location": f"0:0:{i}", "start": 5.0 + i * 0.9,
                      "end": 5.0 + i * 0.9 + 0.7})
    qs = {"kind": "special", "special_type": "Isti'adha+Basmala",
          "time_from": 0.0, "words": words}
    segments, _ = segments_from_qud_alignment([qs], [], [])
    ayahs = sorted({c["ayah"] for c in segments})
    assert ayahs == [0, 1], ayahs
    isti = [c for c in segments if c["ayah"] == 0]
    basm = [c for c in segments if c["ayah"] == 1]
    assert _joined(isti) == _OPENER_RAW["istiadha"].split()
    assert _joined(basm) == load_ayah_texts()["1:1"].split()
    assert max(c["end"] for c in isti) <= min(c["start"] for c in basm) + 0.001


def test_out_of_range_location_raises():
    q = _qseg(59, 2, 1, 2)
    q["words"][0]["location"] = "59:2:999"
    with pytest.raises(QudUnavailable):
        segments_from_qud_alignment([q], [], [])


def test_empty_alignment_raises():
    with pytest.raises(QudUnavailable):
        segments_from_qud_alignment([], [], [])


def test_leading_takbir_is_retagged_to_the_opener():
    """QUD has no takbir special; a recording that opens with takbir must not
    be captioned as whatever canon ayah merely contains those words."""
    seg = {"surah": 59, "ayah": 22, "start": 0.0, "end": 2.0,
           "text_ar": "x", "text_en": "x"}
    ws = [{"text": "ٱللَّهُ", "start": 0.0, "end": 1.0},
          {"text": "أَكْبَرُ", "start": 1.0, "end": 2.0}]
    base = [(seg, ws)]
    _qud_retag_leading_opener(base)
    out, words = base[0]
    assert (out["surah"], out["ayah"]) == (1, 0)
    assert out["text_ar"] == _OPENER_RAW["takbir"]
    assert [w["text"] for w in words] == _OPENER_RAW["takbir"].split()


def test_retag_only_applies_to_the_first_piece():
    opener = {"surah": 1, "ayah": 1, "start": 0.0, "end": 3.0,
              "text_ar": "بسم", "text_en": "b"}
    ows = [{"text": "بِسْمِ", "start": 0.0, "end": 3.0}]
    seg = {"surah": 9, "ayah": 72, "start": 4.0, "end": 6.0,
           "text_ar": "x", "text_en": "x"}
    ws = [{"text": "ٱللَّهُ", "start": 4.0, "end": 5.0},
          {"text": "أَكْبَرُ", "start": 5.0, "end": 6.0}]
    base = [(opener, ows), (seg, ws)]
    _qud_retag_leading_opener(base)
    assert base[1][0]["ayah"] == 72, "a mid-clip takbir is legitimate canon"


def test_caption_never_runs_through_a_measured_breath():
    """QUD's last word end can absorb the trailing pause (measured: a 0.49 s
    breath at 44.25 inside a cue ending 44.77). The cue must be cut back to
    the silence onset so the screen is blank while the reciter breathes."""
    seg = _qseg(59, 15, 1, 6)          # words every 0.8 s, 0.6 s each
    segments, _ = segments_from_qud_alignment([seg], [((2.1), 2.6)], [(0.0, 20.0)])
    first = segments[0]
    assert first["end"] <= 2.101, segments[:2]
    assert first["start"] < 2.1


def test_a_short_articulation_dip_does_not_blank_the_next_words():
    """A 0.38 s VAD dip is not a breath; trimming there left 1.18 s of recited
    "أَنَّهُم" with no caption (real Al-Hashr run at 40.1 s)."""
    seg = _qseg(59, 15, 1, 6)
    segments, _ = segments_from_qud_alignment([seg], [(2.1, 2.48)], [(0.0, 20.0)])
    # the cue keeps its word end (2.2 s) plus lead-out instead of being
    # pulled back to the dip at 2.1
    assert segments[0]["end"] >= 2.2, segments[:2]


def test_trust_times_keeps_the_qud_start_not_the_vad_edge():
    """The word after a long madd starts at 1.2 s; the VAD run began at 0.0
    (continuous voicing). Snapping to the VAD edge would put the next words on
    screen over the held madd — trust_times must keep QUD's phoneme times."""
    seg = _qseg(59, 15, 1, 6, t0=1.2)
    segments, _ = segments_from_qud_alignment([seg], [], [(0.0, 20.0)])
    assert segments[0]["start"] > 1.1, segments[0]


def test_engine_is_reported():
    from backend.main import _align_with_qud

    class FakeQud:
        name = "qud"

        def align(self, path, progress_cb=None):
            return {"segments": [_qseg(59, 1, 1, 11)]}

    resp = _align_with_qud(FakeQud(), "x.wav", None, None, None, None)
    assert resp.engine == "qud"
    assert resp.surah.id == 59
    assert (resp.start_ayah, resp.end_ayah) == (1, 1)
    assert len(resp.words) == 11


def test_coverage_guard_fires_on_a_barely_aligned_clip():
    from backend.main import _align_with_qud

    class FakeQud:
        name = "qud"

        def align(self, path, progress_cb=None):
            return {"segments": [_qseg(59, 1, 1, 11, t0=0.0)]}  # 8.8 s

    with pytest.raises(QudUnavailable) as exc:
        _align_with_qud(FakeQud(), "x.wav", [(0.0, 100.0)], None, None, None)
    assert "voiced audio" in str(exc.value)


def test_coverage_guard_spares_a_partially_aligned_intro():
    """A du'a intro the aligner skipped must not veto a solid Quran stretch."""
    from backend.main import _align_with_qud

    class FakeQud:
        name = "qud"

        def align(self, path, progress_cb=None):
            # 30 s intro the aligner skipped; the Quran stretch is solid
            return {"segments": [_qseg(59, 1, 1, 11, t0=30.0, step=7.0, dur=6.0)]}

    resp = _align_with_qud(FakeQud(), "x.wav", [(0.0, 100.0)], None, None, None)
    assert resp.engine == "qud"


def test_few_quran_words_raises():
    from backend.main import _align_with_qud

    class FakeQud:
        name = "qud"

        def align(self, path, progress_cb=None):
            return {"segments": [_qseg(59, 1, 1, 3)]}

    with pytest.raises(QudUnavailable):
        _align_with_qud(FakeQud(), "x.wav", None, None, None, None)


def test_align_impl_falls_back_to_groq(monkeypatch):
    """A QUD quota/outage must run the unchanged Groq path, not fail."""
    import backend.main as m

    wav = os.path.join("data", "test_multisurah.wav")
    if not os.path.exists(wav):
        pytest.skip("no wav fixture")

    class FakeQud:
        name = "qud"

    class FakeGroq:
        name = "groq"

    sentinel = object()
    calls = {"fallback": 0}

    monkeypatch.setattr(m, "get_asr_provider",
                        lambda name=None: FakeQud() if name == "qud" else FakeGroq())

    def _bad_qud(*a, **k):
        raise QudUnavailable("quota exhausted")

    def _ok_groq(provider, *a, **k):
        calls["fallback"] += 1
        assert isinstance(provider, FakeGroq)
        return sentinel

    monkeypatch.setattr(m, "_align_with_qud", _bad_qud)
    monkeypatch.setattr(m, "_align_with_provider", _ok_groq)
    out = m._align_impl(None, asr_backend="qud", media_path=wav)
    assert out is sentinel
    assert calls["fallback"] == 1


def test_align_impl_reports_fallback_reason(monkeypatch):
    """The response must say WHY the fallback ran: the deployed host cannot
    be probed from outside and the stage text is gone once the job finishes."""
    import backend.main as m
    from backend.models import AlignResponse

    wav = os.path.join("data", "test_multisurah.wav")
    if not os.path.exists(wav):
        pytest.skip("no wav fixture")

    class FakeQud:
        name = "qud"

    class FakeGroq:
        name = "groq"

    monkeypatch.setattr(m, "get_asr_provider",
                        lambda name=None: FakeQud() if name == "qud" else FakeGroq())
    monkeypatch.setattr(m, "_align_with_qud",
                        lambda *a, **k: (_ for _ in ()).throw(
                            QudUnavailable("HTTP 402: GPU quota exhausted")))
    monkeypatch.setattr(
        m, "_align_with_provider",
        lambda *a, **k: AlignResponse(
            surah={"id": 1, "name_ar": "الفاتحة", "name_en": "Al-Fatihah"},
            start_ayah=1, end_ayah=1, confidence=1.0,
            words=[], chunks=[], segments=[], srt="", vtt="", engine="groq"))
    out = m._align_impl(None, asr_backend="qud", media_path=wav)
    assert out.engine == "groq"
    assert "402" in out.engine_note and "QUD unavailable" in out.engine_note


def test_align_impl_reports_qud_failure_when_no_fallback(monkeypatch):
    import backend.main as m

    wav = os.path.join("data", "test_multisurah.wav")
    if not os.path.exists(wav):
        pytest.skip("no wav fixture")

    class FakeQud:
        name = "qud"

    # QUD is configured, but no usable fallback provider exists.
    monkeypatch.setattr(m, "get_asr_provider",
                        lambda name=None: FakeQud() if name == "qud" else None)
    monkeypatch.setattr(m, "_align_with_qud",
                        lambda *a, **k: (_ for _ in ()).throw(
                            QudUnavailable("rate limited")))
    out = m._align_impl(None, asr_backend="qud", media_path=wav)
    assert "QUD aligner failed" in getattr(out, "error", "")
