"""Madd-tail hold: captions must survive long elongations providers cut early.

Steals qurani.io quran-timing-helper's --madd-fix without any heavy deps
(no torch, no CTC, no downloads — pure segmenter math + raw Uthmani marks):

- _vad_clamp_bounds only recovers holds <= VAD_EDGE_SNAP (1.5s) AND only when
  the VAD offset sits that close. In continuous recitation the speech segment
  runs on for ayahs, so a 3-6s mujawwad verse-final madd keeps its early
  provider cut and the caption vanishes mid-vowel.
- _madd_extend holds a madd-word phrase end into CONTINUING voicing (same raw
  VAD seg), stopped by real silence / the VAD offset / the next phrase / the
  cap (1.8s mid-verse, 3.5s verse-final). Never into a pause, never overlap.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.segmenter import (
    MADD_FINAL_EXTEND,
    MADD_MID_EXTEND,
    MIN_GAP,
    _has_madd,
    _madd_extend,
    _phrase_split,
    segments_from_provider_times,
)

assert MADD_MID_EXTEND == 1.8
assert MADD_FINAL_EXTEND == 3.5

# Last words (raw Uthmani): explicit marks vs plain vs unholdable.
W_MARK = "ٱلْعَٰلَمِينَ"      # superscript alef U+0670 + ya
W_VOWEL = "نَسْتَعِينُ"        # long ya, no explicit mark (broad)
W_PLAIN = "مِنَ"               # no long vowel, no mark
W_CONS = "قُلْ"                # consonants only


def mk_seg(text_ar, start=0.0, end=10.0):
    return {"surah": 1, "ayah": 2, "start": start, "end": end,
            "text_ar": text_ar, "text_en": "en"}


def ws_of(spans):
    return [{"start": s, "end": e} for s, e in spans]


def test_has_madd_marks_and_vowels():
    assert _has_madd(W_MARK)
    assert _has_madd(W_VOWEL)
    assert not _has_madd(W_PLAIN)
    assert not _has_madd(W_CONS)
    assert not _has_madd("")


def test_verse_final_madd_holds_to_cap():
    # Provider cuts the ayah-final madd at 5.0, voicing runs on in the same
    # VAD segment (continuous recitation). Caption must hold 3.5s, not drop.
    seg = mk_seg("ٱلْحَمْدُ لِلَّهِ رَبِّ " + W_MARK, end=5.0)
    ws = ws_of([(0.0, 0.5), (0.6, 5.0)])
    out = _phrase_split(seg, ws, silence_runs=[], raw_segs=[(0.0, 20.0)])
    assert len(out) == 1
    assert out[0]["end"] == 8.5, out[0]


def test_verse_final_madd_stops_at_vad_offset():
    # Same cut, but voicing genuinely stops at 6.0 (short VAD seg): the hold
    # must not invent speech past the offset.
    seg = mk_seg("ٱلْحَمْدُ لِلَّهِ رَبِّ " + W_MARK, end=5.0)
    ws = ws_of([(0.0, 0.5), (0.6, 5.0)])
    out = _phrase_split(seg, ws, silence_runs=[], raw_segs=[(0.0, 6.0)])
    assert len(out) == 1
    assert out[0]["end"] == 6.0, out[0]


def test_verse_final_madd_stops_at_silence():
    # Breath pause at 6.0: extension must never bleed into it.
    seg = mk_seg("ٱلْحَمْدُ لِلَّهِ رَبِّ " + W_MARK, end=5.0)
    ws = ws_of([(0.0, 0.5), (0.6, 5.0)])
    out = _phrase_split(seg, ws, silence_runs=[(6.0, 6.5)],
                        raw_segs=[(0.0, 20.0)])
    assert len(out) == 1
    assert out[0]["end"] == 6.0, out[0]


def test_plain_final_word_never_extends():
    # Same early cut, but the last word cannot carry a madd: stamp stands.
    seg = mk_seg("وَمِن شَرِّ حَاسِدٍ إِذَا حَسَدَ", end=5.0)
    assert not _has_madd("حَسَدَ")
    ws = ws_of([(0.0, 0.5), (0.6, 5.0)])
    out = _phrase_split(seg, ws, silence_runs=[], raw_segs=[(0.0, 20.0)])
    assert len(out) == 1
    assert out[0]["end"] == 5.0, out[0]


def test_midverse_cap_is_smaller():
    # First of two breath phrases ends in a madd word: mid-verse cap 1.8s.
    # raw segs end where the breath starts (5.9); provider cut at 3.0.
    seg = mk_seg("ٱهْدِنَا " + W_MARK + " وَمِن شَرِّ حَاسِدٍ إِذَا حَسَدَ")
    ws = ws_of([(0.0, 0.5), (0.6, 3.0), (7.0, 7.5), (7.6, 8.1)])
    out = _phrase_split(seg, ws, silence_runs=[(5.9, 6.6)],
                        raw_segs=[(0.0, 5.9), (6.6, 20.0)])
    assert len(out) == 2
    assert out[0]["end"] == 4.8, out[0]  # 3.0 + 1.8, inside voicing
    assert out[1]["start"] >= out[0]["end"] + MIN_GAP - 1e-9


def test_no_raw_segs_no_change():
    # Local forced-align path (frame-accurate already) and callers without
    # VAD data: behavior byte-identical to before.
    seg = mk_seg("ٱلْحَمْدُ لِلَّهِ رَبِّ " + W_MARK, end=5.0)
    ws = ws_of([(0.0, 0.5), (0.6, 5.0)])
    out = _phrase_split(seg, ws, silence_runs=[])
    assert len(out) == 1
    assert out[0]["end"] == 5.0, out[0]


def test_madd_extend_unit_guards():
    assert _madd_extend(5.0, W_MARK, True, [], [], float("inf")) == 5.0
    assert _madd_extend(5.0, W_PLAIN, True, [], [(0.0, 20.0)],
                        float("inf")) == 5.0
    # End sitting in silence (no covering seg): nothing to hold.
    assert _madd_extend(6.2, W_MARK, True, [(6.0, 6.5)],
                        [(0.0, 6.0), (6.5, 20.0)], float("inf")) == 6.2


def test_provider_path_holds_real_ayah_madd():
    # End-to-end through segments_from_provider_times on 114:1
    # ( ... بِرَبِّ النَّاسِ — alef+seen madd tail), provider cut 2s early.
    from data.quran_text import load_ayah_texts, load_english_texts
    ar = load_ayah_texts()["114:1"]
    en = load_english_texts().get("114:1", "en")
    assert _has_madd(ar.split()[-1])
    words = [
        {"text": "w0", "start": 0.0, "end": 0.5},
        {"text": "w1", "start": 0.6, "end": 1.0},
        {"text": "w2", "start": 1.1, "end": 3.0},
    ]

    class M:
        chain = [(0, 114, 1), (1, 114, 1), (2, 114, 1)]
        surah = 114

    segs = segments_from_provider_times([M()], words, [], [(0.0, 20.0)])
    assert len(segs) == 1
    # 3.0 + verse-final 3.5 hold, then the standard 0.06 display lead-out.
    assert segs[0]["end"] == 6.56, segs[0]
    assert segs[0]["text_ar"] == ar


def test_provider_path_plain_tail_stands():
    # 113:5 ends حَسَدَ (no madd): the same early cut must stand.
    from data.quran_text import load_ayah_texts
    ar = load_ayah_texts()["113:5"]
    assert not _has_madd(ar.split()[-1])
    words = [
        {"text": "w0", "start": 0.0, "end": 0.5},
        {"text": "w1", "start": 0.6, "end": 3.0},
    ]

    class M:
        chain = [(0, 113, 5), (1, 113, 5)]
        surah = 113

    segs = segments_from_provider_times([M()], words, [], [(0.0, 20.0)])
    assert len(segs) == 1
    assert segs[0]["end"] == 3.06, segs[0]  # stamp stands + lead-out only


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
