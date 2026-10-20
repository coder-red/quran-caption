"""Caption DISPLAY invariants: the overlay must be visible and correctly
timed for every recited phrase.

These tests encode the three real defects observed on the Render deployment:

1. MISSING ENGLISH (and sometimes both languages): `_phrase_split` could
   emit a phrase whose proportional English slice was empty when the
   sentence-break snap walked backwards onto/behind the previous boundary
   (a_en == n_en was a no-op for the old guard).

2. INVISIBLE CAPTION: upstream phrases that overlap collapse to a
   zero-duration cue (`start == end`). The player shows a caption only while
   `t >= start and t < end`, so a zero-length cue is NEVER displayed.

3. DELAYED VERSES: with weak/absent silence data, one over-long upstream end
   is not pulled back to a pause edge, so every later caption is pushed late
   and never recovers (forward-only clamp chain).

The invariants below must hold for BOTH the provider path (hosted ASR — what
Render runs) and the local forced-align path.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.segmenter import (
    _phrase_split,
    _snap_en_boundary,
    apply_sync_padding,
)


def words(seq):
    out = []
    t = 0.0
    for dur, gap in seq:
        out.append({"start": round(t, 3), "end": round(t + dur, 3)})
        t += dur + gap
    return out


def mk_seg(text_ar, text_en, start=0.0, end=10.0):
    return {"surah": 49, "ayah": 1, "start": start, "end": end,
            "text_ar": text_ar, "text_en": text_en}


# --------------------------------------------------------------- invariant 1
# every emitted phrase must carry BOTH languages when the source ayah has both
EN_LONG = ("O you who have believed, do not put forward before Allah and His "
           "Messenger. Indeed, Allah is Hearing and Knowing. O you who have "
           "believed, when you are told to make room, then make room.")


def _assert_no_empty_language(phrases, need_en=True, need_ar=True):
    for p in phrases:
        if need_ar:
            assert (p.get("text_ar") or "").strip(), (
                f"phrase {p['start']}-{p['end']} has EMPTY arabic text")
        if need_en:
            assert (p.get("text_en") or "").strip(), (
                f"phrase {p['start']}-{p['end']} has EMPTY english text")


def test_english_never_empty_when_sentence_snap_walks_back():
    """Regression for the reproduced slice=[5:5] empty-English case."""
    en_words = EN_LONG.split()
    n_en = len(en_words)
    for a in range(0, n_en + 1):
        for b in range(a, n_en + 1):
            snapped = _snap_en_boundary(en_words, a, b)
            # The snap must never move the cut to or behind the previous one.
            assert snapped > a or b == a, (
                f"snap returned {snapped} for prev={a} plain={b} (n_en={n_en}) "
                "-> empty english slice")


def test_phrase_split_yields_no_empty_english_phrase():
    """A many-word ayah split into several breath phrases must give every
    phrase non-empty EN text (was: empty on the exhausted-English edge)."""
    # 12 words with REAL pauses (gap > SILENCE_SPLIT = 0.4s) after word 4 / word 8
    seq = [(0.5, 0.1)] * 4 + [(0.5, 0.6)] + [(0.5, 0.1)] * 3 + [(0.5, 0.6)] + [(0.5, 0.1)] * 3
    ws = words(seq)
    seg = mk_seg(" ".join(["و"] * 12), EN_LONG)
    runs = [(ws[4]["end"], ws[5]["start"]), (ws[8]["end"], ws[9]["start"])]
    phrases = _phrase_split(seg, ws, runs, None)
    assert len(phrases) >= 2, "expected the ayah to split at the two pauses"
    _assert_no_empty_language(phrases)


def test_trailing_phrase_keeps_english_real_baqarah_shape():
    """Real production bug (Baqarah 2:15/2:16/2:27 tails): the LAST breath
    phrase of an ayah was emitted with its 2 remaining Arabic words but an
    EMPTY English slice, so the translation vanished on screen exactly when
    the reciter finished the verse. The last phrase must take ALL remaining
    English."""
    ar = "ٱللَّهُ يَسۡتَهۡزِئُ بِهِمۡ وَيَمُدُّهُمۡ فِي طُغۡيَٰنِهِمۡ يَعۡمَهُونَ"
    en = ("[But] Allah mocks them and prolongs them in their transgression "
          "[while] they wander blindly.")
    n_ar = len(ar.split())
    seq = [(0.4, 0.08)] * n_ar
    # inject a real pause before the final 2 words (the real 2:15 shape)
    seq[n_ar - 3] = (0.4, 0.7)
    ws = words(seq)
    seg = mk_seg(ar, en, 0.0, float(ws[-1]["end"]))
    runs = [(ws[n_ar - 3]["end"], ws[n_ar - 2]["start"])]
    phrases = _phrase_split(seg, ws, runs, None)
    assert len(phrases) >= 2, phrases
    _assert_no_empty_language(phrases)
    # the trailing phrase must carry the tail of the ENGLISH sentence too
    tail = phrases[-1]
    assert tail["text_en"].strip(), tail


# --------------------------------------------------------------- invariant 2
def test_no_zero_duration_segment_survives():
    """Overlapping upstream phrases must never produce an undisplayable cue."""
    segs = [
        {"start": 10.0, "end": 10.0, "text_ar": "zero", "text_en": "zero"},
        {"start": 10.0, "end": 12.0, "text_ar": "real", "text_en": "real"},
    ]
    out = apply_sync_padding([dict(s) for s in segs], silence_runs=None)
    for s in out:
        assert s["end"] > s["start"], (
            f"undisplayable cue: {s['text_en']} {s['start']}-{s['end']}")


def test_all_cues_displayable_after_padding():
    """Whatever the input, every output cue must satisfy start < end."""
    segs = [
        {"start": 0.0, "end": 5.0, "text_ar": "a", "text_en": "a"},
        {"start": 4.9, "end": 5.0, "text_ar": "b", "text_en": "b"},
        {"start": 5.0, "end": 5.0, "text_ar": "c", "text_en": "c"},
        {"start": 4.0, "end": 4.5, "text_ar": "d", "text_en": "d"},
    ]
    out = apply_sync_padding([dict(s) for s in segs], silence_runs=None)
    out.sort(key=lambda s: s["start"])
    prev_end = -1.0
    for s in out:
        assert s["end"] > s["start"], f"zero cue {s}"
        assert s["start"] >= prev_end - 1e-9, f"overlap at {s}"
        prev_end = s["end"]


# --------------------------------------------------------------- invariant 3
def test_overlong_end_pulled_back_so_later_captions_not_delayed():
    """An upstream end that swallowed a pause (its end stamp sits INSIDE the
    measured silence) is pulled back to the pause edge, so the next caption is
    not pushed late."""
    segs = [
        {"start": 0.0, "end": 2.0, "text_ar": "a", "text_en": "a"},
        {"start": 1.9, "end": 2.5, "text_ar": "b", "text_en": "b"},  # end mid-pause
        {"start": 3.1, "end": 5.0, "text_ar": "c", "text_en": "c"},
    ]
    runs = [(2.05, 3.0)]
    out = apply_sync_padding([dict(s) for s in segs], silence_runs=runs)
    a, b, c = out[0], out[1], out[2]
    assert b["end"] <= 2.05 + 1e-9, (
        f"caption b end {b['end']} bridged the pause edge 2.05")
    # c must NOT have been pushed forward by b's over-long end
    assert c["start"] < 3.2, f"caption c was pushed late to {c['start']}"


# ------------------------------------------------------------ coverage metric
def caption_coverage(segments, raw_segs):
    """Fraction of voiced (VAD speech) seconds covered by a DISPLAYABLE
    caption. 1.0 means no recited moment is left uncaptioned."""
    import bisect
    voiced = sum(e - s for s, e in raw_segs)
    if voiced <= 0:
        return 1.0
    covered = 0.0
    for s, e in raw_segs:
        acc = 0.0
        for seg in segments:
            if seg["end"] <= seg["start"]:
                continue  # undisplayable
            lo = max(s, seg["start"])
            hi = min(e, seg["end"])
            if hi > lo:
                acc += hi - lo
        covered += min(acc, e - s)
    return round(covered / voiced, 4)


def test_coverage_metric_counts_zero_duration_as_uncovered():
    raw = [(10.0, 12.0)]
    good = [{"start": 10.0, "end": 12.0}]
    assert caption_coverage(good, raw) == 1.0
    assert caption_coverage([{"start": 10.0, "end": 10.0}], raw) == 0.0


def test_coverage_full_for_split_phrases_with_pauses():
    """Voiced seconds across phrases (pauses are unvoiced) are fully covered."""
    ws = words([(0.5, 0.1)] * 8)
    seg = mk_seg(" ".join(["و"] * 8), EN_LONG, 0.0, float(ws[-1]["end"]))
    runs = [(ws[3]["end"], ws[4]["start"])]
    raw = [(0.0, ws[3]["end"]), (ws[4]["start"], float(ws[-1]["end"]))]
    phrases = _phrase_split(seg, ws, runs, None)
    phrases = apply_sync_padding(phrases, runs)
    cov = caption_coverage(phrases, raw)
    assert cov >= 0.98, f"coverage too low: {cov}"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
