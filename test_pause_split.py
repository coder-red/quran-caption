"""Pause-aware caption splitting: ayah captions are cut at reciter "real stops"
(>= PAUSE_GAP silence between words — deliberate breathing pauses/waqf) so text
vanishes during the long pause and resumes where the reciter resumes. Short
breaths (< PAUSE_GAP) keep the caption up. Tiny phrases (few words / short
duration) merge into their neighbors so captions never flash. No models needed
— pure segmenter."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.segmenter import (
    PAUSE_GAP,
    SILENCE_SPLIT,
    _phrase_split,
    _silence_between,
    apply_sync_padding,
    segments_from_provider_times,
    segments_from_word_times,
    snap_words_to_silence,
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


AR = "يَا أَيُّهَا الَّذِينَ آمَنُوا لَا تُقَدِّمُوا بَيْنَ يَدَيِ اللَّهِ وَرَسُولِهِ"
EN = "O you who have believed, do not put forward before Allah and His Messenger"


def test_no_pause_returns_original():
    ws = words([(0.4, 0.1), (0.4, 0.1), (0.4, 0.1), (0.4, 0.1)])
    out = _phrase_split(mk_seg(AR, EN), ws)
    assert len(out) == 1
    assert out[0]["text_ar"] == AR


def test_short_breath_does_not_split():
    # 0.7s gap is a short breath, not a real stop -> caption stays up
    ws = words([(0.4, 0.1), (0.4, 0.7), (0.4, 0.1), (0.4, 0.1)])
    out = _phrase_split(mk_seg(AR, EN), ws)
    assert len(out) == 1


def test_pause_cuts_into_two_phrases():
    # gap 1.2 (>= PAUSE_GAP) after word 2; both halves are 3 words / >= 1.2s
    ws = words([(0.5, 0.1), (0.5, 0.1), (0.5, 1.2), (0.5, 0.1), (0.5, 0.1), (0.5, 0.1)])
    out = _phrase_split(mk_seg(AR, EN), ws)
    assert len(out) == 2
    assert out[0]["start"] == 0.0
    assert 1.5 <= out[0]["end"] <= 1.7  # w2 end
    assert 2.8 <= out[1]["start"] <= 3.0  # w3 start after 1.2s gap
    # Arabic split is proportional by word count (3 words / 6 words -> 5/10)
    assert "الَّذِينَ" in out[0]["text_ar"]
    assert out[0]["text_ar"].startswith("يَا")
    assert "وَرَسُولِهِ" in out[1]["text_ar"]
    # English snapped to the comma after "believed", not mid-clause
    assert out[0]["text_en"] == "O you who have believed,"
    assert out[1]["text_en"] == ("do not put forward before Allah and "
                                 "His Messenger")


def test_phrases_do_not_overlap_after_padding():
    from backend.segmenter import apply_sync_padding
    ws = words([(0.5, 0.1), (0.5, 0.1), (0.5, 1.2), (0.5, 0.1), (0.5, 0.1), (0.5, 0.1)])
    out = apply_sync_padding(_phrase_split(mk_seg(AR, EN), ws))
    for a, b in zip(out, out[1:]):
        assert a["end"] < b["start"]


def test_small_leading_phrase_folds_into_next():
    # first phrase is a single short word then a real stop -> folded forward
    ws = words([(0.3, 1.2), (0.5, 0.1), (0.5, 0.1), (0.5, 1.2), (0.5, 0.1), (0.5, 0.1), (0.5, 0.1)])
    out = _phrase_split(mk_seg(AR, EN), ws)
    assert len(out) == 2
    assert out[0]["text_ar"].startswith("يَا أَيُّهَا الَّذِينَ")


def test_tiny_mid_phrase_merges_into_previous():
    ws = words([(0.5, 0.1), (0.5, 0.1), (0.5, 1.1), (0.2, 1.1), (0.5, 0.1), (0.5, 0.1), (0.5, 0.1)])
    out = _phrase_split(mk_seg(AR, EN), ws)
    assert len(out) == 2


def test_provider_path_splits_via_chain():
    c = [(0, 49, 1), (1, 49, 1), (2, 49, 1), (3, 49, 1), (4, 49, 1), (5, 49, 1)]
    w = words([(0.5, 0.1), (0.5, 0.1), (0.5, 1.2), (0.5, 0.1), (0.5, 0.1), (0.5, 0.1)])
    for i, d in enumerate(w):
        d["text"] = f"w{i}"
    class M:
        chain = c
        surah = 49
    segs = segments_from_provider_times([M()], w)
    assert len(segs) >= 2
    starts = [s["start"] for s in segs]
    assert starts == sorted(starts)


def test_local_path_splits_with_opener_tags():
    w = [
        {"ayah": -1, "text": "الله", "start": 0.5, "end": 1.3},
        {"ayah": -1, "text": "اكبر", "start": 1.4, "end": 2.0},
        {"ayah": 0, "text": "بسم", "start": 2.1, "end": 2.9},
        {"ayah": 0, "text": "الله", "start": 2.95, "end": 3.4},
        {"ayah": 0, "text": "الرحمن", "start": 4.6, "end": 5.4},
        {"ayah": 0, "text": "الرحيم", "start": 6.6, "end": 7.4},
        {"ayah": 1, "text": "قل", "start": 7.6, "end": 8.2},
        {"ayah": 1, "text": "هو", "start": 8.3, "end": 8.8},
        {"ayah": 1, "text": "الله", "start": 9.6, "end": 10.4},
        {"ayah": 1, "text": "أحد", "start": 10.5, "end": 11.2},
    ]
    segs = segments_from_word_times(112, 1, 1, w)
    # takbir (2 words, no split) + bismillah (4 words: بسم الله + الرحمن الرحيم
    # regions would need >= 3 words, so the 2-word head folds -> 1 segment) +
    # ayah 1 (no real stop) = 3 segments; bismillah text stays complete.
    assert len(segs) == 3, [s["surah"] for s in segs]
    assert segs[0]["text_ar"] == "ٱللَّهُ أَكْبَرُ"
    assert "بِسْمِ ٱللَّهِ" in segs[1]["text_ar"]
    assert "ٱلرَّحِيمِ" in segs[1]["text_ar"]
    # no overlapping windows
    for a, b in zip(segs, segs[1:]):
        assert a["start"] < b["start"]
        assert a["end"] < b["start"] or a["end"] <= b["end"]
    assert segs[-1]["end"] > 10.0


def test_surah_1_bismillah_display_is_canon():
    w = [{"ayah": 0, "text": "بسم", "start": 0.0, "end": 0.5},
         {"ayah": 0, "text": "الله", "start": 0.6, "end": 1.0},
         {"ayah": 0, "text": "الرحمن", "start": 1.1, "end": 1.5},
         {"ayah": 0, "text": "الرحيم", "start": 1.6, "end": 2.0}]
    segs = segments_from_word_times(1, 1, 1, w)
    assert segs[0]["surah"] == 1
    assert segs[0]["ayah"] == 1
    assert segs[0]["text_en"] and "Allah" in segs[0]["text_en"]


def test_silence_between_measures_real_silence():
    a = {"start": 1.0, "end": 1.5}
    b = {"start": 3.0, "end": 3.5}
    runs = [(1.6, 2.9)]
    got = _silence_between(a, b, runs)
    assert 1.2 <= got <= 1.4  # 1.6..2.9 minus the 0.12 pad on each side


def test_silence_runs_split_where_whisper_gap_does_not():
    # Whisper stamps a 0.7s gap (drift; < PAUSE_GAP so fallback would NOT
    # split), but the audio has a real 0.6s silence inside it -> audio truth
    # splits. Both halves have 3 words / >= 1.2s so nothing merges.
    ws = [
        {"start": 0.0, "end": 0.5}, {"start": 0.5, "end": 1.0},
        {"start": 1.0, "end": 1.3}, {"start": 2.0, "end": 2.5},
        {"start": 2.5, "end": 3.0}, {"start": 3.0, "end": 3.5},
    ]
    assert len(_phrase_split(mk_seg(AR, EN), ws)) == 1  # fallback: gap 0.7
    out = _phrase_split(mk_seg(AR, EN), ws, silence_runs=[(1.35, 1.95)])
    assert len(out) == 2


def test_silence_runs_do_not_split_fake_gaps():
    # Whisper gap reads 1.2s but the audio is continuous -> NO split (whisper
    # timestamps drift and fake pauses; real silence is the only truth)
    ws = words([(0.5, 0.1), (0.5, 0.1), (0.5, 1.2), (0.5, 0.1), (0.5, 0.1), (0.5, 0.1)])
    out = _phrase_split(mk_seg(AR, EN), ws, silence_runs=[(1.6, 1.7)])
    assert len(out) == 1


def test_snap_words_to_silence_pulls_boundaries():
    ws = words([(0.5, 0.1), (0.5, 0.1)])
    ws[0]["end"] = 1.24  # bleeds into the silence run starting at 1.2
    ws[1]["start"] = 2.76  # starts inside the run ending at 2.8
    out = snap_words_to_silence(ws, [(1.2, 2.8)])
    assert out[0]["end"] == 1.2
    assert out[1]["start"] == 2.8
    # boundaries far from any run are untouched
    far = snap_words_to_silence([{"start": 5.0, "end": 5.4}], [(1.2, 2.8)])
    assert far[0]["end"] == 5.4


def test_snap_truncates_word_containing_a_breath():
    # Groq stamps one word across a whole breath (drift): the 1.0s pause
    # (1.2,2.2) sits strictly inside word 2's [1.0,2.6] span. The word end is
    # truncated to the pause start, so the splitter then measures the full
    # pause and cuts — caption blanks instead of riding through.
    ws = [
        {"start": 0.0, "end": 0.5}, {"start": 0.5, "end": 1.0},
        {"start": 1.0, "end": 2.6}, {"start": 2.6, "end": 3.1},
        {"start": 3.1, "end": 3.6}, {"start": 3.6, "end": 4.1},
    ]
    runs = [(1.2, 2.2)]
    assert len(_phrase_split(mk_seg(AR, EN), ws, silence_runs=runs)) == 1
    snapped = snap_words_to_silence(ws, runs)
    assert snapped[2]["end"] == 1.2
    out = _phrase_split(mk_seg(AR, EN), snapped, silence_runs=runs)
    assert len(out) == 2
    assert out[0]["end"] <= 1.2 + 1e-9
    assert out[1]["start"] >= 2.2 - 1e-9


def test_snap_midpoint_straddlers_shrink_away():
    # Drifted stamps straddling a 1.0s breath (1.2,2.2): the early-centered
    # word truncates to the pause start, the late-centered one pushes to the
    # pause end — neither may cover the midpoint afterwards.
    ws = [
        {"start": 0.0, "end": 0.5},
        {"start": 0.9, "end": 2.0},   # mid 1.45 <= 1.7 -> before
        {"start": 1.5, "end": 2.6},   # mid 2.05 > 1.7 -> after
        {"start": 2.6, "end": 3.1},
    ]
    runs = [(1.2, 2.2)]
    out = snap_words_to_silence(ws, runs)
    assert out[1]["end"] <= 1.2 + 1e-9
    assert out[2]["start"] >= 2.2 - 1e-9
    for w in out:
        assert not (w["start"] < 1.7 < w["end"])


def test_snap_words_gap_becomes_measured_silence():
    # Whisper BRIDGES a real 0.7s pause with a tiny 0.4s gap (drift; <
    # PAUSE_GAP, so fallback does NOT split): w2.end=1.35 bleeds 0.15 into the
    # run (1.2,1.9) and w3.start=1.75 bleeds 0.15 into it too. Snapping pulls
    # them to the run edges -> measured silence 0.7s -> the silence-based
    # splitter splits into two solid 3-word phrases.
    ws = [
        {"start": 0.0, "end": 0.5}, {"start": 0.5, "end": 1.0},
        {"start": 1.0, "end": 1.35}, {"start": 1.75, "end": 2.25},
        {"start": 2.25, "end": 2.75}, {"start": 2.75, "end": 3.25},
    ]
    runs = [(1.2, 1.9)]
    assert len(_phrase_split(mk_seg(AR, EN), ws)) == 1  # fallback: gap 0.4
    snapped = snap_words_to_silence(ws, runs)
    assert snapped[2]["end"] == 1.2 and snapped[3]["start"] == 1.9
    out = _phrase_split(mk_seg(AR, EN), snapped, silence_runs=runs)
    assert len(out) == 2


def test_measured_tiny_phrase_is_not_merged():
    # Audio truth: a real 1.0s breath after a single short word. The 1-word
    # head MUST stay its own caption (show briefly, then blank) — merging it
    # would keep text visible through the reciter's pause.
    ws = [
        {"start": 0.0, "end": 0.3},
        {"start": 1.3, "end": 1.8}, {"start": 1.8, "end": 2.3},
        {"start": 2.3, "end": 2.8}, {"start": 2.8, "end": 3.3},
    ]
    out = _phrase_split(mk_seg(AR, EN), ws, silence_runs=[(0.3, 1.3)])
    assert len(out) == 2
    assert out[0]["end"] <= 0.3 + 1e-9
    assert out[1]["start"] >= 1.3 - 1e-9


def test_empty_silence_runs_means_continuous_no_split():
    # VAD found zero pauses: the audio is continuous, so even a
    # whisper-looking 1.2s gap must NOT split (it is drift, not a pause).
    ws = words([(0.5, 0.1), (0.5, 0.1), (0.5, 1.2), (0.5, 0.1), (0.5, 0.1), (0.5, 0.1)])
    out = _phrase_split(mk_seg(AR, EN), ws, silence_runs=[])
    assert len(out) == 1


def test_breath_threshold_short_dip_keeps_caption_real_breath_splits():
    # 6 solid words; only the middle gap varies. A 0.3s articulation dip
    # keeps one caption; a 0.4s breath cuts into two.
    base = [(0.5, 0.1), (0.5, 0.1), (0.5, None), (0.5, 0.1), (0.5, 0.1), (0.5, 0.1)]
    ws_dip = words([(d, g if g is not None else 0.3) for d, g in base])
    ws_breath = words([(d, g if g is not None else 0.4) for d, g in base])
    dip_runs = [(ws_dip[2]["end"], ws_dip[3]["start"])]
    breath_runs = [(ws_breath[2]["end"], ws_breath[3]["start"])]
    assert SILENCE_SPLIT == 0.4
    assert len(_phrase_split(mk_seg(AR, EN), ws_dip, silence_runs=dip_runs)) == 1
    assert len(_phrase_split(mk_seg(AR, EN), ws_breath, silence_runs=breath_runs)) == 2


def test_padding_never_bleeds_into_silence():
    # Lead-out / MIN_DUR must not drag a caption into a measured pause:
    # phrase ends 10.0, reciter breathes 10.0-11.0, next starts 11.0.
    segs = [
        {"surah": 112, "ayah": 1, "start": 9.0, "end": 10.0,
         "text_ar": "قُلْ", "text_en": "Say"},
        {"surah": 112, "ayah": 2, "start": 11.0, "end": 12.0,
         "text_ar": "اللَّهُ", "text_en": "Allah"},
    ]
    out = apply_sync_padding(segs, silence_runs=[(10.0, 11.0)])
    assert out[0]["end"] <= 10.0 + 1e-9
    assert out[1]["start"] >= 11.0 - 1e-9
    assert out[0]["end"] < out[1]["start"]


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))