"""Every recited second must get a caption.

Two real defects, both found by auditing a full 8-minute Al-Hashr
recitation against the audio:

1. The DP alignment built its canon lookup from the UTHMANI normalization
   (superscript alef U+0670 deleted), but the ASR writes the plain alef.
   So "ٱلسَّمَٰوَٰتِ" was indexed as "السموت" while the ASR said "السماوات":
   the word could never match, the chain skipped it, and the seconds it
   covered had no caption at all. The DP now indexes BOTH forms.

2. A passage the reciter REPEATED (or words the ASR misheard) is left out
   of the chain, leaving a hole between two captions — the screen went
   blank while he was plainly still reciting (19 s of an 8-minute clip).
   The unmatched words are now assigned to the ayah whose canon text they
   match, and a stretch with no words at all is bridged when the VAD says
   the reciter was speaking there (a real breath still blanks the caption).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.detector import SurahDetector
from backend.segmenter import (_close_chain_holes, _close_wordless_holes,
                               segments_from_provider_times)


def _canon_words(text):
    return text.split()


def test_dp_matches_superscript_alef_words():
    """The ASR writes plain alefs; the canon uses U+0670. Both must align."""
    d = SurahDetector()
    d._build_index()
    # "السماوات" (ASR spelling) must be reachable, and so must the
    # Uthmani form the canon actually stores.
    assert d._word_pos.get("السماوات"), "standard spelling must be indexed"
    assert d._flat_std is not None
    # the standard form must differ from the Uthmani form for this word
    diffs = [(i, f[2], s) for i, (f, s) in
             enumerate(zip(d._flat, d._flat_std)) if f[2] != s]
    assert diffs, "some canon words must differ between the two spellings"
    # every differing word must have its standard form indexed too
    for i, uth, std in diffs[:200]:
        assert std in d._word_pos, (uth, std)


def test_dp_chain_covers_superscript_alef_words():
    """A transcript using plain-alef spelling must chain the U+0670 words."""
    d = SurahDetector()
    # 59:1 = "سبح لله ما في السماوات وما في الأرض وهو العزيز الحكيم"
    transcript = "سبح لله ما في السماوات وما في الأرض وهو العزيز الحكيم"
    m = d.detect_all(transcript)
    assert m, "must detect the surah"
    chained = {c[0] for c in m[0].chain}
    words = transcript.split()
    idx = words.index("السماوات")
    assert idx in chained, (
        f"word {idx} ('السماوات') was skipped by the chain: {m[0].chain}")


def test_repeated_passage_gets_a_caption():
    """A passage said twice must not leave a hole: both recitations are
    captioned under the same ayah."""
    text_ar = {
        "59:15": "كَمَثَلِ ٱلَّذِينَ مِن قَبْلِهِمْ قَرِيبًا ۖ ذَاقُوا۟ وَبَالَ "
                 "أَمْرِهِمْ وَلَهُمْ عَذَابٌ أَلِيمٌ",
    }
    text_en = {"59:15": "Like those before them; they taste the evil of their deed."}
    w = ["كمثل", "الذين", "من", "قبلهم", "قريبا", "ذاقوا", "وبال",
         "أمرهم", "ولهم", "عذاب", "أليم"]
    # recited twice: 0-2.0 s and 3.0-5.0 s
    words = []
    for k in range(2):
        t = k * 3.0
        for i, x in enumerate(w):
            words.append({"text": x, "start": round(t + i * 0.18, 3),
                          "end": round(t + i * 0.18 + 0.16, 3)})
    matches = [type("M", (), {
        "chain": [(i, 59, 15) for i in range(len(w))],   # only the FIRST pass
        "surah": type("S", (), {"id": 59, "name_ar": "الحشر", "name_en": "Al-Hashr"})(),
        "start_ayah": 15, "end_ayah": 15, "confidence": 1.0,
    })()]
    segs = segments_from_provider_times(
        matches, words, silence_runs=[], raw_segs=[(0.0, 5.2)]
    )
    assert segs, "expected segments"
    # the second recitation must be covered
    covered = [s for s in segs if s["end"] > 3.0]
    assert covered, f"the repeat was not captioned: {[(s['start'], s['end']) for s in segs]}"
    assert min(s["start"] for s in covered) < 3.2, covered
    assert max(s["end"] for s in covered) > 4.5, covered


def test_wordless_gap_bridged_when_speaking_but_not_at_a_breath():
    """A stretch with no words is bridged when the VAD says speech, and left
    blank when the VAD says silence."""
    segs = [
        {"surah": 59, "ayah": 1, "start": 0.0, "end": 2.0, "text_ar": "a", "text_en": "a"},
        {"surah": 59, "ayah": 1, "start": 5.0, "end": 7.0, "text_ar": "b", "text_en": "b"},
    ]
    # speech run spanning the gap -> bridge it
    _close_wordless_holes(segs, [(0.0, 7.0)])
    assert segs[0]["end"] == 4.999, segs[0]["end"]

    segs2 = [
        {"surah": 59, "ayah": 1, "start": 0.0, "end": 2.0, "text_ar": "a", "text_en": "a"},
        {"surah": 59, "ayah": 1, "start": 5.0, "end": 7.0, "text_ar": "b", "text_en": "b"},
    ]
    # a real breath between the two runs -> leave the caption off
    _close_wordless_holes(segs2, [(0.0, 2.0), (5.0, 7.0)])
    assert segs2[0]["end"] == 2.0, segs2[0]["end"]


def test_wordless_gap_with_breath_then_voiced_island():
    """A gap that starts with a real breath does NOT get painted across it.

    Measured on the 8-min Al-Hashr clip (59:7): the phrase before the gap
    ended at 145.470, a 0.55 s breath followed, and the ASR then produced no
    word for 3.2 s while the VAD (and the audio level) said the reciter was
    still voicing. The old bridge ran the caption straight over the breath
    (145.470 -> 149.194). Now the caption stops at the breath and comes back
    over the voiced island only.
    """
    segs = [
        {"surah": 59, "ayah": 7, "start": 136.77, "end": 145.47,
         "text_ar": "ما أفاء الله على رسوله من أهل القرى فلله وللرسول ولذي",
         "text_en": "And what Allah restored ... and for the Messenger"},
        {"surah": 59, "ayah": 7, "start": 149.264, "end": 173.734,
         "text_ar": "القربى واليتامى والمساكين ...",
         "text_en": "relatives and orphans ..."},
    ]
    raw = [(136.77, 145.47), (146.018, 154.206)]
    sil = [(145.47, 146.018)]
    _close_wordless_holes(segs, raw, sil)
    assert len(segs) == 3, [(s["start"], s["end"]) for s in segs]
    a, island, b = segs
    assert a["end"] == 145.47, a["end"]          # blank on the breath
    assert island["start"] == 146.018            # caption back with the voice
    assert island["end"] == 149.263              # ... right up to the next phrase
    assert island["text_ar"] == a["text_ar"]     # it is what he is still saying
    assert b["start"] == 149.264 and b["end"] == 173.734
    # no emitted caption may cover the breath
    for s in segs:
        assert not (s["start"] < 146.018 and s["end"] > 145.47)


def test_wordless_gap_with_several_breaths_keeps_each_island():
    """Every voiced island between breaths is captioned, every breath blank."""
    segs = [
        {"surah": 59, "ayah": 1, "start": 0.0, "end": 1.5, "text_ar": "a", "text_en": "a"},
        {"surah": 59, "ayah": 1, "start": 8.0, "end": 9.0, "text_ar": "b", "text_en": "b"},
    ]
    raw = [(0.0, 1.5), (2.5, 4.0), (5.5, 8.0)]
    sil = [(1.5, 2.5), (4.0, 5.5)]
    _close_wordless_holes(segs, raw, sil)
    iv = [(round(s["start"], 3), round(s["end"], 3)) for s in segs]
    assert iv == [(0.0, 1.5), (2.5, 3.999), (5.5, 7.999), (8.0, 9.0)], iv
    for s in segs:  # nothing over 1.5-2.5 or 4.0-5.5
        assert not (s["start"] < 1.5 and s["end"] > 1.5)
        assert not (s["start"] < 4.0 and s["end"] > 4.0)



def test_chain_skipped_word_inside_ayah_gets_caption():
    """A word the chain skipped MID-ayah must be added to that ayah's words.

    The old hole-closer only looked BETWEEN ayah groups, so a misheard word
    with chained neighbours on both sides belonged to no group. On the 8-min
    Al-Hashr clip this left "يحتسموا" (49.887-51.347, 1.42 s of voiced
    recitation) with no caption at all: the segment ended at the previous
    word (49.871) and the next phrase began at 51.874.
    """
    from backend.segmenter import _close_chain_holes

    words = [
        {"text": "أ", "start": 46.466, "end": 49.871},
        {"text": "يحتسموا", "start": 49.887, "end": 51.347},  # chain-skipped
        {"text": "ب", "start": 51.874, "end": 52.565},
    ]
    ordered = [((59, 2), [words[0], words[2]])]
    used = {0, 2}
    _close_chain_holes(ordered, words, used, {})
    assert [w["text"] for w in ordered[0][1]] == ["أ", "يحتسموا", "ب"], ordered
    assert 1 in used


def test_midayah_skipped_word_is_covered_end_to_end():
    """segments_from_provider_times must cover the skipped mid-ayah word."""
    w = [
        {"text": "ما", "start": 0.0, "end": 0.5},
        {"text": "يحتسموا", "start": 0.6, "end": 1.8},   # not in the chain
        {"text": "وقذف", "start": 2.0, "end": 2.6},
    ]
    matches = [type("M", (), {
        "chain": [(0, 59, 2), (2, 59, 2)],
        "surah": type("S", (), {"id": 59, "name_ar": "", "name_en": ""})(),
        "start_ayah": 2, "end_ayah": 2, "confidence": 1.0,
    })()]
    segs = segments_from_provider_times(
        matches, w, silence_runs=[], raw_segs=[(0.0, 2.6)]
    )
    # Word-tracked cues may cut the phrase in several pieces; the invariant is
    # that the skipped word's AUDIO stays covered and its text is shown.
    assert segs[0]["start"] <= 0.05, segs[0]
    assert segs[-1]["end"] >= 1.8, [(s["start"], s["end"]) for s in segs]
    joined = " ".join(s["text_ar"] for s in segs).split()
    from data.quran_text import load_ayah_texts
    assert joined == load_ayah_texts()["59:2"].split(), joined


def test_repeat_does_not_shift_text_ahead_of_voice():
    """A repeated passage must show ITS words, not the next words of the ayah.

    Measured on the 8-min Al-Hashr clip (59:15): the whole ayah was recited,
    then its opening was repeated. The ASR word count (15) exceeded the canon
    count (11), so the proportional text cut put "ذاقوا وبال" on screen while
    the reciter was saying "كمثل الذين من قبلهم". The caption text now follows
    the words actually being recited.
    """
    from data.quran_text import _normalize, load_ayah_texts
    from backend.segmenter import _map_words_to_canon

    canon = load_ayah_texts()["59:15"].split()
    assert len(canon) >= 11
    seq = list(range(11)) + [0, 1, 2, 3]      # full ayah, then the opening again
    words = []
    t = 0.0
    for k, i in enumerate(seq):
        if k == 11:                           # the breath before the repeat
            t += 0.6
        words.append({"text": _normalize(canon[i]),
                      "start": round(t, 3), "end": round(t + 0.25, 3)})
        t += 0.3
    pos = _map_words_to_canon(words, " ".join(canon))
    assert pos[:11] == list(range(11)), pos
    assert pos[11:] == [0, 1, 2, 3], pos

    matches = [type("M", (), {
        "chain": [(i, 59, 15) for i in range(len(words))],
        "surah": type("S", (), {"id": 59, "name_ar": "", "name_en": ""})(),
        "start_ayah": 15, "end_ayah": 15, "confidence": 1.0,
    })()]
    sil = [(3.3, 3.9)]
    segs = segments_from_provider_times(
        matches, words, silence_runs=sil, raw_segs=[(0.0, 3.3), (3.9, 6.0)]
    )
    assert len(segs) >= 2, [(s["start"], s["end"]) for s in segs]
    before = [s for s in segs if s["start"] < 3.3]
    after = [s for s in segs if s["start"] >= 3.3]
    assert before and after
    # Concatenated cue text (cues are now word-sized) must recite the ayah in
    # order before the breath, and the repeated opening after it.
    first = " ".join(s["text_ar"] for s in before).split()
    last = " ".join(s["text_ar"] for s in after).split()
    assert first == canon[:11], first
    # the repeated phrase shows the repeated opening — never the ayah tail
    assert last == canon[:4], last


def test_last_caption_is_not_held_over_trailing_silence():
    """The last caption must not stay on for the length of a file that is
    padded with silence.

    data/test_multisurah.wav ends with ~5.5 s of digital silence; holding
    the last caption to the end of the FILE left it on screen over pure
    silence for that whole time. The hold limit is now the voice (neural
    VAD) and the audible audio, not the file length.
    """
    import soundfile as sf

    wav = "data/test_multisurah.wav"
    if not os.path.exists(wav):
        return
    dur = sf.info(wav).duration
    from backend.main import _align_impl

    resp = _align_impl(None, asr_backend="groq", media_path=wav)
    data = resp.model_dump() if hasattr(resp, "model_dump") else resp
    if isinstance(data, dict) and data.get("error"):
        raise AssertionError(data["error"])
    last = data["segments"][-1]
    assert last["end"] < dur - 0.5, (
        f"the last caption is held to {last['end']} in a {dur:.2f}s file - "
        "it is sitting over the trailing silence")
    assert last["end"] > dur - 8.0, (
        f"the last caption ends at {last['end']}, too early for a {dur:.2f}s file")


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
