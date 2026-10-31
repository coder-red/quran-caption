"""Word-tracked caption cues.

An ayah caption that stays on screen while the reciter reads through it has
text the voice has not reached yet — measured on the 8-min Al-Hashr clip:
median cue 10.2s, longest 29.2s, 20/39 over 10s. Captions are now cut into
short cues at the provider's own word stamps (WORD_CUE_WORDS / WORD_CUE_GAP)
so the text on screen is the words being recited.
"""
import pytest

from backend.segmenter import (
    MIN_GAP,
    WORD_CUE_GAP,
    WORD_CUE_WORDS,
    _phrase_split,
    _word_cues,
    apply_sync_padding,
    segments_from_provider_times,
    subdivide_word_cues,
)
from data.quran_text import _normalize, load_ayah_texts, load_english_texts


def _ayah_words(key, n, step=0.8, gap_at=None, gap=0.0):
    """Word stamps for the first n canon words of an ayah, reciting in order."""
    canon = load_ayah_texts()[key].split()[:n]
    words, t = [], 0.0
    for i, w in enumerate(canon):
        if gap_at is not None and i == gap_at:
            t += gap
        words.append({"text": _normalize(w), "start": round(t, 3),
                      "end": round(t + step * 0.7, 3)})
        t += step
    return words


def _seg(key, words=None):
    ar = load_ayah_texts()[key]
    start = words[0]["start"] if words else 0.0
    end = words[-1]["end"] if words else 20.0
    return {"surah": int(key.split(":")[0]), "ayah": int(key.split(":")[1]),
            "start": start, "end": end, "text_ar": ar,
            "text_en": "en " * len(ar.split())}


def test_cues_are_short_and_keep_the_full_canon_text():
    key = "59:15"
    words = _ayah_words(key, 11)
    seg = _seg(key, words)
    cues = apply_sync_padding(_word_cues(seg, words), silence_runs=[])
    assert len(cues) >= 4, [(c["start"], c["text_ar"]) for c in cues]
    for cue in cues:
        assert 1 <= len(cue["text_ar"].split()) <= WORD_CUE_WORDS, cue["text_ar"]
        assert cue["text_ar"].strip()
        assert cue["text_en"].strip()
    # text is canon, complete and in order
    assert " ".join(c["text_ar"] for c in cues).split() == seg["text_ar"].split()
    # and the cues are short: no caption outlives ~3 words of recitation
    assert max(c["end"] - c["start"] for c in cues) < 4.0


def test_cue_edges_follow_their_own_word_stamps():
    key = "59:15"
    step, n = 0.8, 11
    words = _ayah_words(key, n, step=step)
    cues = apply_sync_padding(_word_cues(_seg(key, words), words), silence_runs=[])
    for cue in cues:
        # first word that starts at/after this cue's onset
        shown = [w for w in words if w["start"] >= cue["start"] - 0.06]
        assert shown, cue
        assert cue["start"] <= shown[0]["start"] + 0.06, (cue, shown[0])
        last = max(w["end"] for w in words if w["start"] < cue["end"] + 0.1)
        assert cue["end"] <= last + 0.2, (cue, last)


def test_articulation_gap_splits_inside_a_three_word_group():
    key = "59:15"
    canon = load_ayah_texts()[key].split()
    words = _ayah_words(key, 11, gap_at=1, gap=WORD_CUE_GAP + 0.1)
    cues = _word_cues(_seg(key, words), words)
    first, second = cues[0], cues[1]
    assert first["text_ar"] == canon[0]
    assert second["text_ar"] == " ".join(canon[1:4])
    assert second["start"] - first["end"] >= WORD_CUE_GAP - 0.05


def test_cues_never_overlap_and_are_displayable():
    key = "59:15"
    words = _ayah_words(key, 11, step=0.3)  # fast recitation: crowded cues
    cues = apply_sync_padding(_word_cues(_seg(key, words), words), silence_runs=[])
    for a, b in zip(cues, cues[1:]):
        assert a["end"] <= b["start"] + 1e-6, (a, b)
    for cue in cues:
        assert cue["end"] > cue["start"], cue


def test_provider_path_emits_word_cues():
    key = "112:1"
    words = _ayah_words(key, 4)
    class M:
        chain = [(i, 112, 1) for i in range(4)]
        surah = 112
    segs = segments_from_provider_times([M()], words, silence_runs=[], raw_segs=[(0.0, 5.0)])
    assert len(segs) >= 2, segs
    assert " ".join(s["text_ar"] for s in segs).split() == load_ayah_texts()[key].split()
    for a, b in zip(segs, segs[1:]):
        assert a["end"] <= b["start"] + 1e-6
        assert b["start"] - a["end"] >= MIN_GAP - 1e-6


def test_a_stretched_phrase_end_does_not_pull_in_the_next_ayahs_word():
    """A verse-final madd hold can stretch a phrase end past the next ayah's
    first word. Word membership by time window would then cut a cue around
    that foreign word and steal this ayah's tail text (measured: 59:4 showed
    only "اللَّهَ" and its شَدِيدُ ٱلْعِقَابِ appeared over 59:5's "مَا")."""
    key = "59:15"
    words = _ayah_words(key, 4, step=1.0)   # four one-word cues (gap 0.44)
    foreign = {"text": "وما", "start": words[-1]["end"] + 0.3,
               "end": words[-1]["end"] + 0.8}
    seg = _seg(key, words)
    seg["text_ar"] = " ".join(load_ayah_texts()[key].split()[:8])
    seg["end"] = foreign["end"]              # madd hold ran past the next ayah
    seg["_ws"] = list(words)                 # the phrase's OWN recited words
    cues = subdivide_word_cues([seg], words + [foreign], silence_runs=[],
                               raw_segs=[(0.0, 6.0)])
    # every cue is anchored to one of this phrase's own words
    assert all(c["start"] <= words[-1]["end"] for c in cues), cues
    assert all(c["start"] <= words[-1]["end"] for c in cues[1:])
    # the foreign word never heads a cue, so the text stays canon and in order
    assert " ".join(c["text_ar"] for c in cues).split() == seg["text_ar"].split()


def test_misheard_word_between_two_phrases_is_not_dropped_from_the_text():
    """The canon word a misheard ASR token stands for maps to None; if the
    breath cut lands right before it, a phrase-span of min(got) would start
    after it and the word would never be shown (measured: 59:14 lost
    جُدُرِۭۚ between two phrases)."""
    key = "112:1"
    canon = load_ayah_texts()[key]
    words = [
        {"text": canon.split()[0], "start": 0.0, "end": 0.6},
        {"text": canon.split()[1], "start": 0.8, "end": 1.4},
        {"text": "دجه", "start": 2.2, "end": 2.8},     # misheard -> cpos None
        {"text": canon.split()[3], "start": 3.0, "end": 3.6},
    ]
    seg = {"surah": 112, "ayah": 1, "start": 0.0, "end": 3.6,
           "text_ar": canon, "text_en": ""}
    phrases = _phrase_split(seg, words, [(1.45, 2.15)],
                            raw_segs=[(0.0, 1.4), (2.2, 3.6)],
                            canon_pos=[0, 1, None, 3])
    assert len(phrases) == 2
    shown = " ".join(p["text_ar"] for p in phrases).split()
    assert shown == canon.split()


def test_every_cue_keeps_its_english_when_the_snap_hits_the_final_period():
    """The English cut is snapped forward to a sentence break; without a
    reservation that snap can jump to the ayah's closing period and leave
    every later cue with Arabic but no translation (measured: 14 cues)."""
    key = "59:5"
    en = load_english_texts()[key]
    words = _ayah_words(key, len(load_ayah_texts()[key].split()))
    seg = _seg(key, words)
    seg["text_en"] = en
    cues = apply_sync_padding(_word_cues(seg, words), silence_runs=[])
    assert all(c["text_en"].strip() for c in cues), cues
    assert " ".join(c["text_en"] for c in cues).split() == en.split()


def test_vocative_hole_words_join_the_ayah_they_open():
    """ASR emits يَـٰٓأَيُّهَا as two tokens; if the detector chain leaves
    them out of a hole, they must be assigned to the ayah they open (the one
    whose canon text contains the compound) — not to the preceding ayah,
    where their audio would show the previous verse's tail (measured on
    59:17/18: "جَزَـٰٓؤُاْ" shown while "يَا أَيُّهَا" was recited)."""
    from backend.segmenter import _close_chain_holes

    texts_ar = load_ayah_texts()
    tail = _ayah_words("59:17", 10)          # ... وَذَٰلِكَ جَزَٰٓؤُاْ ٱلظَّٰلِمِينَ
    base = tail[-1]["end"] + 0.4
    hole = [
        {"text": "يا", "start": base, "end": base + 0.4},
        {"text": "ايها", "start": base + 0.5, "end": base + 1.2},
    ]
    head = []
    t = base + 1.5
    for w in load_ayah_texts()["59:18"].split()[1:5]:  # after the compound
        head.append({"text": _normalize(w), "start": round(t, 3),
                     "end": round(t + 0.5, 3)})
        t += 0.7
    words = tail + hole + head
    ordered = [((59, 17), list(tail)), ((59, 18), list(head))]
    used = set(range(len(tail))) | set(range(len(tail) + len(hole), len(words)))
    _close_chain_holes(ordered, words, used, texts_ar)
    assert len(ordered[0][1]) == len(tail), "hole words leaked into 59:17"
    assert len(ordered[1][1]) == len(head) + 2, "hole words never assigned"


def test_provider_path_word_cues_the_reciter_opener():
    """The Bismillah is stripped from the detector chain, so it is injected
    after matching. Injected AFTER the other stages it used to stay one whole
    phrase AND, worse, the words it doesn't cover were attributed to it: the
    gap it leaves looked wordless, _close_wordless_holes painted the previous
    ayah's text over the opener, and padding absorbed that text into the
    Bismillah cue (measured on the deployed UI: 108:1 read
    "بِسۡمِ ٱللَّهِ ... أَحَدُۢ" — Kawthar's Bismillah announced 112:4)."""
    from backend.segmenter import _normalize as _n

    kh = load_ayah_texts()["112:1"].split()
    bi = load_ayah_texts()["1:1"].split()
    kw = load_ayah_texts()["108:1"].split()
    words, t = [], 10.0
    for w in kh:                                   # 112:1
        words.append({"text": _normalize(w), "start": round(t, 3),
                      "end": round(t + 0.6, 3)})
        t += 0.9
    t += 2.0                                       # breath before the Bismillah
    for w in bi:                                   # the opener
        words.append({"text": _normalize(w), "start": round(t, 3),
                      "end": round(t + 0.6, 3)})
        t += 0.9
    t += 1.0
    for w in kw:                                   # 108:1
        words.append({"text": _normalize(w), "start": round(t, 3),
                      "end": round(t + 0.6, 3)})
        t += 0.9
    opener_start, opener_end = words[4]["start"], words[7]["end"]

    class M1:
        chain = [(i, 112, 1) for i in range(4)]
        surah = 112

    class M2:
        chain = [(8 + i, 108, 1) for i in range(3)]
        surah = 108

    raw_segs = [(10.0, 13.75), (opener_start, opener_end), (words[8]["start"], t)]
    silences = [(13.75, opener_start), (opener_end, words[8]["start"])]
    segs = segments_from_provider_times([M1(), M2()], words, silence_runs=silences,
                                        raw_segs=raw_segs)
    opener = [s for s in segs
              if s["start"] >= opener_start - 1e-6 and s["end"] <= opener_end + 1e-6]
    assert opener, segs
    assert len(opener) >= 2, "opener is still one whole-phrase caption"
    for cue in opener:
        assert 1 <= len(cue["text_ar"].split()) <= WORD_CUE_WORDS, cue
    assert " ".join(c["text_ar"] for c in opener).split() == bi
    # nothing from the previous ayah is painted over the opener's audio
    for cue in segs:
        if cue["start"] < opener_end and cue["end"] > opener_start:
            assert set(cue["text_ar"].split()) <= set(bi), cue


def test_a_squeezed_cue_keeps_recitation_order_when_carried_forward():
    """Two words can share one VAD onset (provider stamps overlap). The
    earlier cue then has no room and its text is carried into the next one —
    it must still read first, in the order the reciter said it."""
    a = {"surah": 108, "ayah": 1, "start": 5.0, "end": 5.3,
         "text_ar": "الكوثر", "text_en": "al-Kawthar."}
    b = {"surah": 108, "ayah": 2, "start": 5.0, "end": 5.8,
         "text_ar": "فصل لربك", "text_en": "So pray to your Lord"}
    out = apply_sync_padding([a, b], silence_runs=[])
    assert len(out) == 1, out
    assert out[0]["text_ar"].split()[0] == "الكوثر", out[0]["text_ar"]
    assert out[0]["text_en"].startswith("al-Kawthar"), out[0]["text_en"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
