from bisect import bisect_left, bisect_right
from collections import Counter

from rapidfuzz import fuzz

from data.quran_text import load_ayah_texts, load_english_texts, _normalize, _normalize_std


LEAD_IN = 0.02
LEAD_OUT = 0.06
MIN_GAP = 0.05
MIN_DUR = 0.5

# Reciters pause between breath phrases (waqf) mid-ayah. When a TRUE silence
# (measured from the audio via VAD — backend.vad.silence_runs_from_segments,
# NOT from Whisper word-gap math, which drifts 200-500ms) >= SILENCE_SPLIT
# sits between two words, the caption for that ayah is cut into separate
# phrases so text disappears during the pause and reappears where the reciter
# resumes. Madd (elongation) is voiced, so VAD never reports it as silence
# and it can never fake a pause; only real breath silences split (>= 0.4s —
# shorter inter-word dips are articulation, not pauses).
# PAUSE_GAP is the conservative fallback threshold used ONLY when no silence
# runs are available (tests, callers without VAD data): whisper gaps are
# unreliable evidence, so they must clear a higher bar to split.
SILENCE_SPLIT = 0.4
PAUSE_GAP = 1.0
# Minimum VAD gap that counts as a real breath when a wordless stretch is
# bridged (_close_wordless_holes). Matches silence_runs_from_segments'
# min_silence: shorter dips are articulation and may stay captioned.
BREATH_MIN = 0.25
MIN_PHRASE_WORDS = 3
MIN_PHRASE_DUR = 1.2
# Max disagreement (seconds) between a provider word edge and the VAD speech
# edge it is snapped to in _vad_clamp_bounds. Beyond this the word/segment
# association is dubious (neighbor phrase's speech) and the word stamp stands.
VAD_CLAMP_TOL = 0.6
# Wider tolerance (seconds) applied when the word edge sits INSIDE a VAD
# speech segment (containing case). Same-segment means continuous voicing
# with no silence between the VAD edge and the word, so the edge is almost
# certainly the phrase's true edge: a first word misheard by ASR (caption
# would start ~1 word late) or a madd elongation Whisper cut early (caption
# would vanish while the reciter still holds the vowel). Still bounded —
# in continuous recitation a phrase sits mid-segment with the edge far away,
# and then the stamp correctly stands (plus apply_sync_padding clamps any
# residual overlap with the next phrase).
VAD_EDGE_SNAP = 1.5
_EN_BREAK_CHARS = set(".;!?،،:,")

# Madd (elongation) tail hold — idea stolen from qurani.io's quran-timing-helper
# (--madd-fix): a reciter holding a madd vowel for seconds while the provider
# stamps the word end early leaves the caption dropping mid-vowel.
# _vad_clamp_bounds already recovers holds <= VAD_EDGE_SNAP, but ONLY when the
# VAD segment offset sits that close; in continuous recitation the speech
# segment runs on for ayahs and the early stamp stands. Madd is voiced, so the
# fix extends a madd-word phrase end into CONTINUING voicing (the same raw VAD
# segment still covers the edge), stopped by real silence, the next phrase, or
# the cap below — never into a pause, never overlapping. Mid-verse cap 1.8s,
# verse-final cap 3.5s (mujawwad holds; murattal rarely exceeds 1s).
MADD_MID_EXTEND = 1.8
MADD_FINAL_EXTEND = 3.5
# Explicit Uthmani madd marks: maddah-above U+0653, superscript alef U+0670
# (very common in Uthmani: الرحمن, الصراط, يايها...).
_MADD_MARKS = {"\u0653", "\u0670"}
# Long-vowel letters that can carry a held madd (broad check, qurani "broad"
# mode): plain + wasla/maddah/hamza alef forms, waw, ya.
_MADD_VOWELS = {"\u0627", "\u0671", "\u0622", "\u0623", "\u0625",
                "\u0648", "\u064a"}

# Uthmani orthography writes some long-a words with a superscript alif
# (U+0670, e.g. ٱلشَّيْطَٰنِ) that _normalize strips but ASR reads back as a
# real alif (الشيطان). _opener_matches @checks BOTH spellings: canon is the
# Uthmani-stripped form while the ASR form carries the extra alif — and the
# reverse also happens (ASR dropping the alif like الرحمن/الرحمان).
BISMILLAH_WORDS = _normalize("بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ").split()
TAKBIR_WORDS = _normalize("ٱللَّهُ أَكْبَرُ").split()
ISTIADHA_WORDS = _normalize("أَعُوذُ بِٱللَّهِ مِنَ ٱلشَّيْطَٰنِ ٱلرَّجِيمِ").split()

_OPENER_RAW: dict[str, str] = {
    "bismillah": "بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ",
    "takbir": "ٱللَّهُ أَكْبَرُ",
    "istiadha": "أَعُوذُ بِٱللَّهِ مِنَ ٱلشَّيْطَٰنِ ٱلرَّجِيمِ",
}
# The "plain-alif" reading of each opener's canon words (U+0670 -> U+0627).
_STD_WORDS: dict[str, list[str]] = {
    kind: _normalize_std(raw).split() for kind, raw in _OPENER_RAW.items()
}

OPENERS: list[tuple[str, list[str], str, str]] = [
    ("bismillah", BISMILLAH_WORDS, _OPENER_RAW["bismillah"],
     "In the name of Allah, the Entirely Merciful, the Especially Merciful."),
    ("takbir", TAKBIR_WORDS, _OPENER_RAW["takbir"], "Allah is the Greatest."),
    ("istiadha", ISTIADHA_WORDS, _OPENER_RAW["istiadha"],
     "I seek refuge in Allah from the accursed Satan."),
]

BISMILLAH_AYAH = "1:1"

# Word-tracked cues. An ayah caption that sits still while the reciter reads
# through it (measured on the 8-min Al-Hashr clip: median cue 10.2s, longest
# 29.2s) is what makes a frame-accurate caption still FEEL out of sync — the
# text on screen has words in it the voice has not reached yet. Each phrase is
# therefore cut into short cues at the provider's own word timestamps: at most
# WORD_CUE_WORDS canon words, or at a WORD_CUE_GAP articulation break,
# whichever comes first. Cue text comes from the same canon span
# _phrase_split resolved for the phrase, so nothing is lost, repeated or
# reordered. No duration cap: a long madd hold must KEEP its word on screen.
WORD_CUE_WORDS = 3
WORD_CUE_GAP = 0.25


def snap_words_to_silence(
    words: list[dict], runs: list[tuple[float, float]], tol: float = 0.4
) -> list[dict]:
    """Snap Whisper word boundaries to real silence edges (stable-ts /
    whisper-sync style). Whisper's timestamp head drifts 200-500ms and bleeds
    word ends into silence — a word whose end falls right before a silence run
    is moved to the run's start, and a word whose start falls right after a
    run is moved to the run's end. Boundaries that don't touch a silence edge
    are left alone (snapping only ever pulls a bled boundary back to the
    edge, never stretches audio).

    A word whose stamped span fully CONTAINS a real breath pause (>=
    SILENCE_SPLIT of true silence strictly inside the word) is mis-stamped:
    no single word contains a breath. Its end is truncated to the pause start
    (reciters pause AFTER words, so the utterance belongs before the breath).
    Without this, the caption would ride straight through the pause and no
    phrase split could fire (both neighbors sit outside the run). Madd is
    voiced, so VAD never reports a run inside it — this only bites on
    drifted stamps.

    Finally, any word still covering a pause MIDPOINT (drifted stamp
    straddling the breath) is shrunk away from the midpoint: a word centered
    before it is truncated to the pause start, one centered after is pushed
    to the pause end (ties go before — reciters pause after words). The word
    may collapse to zero length; its text still captions via the canon slice,
    but it can no longer hold the overlay up mid-pause. Returns new word
    dicts (input untouched)."""
    if not runs:
        return words
    out = []
    for w in words:
        w = dict(w)
        s, e = float(w["start"]), float(w["end"])
        for rs, re in runs:
            if rs > e + tol or re < s - tol:
                continue
            if abs(e - rs) <= tol:
                e = min(e, rs)
            if abs(s - re) <= tol:
                s = max(s, re)
            # epsilon: float jitter (1.2-1.0 = 0.1999...) must not flip the
            # exact-boundary comparison
            if re - rs >= SILENCE_SPLIT - 1e-9 and rs - s >= 0.2 - 1e-9 and e > rs:
                e = min(e, rs)
        if e < s:
            e = s
        w["start"] = round(s, 3)
        w["end"] = round(e, 3)
        out.append(w)
    # Midpoint pass: no word may cover the middle of a real breath pause.
    big = [(rs, re) for rs, re in runs if re - rs >= SILENCE_SPLIT - 1e-9]
    for w in out:
        s, e = float(w["start"]), float(w["end"])
        for rs, re in big:
            mid = (rs + re) / 2
            if not (s < mid < e):
                continue
            if (s + e) / 2 <= mid:
                e = min(e, rs)
                if e < s:
                    e = s
            else:
                s = max(s, re)
                if s > e:
                    s = e
            w["start"] = round(s, 3)
            w["end"] = round(e, 3)
    return out


def _has_madd(raw_word: str) -> bool:
    """True when a RAW Uthmani word can carry a held elongation: an explicit
    madd mark (maddah U+0653, superscript alef U+0670) or a long-vowel letter
    (the usual verse-final madd-عارض landing spot: نستعين، الرحيم). Raw text
    only — _normalize strips every one of these marks, so this must run on
    seg["text_ar"] (which is raw Uthmani), never on normalized text."""
    if not raw_word:
        return False
    for ch in raw_word:
        if ch in _MADD_MARKS or ch in _MADD_VOWELS:
            return True
    return False


def _madd_extend(end: float, raw_word: str, verse_final: bool,
                 silence_runs: list | None,
                 raw_segs: list[tuple[float, float]],
                 cap_next: float) -> float:
    """Hold a caption over a madd elongation the provider cut early. Extends
    end into continuing VOICING only: the same raw VAD segment must still
    cover end (madd is voiced — a segment edge here means voicing stopped and
    there is nothing to hold). Stops at the first real silence, at the VAD
    segment offset, at cap_next (next phrase start - MIN_GAP), or after
    MADD_FINAL_EXTEND (verse-final phrase) / MADD_MID_EXTEND. Returns end
    unchanged when the last word carries no madd evidence. Rounded."""
    if not raw_segs or not _has_madd(raw_word):
        return end
    covers = [re for rs, re in raw_segs if rs <= end <= re]
    if not covers:
        return end
    limit = min(end + (MADD_FINAL_EXTEND if verse_final else MADD_MID_EXTEND),
                max(covers), cap_next)
    if silence_runs:
        for rs, re in silence_runs:
            if rs > end - 0.03 and rs < limit:
                limit = min(limit, rs)
    return round(max(end, limit), 3)


def _vad_clamp_bounds(
    start: float, end: float,
    first: dict, last: dict,
    raw_segs: list[tuple[float, float]],
) -> tuple[float, float]:
    """Snap a phrase's word-derived (start, end) to real speech edges: the
    overlapping raw-VAD segment's onset/offset. Provider word stamps are
    ±200-500ms noisy in both directions while Silero onsets/offsets sit
    within tens of ms of true voicing, so the VAD edge wins.

    Two tolerances: a word edge sitting INSIDE a speech segment snaps to
    that segment's edge up to VAD_EDGE_SNAP (same-segment = continuous
    voicing, no silence between — recovers a misheard first word that would
    otherwise start the caption ~1 word late, and holds a madd elongation
    Whisper cut early instead of dropping the caption mid-vowel). A word
    edge in silence falls back to the nearest neighbor edge within the
    stricter VAD_CLAMP_TOL (bigger gaps mean the word belongs to a
    neighboring phrase and the stamp is kept). Returns rounded bounds."""
    try:
        containing_start = [rs for rs, re in raw_segs
                            if rs <= first["start"] <= re]
        if containing_start:
            cand = min(containing_start)
            if abs(cand - start) <= VAD_EDGE_SNAP:
                start = cand
        else:
            upcoming = [rs for rs, re in raw_segs
                        if re >= first["start"] - VAD_CLAMP_TOL
                        and rs <= first["end"]]
            cand = min(upcoming) if upcoming else None
            if cand is not None and abs(cand - start) <= VAD_CLAMP_TOL:
                start = cand
    except (KeyError, TypeError, ValueError):
        pass
    try:
        containing_end = [re for rs, re in raw_segs
                          if rs <= last["end"] <= re]
        if containing_end:
            cand = max(containing_end)
            if abs(cand - end) <= VAD_EDGE_SNAP:
                end = cand
        else:
            preceding = [re for rs, re in raw_segs
                         if rs <= last["end"] + VAD_CLAMP_TOL
                         and re >= last["start"]]
            cand = max(preceding) if preceding else None
            if cand is not None and abs(cand - end) <= VAD_CLAMP_TOL:
                end = cand
    except (KeyError, TypeError, ValueError):
        pass
    return round(start, 3), round(end, 3)


def _trim_texts(text_ar: str, text_en: str,
                off: tuple[int, int] | None) -> tuple[str, str]:
    """Clip display texts to the recited word range (mid-ayah clip cuts).
    Canon words are never reworded — only the unrecited head/tail is cut.
    EN uses the proportional rule (EN word counts differ from AR)."""
    if not off:
        return text_ar, text_en
    ar_words = text_ar.split()
    if not ar_words:
        return text_ar, text_en
    first, last = off
    ar_total = len(ar_words)
    ar = ar_words[first:last]
    en_words = text_en.split()
    if en_words and len(en_words) != ar_total:
        f = round(first * len(en_words) / ar_total)
        l = round(last * len(en_words) / ar_total)
        en = en_words[f:l]
    else:
        en = en_words[first:last]
    return " ".join(ar), " ".join(en)


def _silence_between(
    a: dict, b: dict, runs: list[tuple[float, float]], pad: float = 0.12
) -> float:
    """Real audio silence (seconds) separating word a from word b: overlap of
    the silence runs with the [a.end, b.start] region, padded slightly so a
    run that ends 50ms into a word's tail still counts (silence edges are
    frame-quantized)."""
    lo = float(a["end"]) - pad
    hi = float(b["start"]) + pad
    total = 0.0
    for rs, re in runs:
        if re <= lo or rs >= hi:
            continue
        total += min(re, hi) - max(rs, lo)
    return total


def _canon_forms(raw: str) -> set[str]:
    """Normalized lookup forms of one canon word. A vocative compound
    (يَـٰٓأَيُّهَا = يَا أَيُّهَا) is ONE canon word but TWO ASR tokens, so the
    split forms are registered too — otherwise the ASR's "يا"/"أيها" matches
    neither the ayah they open nor its predecessor."""
    n = _normalize(raw)
    forms = {n, _normalize_std(raw)}
    if "\u0670" in raw and n.startswith("يا") and len(n) > 3:
        forms.add("يا")
        forms.add("ا" + n[2:])
    return forms


def _map_words_to_canon(ws: list[dict], text_ar: str) -> list[int | None] | None:
    """Map ASR words to their canon words inside one ayah's text.

    Returns one entry per ASR word: the index of the canon word it recites, or
    None when the word could not be matched. Used to cut the DISPLAY text of a
    breath phrase at the exact words being recited instead of by proportional
    word counts: a reciter who REPEATS a passage makes the word count longer
    than the canon ayah, so proportional slicing shifts every later phrase's
    text ahead of the voice (measured on 59:15: caption read "ذاقوا وبال"
    while the reciter was saying "كمثل الذين من قبلهم").

    Matching is forward-preferring but allows a jump BACK (a repeat), and
    falls back to a fuzzy window around the expected position for misheard
    words. Returns None when the ayah text is empty."""
    canon = text_ar.split()
    if not canon or not ws:
        return None
    lookup: dict[str, list[int]] = {}
    for i, raw in enumerate(canon):
        for v in _canon_forms(raw):
            lookup.setdefault(v, []).append(i)
    # per-index comparable form for the fuzzy fallback
    uth = [_normalize(w) for w in canon]
    std = [_normalize_std(w) for w in canon]

    out: list[int | None] = []
    pos = 0
    for w in ws:
        t = _normalize(w.get("text") or "")
        if not t:
            out.append(None)
            continue
        cands = lookup.get(t)
        if cands:
            # nearest canon occurrence; forward wins a tie (normal recitation
            # advances word by word, a repeat jumps back).
            j = min(cands, key=lambda c: (abs(c - pos), c < pos))
            out.append(j)
            pos = j + 1
            continue
        # fuzzy window around where the next word should be
        lo = max(0, pos - 10)
        hi = min(len(canon), pos + 10)
        best, best_score = None, 0.0
        for j in range(lo, hi):
            score = max(fuzz.ratio(t, uth[j]), fuzz.ratio(t, std[j]))
            if score > best_score:
                best, best_score = j, score
        if best is not None and best_score >= 78:
            out.append(best)
            pos = best + 1
        else:
            out.append(None)
    return out


def _phrase_split(seg: dict, words: list[dict], silence_runs: list | None = None,
                  raw_segs: list[tuple[float, float]] | None = None,
                  canon_pos: list[int | None] | None = None,
                  trust_times: bool = False) -> list[dict]:
    """Split one ayah segment into breath-phrase segments at REAL pauses
    (inter-word silences >= SILENCE_SPLIT measured from the audio), so the
    caption vanishes while the reciter breathes and resumes where they resume.

    words: chronological word timestamps {"start", "end"} for this ayah
    (they may carry "text"; unused here). silence_runs: (start, end) silence
    regions from backend.vad.silence_runs — the pause ground truth. An empty
    list means the audio is truly continuous (VAD found no pause), so nothing
    splits; only None (no audio data, e.g. tests) falls back to the Whisper
    word gap as weak evidence. Canon text is re-sliced to each phrase by
    word-count proportion (recitation order == text order); English uses the
    same proportional cuts, snapped to the nearest sentence break so
    translations don't split mid-sentence. canon_pos (provider path) replaces
    the proportional cut with the exact canon span each phrase recites, so a
    REPEATED passage cannot shift the text ahead of the voice. raw_segs: raw
    (unmerged) VAD speech segments — when given (provider path), each
    phrase's word-derived bounds are VAD-clamped to real speech edges
    (provider word stamps are ±200-500ms noisy; the local forced-align path
    skips this because wav2vec2 times are already frame-accurate).

    Audio-evidenced cuts are ALWAYS kept — even a 1-word phrase before a real
    breath shows briefly then blanks, and never bridges the pause. Tiny-phrase
    merging applies to the fallback path only (fake whisper gaps must not
    shatter captions). With raw_segs, a phrase ending in a madd word is HELD
    past an early provider cut into continuing voicing (_madd_extend) so the
    caption survives long mujawwad elongations instead of vanishing mid-vowel.
    trust_times: the word stamps are already phoneme-accurate (QUD aligner) and
    include the madd — skip the VAD clamp and the madd hold, which could
    otherwise pull a cue back onto a previous held elongation.

    Returns [seg] unchanged when there is no pause to split on."""
    if len(words) < 2:
        return [seg]
    ws = sorted(words, key=lambda w: float(w["start"]))
    if silence_runs is not None:
        gaps = [
            _silence_between(ws[i], ws[i + 1], silence_runs)
            for i in range(len(ws) - 1)
        ]
        thr = SILENCE_SPLIT
    else:
        # No audio-truth data: fall back to the raw word gap, but clear a
        # HIGHER bar — whisper stamps drift and fake gaps, so only large ones
        # are trusted to split.
        gaps = [float(ws[i + 1]["start"]) - float(ws[i]["end"]) for i in range(len(ws) - 1)]
        thr = PAUSE_GAP
    # epsilon: float subtraction jitter (e.g. 1.95-1.35 = 0.5999...) must not
    # flip the exact-boundary comparison
    cuts = [i + 1 for i, g in enumerate(gaps) if g >= thr - 1e-9]
    if not cuts:
        if raw_segs:
            s, e = float(seg["start"]), float(seg["end"])
            if not trust_times:
                s, e = _vad_clamp_bounds(
                    seg["start"], seg["end"], ws[0], ws[-1], raw_segs)
                # Single-phrase ayah (the common verse-final madd case): hold a
                # madd tail the provider cut early. verse_final: one region covers
                # the whole segment, whose text is the full canon ayah here.
                raw_tail = (seg.get("text_ar") or "").split()
                e = _madd_extend(e, raw_tail[-1] if raw_tail else "",
                                 True, silence_runs, raw_segs, float("inf"))
            # _ws keeps this phrase's OWN words for the word-cue pass: a madd
            # hold can push the phrase end past the next ayah's first word, and
            # time-window membership would then pull that foreign word in.
            seg = dict(seg, start=s, end=e, _ws=ws)
        return [seg]

    # Regions between cuts.
    regions: list[list[int]] = []
    lo = 0
    for c in cuts + [len(ws)]:
        regions.append([lo, c - 1])
        lo = c

    # Merge too-small regions (few words or short) into their neighbors —
    # FALLBACK PATH ONLY. A region that fails the threshold is absorbed by
    # the previous one, or the next one when it is leading (so one-word
    # "يا أيها" whisper-gap tails don't flash a caption and nothing is
    # dropped at the start). With audio truth every measured pause is a real
    # breath, so small regions stay split and the caption blanks on time.
    def small(r: list[int]) -> bool:
        wc = r[1] - r[0] + 1
        dur = float(ws[r[1]]["end"]) - float(ws[r[0]]["start"])
        # epsilon: float subtraction jitter (e.g. 3.3-2.9 = 0.3999...) must
        # not flip the exact-boundary comparison
        return wc < MIN_PHRASE_WORDS or dur < MIN_PHRASE_DUR - 1e-6

    if silence_runs is None:
        i = 0
        while i < len(regions):
            if not small(regions[i]):
                i += 1
                continue
            if i > 0:  # absorb into previous
                regions[i - 1][1] = regions[i][1]
                del regions[i]
            elif i + 1 < len(regions):  # fold leading into next
                regions[i + 1][0] = regions[i][0]
                del regions[i]
            else:  # only region -> nothing to split
                return [seg]
    if len(regions) <= 1:
        return [seg]

    text_ar = seg.get("text_ar") or ""
    text_en = seg.get("text_en") or ""
    ar_words = text_ar.split()
    en_words = text_en.split()
    total = len(ws)
    n_ar, n_en = len(ar_words), len(en_words)

    def cut(seq_len: int, hi: int) -> int:
        k = int(seq_len * (hi + 1) / total)
        return max(0, min(seq_len, k))

    # Repeat-proof text spans: cut each phrase's display text at the canon
    # words it actually recites. A repeated passage makes the ASR word count
    # exceed the canon word count, so proportional cutting shifts every later
    # phrase's text ahead of the voice. Only used when EVERY phrase has at
    # least one recognised word — otherwise proportional stays the safe
    # default (a misheard phrase must never blank a language).
    n_regions = len(regions)
    spans: list[tuple[int, int]] | None = None
    if canon_pos and len(canon_pos) == total and n_ar:
        per: list[tuple[int, int]] = []
        for idx, (lo, hi) in enumerate(regions):
            got = [p for p in canon_pos[lo:hi + 1] if p is not None]
            if not got:
                per = []
                break
            m0, m1 = min(got), max(got)
            if idx == 0 and m0 <= 2:
                m0 = 0                 # recover a misheard head (not a mid-ayah clip)
            if idx == n_regions - 1 and (n_ar - 1 - m1) <= 2:
                m1 = n_ar - 1          # recover a misheard tail (not a repeated ending)
            per.append((m0, m1 + 1))
        if per:
            # A phrase must begin where its predecessor ended: a canon word
            # the ASR misheard between two phrases maps to no word, so
            # min(got) would step past it and the word would never be shown
            # (measured: 59:14 lost جُدُرِۭۚ between two breath phrases).
            fixed, prev_end = [], 0
            for a, b in per:
                fixed.append((min(a, prev_end), b))
                prev_end = b
            spans = fixed

    out = []
    prev_b_ar = 0
    prev_b_en = 0
    for idx, (lo, hi) in enumerate(regions):
        last = idx == n_regions - 1
        if spans is not None:
            a_ar, b_ar = spans[idx]
            a_en = (round(a_ar * n_en / n_ar) if n_en else 0)
            b_en = (round(b_ar * n_en / n_ar) if n_en else 0)
            if n_en and b_en <= a_en:
                b_en = min(a_en + 1, n_en)
            prev_b_ar, prev_b_en = b_ar, b_en
        else:
            # Reserve one word per remaining phrase so no phrase is starved:
            # the proportional cut must never consume words a later phrase
            # still needs.
            regions_left = n_regions - idx - 1
            if last:
                # Last phrase takes EVERYTHING left in both languages, so the
                # tail of the ayah can never lose its text (this was the
                # "English did not show" bug: the final 1-2 Arabic words were
                # emitted with an empty English slice because the proportional
                # EN cut had already been consumed by the previous phrase).
                b_ar, b_en = n_ar, n_en
            else:
                b_ar = min(cut(n_ar, hi), n_ar - regions_left)
                b_en_plain = cut(n_en, hi)
                b_en = _snap_en_boundary(en_words, prev_b_en, b_en_plain)
                b_en = min(b_en, n_en - regions_left)
            a_ar, a_en = prev_b_ar, prev_b_en
            # Never emit a phrase with a language it still has words for.
            if n_ar and b_ar <= a_ar:
                b_ar = min(a_ar + 1, n_ar)
            if n_en and b_en <= a_en:
                b_en = min(a_en + 1, n_en)
            prev_b_ar, prev_b_en = b_ar, b_en
        phrase = dict(seg)
        pstart = round(float(ws[lo]["start"]), 3)
        pend = round(float(ws[hi]["end"]), 3)
        if raw_segs and not trust_times:
            pstart, pend = _vad_clamp_bounds(
                pstart, pend, ws[lo], ws[hi], raw_segs)
            # Hold a madd tail the provider cut early (mid-verse cap, bigger
            # verse-final cap). Capped at the next phrase's start so regions
            # never overlap here either (apply_sync_padding re-enforces).
            cap_next = (float(ws[regions[idx + 1][0]]["start"]) - MIN_GAP
                        if not last else float("inf"))
            pend = _madd_extend(pend, ar_words[b_ar - 1] if b_ar else "",
                               last, silence_runs, raw_segs, cap_next)
        phrase.update({
            "start": pstart,
            "end": pend,
            "text_ar": " ".join(ar_words[a_ar:b_ar]).strip(),
            "text_en": " ".join(en_words[a_en:b_en]).strip(),
        })
        if raw_segs:
            phrase["_ws"] = [ws[i] for i in range(lo, hi + 1)]
        out.append(phrase)
    return out


def _snap_en_boundary(en_words: list[str], a: int, b: int) -> int:
    """Nudge an English proportional cut to the nearest word ending a sentence
    (., !, ?) within a small window, so phrases split at sentence breaks."""
    if b <= a + 1 or b >= len(en_words):
        return b
    lo, hi = max(a + 1, b - 3), min(len(en_words), b + 3)
    for j in range(lo, hi + 1):
        prev = en_words[j - 1].rstrip('"')
        if prev and prev[-1] in _EN_BREAK_CHARS:
            return j
    return b


def split_segments_at_pauses(
    segments: list[dict], words_by_ayah: dict, silence_runs: list | None = None
) -> list[dict]:
    """Apply phrase splitting to every segment. words_by_ayah: seg key ->
    chronological word timestamps for that ayah. silence_runs: audio-truth
    silence regions (backend.vad.silence_runs); when given, pause evidence
    comes from measured silence instead of Whisper word gaps. Segments
    without entries are kept as-is (e.g. openers with no word times
    available)."""
    out = []
    for seg in segments:
        key = seg.get("surah"), seg.get("ayah")
        words = words_by_ayah.get(key)
        out.extend(_phrase_split(seg, words, silence_runs) if words else [seg])
    return out


def apply_sync_padding(
    segments: list[dict], silence_runs: list | None = None
) -> list[dict]:
    """Final caption timing polish: a tiny lead-in (~30ms) so text is visible
    a hair before the first word, and a short lead-out (~120ms) so the caption
    lingers over the reciter's held final syllable. Enforces non-overlap +
    min duration.

    silence_runs: audio-truth pauses — padding NEVER bleeds into them. A
    caption whose lead-out would spill into a pause is pulled back to the
    pause edge, and one that would pop in mid-pause is pushed to the pause
    end, so text is strictly blank while the reciter breathes. MIN_DUR and
    the raw word end never cover the NEXT caption's onset either: every end
    is capped at the next caption's start − MIN_GAP first, so a forced
    minimum duration can delay nothing — it only fills silence-free room
    that is actually free. Without the cap, a 0.5s minimum on a crowded
    phrase punched past the next onset and the next caption was pushed late
    (appears after the voice) and squeezed to a flash/zero-length cue the
    live player can never display (`t < end` is never true).

    Every emitted cue is guaranteed DISPLAYABLE (start < end): the live player
    matches on `t >= start && t < end`, so a zero-length cue is never shown at
    all. When an upstream phrase has no free room (its start was pushed onto or
    past its own end because the previous caption's end ran into it), its text
    is CARRIED FORWARD into the next caption that does show, instead of being
    emitted as an invisible stub — no recited moment is ever left uncaptioned.
    Real speech alignment is the job of the ASR path upstream; this function
    only nudges *display* timing."""
    segs = sorted(segments, key=lambda s: s["start"])
    runs = sorted(silence_runs) if silence_runs else []

    def pull_end(t: float) -> float:
        for rs, re in runs:
            if rs - 0.03 <= t < re:
                return min(t, rs)
        return t

    def push_start(t: float) -> float:
        for rs, re in runs:
            if rs < t <= re + 0.03:
                return max(t, re)
        return t

    raw_end = [seg["end"] for seg in segs]
    pushed_start = [
        push_start(max(0.0, round(seg["start"] - LEAD_IN, 3))) for seg in segs
    ]

    out: list[dict] = []
    prev_end = 0.0
    # Text of phrases that had no displayable room, carried forward into the
    # next cue that does show. Order-preserving: no recited text is lost.
    carry_ar: list[str] = []
    carry_en: list[str] = []

    def absorb(target: dict, src: dict) -> dict:
        merged = dict(target)
        for key in ("text_ar", "text_en"):
            a = (merged.get(key) or "").strip()
            b = (src.get(key) or "").strip()
            if b and b not in a:
                # The carried cue is EARLIER in the recitation, so its text
                # has to read first: appending it produced the next ayah's
                # word followed by the squeezed one (measured: 108:1/2 read
                # "فَصَلِّ ٱلْكَوْثَرَ" where the voice says الكوثر then فصل).
                merged[key] = (b + " " + a).strip() if a else b
        return merged

    for i, seg in enumerate(segs):
        start = max(round(prev_end + MIN_GAP, 3), pushed_start[i])
        # Next caption's onset is sacred: nothing here may cover it. The next
        # pushed start only ever moves forward (prev_end clamp above), so
        # capping at it can leave a wider gap but never an overlap.
        ncap = (round(pushed_start[i + 1] - MIN_GAP, 3)
                if i + 1 < len(segs) else float("inf"))
        end = pull_end(min(round(raw_end[i] + LEAD_OUT, 3), ncap))
        end = max(end,
                  pull_end(min(round(raw_end[i], 3), ncap)),
                  pull_end(min(round(start + MIN_DUR, 3), ncap)))
        if end <= start and i + 1 < len(segs):
            # No displayable room: this cue would be zero-length and thus
            # invisible forever (`t < end` never true). Carry its text into
            # the NEXT cue that shows instead of emitting a stub.
            carry_ar.append(seg.get("text_ar") or "")
            carry_en.append(seg.get("text_en") or "")
            continue
        if end <= start:
            # Last phrase with no room: keep a minimal displayable span so the
            # final words still appear (nothing follows to absorb them).
            end = round(start + max(MIN_DUR, 0.12), 3)
        if carry_ar or carry_en:
            seg = absorb(seg, {"text_ar": " ".join(carry_ar),
                               "text_en": " ".join(carry_en)})
            carry_ar, carry_en = [], []
        seg["start"] = start
        seg["end"] = end
        out.append(seg)
        prev_end = end
    return out


def build_segments(
    surah: int,
    start_ayah: int,
    end_ayah: int,
    chunks: list[dict],
) -> list[dict]:
    """Map ASR chunks to ayahs (greedy sequential matching against canonical
    Quran text) and return ayah-level segments with start/end times."""
    texts_ar = load_ayah_texts()
    texts_en = load_english_texts()

    canon: list[tuple[int, str]] = []
    for ayah in range(start_ayah, end_ayah + 1):
        text = texts_ar.get(f"{surah}:{ayah}", "")
        for word in _normalize(text).split():
            canon.append((ayah, word))

    if not canon:
        return []

    assigned: list[tuple[float, float, int, int]] = []
    cursor = 0
    for chunk in chunks:
        text = _normalize(chunk["text"])
        words = text.split()
        if not words:
            continue

        best_pos, best_score = None, 0.0
        max_window = min(len(canon), cursor + max(len(words) + 2, 8))
        for pos in range(cursor, max_window):
            end_pos = min(pos + len(words), len(canon))
            slice_text = " ".join(canon[i][1] for i in range(pos, end_pos))
            score = fuzz.ratio(text, slice_text)
            if score > best_score:
                best_score, best_pos = score, pos

        if best_pos is not None and best_score >= 55:
            end_pos = min(best_pos + len(words), len(canon))
            assigned.append((chunk["start"], chunk["end"], best_pos, end_pos))
            cursor = max(cursor, end_pos)

    ayah_times: dict[int, list[float]] = {}
    for cs, ce, bp, ep in assigned:
        ayah = Counter(canon[i][0] for i in range(bp, ep)).most_common(1)[0][0]
        if ayah not in ayah_times:
            ayah_times[ayah] = [cs, ce]
        else:
            ayah_times[ayah][1] = ce

    segments = []
    for ayah in sorted(ayah_times):
        start, end = ayah_times[ayah]
        segments.append({
            "surah": surah,
            "ayah": ayah,
            "start": round(start, 3),
            "end": round(end, 3),
            "text_ar": texts_ar.get(f"{surah}:{ayah}", ""),
            "text_en": texts_en.get(f"{surah}:{ayah}", ""),
        })
    return segments


def _opener_word_matches(word: str, canon: str, std: str) -> bool:
    """Tolerant single-word match: ASR may spell a superscript-alif word with a
    plain alif (الشيطان vs canon الشيطن), or drop an alif (الرحمان vs
    الرحمن). Accept the strict form or the plain-alif (_normalize_std) form."""
    if not word or not canon:
        return False
    n = _normalize(word)
    return n == canon or n == std or n == _normalize(std)


def _leading_opener(words: list[dict]) -> tuple[str, str, str, float, float] | None:
    """If the word list opens with a reciter opener (takbir / isti'adha /
    Bismillah) that detector.py strips from matching, return its display
    texts + times so it can still be captioned. Leading-only for takbir and
    isti'adha (they are legitimate canon mid-clip); Bismillah everywhere
    (detector strips every occurrence)."""
    for kind, canon, text_ar, text_en in OPENERS:
        if len(words) < len(canon):
            continue
        std = _STD_WORDS[kind]
        if all(
            _opener_word_matches(words[i]["text"], canon[i], std[i])
            for i in range(len(canon))
        ):
            return kind, text_ar, text_en, words[0]["start"], words[len(canon) - 1]["end"]
    return None


def match_leading_openers(tokens: list[str]) -> list[str]:
    """Which opener phrases LEAD a decoded token list, in recitation order
    (takbir -> isti'adha -> Bismillah). Tolerant of ASR spelling variants;
    returns [] when the text does not start with any opener. Pure function —
    no models, unit-testable."""
    canon_by_kind = {kind: canon for kind, canon, *_rest in OPENERS}
    found: list[str] = []
    idx = 0
    for kind in ("takbir", "istiadha", "bismillah"):
        canon = canon_by_kind[kind]
        std = _STD_WORDS[kind]
        if idx + len(canon) > len(tokens):
            continue
        if all(
            _opener_word_matches(tokens[idx + i], canon[i], std[i])
            for i in range(len(canon))
        ):
            found.append(kind)
            idx += len(canon)
    return found


def inject_opener_segments(
    matches: list, words: list[dict], segments: list[dict],
    silence_runs: list | None = None,
    raw_segs: list[tuple[float, float]] | None = None,
) -> list[dict]:
    """Caption reciter openers (leading takbir / isti'adha, every Bismillah)
    that are otherwise stripped from matching: they ARE audible recitation and
    viewers expect a caption from the first word. Opener segments reuse the
    canon Fatihah 1:1 text for Bismillah (it IS that text); takbir/isti'adha
    use plain display text. Tagged surah = the match the opener precedes."""
    if not words or not segments:
        return segments
    texts_ar = load_ayah_texts()
    texts_en = load_english_texts()
    bi_ar = texts_ar.get(BISMILLAH_AYAH, "") or OPENERS[0][2]
    bi_en = texts_en.get(BISMILLAH_AYAH, "") or OPENERS[0][3]

    added = []
    n = len(words)
    i = 0
    lead_run = 0  # words consumed by contiguous leading openers
    while i < n:
        opener = _leading_opener(words[i:])
        if opener is None:
            i += 1
            continue
        kind, text_ar, text_en, start, end = opener
        # Takbir / isti'adha only caption as a leading run — mid-clip they are
        # legitimate canon (9:72, 29:45, ...) and detector keeps them there;
        # captioning them again would double-caption real ayahs. Bismillah is
        # everywhere (detector strips it everywhere, so it must be re-injected
        # at every occurrence).
        if kind != "bismillah" and i != lead_run:
            i += 1
            continue
        kind_idx = next(k for k, (k2, *_rest) in enumerate(OPENERS) if k2 == kind)
        if kind == "bismillah":
            text_ar, text_en = bi_ar, bi_en
            ayah = 1
        else:
            ayah = 0
        opener_len = len(OPENERS[kind_idx][1])
        added.append((i, ayah, float(start), float(end), text_ar, text_en,
                      words[i:i + opener_len]))
        i += opener_len
        if kind != "bismillah":
            lead_run = i
    if not added:
        return segments

    ordered = sorted(
        matches, key=lambda m: (m.chain[0][0] if getattr(m, "chain", None) else 0)
    )
    first_surah = ordered[0].surah if ordered else 1

    def surah_after(idx: int) -> int:
        for m in ordered:
            chain = getattr(m, "chain", None) or []
            if chain and chain[0][0] > idx:
                return m.surah
        return first_surah

    for idx, ayah, start, end, text_ar, text_en, opener_words in added:
        seg = {
            "surah": surah_after(idx),
            "ayah": ayah,
            "start": round(start, 3),
            "end": round(end, 3),
            "text_ar": text_ar,
            "text_en": text_en,
        }
        # Openers get phrase-split on their own word times too (a Bismillah
        # recited in two breaths becomes two captions).
        segments.extend(_phrase_split(seg, opener_words, silence_runs, raw_segs))
    return segments


def segments_from_word_times(
    surah: int,
    start_ayah: int,
    end_ayah: int,
    word_times: list[dict],
    word_off: dict[int, tuple[int, int]] | None = None,
    silence_runs: list | None = None,
) -> list[dict]:
    """Group forced-aligned word times into ayah segments. Opener tags from
    the local forced-align path: ayah -1 = takbir, -2 = isti'adha, 0 =
    Bismillah — each becomes its own captioned segment (with EN) so reciter
    openers are never dropped from the captions.

    word_off: ayah -> (first_canon_word, last_canon_word+1) actually recited
    (mid-ayah clip cuts). Display text is trimmed to that range so captions
    never show unrecited words."""
    texts_ar = load_ayah_texts()
    texts_en = load_english_texts()

    word_off = word_off or {}
    opener_texts = {kind: (ar, en) for kind, _c, ar, en in OPENERS}

    def seg_key(ayah: int) -> tuple[int, int]:
        if ayah <= 0:  # takbir -1, isti'adha -2, opening Bismillah 0
            return 1, ayah
        return surah, ayah

    def seg_display(key: tuple[int, int]) -> tuple[int, int]:
        # -2 isti'adha / -1 takbir -> 1:0; 0 Bismillah -> 1:1 (canon Fatihah)
        if key[1] == 0:
            return 1, 1
        if key[1] < 0:
            return 1, 0
        return key

    words_by_ayah: dict[tuple[int, int], list[dict]] = {}
    for w in word_times:
        ayah = int(w["ayah"])
        words_by_ayah.setdefault(seg_key(ayah), []).append(w)

    segments = []
    for key, ws in words_by_ayah.items():
        start = min(float(w["start"]) for w in ws)
        end = max(float(w["end"]) for w in ws)
        seg_surah, seg_ayah = seg_display(key)
        if seg_ayah <= 0 or (seg_surah, seg_ayah) == (1, 1) and key[1] == 0:
            opener_kind = {0: "bismillah", -1: "takbir", -2: "istiadha"}.get(key[1], "bismillah")
            text_ar, text_en = opener_texts[opener_kind]
        else:
            text_ar = texts_ar.get(f"{seg_surah}:{seg_ayah}", "")
            text_en = texts_en.get(f"{seg_surah}:{seg_ayah}", "")
        off = word_off.get(seg_ayah)
        if off:
            first, last = off
            # a mid-word clip cut drops the final aligned word(s) — never
            # display text beyond what actually has audio
            last = min(last, first + len(ws))
            off = (first, last)
        text_ar, text_en = _trim_texts(text_ar, text_en, off)
        seg = {
            "surah": seg_surah,
            "ayah": seg_ayah,
            "start": round(start, 3),
            "end": round(end, 3),
            "text_ar": text_ar,
            "text_en": text_en,
        }
        # Key must stay the TAG (1,-2)/(1,-1)/(1,0) — the display ayah (1,0)
        # is shared by takbir and isti'adha and would merge their words.
        segments.extend(_phrase_split(seg, ws, silence_runs))
    return apply_sync_padding(segments, silence_runs)


def _cue_groups(ws: list[dict], cpos: list[int | None]) -> list[list[int]]:
    """Group one phrase's words into display cues: at most WORD_CUE_WORDS
    canon words, or up to a WORD_CUE_GAP articulation break. A repeat (canon
    index jumping back) always starts a new cue so its text cannot be folded
    into the passage before it."""
    groups: list[list[int]] = [[0]]
    first = cpos[0]
    for i in range(1, len(ws)):
        cur = groups[-1]
        canon = cpos[i]
        gap = float(ws[i]["start"]) - float(ws[i - 1]["end"])
        repeat = canon is not None and first is not None and canon < first
        wide = (canon is not None and first is not None
                and canon - first + 1 > WORD_CUE_WORDS)
        if gap >= WORD_CUE_GAP or repeat or wide:
            groups.append([i])
            first = canon
        else:
            cur.append(i)
    return groups


def _word_cues(seg: dict, ws: list[dict], silence_runs: list | None = None,
               raw_segs: list | None = None,
               trust_times: bool = False) -> list[dict]:
    """Cut one phrase caption into word-tracked cues (see WORD_CUE_*).

    Text stays canon and complete: each cue takes the span of canon words its
    own word stamps recite (via _map_words_to_canon, so a REPEAT cuts at the
    repeated words instead of by word count), every cue keeps at least one
    word, and the last cue takes the remainder. Timing: the first cue keeps
    the phrase onset (VAD-clamped head recovery) and the last keeps the phrase
    end (madd hold); interior cues use their own word stamps, with a madd
    word held into continuing voicing up to the next cue's onset."""
    if len(ws) < 2:
        return [seg]
    ar_words = (seg.get("text_ar") or "").split()
    en_words = (seg.get("text_en") or "").split()
    n_ar, n_en = len(ar_words), len(en_words)
    if not n_ar:
        return [seg]
    cpos = _map_words_to_canon(ws, seg.get("text_ar") or "")
    if cpos is None:
        return [seg]
    groups = _cue_groups(ws, cpos)
    while len(groups) > n_ar:  # more cues than words: fold the tail cues
        groups[-2].extend(groups.pop())

    # Canon span per cue. A cue starts at its first recognised word, but never
    # before the previous cue's end and always leaving a word for every later
    # cue; the last cue takes everything left.
    bounds = [0]
    for gi in range(1, len(groups)):
        got = [cpos[i] for i in groups[gi] if cpos[i] is not None]
        a = min(got) if got else bounds[-1] + 1
        a = max(a, bounds[-1] + 1)
        a = min(a, n_ar - (len(groups) - gi))
        bounds.append(min(n_ar, max(bounds[-1] + 1, a)))
    bounds.append(n_ar)

    cues: list[dict] = []
    prev_en = 0
    last_gi = len(groups) - 1
    for gi, g in enumerate(groups):
        a_ar, b_ar = bounds[gi], bounds[gi + 1]
        if n_en:
            a_en = prev_en
            if gi == last_gi:
                b_en = n_en
            else:
                left = last_gi - gi
                b_en = round(b_ar * n_en / n_ar)
                b_en = _snap_en_boundary(en_words, a_en, b_en)
                # Never eat an English word a later cue still needs: the snap
                # can run forward to the ayah's closing period and leave every
                # cue after it with Arabic but no translation (measured: 14).
                b_en = min(b_en, n_en - left)
                b_en = max(b_en, min(a_en + 1, n_en))
            prev_en = b_en
        else:
            a_en = b_en = 0
        start = float(ws[g[0]]["start"])
        if gi == 0:
            start = min(start, float(seg["start"]))
        end = float(ws[g[-1]]["end"])
        if gi == last_gi:
            end = float(seg["end"])
        elif raw_segs and not trust_times:
            cap_next = float(ws[groups[gi + 1][0]]["start"]) - MIN_GAP
            end = _madd_extend(end, ar_words[b_ar - 1], False,
                               silence_runs, raw_segs, cap_next)
        cue = dict(seg)
        cue.update({
            "start": round(start, 3),
            "end": round(end, 3),
            "text_ar": " ".join(ar_words[a_ar:b_ar]).strip(),
            "text_en": " ".join(en_words[a_en:b_en]).strip(),
        })
        cues.append(cue)
    return cues


def subdivide_word_cues(segments: list[dict], words: list[dict],
                        silence_runs: list | None = None,
                        raw_segs: list | None = None,
                        trust_times: bool = False) -> list[dict]:
    """Re-cut every phrase caption into word-tracked cues. Segments without
    word stamps (openers, wordless bridges) are kept as they are.

    A segment that carries "_ws" (provider path: the phrase's own recited
    words) is cut from exactly those words — a window by time would also pick
    up a neighbouring ayah's word whenever a madd hold stretches the phrase
    end past it. Legacy callers fall back to midpoint membership."""
    if not words:
        return segments
    ws_all = sorted(words, key=lambda w: float(w.get("start", 0.0)))
    mids = [(float(w.get("start", 0.0)) + float(w.get("end", 0.0))) / 2
            for w in ws_all]
    out: list[dict] = []
    for seg in segments:
        own = seg.get("_ws")
        if own is None:
            lo = bisect_left(mids, float(seg["start"]) - 0.02)
            hi = bisect_right(mids, float(seg["end"]) + 0.02)
            own = ws_all[lo:hi]
        out.extend(_word_cues(seg, own, silence_runs, raw_segs, trust_times))
    return out


def segments_from_provider_times(
    matches: list, words: list[dict], silence_runs: list | None = None,
    raw_segs: list[tuple[float, float]] | None = None,
) -> list[dict]:
    """Build ayah segments straight from hosted-ASR word timestamps.
    Each AyahMatch carries chain=[(word_idx, surah, ayah)]; words[idx] gives
    {text, start, end}. One segment per ayah, times clamped to the recited
    words — no forced alignment needed. silence_runs: audio-truth pause
    evidence for phrase splitting (provider word times drift into silence).
    raw_segs: raw (unmerged) VAD speech segments — phrase bounds are
    VAD-clamped to real speech edges (provider word stamps are noisy).
    Display text is the full canon ayah text (as before): the detector
    chain routinely skips misheard words mid-ayah, so it cannot reliably
    tell unrecited words apart from misheard-but-recited ones — trimming to
    it would hide recited words (measured: 20/24 ayahs gutted on a full
    recitation). Times still come only from the recited words."""
    texts_ar = load_ayah_texts()
    texts_en = load_english_texts()

    segments: list[dict] = []
    all_word_groups: list[tuple[dict, list[dict]]] = []
    for m in matches:
        chain = getattr(m, "chain", None) or []
        if not chain:
            continue
        words_by_ayah: dict[tuple[int, int], list[dict]] = {}
        used_idx: set[int] = set()
        for word_idx, surah, ayah in chain:
            if not (0 <= word_idx < len(words)):
                continue
            words_by_ayah.setdefault((surah, ayah), []).append(words[word_idx])
            used_idx.add(word_idx)

        ordered = sorted(
            words_by_ayah.items(), key=lambda kv: float(kv[1][0]["start"])
        )
        _close_chain_holes(ordered, words, used_idx, texts_ar)

        for (surah, ayah), ws in ordered:
            start = min(float(w.get("start", 0.0)) for w in ws)
            end = max(float(w.get("end", start)) for w in ws)
            seg = {
                "surah": surah,
                "ayah": ayah,
                "start": round(start, 3),
                "end": round(end, 3),
                "text_ar": texts_ar.get(f"{surah}:{ayah}", ""),
                "text_en": texts_en.get(f"{surah}:{ayah}", ""),
            }
            segments.append(seg)
            all_word_groups.append((seg, sorted(ws, key=lambda w: float(w["start"]))))

    segments = []
    for seg, ws in all_word_groups:
        cpos = _map_words_to_canon(ws, seg.get("text_ar") or "")
        segments.extend(_phrase_split(seg, ws, silence_runs, raw_segs, cpos))
    segments.sort(key=lambda s: s["start"])
    # Openers first: a Bismillah's words are stripped from the detector chain,
    # so before injection the gap they occupy looks wordless and
    # _close_wordless_holes paints the PREVIOUS ayah's text over the reciter's
    # opener (measured: Kawthar's Bismillah announced 112:4). Injecting them
    # here also submits them to subdivide_word_cues like any other phrase.
    segments = inject_opener_segments(matches, words, segments, silence_runs, raw_segs)
    segments.sort(key=lambda s: s["start"])
    _close_wordless_holes(segments, raw_segs, silence_runs)
    segments = subdivide_word_cues(segments, words, silence_runs, raw_segs)
    segments = apply_sync_padding(segments, silence_runs)
    for s in segments:
        s.pop("_ws", None)
    return segments


def _close_chain_holes(ordered, words, used_idx: set[int], texts_ar) -> None:
    """Give every recited word a caption by closing the detector's holes.

    The detector's chain is sparse: a word the ASR misheard, or a passage
    the reciter REPEATED, is left out of it. Those words are still in the
    transcript with correct timestamps, but no segment covered them — so
    the screen went blank for those seconds while the reciter was plainly
    still reciting (measured: 19 s of an 8-minute recitation).

    For every hole between two consecutive ayah groups, the unmatched words
    are assigned to the ayah whose canon text they match best — a repeated
    passage belongs to the repeated ayah, so its caption simply extends
    over the repeat. When nothing matches (a genuinely misheard tail) they
    go to the preceding ayah, whose caption then covers its own tail. Either
    way no recited second is left uncaptioned.

    Words are also filled INSIDE a group's own time span: the chain skips a
    word mid-ayah (misheard) while its neighbours on both sides are chained,
    so it is not in any "hole between groups" yet its seconds are voiced.
    Measured on 59:2: "يحتسموا" (49.887-51.347) was skipped this way and left
    1.42 s of blank screen — the old wordless-gap bridge hid it by accident.
    """
    n = len(words)
    if not n:
        return

    # (a) chain-skipped words that sit INSIDE a group's span belong to it
    for _key, ws in ordered:
        if not ws:
            continue
        lo = min(float(w.get("start", 0.0)) for w in ws)
        hi = max(float(w.get("end", 0.0)) for w in ws)
        for j in range(n):
            if j in used_idx:
                continue
            w = words[j]
            if (lo - 0.05 <= float(w.get("start", 0.0))
                    and float(w.get("end", 0.0)) <= hi + 0.05):
                used_idx.add(j)
                ws.append(w)
        ws.sort(key=lambda w: float(w.get("start", 0.0)))

    if len(ordered) < 2:
        return

    def canon_words(surah: int, ayah: int) -> set[str]:
        txt = texts_ar.get(f"{surah}:{ayah}", "")
        forms: set[str] = set()
        for w in txt.split():
            if w:
                forms |= _canon_forms(w)
        return forms

    for i in range(len(ordered) - 1):
        (key_a, ws_a), (key_b, ws_b) = ordered[i], ordered[i + 1]
        end_a = max(float(w.get("end", 0.0)) for w in ws_a)
        start_b = min(float(w.get("start", 0.0)) for w in ws_b)
        if start_b - end_a <= 0.05:
            continue
        # the unmatched transcript words inside the hole
        hole = []
        for j in range(n):
            if j in used_idx:
                continue
            w = words[j]
            if end_a - 0.05 <= float(w.get("start", 0.0)) < start_b + 0.05:
                hole.append((j, w))
        if not hole:
            continue
        hole_norms = [_normalize(w["text"]) for _, w in hole if w.get("text")]
        if not hole_norms:
            continue
        # best-matching neighbour ayah
        best_key, best_score = None, 0.0
        for key in (key_b, key_a):
            cw = canon_words(*key)
            if not cw:
                continue
            hit = sum(1 for h in hole_norms if h in cw)
            score = hit / len(hole_norms)
            if score > best_score:
                best_key, best_score = key, score
        if best_key is None or best_score < 0.5:
            best_key = key_a  # unmatched tail -> the preceding ayah
        ordered_group = ws_a if best_key == key_a else ws_b
        for j, w in hole:
            used_idx.add(j)
            ordered_group.append(w)
        ordered_group.sort(key=lambda w: float(w.get("start", 0.0)))


def _close_wordless_holes(segments: list[dict], raw_segs,
                          silence_runs: list | None = None) -> None:
    """Keep the caption on across a stretch the ASR gave no words for.

    Some stretches produce no provider word at all (the ASR swallowed them
    into a neighbouring word's time range). If the VAD says the reciter was
    SPEAKING there, the caption must not go blank in the middle of his
    recitation — the gap is bridged. Where the VAD says silence, the gap is
    a real breath and the caption correctly disappears.

    A gap can contain BOTH: a breath, then voiced-but-wordless recitation
    (madd-held phrase, repeated passage the ASR dropped). Bridging the whole
    gap used to paint the caption straight across the breath (measured: two
    >=0.4s pauses held on the 8-min Al-Hashr clip). Now each VOICED island
    inside the gap gets a caption of its own — same text as the phrase before
    it, which is what the reciter is still saying — and the breath stays
    blank.

    silence_runs: audio-truth pauses (backend.vad.silence_runs). When omitted
    they are derived from raw_segs, keeping the legacy call shape.
    """
    if not raw_segs or len(segments) < 2:
        return
    if silence_runs is None:
        silence_runs = []
        for (_, e1), (s2, _) in zip(raw_segs, raw_segs[1:]):
            if s2 - e1 >= BREATH_MIN:
                silence_runs.append((e1, s2))
    silences = sorted((float(s), float(e)) for s, e in silence_runs
                      if float(e) - float(s) >= BREATH_MIN)
    out: list[dict] = []
    for idx, a in enumerate(segments):
        out.append(a)
        if idx + 1 >= len(segments):
            break
        b = segments[idx + 1]
        if b["start"] - a["end"] <= 0.10:
            continue
        # breaths that touch this gap (start sorted)
        inner = [(s, e) for s, e in silences
                 if s < b["start"] - 0.02 and e > a["end"] + 0.02]
        if not inner:
            # continuous voicing: nothing to blank, hold the caption through
            a["end"] = round(b["start"] - 0.001, 3)
            continue
        # stop the caption at the first breath it runs into — and hold it
        # through any voiced-but-wordless stretch before that breath ...
        first_start = inner[0][0]
        if first_start > a["start"] + 0.02:
            a["end"] = round(first_start, 3)
        # ... then re-show it over every voiced island after a breath
        for k, (_, re) in enumerate(inner):
            island_start = max(re, a["end"])
            island_end = (inner[k + 1][0] if k + 1 < len(inner)
                          else b["start"])
            if island_end - island_start <= 0.10:
                continue
            out.append(dict(a, start=round(island_start, 3),
                            end=round(island_end - 0.001, 3), _ws=[]))
    segments[:] = out


def build_srt(segments: list[dict], language: str = "ar") -> str:
    """Build SRT from ayah segments. language: 'ar' | 'en' | 'both'."""
    lines = []
    for i, seg in enumerate(segments, 1):
        lines.append(str(i))
        lines.append(f"{_fmt_ts(seg['start'])} --> {_fmt_ts(seg['end'])}")
        if language == "en":
            lines.append(seg.get("text_en") or seg.get("text_ar") or "")
        elif language == "both":
            if seg.get("text_ar"):
                lines.append(seg["text_ar"])
            if seg.get("text_en"):
                lines.append(seg["text_en"])
        else:
            lines.append(seg.get("text_ar") or "")
        lines.append("")
    return "\n".join(lines)


def _fmt_ts(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# ----------------------------------------------------------------- QUD aligner

def _qud_location(location: str) -> tuple[int, int, int]:
    parts = (location or "").split(":")
    if len(parts) != 3:
        raise ValueError(f"bad word location: {location!r}")
    return int(parts[0]), int(parts[1]), int(parts[2])


def _qud_word(location: str, rel_start, rel_end, t0: float,
              text: str = "", cpos: int | None = None) -> dict:
    try:
        start = t0 + float(rel_start)
        end = t0 + float(rel_end)
    except (TypeError, ValueError):
        raise ValueError(f"bad word timing at {location!r}")
    if end < start:
        end = start
    word = {"text": text, "start": round(start, 3), "end": round(end, 3)}
    if cpos is not None:
        word["_cpos"] = cpos
    return word


def _qud_ayah_pieces(words: list[dict], t0: float, texts_ar: dict):
    """Split one QUD quran segment into consecutive ayah pieces.

    A new piece starts whenever the ayah changes, so a repeat inside a single
    segment keeps recitation order instead of being merged into the earlier
    piece. Word text is OUR canon token at the QUD location (verified to have
    the same numbering); a location past the ayah's word count means the two
    tokenizations disagree and the result cannot be trusted."""
    pieces: list[tuple[int, int, list[dict]]] = []
    for raw in sorted(words, key=lambda w: float(w.get("start") or 0.0)):
        surah, ayah, wpos = _qud_location(raw.get("location"))
        if surah < 1:
            raise ValueError(f"unexpected non-Quran location {raw.get('location')!r}")
        canon = (texts_ar.get(f"{surah}:{ayah}") or "").split()
        if wpos < 1 or wpos > len(canon):
            raise ValueError(
                f"QUD word numbering does not match the Quran text at "
                f"{raw.get('location')} ({len(canon)} words)")
        if not pieces or (pieces[-1][0], pieces[-1][1]) != (surah, ayah):
            pieces.append((surah, ayah, []))
        pieces[-1][2].append(_qud_word(
            raw.get("location"), raw.get("start"), raw.get("end"),
            t0, canon[wpos - 1], wpos - 1))
    return pieces


def _qud_special_pieces(qs: dict, words: list[dict], t0: float,
                        texts_ar: dict, texts_en: dict):
    """Turn a `kind=special` segment (Basmala / Isti'adha) into display
    segments. QUD returns openers as specials instead of mushaf refs, so this
    replaces the old ASR-spelling opener scanner for the QUD path."""
    stype = (qs.get("special_type") or "").strip().lower()
    has_isti = "isti" in stype
    has_basmala = "basmala" in stype
    by_prefix: dict[str, list[dict]] = {}
    for raw in words:
        prefix = ":".join(str(raw.get("location") or "").split(":")[:2])
        by_prefix.setdefault(prefix, []).append(raw)
    if not has_isti and "0:1" in by_prefix:
        has_isti = True
    if not has_basmala and ("0:0" in by_prefix or "0:2" in by_prefix):
        has_basmala = True
    if not has_isti and not has_basmala:
        raise ValueError(f"unsupported QUD special {qs.get('special_type')!r}")

    groups: list[tuple[str, list[dict]]] = []
    if has_isti and has_basmala:
        isti, rest = [], []
        for w in words:
            prefix = ":".join(str(w.get("location") or "").split(":")[:2])
            (isti if prefix == "0:1" else rest).append(w)
        if isti and rest:
            groups = [("istiadha", isti), ("basmala", rest)]
        else:
            cut = 5 if len(words) >= 9 else 0
            groups = [("istiadha", list(words[:cut])),
                      ("basmala", list(words[cut:]))]
    elif has_isti:
        groups = [("istiadha", list(words))]
    else:
        groups = [("basmala", list(words))]

    en_openers = {op[0]: op[3] for op in OPENERS}
    out = []
    for kind, ws in groups:
        if not ws:
            continue
        if kind == "basmala":
            surah, ayah = 1, 1
            disp_ar = texts_ar.get(BISMILLAH_AYAH) or _OPENER_RAW["bismillah"]
            disp_en = texts_en.get(BISMILLAH_AYAH) or en_openers["bismillah"]
        else:
            surah, ayah = 1, 0
            disp_ar = _OPENER_RAW["istiadha"]
            disp_en = en_openers["istiadha"]
        display = disp_ar.split()
        row_words = []
        for i, raw in enumerate(sorted(ws, key=lambda w: float(w.get("start") or 0.0))):
            row_words.append(_qud_word(
                raw.get("location"), raw.get("start"), raw.get("end"), t0,
                display[i] if i < len(display) else "", i))
        out.append((
            {
                "surah": surah,
                "ayah": ayah,
                "start": round(float(row_words[0]["start"]), 3),
                "end": round(max(float(w["end"]) for w in row_words), 3),
                "text_ar": disp_ar,
                "text_en": disp_en,
            },
            row_words,
        ))
    if not out:
        raise ValueError(f"QUD special {qs.get('special_type')!r} had no words")
    return out


def _qud_retag_leading_opener(base: list[tuple[dict, list[dict]]]) -> None:
    """QUD has no takbir special, so a recording that opens with takbir can be
    aligned to a canon ayah that merely contains those words. Retag ONLY the
    first piece when its words are exactly the takbir/isti'adha — the same
    lead-run rule the old opener injection used (mid-clip they are legit)."""
    if not base:
        return
    first = min(range(len(base)), key=lambda i: float(base[i][0]["start"]))
    seg, ws = base[first]
    if seg.get("surah", 0) < 1 or seg.get("ayah", 0) < 1:
        return
    forms = [_normalize(w.get("text") or "") for w in ws]
    for kind, canon_words, raw_ar, raw_en in OPENERS:
        if kind == "bismillah" or forms != list(canon_words):
            continue
        display = raw_ar.split()
        for i, w in enumerate(ws):
            w["text"] = display[i] if i < len(display) else ""
            w["_cpos"] = i
        base[first] = (
            dict(seg, surah=1, ayah=0, text_ar=raw_ar, text_en=raw_en),
            ws,
        )
        return


def _qud_trim_silence_overhang(segments: list[dict], silence_runs) -> list[dict]:
    """Cut a cue back to the silence onset when its end reaches past a measured
    breath. QUD's word durations can absorb the trailing pause into the last
    word (measured: a 0.49 s breath at 44.25 sat inside a cue ending 44.77,
    i.e. the previous words stayed on screen while the reciter breathed).
    Only cues that STRADDLE the run are touched — a cue already ending before
    it is left alone, and a start inside a run is handled by padding."""
    if not silence_runs:
        return segments
    runs = sorted(silence_runs)
    for seg in segments:
        a, b = seg["start"], seg["end"]
        for rs, _re in runs:
            if b <= rs:
                break
            if a < rs < b:
                seg["end"] = round(rs, 3)
                break
    return segments


def segments_from_qud_alignment(
    qud_segments: list[dict],
    silence_runs: list | None = None,
    raw_segs: list[tuple[float, float]] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Build caption segments straight from a QUD aligner response.

    QUD already tells us WHICH words were recited and WHEN (ayah + word refs
    and phoneme-accurate word times), so there is no detector, no chain-hole
    filling and no VAD snapping: the display text is our canon text sliced to
    exactly the recited words, repeats become their own captions, and a held
    madd is already inside its word's duration (trust_times).

    Returns (segments, words) where words are absolute-time {text,start,end}
    entries in recitation order (repeats included)."""
    from backend.providers.qud import QudUnavailable

    texts_ar = load_ayah_texts()
    texts_en = load_english_texts()

    base: list[tuple[dict, list[dict]]] = []
    words_out: list[dict] = []

    try:
        for qs in qud_segments or []:
            if not isinstance(qs, dict):
                continue
            seg_words = qs.get("words") or []
            if not seg_words:
                continue
            try:
                t0 = float(qs.get("time_from") or 0.0)
            except (TypeError, ValueError):
                t0 = 0.0
            kind = str(qs.get("kind") or "quran").strip().lower()
            if kind == "special":
                base.extend(_qud_special_pieces(
                    qs, seg_words, t0, texts_ar, texts_en))
                continue
            for surah, ayah, ws in _qud_ayah_pieces(seg_words, t0, texts_ar):
                ws.sort(key=lambda w: float(w["start"]))
                cpos = [w["_cpos"] for w in ws]
                text_ar, text_en = _trim_texts(
                    texts_ar.get(f"{surah}:{ayah}") or "",
                    texts_en.get(f"{surah}:{ayah}") or "",
                    (min(cpos), max(cpos) + 1))
                seg = {
                    "surah": surah,
                    "ayah": ayah,
                    "start": round(float(ws[0]["start"]), 3),
                    "end": round(max(float(w["end"]) for w in ws), 3),
                    "text_ar": text_ar,
                    "text_en": text_en,
                }
                words_out.extend(ws)
                base.append((seg, ws))
    except ValueError as exc:
        raise QudUnavailable(str(exc)) from exc

    if not base:
        raise QudUnavailable("QUD returned no usable segments")

    _qud_retag_leading_opener(base)

    segments: list[dict] = []
    for seg, ws in base:
        cpos = [w.get("_cpos") for w in ws]
        segments.extend(_phrase_split(
            seg, ws, silence_runs, raw_segs, cpos, trust_times=True))
    segments.sort(key=lambda s: s["start"])
    segments = subdivide_word_cues(
        segments, words_out, silence_runs, raw_segs, trust_times=True)
    # A caption blanks for a BREATH, not for every 0.2-0.4 s articulation dip:
    # a 0.38 s VAD dip inside a QUD word pulled the cue end back and left
    # 1.18 s of recited "أَنَّهُم" with no caption (real Al-Hashr at 40.1 s).
    breaths = [(s, e) for s, e in (silence_runs or []) if e - s >= SILENCE_SPLIT]
    segments = _qud_trim_silence_overhang(segments, breaths)
    segments = apply_sync_padding(segments, breaths)
    for s in segments:
        s.pop("_ws", None)
    words_out.sort(key=lambda w: float(w["start"]))
    for w in words_out:
        w.pop("_cpos", None)
    return segments, words_out
