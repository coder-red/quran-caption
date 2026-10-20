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


def _phrase_split(seg: dict, words: list[dict], silence_runs: list | None = None,
                  raw_segs: list[tuple[float, float]] | None = None) -> list[dict]:
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
    translations don't split mid-sentence. raw_segs: raw (unmerged) VAD
    speech segments — when given (provider path), each phrase's word-derived
    bounds are VAD-clamped to real speech edges (provider word stamps are
    ±200-500ms noisy; the local forced-align path skips this because
    wav2vec2 times are already frame-accurate).

    Audio-evidenced cuts are ALWAYS kept — even a 1-word phrase before a real
    breath shows briefly then blanks, and never bridges the pause. Tiny-phrase
    merging applies to the fallback path only (fake whisper gaps must not
    shatter captions). With raw_segs, a phrase ending in a madd word is HELD
    past an early provider cut into continuing voicing (_madd_extend) so the
    caption survives long mujawwad elongations instead of vanishing mid-vowel.

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
            s, e = _vad_clamp_bounds(
                seg["start"], seg["end"], ws[0], ws[-1], raw_segs)
            # Single-phrase ayah (the common verse-final madd case): hold a
            # madd tail the provider cut early. verse_final: one region covers
            # the whole segment, whose text is the full canon ayah here.
            raw_tail = (seg.get("text_ar") or "").split()
            e = _madd_extend(e, raw_tail[-1] if raw_tail else "",
                             True, silence_runs, raw_segs, float("inf"))
            seg = dict(seg, start=s, end=e)
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

    out = []
    prev_b_ar = 0
    prev_b_en = 0
    n_regions = len(regions)
    for idx, (lo, hi) in enumerate(regions):
        last = idx == n_regions - 1
        # Reserve one word per remaining phrase so no phrase is starved: the
        # proportional cut must never consume words a later phrase still needs.
        regions_left = n_regions - idx - 1
        if last:
            # Last phrase takes EVERYTHING left in both languages, so the tail
            # of the ayah can never lose its text (this was the "English did
            # not show" bug: the final 1-2 Arabic words were emitted with an
            # empty English slice because the proportional EN cut had already
            # been consumed by the previous phrase).
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
        if raw_segs:
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
                merged[key] = (a + " " + b).strip() if a else b
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

    segments = [p for seg, ws in all_word_groups
                for p in _phrase_split(seg, ws, silence_runs, raw_segs)]
    segments.sort(key=lambda s: s["start"])
    _close_wordless_holes(segments, raw_segs)
    return apply_sync_padding(
        inject_opener_segments(matches, words, segments, silence_runs, raw_segs),
        silence_runs,
    )


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
    """
    if len(ordered) < 2:
        return

    def canon_words(surah: int, ayah: int) -> set[str]:
        txt = texts_ar.get(f"{surah}:{ayah}", "")
        return {_normalize(w) for w in txt.split() if w}

    n = len(words)
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


def _close_wordless_holes(segments: list[dict], raw_segs) -> None:
    """Keep the caption on across a stretch the ASR gave no words for.

    Some stretches produce no provider word at all (the ASR swallowed them
    into a neighbouring word's time range). If the VAD says the reciter was
    SPEAKING there, the caption must not go blank in the middle of his
    recitation — the gap is bridged. Where the VAD says silence, the gap is
    a real breath and the caption correctly disappears.
    """
    if not raw_segs or len(segments) < 2:
        return
    # the silences between the VAD speech runs = the real breaths
    silences = []
    for (s1, e1), (s2, e2) in zip(raw_segs, raw_segs[1:]):
        if e2 - s1 >= 0.10:
            silences.append((e1, s2))
    for a, b in zip(segments, segments[1:]):
        gap = b["start"] - a["end"]
        if gap <= 0.10:
            continue
        # a real breath explains the gap -> the caption correctly disappears
        if any(ps <= a["end"] + 0.15 and pe >= b["start"] - 0.15
               for ps, pe in silences):
            continue
        a["end"] = round(b["start"] - 0.001, 3)


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
