from collections import Counter

from rapidfuzz import fuzz

from data.quran_text import load_ayah_texts, load_english_texts, _normalize, _normalize_std


LEAD_IN = 0.05
LEAD_OUT = 0.25
MIN_GAP = 0.05
MIN_DUR = 0.5

# Reciters pause between breath phrases (waqf) mid-ayah. When a TRUE silence
# (measured from the audio via VAD — backend.vad.silence_runs_from_segments,
# NOT from Whisper word-gap math, which drifts 200-500ms) >= SILENCE_SPLIT
# sits between two words, the caption for that ayah is cut into separate
# phrases so text disappears during the pause and reappears where the reciter
# resumes. Madd (elongation) extends a word's own end, so it never fakes a
# pause; only real silences split. Short breaths (< SILENCE_SPLIT) keep the
# caption up, so captions don't flicker off mid-ayah at every breath (seen on
# 2h Baqarah: 207 tiny 1-3 word captions from 0.5-0.9s breaths).
# PAUSE_GAP is the conservative fallback threshold used ONLY when no silence
# runs are available (tests, callers without VAD data): whisper gaps are
# unreliable evidence, so they must clear a higher bar to split.
SILENCE_SPLIT = 0.6
PAUSE_GAP = 1.0
MIN_PHRASE_WORDS = 3
MIN_PHRASE_DUR = 1.2
_EN_BREAK_CHARS = set(".;!?،،:,")

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
    words: list[dict], runs: list[tuple[float, float]], tol: float = 0.25
) -> list[dict]:
    """Snap Whisper word boundaries to real silence edges (stable-ts /
    whisper-sync style). Whisper's timestamp head drifts 200-500ms and bleeds
    word ends into silence — a word whose end falls right before a silence run
    is moved to the run's start, and a word whose start falls right after a
    run is moved to the run's end. Boundaries that don't touch a silence edge
    are left alone. Returns new word dicts (input untouched)."""
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
        w["start"] = round(s, 3)
        w["end"] = round(e, 3)
        out.append(w)
    return out


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


def _phrase_split(seg: dict, words: list[dict], silence_runs: list | None = None) -> list[dict]:
    """Split one ayah segment into breath-phrase segments at REAL pauses
    (inter-word silences >= SILENCE_SPLIT measured from the audio), so the
    caption vanishes while the reciter breathes and resumes where they resume.

    words: chronological word timestamps {"start", "end"} for this ayah
    (they may carry "text"; unused here). silence_runs: (start, end) silence
    regions from backend.vad.silence_runs — the pause ground truth. Without
    them (e.g. tests), the Whisper word gap is used as weak evidence instead.
    Canon text is re-sliced to each phrase by word-count proportion
    (recitation order == text order); English uses the same proportional
    cuts, snapped to the nearest sentence break so translations don't split
    mid-sentence.

    Returns [seg] unchanged when there is no pause to split on."""
    if len(words) < 2:
        return [seg]
    ws = sorted(words, key=lambda w: float(w["start"]))
    if silence_runs:
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
        return [seg]

    # Regions between cuts.
    regions: list[list[int]] = []
    lo = 0
    for c in cuts + [len(ws)]:
        regions.append([lo, c - 1])
        lo = c

    # Merge too-small regions (few words or short) into their neighbors:
    # a region that fails the threshold is absorbed by the previous one, or
    # the next one when it is leading (so one-word "يا أيها" pause tails
    # don't flash a caption and nothing is dropped at the start).
    def small(r: list[int]) -> bool:
        wc = r[1] - r[0] + 1
        dur = float(ws[r[1]]["end"]) - float(ws[r[0]]["start"])
        # epsilon: float subtraction jitter (e.g. 3.3-2.9 = 0.3999...) must
        # not flip the exact-boundary comparison
        return wc < MIN_PHRASE_WORDS or dur < MIN_PHRASE_DUR - 1e-6

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
    for idx, (lo, hi) in enumerate(regions):
        last = idx == len(regions) - 1
        b_ar = cut(n_ar, hi)
        b_en_plain = cut(n_en, hi)
        b_en = b_en_plain if last else _snap_en_boundary(en_words, prev_b_en, b_en_plain)
        a_ar, a_en = prev_b_ar, prev_b_en
        # never emit an empty phrase
        b_ar = max(b_ar, min(a_ar + 1, n_ar)) if b_ar <= a_ar else b_ar
        b_en = max(b_en, min(a_en + 1, n_en)) if b_en <= a_en else b_en
        prev_b_ar, prev_b_en = b_ar, b_en
        phrase = dict(seg)
        phrase.update({
            "start": round(float(ws[lo]["start"]), 3),
            "end": round(float(ws[hi]["end"]), 3),
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


def apply_sync_padding(segments: list[dict]) -> list[dict]:
    """Final caption timing polish: a tiny lead-in (~50ms) so text is visible
    a hair before the first word, and a lead-out so the caption lingers over
    the reciter's held final syllable. Enforces non-overlap + min duration.

    Real speech alignment is the job of the ASR path upstream — Groq provider
    words are VAD-anchored (Silero slice bounds) to within ~50ms of wav2vec2
    forced alignment; this function only nudges *display* timing."""
    segs = sorted(segments, key=lambda s: s["start"])
    raw_end = [seg["end"] for seg in segs]
    for i, seg in enumerate(segs):
        seg["start"] = max(0.0, round(seg["start"] - LEAD_IN, 3))
    prev_end = 0.0
    for i, seg in enumerate(segs):
        seg["start"] = max(round(prev_end + MIN_GAP, 3), seg["start"])
        end = round(raw_end[i] + LEAD_OUT, 3)
        if i + 1 < len(segs):
            end = min(end, round(segs[i + 1]["start"] - MIN_GAP, 3))
        end = max(end, round(raw_end[i], 3), round(seg["start"] + MIN_DUR, 3))
        seg["end"] = end
        prev_end = seg["end"]
    return segs


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
        segments.extend(_phrase_split(seg, opener_words, silence_runs))
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

    def trim(text: str, off: tuple[int, int] | None, ar_total: int | None = None) -> str:
        if not off:
            return text
        words = text.split()
        if not words:
            return text
        first, last = off
        if ar_total and len(words) != ar_total:
            # EN text: word counts differ from AR — trim proportionally.
            first = round(first * len(words) / ar_total)
            last = round(last * len(words) / ar_total)
        return " ".join(words[first:last])

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
        ar_total = len(text_ar.split())
        off = word_off.get(seg_ayah)
        if off:
            first, last = off
            # a mid-word clip cut drops the final aligned word(s) — never
            # display text beyond what actually has audio
            last = min(last, first + len(ws))
            off = (first, last)
        seg = {
            "surah": seg_surah,
            "ayah": seg_ayah,
            "start": round(start, 3),
            "end": round(end, 3),
            "text_ar": trim(text_ar, off),
            "text_en": trim(text_en, off, ar_total),
        }
        # Key must stay the TAG (1,-2)/(1,-1)/(1,0) — the display ayah (1,0)
        # is shared by takbir and isti'adha and would merge their words.
        segments.extend(_phrase_split(seg, ws, silence_runs))
    return apply_sync_padding(segments)


def segments_from_provider_times(
    matches: list, words: list[dict], silence_runs: list | None = None
) -> list[dict]:
    """Build ayah segments straight from hosted-ASR word timestamps.
    Each AyahMatch carries chain=[(word_idx, surah, ayah)]; words[idx] gives
    {text, start, end}. One segment per ayah, times clamped to the recited
    words — no forced alignment needed. silence_runs: audio-truth pause
    evidence for phrase splitting (provider word times drift into silence)."""
    texts_ar = load_ayah_texts()
    texts_en = load_english_texts()

    segments: list[dict] = []
    all_word_groups: list[tuple[dict, list[dict]]] = []
    for m in matches:
        chain = getattr(m, "chain", None) or []
        if not chain:
            continue
        words_by_ayah: dict[tuple[int, int], list[dict]] = {}
        for word_idx, surah, ayah in chain:
            if not (0 <= word_idx < len(words)):
                continue
            words_by_ayah.setdefault((surah, ayah), []).append(words[word_idx])

        for (surah, ayah), ws in sorted(
            words_by_ayah.items(), key=lambda kv: float(kv[1][0]["start"])
        ):
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

    segments = [p for seg, ws in all_word_groups for p in _phrase_split(seg, ws, silence_runs)]
    segments.sort(key=lambda s: s["start"])
    return apply_sync_padding(
        inject_opener_segments(matches, words, segments, silence_runs)
    )


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
