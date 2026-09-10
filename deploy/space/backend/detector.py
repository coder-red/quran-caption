import re
from rapidfuzz import fuzz, process
from data.quran_text import SURAH_NAMES, get_surah_text, _normalize


class AyahMatch:
    def __init__(
        self,
        surah: int,
        start_ayah: int,
        end_ayah: int,
        score: float,
        text: str,
        word_span: tuple[int, int] | None = None,
        chain: list[tuple[int, int, int]] | None = None,
        word_off: dict[int, tuple[int, int]] | None = None,
    ):
        self.surah = surah
        self.start_ayah = start_ayah
        self.end_ayah = end_ayah
        self.score = score
        self.text = text
        self.word_span = word_span
        # (transcript index, surah, ayah) for every matched transcript word.
        # Lets hosted ASR paths cut per-ayah segments straight from word
        # timestamps without forced alignment.
        self.chain = chain
        # ayah -> (first_canon_word, last_canon_word+1) actually recited, as
        # offsets into that ayah's canon text. Mid-ayah clip cuts make the
        # run start/end inside an ayah: forced alignment must only cover the
        # recited words (else unrecited tails fail Viterbi / phantom-stretch).
        self.word_off = word_off

    def __repr__(self):
        return f"AyahMatch(surah={self.surah}, ayahs={self.start_ayah}-{self.end_ayah}, score={self.score:.3f})"


class SurahDetector:
    def __init__(self, debug_chain: bool = False):
        self._candidates: list[dict] | None = None
        self._flat: list[tuple[int, int, str]] | None = None
        self._word_pos: dict[str, list[int]] | None = None
        self._debug_chain = debug_chain

    def _build_candidates(self):
        if self._candidates is not None:
            return self._candidates

        candidates = []
        for surah_info in SURAH_NAMES:
            surah_id = surah_info["id"]
            ayahs = get_surah_text(surah_id)
            combined_text = " ".join(a["text"] for a in ayahs)
            candidates.append({
                "surah": surah_id,
                "text": combined_text,
                "normalized": _normalize(combined_text),
            })
        self._candidates = candidates
        return candidates

    def _build_index(self):
        """Flat canon (surah, ayah, word) array + word -> positions index.
        Words are indexed under BOTH orthography normalizations: Uthmani
        (dagger alif stripped: الصلوه, ملك) and standard-spelling
        (dagger alif -> alif: الصلاه, مالك). Whisper emits standard
        orthography, so the standard variant recovers matches.

        Fatihah 1:1 (the Bismillah) is deliberately NOT indexed: it is always
        stripped from provider/local transcripts as a reciter's opener, so the
        slot can never be a legitimate match — and anchoring a leading takbir
        "الله أكبر" there produced a phantom 1:1 caption over takbir audio."""
        if self._flat is not None:
            return
        from data.quran_text import load_ayah_texts, _normalize, _normalize_std
        texts = load_ayah_texts()
        flat = []
        word_pos: dict[str, list[int]] = {}
        compound_idx: list[int] = []
        ayah_start: dict[tuple[int, int], int] = {}
        for key, text in texts.items():
            surah, ayah = (int(p) for p in key.split(":"))
            if surah == 1 and ayah == 1:
                continue
            ayah_start[(surah, ayah)] = len(flat)
            for raw_word in text.split():
                idx = len(flat)
                flat.append((surah, ayah, _normalize(raw_word)))
                for v in {_normalize(raw_word), _normalize_std(raw_word)}:
                    word_pos.setdefault(v, []).append(idx)
                # Vocative phrases are written as ONE Uthmani word with a
                # superscript alef: يَـٰٓأَيُّهَا ("يَا أَيُّهَا"), يَـٰٓآدَمُ
                # ("يَا آدَمُ")... Verbs (يَأۡكُلُ, يَأۡمُرُ) use a plain hamza
                # and are NOT vocatives — the U+0670 test keeps them out.
                if "\u0670" in raw_word:
                    n = _normalize(raw_word)
                    if n.startswith("يا") and len(n) > 3:
                        compound_idx.append(idx)
        # ASR splits the compound into two tokens ("يا أيها"); register both
        # tokens at the compound's slot so chains attach the ayah's leading
        # words and the first caption is not ~a word late. The second token's
        # alef merges into the compound ("يايها" = يا + أيها), so restore it.
        split_pos: set[int] = set()
        for idx in compound_idx:
            w = flat[idx][2]
            for tok in ("يا", "ا" + w[2:]):
                word_pos.setdefault(tok, []).append(idx)
            split_pos.add(idx)
        for lst in word_pos.values():
            if len(lst) > 1:
                lst.sort()
        self._split_pos = split_pos
        self._flat = flat
        self._word_pos = word_pos
        self._ayah_start = ayah_start

    def detect_all(self, transcript: str, min_score: float = 0.55) -> list[AyahMatch]:
        """Global word-anchor alignment of the transcript against the canon.
        Returns chronologically ordered matches (one per surah range found).
        Handles mid-ayah starts/ends and multi-surah recitations (any order)."""
        self._build_index()
        matches: list[AyahMatch] = []
        norm = _normalize(transcript)
        bismillah_words = _normalize("بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ").split()
        twords = [(i, w) for i, w in enumerate(norm.split())]
        # Bismillah is a reciter's opener — strip every occurrence (lead-in,
        # pre-ayah in shorter surahs, etc.) so it never becomes a content
        # match like Al-Fatihah 1:1, which is canonically the Bismillah text.
        if bismillah_words:
            _norm_words = []
            i = 0
            n = len(twords)
            while i < n:
                if (
                    i + len(bismillah_words) <= n
                    and [twords[i + j][1] for j in range(len(bismillah_words))]
                    == bismillah_words
                ):
                    i += len(bismillah_words)
                    continue
                _norm_words.append(twords[i])
                i += 1
            twords = _norm_words
# Reciters also open with a takbir ("الله أكبر") or isti'adha
        # ("أعوذ بالله من الشيطان الرجيم"). Those are only stripped when they
        # LEAD the transcript (never mid-clip: "الله أكبر" is legitimate canon
        # in 9:72, 29:45, 40:10), else they anchor a phantom 1:1 caption over
        # the takbir audio. Tolerant matching: ASR spells الشيطان with a plain
        # alef while canon strips the superscript alef (الشيطن).
        from backend.segmenter import match_leading_openers
        lead_canon_len = {
            "takbir": 2, "istiadha": 5, "bismillah": len(bismillah_words),
        }
        while True:
            kinds = match_leading_openers([w for _, w in twords])
            if not kinds:
                break
            n_strip = sum(lead_canon_len[k] for k in kinds)
            twords = twords[n_strip:]
        self._collect(twords, matches, min_score, depth=0)
        return matches

    def _collect(
        self,
        twords: list[tuple[int, str]],
        matches: list[AyahMatch],
        min_score: float,
        depth: int,
    ):
        if depth > 4:
            return
        from bisect import bisect_right, bisect_left
        flat = self._flat
        word_pos = self._word_pos
        WINDOW = 25  # max canon gap between consecutive matched words

        twords = [t for t in twords if t[1] in word_pos]
        if not twords:
            return

        # For each anchor word (first 40 transcript words), walk the
        # transcript BOTH backward and forward from the anchor, extending the
        # chain only to the nearest canon occurrence within a locality window.
        # Beyond the window the chain stops (surah boundary / recitation gap)
        # instead of jumping to a far-away occurrence of a common word.
        best_chain: list[tuple[int, int]] | None = None
        best_adj = -1
        for ti in range(min(40, len(twords))):
            occs = word_pos[twords[ti][1]]
            if len(occs) > 100:
                continue
            for start_pos in occs:
                beams: list[tuple[int, int, list[tuple[int, int]]]] = [
                    (start_pos, 0, [(ti, start_pos)])
                ]
                for tj in range(ti - 1, -1, -1):
                    cands = word_pos[twords[tj][1]]
                    new_beams = []
                    for last, adj, chain in beams:
                        new_beams.append((last, adj, chain))
                        i = bisect_right(cands, last)
                        if i > 0:
                            prev = cands[i - 1]
                            if last - prev <= WINDOW and (
                                prev != last or last in self._split_pos
                            ):
                                new_beams.append(
                                    (prev,
                                     adj + (1 if prev == last - 1 else 0),
                                     [(tj, prev)] + chain)
                                )
                    beams = sorted(
                        new_beams,
                        key=lambda b: (b[1], len(b[2])),
                        reverse=True,
                    )[:6]
                for tj in range(ti + 1, len(twords)):
                    cands = word_pos[twords[tj][1]]
                    new_beams = []
                    for last, adj, chain in beams:
                        new_beams.append((last, adj, chain))
                        i = bisect_right(cands, last)
                        if (
                            i < len(cands)
                            and cands[i] - last <= WINDOW
                            and cands[i] > start_pos
                        ):
                            new_beams.append(
                                (cands[i], adj + (1 if cands[i] == last + 1 else 0),
                                 chain + [(tj, cands[i])])
                            )
                    beams = sorted(
                        new_beams,
                        key=lambda b: (b[1], len(b[2])),
                        reverse=True,
                    )[:6]
                for last, adj, chain in beams:
                    if adj > best_adj or (
                        adj == best_adj
                        and (best_chain is None or len(chain) > len(best_chain))
                    ):
                        best_adj = adj
                        best_chain = chain

        if best_chain is None or best_adj < 3:
            return

        if self._debug_chain:
            print(f"  [depth {depth}] best chain len={len(best_chain)} adj={best_adj}")
            prev = None
            for ti, p in best_chain:
                s, a, w = flat[p]
                gap = "" if prev is None or p - prev == 1 else f" gap={p - prev}"
                print(f"    t{ti} {w} @ {s}:{a} (pos {p}){gap}")
                prev = p

        # Split chain into consecutive runs (gap <= 25 tolerated deletions,
        # runs never cross surah boundaries). Short chains additionally split
        # at large canon jumps (gap > 6) — such chains are usually a dense
        # start plus a sparse fragment (repeat), which must not be scored as
        # one sparse run. Long chains keep the 25 tolerance so recitation
        # repeats mid-surah do not fragment the main match.
        runs: list[list[tuple[int, int]]] = [[best_chain[0]]]
        for item in best_chain[1:]:
            ti, p = item
            last_ti, last_p = runs[-1][-1]
            if (
                p - last_p <= 25
                and flat[p][0] == flat[last_p][0]
                and (p - last_p <= 6 or len(runs[-1]) >= 20)
            ):
                runs[-1].append(item)
            else:
                runs.append([item])

        consumed: set[int] = set()
        for run in runs:
            if len(run) < 4:
                continue
            s_p = run[0][1]
            e_p = run[-1][1]
            span_words = e_p - s_p + 1
            score = min(1.0, len(run) / max(1, span_words))
            if score < min_score:
                continue
            # Recited canon word offsets per ayah (mid-ayah clip cuts). The
            # run's flat span [s_p, e_p] is contiguous canon, so the word
            # offset of each end within its ayah = position - ayah_start.
            # Middle ayahs of a multi-ayah run are fully covered.
            word_off: dict[int, tuple[int, int]] = {}
            pos = s_p
            while pos <= e_p:
                s_, a_ = flat[pos][0], flat[pos][1]
                a_start = self._ayah_start[(s_, a_)]
                # word count of this ayah in flat: walk until the ayah changes
                a_end = a_start
                while a_end < len(flat) and flat[a_end][:2] == (s_, a_):
                    a_end += 1
                first = max(pos, a_start) - a_start
                last = min(e_p, a_end - 1) - a_start + 1
                word_off[a_] = (first, last)
                pos = a_end
            matches.append(AyahMatch(
                surah=flat[s_p][0],
                start_ayah=flat[s_p][1],
                end_ayah=flat[e_p][1],
                score=score,
                text=" ".join(w for _, _, w in flat[s_p:e_p + 1]),
                word_span=(twords[run[0][0]][0], twords[run[-1][0]][0]),
                chain=[(twords[ti][0], flat[pos][0], flat[pos][1]) for ti, pos in run],
                word_off=word_off,
            ))
            consumed.update(ti for ti, _ in run)

        if not consumed:
            return

        # Recurse on transcript words not consumed (next surah / trailing
        # recitation, possibly out of canon order).
        remainder = [t for i, t in enumerate(twords) if i not in consumed]
        if remainder:
            self._collect(remainder, matches, min_score, depth + 1)

    def detect(self, transcript: str, top_n: int = 5) -> list[AyahMatch]:
        normalized_transcript = _normalize(transcript)
        candidates = self._build_candidates()

        texts = [c["normalized"] for c in candidates]
        scores = process.extract(normalized_transcript, texts, scorer=fuzz.partial_ratio, limit=top_n)

        results = []
        for match_text, score, idx in scores:
            candidate = candidates[idx]
            surah_id = candidate["surah"]
            surah_text = candidate["normalized"]

            start = surah_text.find(match_text[:50])
            if start == -1:
                start = 0

            prefix = surah_text[:max(0, start)]
            ayah_count = prefix.count(" ") + 1
            start_ayah = max(1, ayah_count)

            total_ayahs = sum(1 for c in match_text if c == " ") + 1
            end_ayah = min(start_ayah + total_ayahs - 1, SURAH_NAMES[surah_id - 1]["ayah_count"])

            results.append(AyahMatch(
                surah=surah_id,
                start_ayah=start_ayah,
                end_ayah=end_ayah,
                score=score / 100.0,
                text=candidate["text"][:500],
            ))

        results.sort(key=lambda x: x.score, reverse=True)
        return results

    def detect_tight(self, transcript: str, window_sizes: list[int] = None) -> AyahMatch | None:
        if window_sizes is None:
            window_sizes = [12, 8, 5, 3, 2, 1]

        from data.quran_text import load_ayah_texts, _normalize
        from bisect import bisect_left
        texts = load_ayah_texts()

        all_entries = []
        for key, text in texts.items():
            parts = key.split(":")
            all_entries.append((int(parts[0]), int(parts[1]), text))

        all_entries.sort(key=lambda x: (x[0], x[1]))
        word_counts = [len(_normalize(e[2]).split()) for e in all_entries]
        pref = [0]
        for c in word_counts:
            pref.append(pref[-1] + c)

        normalized_transcript = _normalize(transcript)

        # Recitations open with the Bismillah, which is not part of the target
        # ayahs' canon text — strip a leading one so word-count window matching
        # lines up with the surah text. Keep the stripped word count so chain
        # indices still reference the FULL transcript (provider word lists
        # include the Bismillah).
        bismillah = _normalize("بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ")
        bi_skip = 0
        if normalized_transcript.startswith(bismillah):
            normalized_transcript = normalized_transcript[len(bismillah):].strip()
            bi_skip = len(bismillah.split())

        n = len(normalized_transcript.split())
        if n == 0:
            return None

        def make_match(entries, score) -> AyahMatch:
            s_p = entries[0][0]
            e_p = entries[-1][0]
            n_use = min(len(entries), n)
            word_off = {}
            for (_, ayah, text) in entries[:n_use]:
                word_off[ayah] = (0, len(_normalize(text).split()))
            return AyahMatch(
                surah=s_p,
                start_ayah=entries[0][1],
                end_ayah=entries[-1][1],
                score=score,
                text=" ".join(e[2] for e in entries),
                word_span=(0, max(0, n_use - 1)),
                chain=[(i + bi_skip, s_p, ayah) for i, (_, ayah, _) in enumerate(entries[:n_use])],
                word_off=word_off,
            )

        def windows_with_words(target: int) -> list[list[tuple]]:
            out = []
            for i in range(len(all_entries)):
                j = bisect_left(pref, pref[i] + target)
                if j > i and j <= len(all_entries) and pref[j] == pref[i] + target:
                    out.append(all_entries[i:j])
            return out

        # 1) Window with exactly the transcript's word count, whole-string ratio.
        windows = windows_with_words(n)
        if windows:
            window_texts = [_normalize(" ".join(e[2] for e in w)) for w in windows]
            best = process.extractOne(
                normalized_transcript, window_texts, scorer=fuzz.ratio
            )
            if best:
                score = best[1] / 100.0
                if score >= 0.75:
                    return make_match(windows[best[2]], score)

        # 2) Partial-ratio fallback over coarse windows (ASR errors present).
        for window_size in window_sizes:
            windows = windows_with_words(window_size)
            if not windows:
                continue
            window_texts = [
                _normalize(" ".join(e[2] for e in w)) for w in windows
            ]
            best = process.extractOne(
                normalized_transcript, window_texts, scorer=fuzz.partial_ratio
            )
            if best and best[1] / 100.0 >= 0.9:
                return make_match(windows[best[2]], best[1] / 100.0)

        return None
