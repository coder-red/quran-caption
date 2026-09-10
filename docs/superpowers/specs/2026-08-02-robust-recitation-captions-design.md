# Robust Recitation → Captions Design

> Date: 2026-08-02
> Status: Approved (user approved inline on 2026-08-02)
> Component: Quran Caption App (`C:\Users\hp\caption_quran`)

## Problem

The current `/align` pipeline fails on three documented edge cases
(AGENTS.md Todo: "Handle edge cases (mid-ayah starts, pauses, repeats)"):

1. **Background music / noise** — the energy-based speech gate
   (`aligner.detect_speech_span`) misfires: loud music looks like speech,
   quiet recitation gets cut. Whisper windows are fixed 30s with an RMS gate.
2. **Multiple surahs in one clip** — `detector.detect_tight` returns a SINGLE
   `AyahMatch`. A clip reciting two surahs only matches the best one.
3. **Mid-ayah starts / ends** — exact word-count window matching
   (`windows_with_words`) fails when the transcript begins/ends inside an ayah
   and the word count doesn't match any canon window. ASR errors compound this.

Known constraints from AGENTS.md (MUST be preserved):

- Quran text is the source of truth; ASR is used only for timing, never caption text.
- Whisper timestamps are unusable (1 ts token per clip) → forced CTC alignment
  (wav2vec2-large-xlsr-53-arabic) is the source of truth for timing.
- Alafasy clips open with Bismillah (~0.3–6.2s) not in target ayahs →
  `detect_bismillah()` prepends ayah 0 canon words; segments skip ayah 0.
- `detect_tight` windows slice by WORD count (prefix sums + bisect).
- Target machine: 8GB RAM, CPU-only. Both models can never be resident at once;
  long audio must be sliced before forced alignment (current code does this
  via energy-based `detect_speech_span` + `slice_audio`).

## Research (best practice, 2026-08-02)

- Silero VAD (neural) is the proven standard for speech detection under
  background music; energy/RMS thresholds fail with music. Open-source,
  ~2MB ONNX/torch model, 16kHz input, RTF ~0.004. Sources: picovoice.ai
  VAD comparison, decibri docs, manyeyes manyspeech, CallSphere tuning guide.
- WhisperX (arXiv 2303.00747): VAD pre-segmentation + cut & merge into ~30s
  chunks with boundaries on minimally active speech regions → better WER and
  up to 12x speedup vs fixed windows. Chunk overlap 2–4s recommended.
- Whisper long-form best practices: 30s windows (model limit), VAD-based
  segmentation preferred over fixed chunking, absolute timestamps via
  chunk-offset addition.

## Design

### Stage 1 — `backend/vad.py` (NEW module)

Silero VAD wrapper with energy fallback. Pure functions over `np.ndarray`.

```
audio (16k mono float32) → get_speech_segments() → merge_segments() → chunk_for_whisper()
```

- `get_speech_segments(audio, sr=16000, threshold=0.5) -> list[(start, end)]`
  - Uses silero-vad package (torch mode). Lazy-loads model once (module-level cache).
  - Fallback: RMS energy segmentation (current `detect_speech_span` logic
    extracted), if silero import or inference fails.
- `merge_segments(segments, max_gap=2.0) -> list[(start, end)]`
  - Merge segments separated by <= max_gap seconds (reciter pauses).
- `chunk_for_whisper(segments, target=30.0, max_len=30.0) -> list[(start, end)]`
  - Cut & merge: pack segments into chunks as close to 30s as possible;
    never exceed `max_len`; boundaries land on silence.
- `speech_span(segments) -> (start, end) | None`
  - First-to-last merged segment with 1.5s padding (replaces
    `detect_speech_span` for the alignment slice).

Behavior contract:
- Pure silence → `get_speech_segments` returns [] → `/align` returns
  "No speech detected in audio".
- Speech with 1s pauses → merged into one span.
- Recitation with background music → speech segments still detected
  (neural VAD); chunks may include music but forced alignment tolerates it
  (CTC path over canon words).

### Stage 2 — `backend/detector.py` upgrade: global canon alignment

Add a flat canon index + `detect_all()`. Keep `detect_tight()` as fallback.

```
canon_flat: list[(surah, ayah, word)]        # ~77k words, canon order
word_index: dict[str, list[int]]             # normalized word → flat positions
```

- Build once (module-level lazy cache), same source as `load_ayah_texts()` +
  `_normalize()`.
- `detect_all(transcript, min_score=0.55) -> list[AyahMatch]`
  - Strip leading Bismillah (existing logic).
  - Greedy anchor chaining over transcript words:
    - For each transcript word (skipping OOV), gather candidate flat
      positions from `word_index`.
    - Chain candidates forward: `cand > last_pos`, prefer minimal jump.
    - Consecutive transcript words matching consecutive flat positions form
      a run. Tolerate ASR insertions (unmatched transcript words ≤ 2 in a
      row) and deletions (flat gap ≤ 3) — word-level edit tolerance.
  - Each run → `AyahMatch(surah=run[0].surah, start_ayah, end_ayah, score)`.
    Score = matched words / (matched + skipped) over run window, clamped.
    `start_ayah`/`end_ayah` from first/last canon entry of the run
    (mid-ayah starts/ends handled naturally).
  - Runs sorted by position (chronological).
- `detect_all` may return multiple matches (multi-surah clips) or one
  (single surah). Empty → caller falls back to `detect_tight`, then error.

Behavior contract:
- Mid-ayah transcript → match starts at that ayah (no exact word-count
  requirement).
- Two-surah clip → two matches in canon order.
- Repeated-ayah recitation (repeats within canon) → chain prefers the
  earliest consistent path; score reflects coverage.
- Clean single-surah transcript → one match, score ≥ 0.9 (same as today).

### Stage 3 — `backend/main.py` multi-match pipeline

`/align` rework (single endpoint, same signature):

1. Extract audio (unchanged, ffmpeg → 16k mono wav).
2. `vad.get_speech_segments` → if none → "No speech detected in audio".
3. `vad.chunk_for_whisper` → per-chunk Whisper ASR (existing chunked loop in
   `transcribe_word_timestamps`, now driven by VAD chunk boundaries instead
   of fixed windows + RMS gate). Keep absolute timestamp offsets.
4. `detector.detect_all(full_transcript)` → ordered matches.
   - Fallback: `detect_tight`; if still nothing → "Could not identify any
     surah".
5. Per match (in order):
   - Slice audio to `speech_span` (whole-clip span, one slice — matches are
     sequential within it) → `detect_bismillah` only for the FIRST match.
   - Build canon words for `(surah, start_ayah..end_ayah)` (+ ayah 0 words if
     bismillah detected and match is first).
   - `align_words(sliced)` → offset timestamps back to absolute.
   - Group into ayah segments via `segments_from_word_times`.
6. Response `AlignResponse` (models.py) gains:
   - `matches: list[MatchInfo]` where `MatchInfo = {surah: SurahInfo,
     start_ayah, end_ayah, confidence}`.
   - Existing fields stay populated: `surah`/`start_ayah`/`end_ayah`/
     `confidence` = FIRST match; `segments`/`srt`/`vtt`/`words`/`chunks`
     span ALL matches (chronological). Frontend keeps working unchanged.
7. Error taxonomy (all via existing `ErrorResponse`):
   - "No speech detected in audio" (VAD empty)
   - "Could not identify any surah" (detection failed)
   - "Could not map recitation to specific ayahs" (alignment failed,
     fallback segments empty)

### Stage 4 — Frontend (minimal)

- No changes required for correctness (top match + all segments render).
- Optional: show match count / surah list in the detection card.
  Deferred — not needed for the caption-first goal.

## Dependencies

- `silero-vad` (pip). ~2MB. Torch mode. Falls back to energy if unavailable.

## Testing

- Unit (pytest-style scripts matching repo convention — plain scripts, no
  pytest dep):
  - `test_vad.py`: tone → no speech; ikhlas wav → 1 span; synthetic
    silence-gapped speech → merged; chunk_for_whisper boundaries ≤ 30s.
  - `test_detector_all.py`: clean full surah; mid-ayah start (slice of
    Fatihah 1-2); multi-surah (ikhlas + kawthar transcripts joined);
    ASR-noisy transcript (2 word substitutions); repeated ayah.
- Integration: extend `test_full.py` path or `test_align.py` with synthetic
  multi-surah wav (ikhlas + kawthar concatenated); realistic silence clip;
  user's failing clip (when available).
- Regression: all existing `test_*.py` must still pass; E2E UI test if the
  user's clip is available.

## Out of Scope

- Whisper model upgrade / batching / GPU.
- LLM-based surah disambiguation (future).
- Multiple simultaneous requests / job queue for /align.
