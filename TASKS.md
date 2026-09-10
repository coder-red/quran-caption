# Quran Caption App — Task Plan

Rule: one small task at a time. Research the best practice **before** implementing each group (links below). Verify each task with its check command before moving on. Update checkboxes as you go.

Current verified state (don't re-do):
- /align: 16k WAV convert → whisper ASR → detect_tight (word-count windows) → wav2vec2 CTC forced alignment (pure-torch Viterbi) → ayah segments with real times. Verified on Al-Ikhlas (112:1-4, 94%) + Al-Kawthar (108:1-3, 91%).
- Frontend editor E2E verified (upload → detect → live bilingual captions → editor → SRT/VTT → render h264_qsv).
- BUGFIX backlog from this session: none open. Known: `temp/` wiped on server shutdown (by design).

---

## Group A — Test corpus (foundation; everything else is measured against this)

- [ ] **A1.** Write `scripts/make_corpus.py`: downloads ~8 short clips (30–120s) from `https://cdn.islamic.network/quran/audio-surah/128/ar.<reciter>/<surah>.mp3` + converts to 16k mono WAV. Reciters: `alafasy`, `abdulbasitmurattal`, `husary`, `minshawi`; surahs: 1 (Al-Fatiha, has Bismillah), 93, 94, 97, 99, 100, 108, 112 (short, different rhythm), 55 (Al-Rahman — **repeated phrase** `فَبِأَىِّ ءَالَآءِ رَبِّكُمَا تُكَذِّبَانِ`, the repeat stress-test).
  - Verify: all files exist in `data/corpus/`, each 15k–200k samples.
- [ ] **A2.** Baseline run: `scripts/run_corpus.py` calls `/align` on every clip, writes `temp/corpus_results.csv` (file, surah, ayahs, confidence, segments, error). Run against the running backend.
  - Verify: CSV has 8 rows; eyeball which rows fail. This defines the bug list.
- [ ] **A3.** Pick 1 clip per failure mode (Bismillah-open, mid-ayah start, pause-heavy, repeat-surah, different reciter timbre) and commit them as permanent fixtures in `data/corpus/` (they are small). Everything below is validated against these.

## Group B — ASR robustness (best practice: VAD + segment confidence filter)

Research (read before coding):
- Silence trimming + VAD: https://theneuralbase.com/whisper/learn/intermediate/reducing-hallucinations-with-vad/ — Silero VAD, frame-index gotcha, keep short pauses (<1.5s) intact, cap long gaps to ~0.3s.
- Segment-level filtering (no_speech_prob / avg_logprob / compression_ratio): https://github.com/OpenWhispr/openwhispr/issues/462 and https://theneuralbase.com/whisper/learn/beginner/hallucination-on-silence/

- [ ] **B1.** `pip install silero-vad`. Add `backend/vad.py` with `trim_silence(audio_16k) -> trimmed (keeps short pauses, caps long gaps)`.
  - Verify: on `data/test_kawthar_16k.wav`, trimmed duration < original, speech intact (spot-check via probe print).
- [ ] **B2.** In `backend/aligner.py` whisper call: feed VAD-trimmed audio instead of raw (whisper returns timestamps relative to trimmed audio — add the trim offset back to chunk timestamps; we only use whisper for *text*, alignment comes from forced aligner, so offset only affects chunk.start used by segmenter — keep chunk times in original frame).
  - Verify: Al-Ikhlas still detects 112/1-4, confidence not worse; kawthar same.
- [ ] **B3.** Post-filter whisper segments: drop segments with `no_speech_prob > 0.6` OR (`compression_ratio > 2.2` AND text short). Rebuild transcript from kept segments. Config flags in `config.py`.
  - Verify: transcript of test clips contains no garbage lines; hallucinated-clip test (mp3 without wav conversion, if it still hallucinates) gets filtered.
- [ ] **B4.** Re-run `run_corpus.py`; compare CSV before/after. Any clip newly failing = regression → stop and fix.

## Group C — Detection / segmentation edge cases

Research (read before coding):
- Munajjam (most relevant existing project): 4 alignment strategies, drift correction, ASR-confusion-aware similarity: https://github.com/Itqan-community/Munajjam
- Chunk-boundaries-as-truth + explicit "no linear word interpolation": https://github.com/sayedmahmoud266/quran-ai-transcriping

- [ ] **C1.** **Mid-ayah start**: clip starting at ayah 3 of a surah. detect_tight currently searches whole surah windows; add: if best match starts mid-ayah and scores ≥ threshold, map correctly. Verify with a generated clip (ffmpeg cut of alafasy 112 starting at ayah 2) → expect detected ayahs 2–4 (not 1–4).
- [ ] **C2.** **Repeated phrase surah** (55): segmenter greedy mapping must not re-match the repeated ayah. Add repeat-guard: once an ayah is consumed, its canon words can't match again. Verify: Al-Rahman clip yields all ayahs in order, none duplicated.
- [ ] **C3.** **Pauses**: ayah boundary detected from whisper chunk gaps vs forced-align word gaps — make `segments_from_word_times` resilient when a pause crosses a word (skip trailing-silence words, never let ayah end < previous ayah end). Verify: pause-heavy clip has no negative/overlapping segment times.
- [ ] **C4.** **Multi-surah clips** (continuous recitation): out of scope — ensure graceful error message (already exists) + document. Decision recorded, not implemented.
- [ ] **C5.** Re-run corpus; fix regressions.

## Group D — Alignment hardening

- [ ] **D1.** Relax `_spans_to_words` strictness: instead of `len(words) != len(canon_words) → fail`, allow char-level merge errors by splitting on `|` tokens directly (per-word span lookup), keep failure only when a word gets zero chars. Verify: same outputs on ikhlas/kawthar (identical to before), fewer None failures on corpus.
- [ ] **D2.** Sanity guards on returned word times: monotonic non-decreasing, `end-start >= 0.05s` clamp, word times inside clip. Verify: asserts pass across corpus.
- [ ] **D3.** (Optional experiment) `torchaudio.functional.forced_align` C++ impl — docs say faster and more accurate than hand-rolled Viterbi: https://docs.pytorch.org/audio/stable/tutorials/ctc_forced_alignment_api_tutorial.html. Requires `pip install torchaudio` (~heavy). **Decision needed**: install or keep pure-torch Viterbi (currently correct on all tested clips). If installed: swap only inside `_viterbi`, keep interface; verify byte-identical times ±1 frame on fixtures.

## Group E — Multiple reciters (the user-visible goal)

- [ ] **E1.** For each reciter (alafasy, abdulbasitmurattal, husary, minshawi): one short-surah clip each, run corpus, record quality. Verify: all 4 detect correct surah with conf ≥ 0.8.
- [ ] **E2.** Fix whatever fails (likely: whisper transcript quality differs per reciter → detector threshold tuning; different Bismillah durations → detect window). Log thresholds in config.

## Group F — Performance

- [ ] **F1.** `torch.set_num_threads(psutil.cpu_count(logical=False))` in config load (currently 2 threads). Measure /align wall time on a 120s clip before/after.
- [ ] **F2.** Cache the processor+model instances module-level (already per-class; ensure single shared `ForcedAligner` instance in main.py — check it isn't re-created per request).
- [ ] **F3.** (Experiment) TBOGamer22/wav2vec2-quran-phonetics — wav2vec2-base fine-tuned on Quran recitation, made for forced alignment (smaller/faster than xlsr-large): https://huggingface.co/TBOGamer22/wav2vec2-quran-phonetics. If it aligns fixtures correctly → swap model name, measure speed. Rollback if quality drops.

## Group G — Frontend polish (only after B–E pass)

- [ ] **G1.** Renderer font-size correctness: ASS/force_style FontSize is relative to PlayResY (384 for SRT) — verify our font_size maps sensibly at 720p: https://www.ffmpeg-micro.com/blog/ffmpeg-subtitles-filter-guide. Fix mapping if needed.
- [ ] **G2.** Editor validation: data_editor rows enforce start<end, no negative times; show inline error instead of silent bad render.
- [ ] **G3.** Show per-segment confidence in the editor (add `conf` column from aligner word scores, read-only).

## Group H — Tests & hygiene

- [ ] **H1.** Unit tests (`tests/`) with fixtures: `_normalize_chars`, `detect_tight` windows, `segments_from_word_times`, `_spans_to_words`, SRT builder — all offline/fast, no model.
- [ ] **H2.** `scripts/run_corpus.py` becomes the smoke suite; wire into a `--check` flag: starts backend if down, runs corpus, prints summary. Run after every backend change.
- [ ] **H3.** Cleanup: `test_*.py` at root → move corpus runners into `scripts/`; delete obsolete scratch.
- [ ] **H4.** Update README: pipeline description (VAD → whisper → detect → forced align), corpus instructions, known limitations (single-surah clips, Hafs reciters).
