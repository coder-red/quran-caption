# Caption sync investigation handoff

## Goal

Find and fix why captions do not follow the reciter's voice. Reproduce the user's failure on a real clip, identify whether the error comes from alignment, preview playback, or burned rendering, then make the smallest evidence-based correction. Do not start by changing timing constants globally.

## Current timing path

1. `backend/main.py::_align_impl` reads the media, gets raw speech intervals from VAD, and sends them to the selected ASR provider.
2. `backend/main.py::_align_with_provider` receives provider word timestamps, snaps timestamps around detected silences, detects ayahs, and calls `segments_from_provider_times`.
3. `backend/segmenter.py` derives phrase edges from matched words, applies `_vad_clamp_bounds`, may extend a madd tail with `_madd_extend`, splits at measured pauses, and applies `apply_sync_padding`.
4. The resulting `segments` and SRT/VTT are returned by `/align`. The local forced-align path uses `segments_from_word_times` instead of provider timestamps.
5. The live overlay in `frontend/app.py` compares segment times with `video.currentTime`. The optional burned MP4 uses the returned SRT through the renderer.
6. Deployed preview media is transcoded in `backend/main.py::media_preview`; its timestamps are normalized with `-avoid_negative_ts make_zero`.

## Existing fixes and constraints

- Provider timestamps are noisy. Raw (unmerged) VAD segments are used for anchoring and phrase edge correction; measured silence is the authority for pauses.
- `_vad_clamp_bounds` uses `VAD_EDGE_SNAP` for edges inside a continuous VAD segment and the stricter `VAD_CLAMP_TOL` for neighboring speech across silence.
- `_madd_extend` only extends a madd-ending phrase while the same raw VAD segment still covers it. It is capped and must not cross a real silence or the next phrase.
- `apply_sync_padding` enforces gaps and keeps padding out of measured silences.
- The live player uses `video.currentTime`; its caption timing data is re-parsed when segment/style data changes. Do not assume a stale overlay is an ASR problem.
- Do not remove the existing pause, opener, multi-surah, or mid-ayah behavior while fixing sync. Do not trim canon text based only on detector-chain words; skipped/misheard words can still be recited.
- The working tree already contains user changes across backend, frontend, and deploy files. Inspect `git diff` before editing and preserve unrelated work. `deploy/space` has previously diverged; compare before copying files.

## Investigation steps

1. **Get a reproducible failure.** Use the exact source clip and identify a few timestamps where captions are early, late, disappear during a held syllable, or remain during silence. Record which ASR backend was selected and whether the complaint is in the live preview, downloaded SRT, or burned MP4. If the source clip is unavailable, ask for it and the approximate bad timestamps before guessing.
2. **Trace one bad caption end to end.** Compare the audio's audible word/phrase onset and offset against provider word times, raw VAD intervals, generated segment times, API/SRT times, browser `currentTime`, and (if applicable) rendered output. Determine where the first divergence appears.
3. **Separate the failure classes.**
   - API/SRT segment is already mistimed: inspect ASR word stamps, provider anchoring, detector word chain, VAD edges, phrase splitting, madd extension, and padding.
   - API/SRT is correct but live overlay is not: inspect preview media timestamps, browser `currentTime`, player segment parsing/update behavior, and reruns.
   - Live overlay is correct but burned output is not: inspect renderer input timestamps, SRT cue times, and output A/V start offsets.
4. **Compare providers and paths where practical.** Check whether the same clip fails with Groq, Gemini, and ElevenLabs, and whether the local forced-align path differs. A shared failure points toward common segment construction or playback; a provider-specific failure points upstream.
5. **Measure against audio evidence.** Do not treat VAD boundaries as exact word boundaries or use `test_sync_edges.py::measure_sync` alone as proof: it compares caption edges with coarse speech-segment edges. Use a short manually checked sample with several onsets, offsets, pauses, and held vowels; report signed errors and examples.
6. **Implement only after locating the cause.** Prefer a local correction with a regression case for the reproduced failure. Keep silence exclusion and non-overlap guarantees. Avoid relaxing edge tolerances or extending every caption to a VAD edge without evidence.
7. **Verify the same failure is fixed.** Re-run the exact clip and timestamps through the affected path, inspect API/SRT and playback/render output as relevant, then check existing pause/sync/madd/provider coverage. Report any provider, clip, or deployment path not verified.

## Relevant code and checks

- `backend/main.py`: `_align_impl`, `_align_with_provider`, `_build_align_response`, `/media/{media_id}/preview`
- `backend/segmenter.py`: `snap_words_to_silence`, `_vad_clamp_bounds`, `_madd_extend`, `_phrase_split`, `apply_sync_padding`, `segments_from_provider_times`, `segments_from_word_times`
- `backend/providers/groq.py`: provider word timestamp handling and VAD anchoring
- `backend/providers/gemini.py`, `backend/providers/elevenlabs.py`: other provider timestamp paths
- `frontend/app.py`: caption player (`parseWrap`, `upd`, `video.currentTime`) and preview setup
- Existing focused checks: `test_sync_edges.py`, `test_madd_tails.py`, `test_pause_split.py`, provider tests, and `test_align_multimatch.py`

## Expected handoff result

Return the concrete root cause, the smallest fix, the exact clip/backend/timestamps used to verify it, measured before/after timing evidence, checks run, and any remaining limitations. If no representative clip is available, state what evidence is missing instead of claiming sync is fixed.
