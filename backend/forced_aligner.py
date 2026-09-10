from __future__ import annotations

import re
import gc

import numpy as np
import soundfile as sf

from backend.config import DEVICE, configure_torch

_DIACRITICS = re.compile(
    r"[\u064B-\u065F\u0670\u06D6-\u06ED\u0640\u064E-\u0652\ufdfd\u06E0-\u06E5]"
)
_FRAME_MS = 0.02


class ForcedAligner:
    """Character-level CTC forced alignment (wav2vec2 xlsr-53 Arabic) with a
    pure-torch Viterbi decoder. Produces real per-word timestamps."""

    def __init__(self, model_name: str = "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"):
        self.model_name = model_name
        self.processor = None
        self.model = None
        self.vocab: dict[str, int] = {}
        self.space_id: int | None = None

    def load(self):
        if self.model is not None:
            return
        torch = configure_torch()
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        self._torch = torch
        self.processor = Wav2Vec2Processor.from_pretrained(
            self.model_name, local_files_only=True
        )
        self.model = Wav2Vec2ForCTC.from_pretrained(
            self.model_name, local_files_only=True
        )
        if DEVICE == "cuda":
            self.model = self.model.to("cuda")
        self.model.eval()
        self.vocab = self.processor.tokenizer.get_vocab()
        self.space_id = self.vocab.get("|", self.vocab.get(" ", 4))

    def unload(self):
        """Release the wav2vec2 model + processor from memory."""
        self.processor = None
        self.model = None
        gc.collect()

    def _normalize_chars(self, text: str) -> str:
        text = _DIACRITICS.sub("", text)
        text = text.replace("\u0629", "\u0647").replace("\u0649", "\u064a")
        text = text.replace("\u0671", "\u0627")
        text = text.replace(" ", "|").strip()
        invalid = {c for c in text} - (
            set(self.vocab) - {"<pad>", "<s>", "</s>", "<unk>"}
        )
        if invalid:
            return ""
        return text

    def align_words(self, audio_path: str, canon_words: list[tuple[int, str]]) -> list[dict] | None:
        """canon_words: [(ayah, normalized_word), ...] in recitation order.
        Returns [{ayah, text, start, end}] or None on failure."""
        self.load()
        text = " ".join(w for _, w in canon_words)
        norm = self._normalize_chars(text)
        if not norm:
            return None

        audio, sr = sf.read(audio_path)
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000

        input_values = self.processor(
            audio, sampling_rate=sr, return_tensors="pt"
        ).input_values
        if DEVICE == "cuda":
            input_values = input_values.to("cuda")

        # wav2vec2 self-attention is O(T^2) — long slices (whole-clip matches)
        # would need ~4GB for a 10-min clip. Run the model in 60s chunks and
        # concatenate frame-level logits (CTC logits are frame-local, so chunk
        # boundaries only lose attention context beyond 60s and ~1 conv frame
        # per boundary).
        logits_list = []
        chunk_samples = 60 * 16000
        with self._torch.no_grad():
            for start in range(0, input_values.shape[-1], chunk_samples):
                chunk = input_values[:, start:start + chunk_samples]
                logits_list.append(self.model(chunk).logits[0])
        logits = self._torch.cat(logits_list, dim=0).float().cpu()
        del input_values

        char_ids = [self.vocab[c] for c in norm]
        path = self._viterbi(logits, char_ids)
        del logits
        if path is None or not any(t != 0 for t in path):
            return None

        frame_char = self._state_to_char_index(path, len(char_ids))
        spans = self._char_spans(frame_char)
        words = self._spans_to_words(spans, norm, canon_words)
        return words or None

    def detect_leading_openings(self, audio_path: str, window_s: float = 12.0) -> list[str]:
        """Greedy CTC decode of the first few seconds — which reciter openers
        (takbir / isti'adha / Bismillah) begin the audio, in order. Alafasy
        clips typically open با takbir then Bismillah; matches are tolerant so
        ASR's plain-alif spellings (الشيطان) match Uthmani canon (الشيطن)."""
        self.load()
        audio, sr = sf.read(audio_path)
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000
        n = int(min(window_s, len(audio) / sr) * sr)
        audio = audio[:n]

        input_values = self.processor(
            audio, sampling_rate=sr, return_tensors="pt"
        ).input_values
        if DEVICE == "cuda":
            input_values = input_values.to("cuda")
        with self._torch.no_grad():
            logits = self.model(input_values).logits[0].float()
        del input_values

        ids = logits.argmax(dim=-1).tolist()
        inv = {v: k for k, v in self.vocab.items()}
        letters = []
        prev = -1
        for tok in ids:
            if tok != prev and tok != 0:
                letters.append(inv.get(tok, ""))
            prev = tok
        text = "".join(letters).replace("|", " ")
        text = _DIACRITICS.sub("", text).strip()
        tokens = text.split()

        from backend.segmenter import match_leading_openers

        return match_leading_openers(tokens)

    def _viterbi(self, logits, chars: list[int]) -> list[int] | None:
        """CTC Viterbi over expanded states [B c0 B c1 ... B]. Returns per-frame
        state index (0..2L), or None if no valid path.

        Memory-light: uint8 backtrack matrix + per-frame gather (no (T, 2L+1)
        float materialization), AND windowed: the backtrack matrix and state
        space only span a sliding char window per ~60s chunk. The full-surah
        backtrack would be (frames x 2*chars) — e.g. ~24GB for a full
        Al-Baqarah (30k chars, 2h) — which swaps the machine to death. The
        window is anchored on the previous chunk's best end position with an
        80-char margin, so long clips stay in a few MB of state."""
        torch = self._torch
        T, C = logits.shape
        L = len(chars)
        if T == 0:
            return None
        logsm = torch.log_softmax(logits, dim=-1)

        CHUNK_FRAMES = int(60.0 / _FRAME_MS)  # 3000 frames (~60s at 20ms)
        MARGIN = 80  # chars of drift headroom around the linear anchor
        n_chunks = (T + CHUNK_FRAMES - 1) // CHUNK_FRAMES

        neg = torch.full((1,), -1e30)
        full_states = torch.full((2 * L + 1,), -1e30)
        full_states[0] = 0.0
        anchor = 0  # absolute char position the path is expected to be near
        back_mats: list[torch.Tensor] = []
        windows: list[tuple[int, int, int, int]] = []  # (f0, f1, w0, w1)

        for ci in range(n_chunks):
            f0 = ci * CHUNK_FRAMES
            f1 = min(f0 + CHUNK_FRAMES, T)
            rate = max(1, int(round((f1 - f0) / T * L)))
            w0 = max(0, anchor - MARGIN)
            w1 = min(L, anchor + rate + MARGIN)
            if w1 <= w0:
                w1 = min(L, w0 + 1)
            windows.append((f0, f1, w0, w1))
            nw = 2 * (w1 - w0) + 1
            tok_ids = torch.full((nw,), 0, dtype=torch.long)
            tok_ids[1::2] = torch.tensor(chars[w0:w1], dtype=torch.long)
            lv_all = logsm[f0:f1, tok_ids]
            back = torch.zeros((f1 - f0, nw), dtype=torch.uint8)
            sl = full_states[2 * w0: 2 * w1 + 1]
            for t in range(f1 - f0):
                lv = lv_all[t]
                stay = sl + lv
                move = torch.cat([neg, sl[:-1] + lv[1:]])
                up = move > stay
                sl.copy_(torch.where(up, move, stay))
                back[t] = up.to(torch.uint8)
            anchor = w0 + int(torch.argmax(sl)) // 2
            back_mats.append(back)
        del logsm

        # Only the last chunk's states are valid for the clip end (earlier
        # windows hold frozen boundary values that never decayed).
        w0_last = windows[-1][2]
        best_end = w0_last * 2 + int(torch.argmax(full_states[2 * w0_last:]))
        if full_states[best_end] <= -1e20:
            return None

        path = []
        s = best_end
        for ci in range(n_chunks - 1, -1, -1):
            f0, f1, w0, w1 = windows[ci]
            back = back_mats[ci]
            rel = s - 2 * w0
            for t in range(f1 - 1, f0 - 1, -1):
                path.append(s)
                if rel <= 0:
                    continue
                if back[t - f0, rel] == 1:
                    s -= 1
                    rel -= 1
        path.reverse()
        return path

    def _state_to_char_index(self, path: list[int], L: int) -> list[int]:
        """Map state index -> char index (-1 for blank)."""
        out = []
        for s in path:
            if s % 2 == 1:
                out.append(s // 2)
            else:
                out.append(-1)
        return out

    def _char_spans(self, frame_char: list[int]) -> list[tuple[int, int, int]]:
        spans = []
        cur, start = -1, -1
        for f, c in enumerate(frame_char):
            if c != cur:
                if cur >= 0:
                    spans.append((cur, start, f))
                cur, start = c, f
        if cur >= 0:
            spans.append((cur, start, len(frame_char)))
        return spans

    def _spans_to_words(
        self,
        spans: list[tuple[int, int, int]],
        norm_text: str,
        canon_words: list[tuple[int, str]],
    ) -> list[dict]:
        norm_chars = list(norm_text)
        # canon word index of each normalized char (words are space-separated)
        char_word = []
        wi = 0
        for ch in norm_chars:
            char_word.append(wi if ch not in ("|", " ") else -1)
            if ch == "|":
                wi += 1
        n_canon = len(canon_words)
        covered: set[int] = set()
        words: list[dict] = []
        cur_chars: list[tuple[int, int]] = []
        cur_ci: int | None = None

        for span_ci, s, e in spans:
            if span_ci >= len(norm_chars):
                break
            ch = norm_chars[span_ci]
            if ch in ("|", " "):
                if cur_chars and cur_ci is not None:
                    start = min(fs for fs, _ in cur_chars) * _FRAME_MS
                    end = max(fe for _, fe in cur_chars) * _FRAME_MS
                    words.append({"start": start, "end": end})
                    covered.add(cur_ci)
                    cur_chars = []
                    cur_ci = None
            else:
                cur_chars.append((s, e))
                cur_ci = char_word[span_ci]

        if cur_chars and cur_ci is not None:
            start = min(fs for fs, _ in cur_chars) * _FRAME_MS
            end = max(fe for _, fe in cur_chars) * _FRAME_MS
            words.append({"start": start, "end": end})
            covered.add(cur_ci)

        if len(words) != n_canon:
            # Missing words are only acceptable as a TRAILING run (audio ends
            # mid-ayah / mid-word). A middle gap means misalignment -> fail.
            missing = [i for i in range(n_canon) if i not in covered]
            if missing and missing != list(range(min(missing), n_canon)):
                return []
            if not covered:
                return []
            words = [w for w, ci in zip(words, sorted(covered))][:max(covered) + 1]

        return [
            {"ayah": ayah, "text": word, "start": w["start"], "end": w["end"]}
            for (ayah, word), w in zip(canon_words, words)
        ]
