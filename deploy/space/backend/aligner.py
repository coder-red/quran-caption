import os
import tempfile
import subprocess
import gc
import numpy as np
import soundfile as sf
from backend.config import ASR_MODEL, DEVICE, DTYPE, FFMPEG_PATH, TEMP_DIR, configure_torch


class ASREngine:
    def __init__(self, model_name: str = ASR_MODEL):
        self.model_name = model_name
        self.processor = None
        self.model = None

    def load(self):
        if self.model is not None:
            return
        torch = configure_torch()
        from transformers import WhisperProcessor, WhisperForConditionalGeneration

        self._torch = torch
        self.processor = WhisperProcessor.from_pretrained(
            self.model_name, local_files_only=True
        )
        self.model = WhisperForConditionalGeneration.from_pretrained(
            self.model_name, local_files_only=True
        )
        if DEVICE == "cuda":
            self.model = self.model.to("cuda")
        self.model.config.forced_decoder_ids = None

        gc = self.model.generation_config
        gc.return_timestamps = True
        if not hasattr(gc, "no_timestamps_token_id"):
            gc.no_timestamps_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|notimestamps|>")
        if not hasattr(gc, "begin_suppress_tokens"):
            gc.begin_suppress_tokens = [self.processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")]
        if not hasattr(gc, "decoder_start_token_id"):
            gc.decoder_start_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
        if not hasattr(gc, "forced_decoder_ids"):
            gc.forced_decoder_ids = None

        tok = self.processor.tokenizer
        lang_ar = tok.convert_tokens_to_ids("<|ar|>")
        transcribe = tok.convert_tokens_to_ids("<|transcribe|>")
        start = tok.convert_tokens_to_ids("<|startoftranscript|>")
        self._forced_decoder_ids = [[1, start], [2, lang_ar], [3, transcribe]]

    def unload(self):
        """Release ASR model + processor from memory (call before loading
        the aligner so peak RAM stays ~one model at a time)."""
        self.processor = None
        self.model = None
        gc.collect()

    def transcribe_word_timestamps(self, audio_path: str) -> dict:
        """Chunked Whisper ASR: 30s windows over the whole clip (Whisper can
        only see 30s at a time), skipping silent windows. Returns transcript +
        absolute chunk times."""
        self.load()
        audio_input, sr = sf.read(audio_path)
        if sr != 16000:
            import librosa
            audio_input = librosa.resample(audio_input, orig_sr=sr, target_sr=16000)
            sr = 16000

        if len(audio_input.shape) > 1:
            audio_input = audio_input.mean(axis=1)

        window_s = 30.0
        total = len(audio_input) / sr
        n_windows = max(1, int(-(-total // window_s)))
        rms = []
        for i in range(n_windows):
            s = audio_input[int(i * window_s * sr):int(min(total, (i + 1) * window_s) * sr)]
            rms.append(float(np.sqrt(np.mean(s ** 2))) if len(s) else 0.0)
        threshold = max(0.005, 0.15 * max(rms)) if rms else 0.0

        all_chunks: list[dict] = []
        texts: list[str] = []
        for i in range(n_windows):
            if rms[i] < threshold:
                continue
            start = int(i * window_s * sr)
            end = min(len(audio_input), start + int(window_s * sr))
            seg = audio_input[start:end]
            pos = i * window_s

            input_features = self.processor(
                seg, sampling_rate=sr, return_tensors="pt"
            ).input_features

            with self._torch.no_grad():
                generated = self.model.generate(
                    input_features,
                    forced_decoder_ids=self._forced_decoder_ids,
                    return_timestamps=True,
                    output_attentions=False,
                )

            text = self.processor.tokenizer.decode(
                generated[0], skip_special_tokens=True
            ).strip()
            if text:
                texts.append(text)
            for c in self._extract_timestamps(generated, seg, sr):
                c["start"] = round(c["start"] + pos, 3)
                c["end"] = round(c["end"] + pos, 3)
                all_chunks.append(c)

        return {
            "text": " ".join(texts).strip(),
            "chunks": all_chunks,
            "words": self._words_from_chunks(all_chunks),
        }

    def transcribe_chunks(self, audio_path: str, chunks: list[tuple[float, float]]) -> dict:
        """Whisper ASR over explicit VAD chunks (absolute timestamps)."""
        self.load()
        audio_input, sr = sf.read(audio_path)
        if sr != 16000:
            import librosa
            audio_input = librosa.resample(audio_input, orig_sr=sr, target_sr=16000)
            sr = 16000
        if len(audio_input.shape) > 1:
            audio_input = audio_input.mean(axis=1)
        all_chunks = []
        texts = []
        seg_bounds = []
        for (cs, ce) in chunks:
            seg = audio_input[int(cs * sr):int(ce * sr)]
            if len(seg) < sr * 0.2:
                continue
            feats = self.processor(seg, sampling_rate=sr, return_tensors="pt").input_features
            with self._torch.no_grad():
                generated = self.model.generate(
                    feats,
                    forced_decoder_ids=self._forced_decoder_ids,
                    return_timestamps=True,
                    output_attentions=False,
                )
            text = self.processor.tokenizer.decode(generated[0], skip_special_tokens=True).strip()
            if text:
                texts.append(text)
                seg_bounds.append((cs, ce))
            for c in self._extract_timestamps(generated, seg, sr):
                c["start"] = round(c["start"] + cs, 3)
                c["end"] = round(c["end"] + cs, 3)
                all_chunks.append(c)
        return {
            "text": " ".join(texts).strip(),
            "chunks": all_chunks,
            "words": self._words_from_chunks(all_chunks),
            "chunk_texts": texts,
            "chunk_bounds": seg_bounds,
        }

    def slice_audio(self, audio_path: str, start: float, end: float) -> str:
        """Write audio[start:end] (seconds) to a temp wav; returns the path.
        Keeps forced alignment bounded to the recitation span instead of the
        whole clip (wav2vec2 would otherwise eat GBs of RAM on long videos)."""
        audio, sr = sf.read(audio_path)
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        fd, out = tempfile.mkstemp(suffix=".wav", dir=TEMP_DIR)
        os.close(fd)
        sf.write(out, audio[int(start * sr):int(end * sr)], sr)
        return out

    def detect_speech_span(
        self, audio_path: str, hop_s: float = 0.05, min_sec: float = 4.0
    ) -> tuple[float, float] | None:
        """Rough recitation region via RMS energy. Returns (start, end) in
        seconds, or None if the clip looks silent. Whisper timestamps can't be
        trusted (1 ts token per clip) so we locate speech by energy instead."""
        audio, sr = sf.read(audio_path)
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        frame = max(1, int(hop_s * sr))
        n = len(audio) // frame
        if n == 0:
            return None
        rms = np.sqrt(np.mean(audio[:n * frame].reshape(-1, frame) ** 2, axis=1))
        thr = max(0.001, 0.05 * float(rms.max()))
        voiced = np.where(rms > thr)[0]
        if len(voiced) == 0:
            return None
        merged = []
        s0 = prev = voiced[0]
        for i in voiced[1:]:
            if i - prev > int(1.5 / hop_s):
                merged.append((s0, prev + 1))
                s0 = i
            prev = i
        merged.append((s0, prev + 1))
        start = merged[0][0] * hop_s
        end = merged[-1][1] * hop_s
        if end - start < min_sec:
            return None
        return start, end

    def _extract_timestamps(self, generated, audio_input, sr) -> list[dict]:
        chunks = []
        tokens = generated[0].tolist()
        timestamp_begin = 50257

        segments = []
        current_segment = []
        current_start = None

        for token in tokens:
            if token >= timestamp_begin:
                time = (token - timestamp_begin) * 0.02
                if current_start is None:
                    current_start = time
                else:
                    if current_segment:
                        text = self.processor.tokenizer.decode(
                            current_segment, skip_special_tokens=True
                        ).strip()
                        if text:
                            segments.append({
                                "text": text,
                                "timestamp": (current_start, time),
                            })
                        current_segment = []
                    current_start = time
            else:
                current_segment.append(token)

        if current_segment:
            text = self.processor.tokenizer.decode(
                current_segment, skip_special_tokens=True
            ).strip()
            if text:
                segments.append({
                    "text": text,
                    "timestamp": (current_start, current_start + 0.5),
                })

        for seg in segments:
            text = seg["text"]
            start, end = seg["timestamp"]
            if text:
                chunks.append({
                    "text": text,
                    "start": start,
                    "end": end,
                })

        return chunks

    def _words_from_chunks(self, chunks: list[dict]) -> list[dict]:
        words = []
        for chunk in chunks:
            text = chunk["text"]
            start, end = chunk["start"], chunk["end"]
            parts = text.split()
            if not parts:
                continue
            word_dur = (end - start) / len(parts)
            for i, word in enumerate(parts):
                words.append({
                    "text": word,
                    "start": start + i * word_dur,
                    "end": start + (i + 1) * word_dur,
                })
        return words

    def extract_audio(self, video_path: str, sr: int = 16000) -> tuple[str, float]:
        fd, out_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

        cmd = [
            FFMPEG_PATH,
            "-i", video_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", str(sr),
            "-ac", "1",
            "-y",
            out_path,
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            msg = result.stderr.decode("utf-8", errors="replace")[-600:]
            raise RuntimeError(f"ffmpeg extract_audio failed rc={result.returncode}: {msg}")

        audio_data, sample_rate = sf.read(out_path)
        duration = len(audio_data) / sample_rate
        return out_path, duration

    def cleanup(self, path: str):
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass
