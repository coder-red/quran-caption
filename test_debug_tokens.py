import sys, os, subprocess, tempfile
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import soundfile as sf
import torch
from backend.aligner import ASREngine

CLIP = r"data\Voice from Heart Beautiful Quran Recitation by Sheikh Ahmed Mokhtar _ AWAZ.mp4"
OUT = "temp\\token_debug.txt"

with tempfile.TemporaryDirectory() as td:
    wav = os.path.join(td, "clip.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-i", CLIP, "-ac", "1", "-ar", "16000", "-vn", wav],
        check=True, capture_output=True,
    )
    audio, sr = sf.read(wav)

    eng = ASREngine()
    eng.load()
    seg = audio[int(11.65 * sr):int(41.65 * sr)]
    feats = eng.processor(seg, sampling_rate=sr, return_tensors="pt").input_features
    with torch.no_grad():
        generated = eng.model.generate(
            feats,
            forced_decoder_ids=eng._forced_decoder_ids,
            return_timestamps=True,
        )
    tokens = generated[0].tolist()
    ts_begin = 50257
    lines = []
    lines.append(f"num tokens: {len(tokens)}")
    lines.append(f"timestamp tokens: {sum(1 for t in tokens if t >= ts_begin)}")
    stream = []
    for t in tokens:
        if t >= ts_begin:
            stream.append(f"<TS { (t - ts_begin) * 0.02:.2f}>")
        else:
            s = eng.processor.tokenizer.decode([t], skip_special_tokens=False)
            stream.append(repr(s))
    lines.append(" ".join(stream))
    lines.append("--- full decode skip_special ---")
    lines.append(repr(eng.processor.tokenizer.decode(generated[0], skip_special_tokens=True)))
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("written", OUT)
