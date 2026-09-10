import sys, os, subprocess, tempfile
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import soundfile as sf
from backend.vad import get_speech_segments, merge_segments, chunk_for_whisper
from backend.aligner import ASREngine

CLIP = r"data\Voice from Heart Beautiful Quran Recitation by Sheikh Ahmed Mokhtar _ AWAZ.mp4"

with tempfile.TemporaryDirectory() as td:
    wav = os.path.join(td, "clip.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-i", CLIP, "-ac", "1", "-ar", "16000",
         "-vn", wav],
        check=True, capture_output=True,
    )
    import wave
    with wave.open(wav, "rb") as wf:
        n = wf.getnframes()
    dur = n / 16000
    print(f"wav duration: {dur:.1f}s")

    audio, sr = sf.read(wav)
    speech = merge_segments(get_speech_segments(audio, sr))
    print(f"speech segments: {len(speech)}")
    for s in speech[:15]:
        print(f"  {s[0]:.2f}-{s[1]:.2f}")
    chunks = chunk_for_whisper(speech)
    print(f"chunks: {len(chunks)}")
    for c in chunks:
        print(f"  {c[0]:.2f}-{c[1]:.2f}")

    eng = ASREngine()
    res = eng.transcribe_chunks(wav, chunks)
    text = res["text"]
    chunk_texts = res["chunk_texts"]
    chunk_bounds = res["chunk_bounds"]
    words = res["words"]
    print(f"\nTOTAL TEXT len={len(text)}")
    print(text[:1500])
    print("\n--- per chunk ---")
    for i, (ct, (cs, ce)) in enumerate(zip(chunk_texts, chunk_bounds)):
        print(f"[{i}] {cs:.1f}-{ce:.1f}: {ct[:120]}")
