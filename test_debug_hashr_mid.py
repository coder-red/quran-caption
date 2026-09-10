import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import json
from backend.detector import SurahDetector

d = json.load(open(r"temp\hashr_mid_align.json", encoding="utf-8"))
# Re-run detection to see the chain
from backend.aligner import ASREngine
from backend.vad import get_speech_segments, merge_segments, chunk_for_whisper
import soundfile as sf

wav = r"temp\hashr_mid45.wav"
if not os.path.exists(wav):
    import subprocess
    subprocess.run([r"C:\ffmpeg\ffmpeg-master-latest-win64-gpl\bin\ffmpeg.exe","-y","-v","error","-i",r"temp\hashr_mid45.mp4","-ar","16000","-ac","1",wav], check=True)

audio, sr = sf.read(wav)
print("dur:", len(audio)/sr)
segs = get_speech_segments(audio, sr)
print("speech:", [(round(s,2),round(e,2)) for s,e in segs])
chunks = chunk_for_whisper(merge_segments(segs))
print("chunks:", [(round(s,2),round(e,2)) for s,e in chunks])
eng = ASREngine()
result = eng.transcribe_chunks(wav, chunks)
eng.unload()
print("TRANSCRIPT:", result["text"][:600])

det = SurahDetector(debug_chain=True)
matches = det.detect_all(result["text"])
print("matches:", [(m.surah, m.start_ayah, m.end_ayah, m.score, m.word_off) for m in matches])
