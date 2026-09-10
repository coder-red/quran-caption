import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from backend.forced_aligner import ForcedAligner
from data.quran_text import load_ayah_texts, _normalize

wav = r"temp\hashr_90s.wav"
fa = ForcedAligner()
try:
    texts_ar = load_ayah_texts()
    canon = []
    for ayah in range(1, 5):
        for word in _normalize(texts_ar.get(f"59:{ayah}", "")).split():
            canon.append((ayah, word))
    print("canon words:", len(canon))
    aligned = fa.align_words(wav, canon)
    print("aligned words:", len(aligned))
    if aligned:
        for w in aligned[:80]:
            print(round(w["start"], 2), round(w["end"], 2), w["ayah"], w["text"])
finally:
    fa.unload()
