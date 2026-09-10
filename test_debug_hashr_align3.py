import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from backend.forced_aligner import ForcedAligner
from data.quran_text import load_ayah_texts, _normalize

# Rebuild exactly what main.py feeds: slice 16.44-89.98, canon 73 words
import soundfile as sf
from backend.aligner import ASREngine

fa = ForcedAligner()
eng = ASREngine()
try:
    texts_ar = load_ayah_texts()
    word_off = {1: (0, 11), 2: (0, 40), 3: (0, 14), 4: (0, 8)}
    canon = []
    for ayah, (first, last) in word_off.items():
        words = _normalize(texts_ar.get(f"59:{ayah}", "")).split()
        canon.extend((ayah, w) for w in words[first:last])
    print("canon:", len(canon))

    sliced = eng.slice_audio(r"temp\hashr_90s.wav", 16.44, 89.98)
    print("sliced:", sliced, os.path.getsize(sliced))
    a, sr = sf.read(sliced)
    print("slice dur:", len(a)/sr)

    fa.load()
    text = " ".join(w for _, w in canon)
    norm = fa._normalize_chars(text)
    print("norm chars:", len(norm))
    input_values = fa.processor(a, sampling_rate=sr, return_tensors="pt").input_values
    print("frames:", input_values.shape[-1], "=", round(input_values.shape[-1]*0.02,1), "s")
    chunk_samples = 60 * 16000
    logits_list = []
    with fa._torch.no_grad():
        for start in range(0, input_values.shape[-1], chunk_samples):
            chunk = input_values[:, start:start + chunk_samples]
            logits_list.append(fa.model(chunk).logits[0])
    logits = fa._torch.cat(logits_list, dim=0).float().cpu()
    char_ids = [fa.vocab[c] for c in norm]
    path = fa._viterbi(logits, char_ids)
    if path is None:
        print("VITERBI FAILED")
    else:
        frame_char = fa._state_to_char_index(path, len(char_ids))
        spans = fa._char_spans(frame_char)
        nonblank = [sp for sp in spans if sp[0] >= 0]
        norm_chars = list(norm)
        covered = set(c for c, _, _ in nonblank if c < len(norm_chars))
        miss = [norm_chars[i] for i in range(len(norm_chars)) if i not in covered]
        print("nonblank spans:", len(nonblank), "covered:", len(covered), "/", len(norm_chars))
        print("missing:", repr("".join(miss[:120])))
        words = fa._spans_to_words(spans, norm, canon)
        print("words:", None if not words else len(words))
        if words:
            for w in words[:15]:
                print(round(w["start"],2), round(w["end"],2), w["ayah"], w["text"])
finally:
    fa.unload()
