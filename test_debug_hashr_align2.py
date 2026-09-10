import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from backend.forced_aligner import ForcedAligner
from data.quran_text import load_ayah_texts, _normalize

wav = r"temp\hashr_90s.wav"
fa = ForcedAligner()
try:
    fa.load()
    texts_ar = load_ayah_texts()
    canon = []
    for ayah in range(1, 5):
        for word in _normalize(texts_ar.get(f"59:{ayah}", "")).split():
            canon.append((ayah, word))
    print("canon words:", len(canon))
    text = " ".join(w for _, w in canon)
    norm = fa._normalize_chars(text)
    print("norm ok:", bool(norm), "chars:", len(norm))
    if not norm:
        sys.exit(1)

    import soundfile as sf
    audio, sr = sf.read(wav)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    input_values = fa.processor(audio, sampling_rate=sr, return_tensors="pt").input_values
    print("input frames:", input_values.shape[-1], "=", input_values.shape[-1]*0.02, "s")

    chunk_samples = 60 * 16000
    logits_list = []
    with fa._torch.no_grad():
        for start in range(0, input_values.shape[-1], chunk_samples):
            chunk = input_values[:, start:start + chunk_samples]
            logits_list.append(fa.model(chunk).logits[0])
    logits = fa._torch.cat(logits_list, dim=0).float().cpu()
    print("logits:", tuple(logits.shape))

    char_ids = [fa.vocab[c] for c in norm]
    print("char_ids:", len(char_ids))
    path = fa._viterbi(logits, char_ids)
    print("viterbi path:", None if path is None else len(path), "non-blank frames:", None if path is None else sum(1 for t in path if t % 2 == 1))

    if path is not None:
        frame_char = fa._state_to_char_index(path, len(char_ids))
        spans = fa._char_spans(frame_char)
        words = fa._spans_to_words(spans, norm, canon)
        print("words:", None if not words else len(words))
        nonblank = [sp for sp in spans if sp[0] >= 0]
        print("total spans:", len(spans), "nonblank:", len(nonblank))
        norm_chars = list(norm)
        covered = set(c for c, _, _ in nonblank if c < len(norm_chars))
        print("covered chars:", len(covered), "/", len(norm_chars))
        miss = [norm_chars[i] for i in range(len(norm_chars)) if i not in covered]
        print("missing chars:", len(miss), repr("".join(miss[:120])))
        if words:
            for w in words:
                print(round(w["start"], 2), round(w["end"], 2), w["ayah"], w["text"])
finally:
    fa.unload()
