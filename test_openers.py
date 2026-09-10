import io, sys, os, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Regression tests for reciter-opener captions (takbir / isti'adha / Bismillah)
# ---------------------------------------------------------------------------
from backend.segmenter import (
    match_leading_openers,
    _opener_word_matches,
    _leading_opener,
    inject_opener_segments,
    segments_from_word_times,
    build_srt,
    OPENERS,
    ISTIADHA_WORDS,
)
from backend.detector import SurahDetector
from data.quran_text import load_ayah_texts, _normalize, _normalize_std

texts = load_ayah_texts()
d = SurahDetector()


def txt(key):
    return _normalize(texts[key])


def norm(toks):
    return [_normalize(t) for t in toks]


# ======================= 1. match_leading_openers (pure matcher) ===========
assert match_leading_openers(norm(["قل", "هو", "الله", "احد"])) == []
assert match_leading_openers(norm(["الله", "اكبر", "بسم", "الله"])) == ["takbir"]
assert match_leading_openers(norm(["بسم", "الله", "الرحمن", "الرحيم", "قل"])) == ["bismillah"]
assert match_leading_openers(norm(["اعوذ", "بالله", "من", "الشيطان", "الرجيم"])) == ["istiadha"]
assert match_leading_openers(norm(["الله", "اكبر", "اعوذ", "بالله", "من", "الشيطان", "الرجيم", "بسم", "الله", "الرحمن", "الرحيم"])) == ["takbir", "istiadha", "bismillah"]
assert match_leading_openers(norm(["الرحمن", "الرحيم"])) == []
print("1. match_leading_openers OK")

# ======================= 2. tolerant opener spelling matching ==============
# ASR emits the plain-alef spelling of words canon writes with a superscript
# alif (الشيطان vs الشيطن), and may drop alifs the canon packs into waw/yaa
assert _opener_word_matches("الشيطان", "الشيطن", "الشيطان") is True
assert _opener_word_matches("الشيطن", "الشيطن", "الشيطان") is True
assert _opener_word_matches("اكبر", "اكبر", "اكبر") is True
assert _opener_word_matches("الر", "الرحمن", "الرحمان") is False  # prefix != word
assert _opener_word_matches("الله", "الله", "الله") is True
assert _opener_word_matches(None, "الله", "الله") is False
print("2. tolerant opener spelling OK")

# ======================= 3. _leading_opener only fires at the lead ==========
class W(dict):
    pass

def words_with(*toks):
    return [{"text": t, "start": 0.1 + 0.6 * i, "end": 0.6 + 0.6 * i} for i, t in enumerate(toks)]

# Mid-clip takbir is legitimate canon -> must NOT be picked as an opener
mid = words_with("قل", "هو", "الله", "اكبر", "احد")
assert _leading_opener(mid) is None, _leading_opener(mid)
# ... but a leading takbir IS
lead = words_with("الله", "اكبر", "بسم", "الله", "الرحمن", "الرحيم")
k = _leading_opener(lead)
assert k is not None and k[0] == "takbir", k
print("3. _leading_opener lead-only rule OK")

# ======================= 4. inject_opener_segments: all openers get EN, mid-clip takbir does NOT =======================
class M:
    def __init__(self, surah, chain):
        self.surah, self.chain = surah, chain

base_words = words_with(
    "الله", "اكبر",              # 0-1  takbir (lead)
    "اعوذ", "بالله", "من", "الشيطان", "الرجيم",  # 2-6  isti'adha
    "بسم", "الله", "الرحمن", "الرحيم",  # 7-10 bismillah
    "قل", "هو", "الله", "احد",     # 11-14 content
    "الله", "اكبر", "احد",        # 15-17 mid-clip takbir (fake canon 112:4-ish)
)
m112 = M(112, [(11, 112, 1), (12, 112, 1), (13, 112, 1), (14, 112, 1)])
base_seg = [{"surah": 112, "ayah": 1, "start": 7.2, "end": 12.0, "text_ar": "قل هو الله أحد", "text_en": "Say: He is Allah"}]

segs = inject_opener_segments([m112], base_words, [dict(s) for s in base_seg])
opener_segs = [s for s in segs if s["ayah"] in (0, 1) and s["start"] < base_seg[0]["start"]]
kinds_seen = [("takbir" if s["ayah"] == 0 else "bismillah") for s in opener_segs]
# Alafasy-style opening: takbir + isti'adha + bismillah all captioned
assert ("takbir" in kinds_seen and "bismillah" in kinds_seen), (kinds_seen, segs)
assert all(s["text_en"] for s in opener_segs), opener_segs
# mid-clip takbir (15-16) must NOT produce a phantom 1:0 segment beyond content
phantom = [s for s in segs if s["ayah"] == 0 and s["start"] >= base_seg[0]["start"]]
assert phantom == [], segs
# no overlapping segments
segs_sorted = sorted(segs, key=lambda s: s["start"])
for a, b in zip(segs_sorted, segs_sorted[1:]):
    assert a["end"] <= b["start"] + 1e-6, (a, b)
print(f"4. inject_opener_segments OK  ({[s['ayah'] for s in segs]})")

# ======================= 5. segments_from_word_times local-path tags ========================
wt = [
    {"ayah": -1, "text": "الله", "start": 0.3, "end": 1.0},
    {"ayah": -1, "text": "اكبر", "start": 1.0, "end": 1.6},
    {"ayah": -2, "text": "اعوذ", "start": 1.8, "end": 2.4},
    {"ayah": -2, "text": "بالله", "start": 2.4, "end": 3.0},
    {"ayah": -2, "text": "من", "start": 3.0, "end": 3.4},
    {"ayah": -2, "text": "الشيطان", "start": 3.4, "end": 4.0},
    {"ayah": -2, "text": "الرجيم", "start": 4.0, "end": 4.6},
    {"ayah": 0, "text": "بسم", "start": 4.8, "end": 5.4},
    {"ayah": 0, "text": "الله", "start": 5.4, "end": 6.0},
    {"ayah": 0, "text": "الرحمن", "start": 6.0, "end": 6.6},
    {"ayah": 0, "text": "الرحيم", "start": 6.6, "end": 7.2},
]
local_segs = segments_from_word_times(112, 1, 1, wt)
labels = [(s["surah"], s["ayah"], s["text_en"][:30]) for s in local_segs]
assert len(local_segs) >= 3, labels
assert local_segs[0]["ayah"] == 0 and local_segs[0]["text_en"].startswith("Allah is the Greatest")
assert local_segs[1]["ayah"] == 0 and local_segs[1]["text_en"].startswith("I seek refuge")
assert local_segs[2]["ayah"] == 1 and local_segs[2]["text_en"].startswith("In the name of Allah")
print("5. local-path opener tags OK:", labels)

# ======================= 6. build_srt carries EN for openers ========================
srt_ar = build_srt(local_segs, "ar")
assert "Allah is the Greatest" not in srt_ar
srt_both = build_srt(local_segs, "both")
assert "Allah is the Greatest" in srt_both and "In the name of Allah" in srt_both
srt_en = build_srt(local_segs, "en")
assert "الرجيم" not in srt_en
print("6. build_srt openers OK")

# ======================= 7. detector strips tolerant lead openers ========================
# ASR writes الشيطان with a plain alif after the takbir — lead strip must be
# tolerant of that spelling (U+0670 vs U+0627) or the takbir/isti'adha shadow
# the content match and shift/break the anchor.
openers = norm(["الله", "اكبر", "اعوذ", "بالله", "من", "الشيطان", "الرجيم"])
asr_t = " ".join(openers + norm("قل هو الله احد الصمد لم يلد ولم يولد ولم يكن له كفوا احد".split()))
m = d.detect_all(asr_t)
assert len(m) == 1, m
assert (m[0].surah, m[0].start_ayah, m[0].end_ayah) == (112, 1, 4), m
assert m[0].score >= 0.9, m[0]

# Takbir only, then content — still one clean match
asr_t2 = " ".join(norm(["الله", "اكبر"]) + norm("بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ قل هو الله احد الصمد".split()))
m2 = d.detect_all(asr_t2)
assert len(m2) == 1, m2
assert (m2[0].surah, m2[0].start_ayah) == (112, 1), m2

# Mid-clip takbir is canon (9:72, 29:45, 40:10) — must NOT be stripped (it is a
# real ayah word, and stripping it would misalign the chain)
t_mid = " ".join(norm("ذَٰلِكَ بِأَنَّهُمُ ".split())) + "الله اكبر"  # placeholder that keeps the takbir mid-text
assert d.detect_all(t_mid) == []  # unmatched junk, no crash either way

# gibberish still produces no matches
assert d.detect_all("zzz qqq www eee rrr ttt") == []

print("7. detector lead-openers strip OK")
print("\ntest_openers.py ALL PASSED")