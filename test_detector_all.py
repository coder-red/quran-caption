import sys, os, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from backend.detector import SurahDetector
from data.quran_text import load_ayah_texts, _normalize

texts = load_ayah_texts()
d = SurahDetector()


def txt(key):
    return _normalize(texts[key])


# 1. Clean full surah (Ikhlas) -> one match 112:1-4, score >= 0.9
t = " ".join(txt(f"112:{a}") for a in range(1, 5))
m = d.detect_all(t)
assert len(m) == 1 and m[0].surah == 112 and m[0].start_ayah == 1 and m[0].end_ayah == 4
assert m[0].score >= 0.9, m[0]

# 2. With leading Bismillah -> same match (bismillah stripped)
t = " ".join([txt("1:1")] + [txt(f"112:{a}") for a in range(1, 5)])
m = d.detect_all(t)
assert len(m) == 1 and m[0].surah == 112 and m[0].start_ayah == 1 and m[0].end_ayah == 4

# 3. Mid-ayah start (Fatihah ayah 2 onwards, no ayah 1)
t = " ".join(txt(f"1:{a}") for a in range(2, 8))
m = d.detect_all(t)
assert len(m) == 1 and m[0].surah == 1 and m[0].start_ayah == 2 and m[0].end_ayah == 7, m

# 4. Mid-ayah END (Ikhlas cut at 2.5 words -> ayah 1..3, partial 4th)
t = " ".join([txt(f"112:{a}") for a in range(1, 4)] + [txt("112:4").split()[0]])
m = d.detect_all(t)
assert len(m) == 1 and m[0].surah == 112 and m[0].start_ayah == 1, m

# 5. Multi-surah (Ikhlas then Kawthar) -> 2 matches in order
t = " ".join(txt(f"112:{a}") for a in range(1, 5)) + " " + " ".join(txt(f"108:{a}") for a in range(1, 4))
m = d.detect_all(t)
assert len(m) == 2, m
assert (m[0].surah, m[0].start_ayah, m[0].end_ayah) == (112, 1, 4), m[0]
assert (m[1].surah, m[1].start_ayah, m[1].end_ayah) == (108, 1, 3), m[1]

# 6. ASR noise: 2 substituted words still detected (score drops but >= 0.55)
words = txt("112:1").split() + txt("112:2").split() + txt("112:3").split() + txt("112:4").split()
words[3] = "xxx"
words[7] = "yyy"
t = " ".join(words)
m = d.detect_all(t)
assert len(m) == 1 and m[0].surah == 112 and m[0].score >= 0.55, m

# 7. Repeated ayah text (recitation repeats) -> single match, no crash
t = txt("112:1") + " " + txt("112:1")
m = d.detect_all(t)
assert len(m) >= 1 and all(x.surah == 112 for x in m)

# 8. Gibberish -> no matches
m = d.detect_all("zzz qqq www eee rrr ttt")
assert m == [], m

print("test_detector_all.py ALL PASSED")
