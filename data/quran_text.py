import json
import os
from typing import Optional


SURAH_NAMES = [
    {"id": 1, "name_ar": "الفاتحة", "name_en": "Al-Fatihah", "ayah_count": 7, "type": "Meccan"},
    {"id": 2, "name_ar": "البقرة", "name_en": "Al-Baqarah", "ayah_count": 286, "type": "Medinan"},
    {"id": 3, "name_ar": "آل عمران", "name_en": "Aal-e-Imran", "ayah_count": 200, "type": "Medinan"},
    {"id": 4, "name_ar": "النساء", "name_en": "An-Nisa", "ayah_count": 176, "type": "Medinan"},
    {"id": 5, "name_ar": "المائدة", "name_en": "Al-Maidah", "ayah_count": 120, "type": "Medinan"},
    {"id": 6, "name_ar": "الأنعام", "name_en": "Al-Anaam", "ayah_count": 165, "type": "Meccan"},
    {"id": 7, "name_ar": "الأعراف", "name_en": "Al-Araf", "ayah_count": 206, "type": "Meccan"},
    {"id": 8, "name_ar": "الأنفال", "name_en": "Al-Anfal", "ayah_count": 75, "type": "Medinan"},
    {"id": 9, "name_ar": "التوبة", "name_en": "At-Tawbah", "ayah_count": 129, "type": "Medinan"},
    {"id": 10, "name_ar": "يونس", "name_en": "Yunus", "ayah_count": 109, "type": "Meccan"},
    {"id": 11, "name_ar": "هود", "name_en": "Hud", "ayah_count": 123, "type": "Meccan"},
    {"id": 12, "name_ar": "يوسف", "name_en": "Yusuf", "ayah_count": 111, "type": "Meccan"},
    {"id": 13, "name_ar": "الرعد", "name_en": "Ar-Rad", "ayah_count": 43, "type": "Medinan"},
    {"id": 14, "name_ar": "إبراهيم", "name_en": "Ibrahim", "ayah_count": 52, "type": "Meccan"},
    {"id": 15, "name_ar": "الحجر", "name_en": "Al-Hijr", "ayah_count": 99, "type": "Meccan"},
    {"id": 16, "name_ar": "النحل", "name_en": "An-Nahl", "ayah_count": 128, "type": "Meccan"},
    {"id": 17, "name_ar": "الإسراء", "name_en": "Al-Isra", "ayah_count": 111, "type": "Meccan"},
    {"id": 18, "name_ar": "الكهف", "name_en": "Al-Kahf", "ayah_count": 110, "type": "Meccan"},
    {"id": 19, "name_ar": "مريم", "name_en": "Maryam", "ayah_count": 98, "type": "Meccan"},
    {"id": 20, "name_ar": "طه", "name_en": "Ta-Ha", "ayah_count": 135, "type": "Meccan"},
    {"id": 21, "name_ar": "الأنبياء", "name_en": "Al-Anbiya", "ayah_count": 112, "type": "Meccan"},
    {"id": 22, "name_ar": "الحج", "name_en": "Al-Hajj", "ayah_count": 78, "type": "Medinan"},
    {"id": 23, "name_ar": "المؤمنون", "name_en": "Al-Muminoon", "ayah_count": 118, "type": "Meccan"},
    {"id": 24, "name_ar": "النور", "name_en": "An-Nur", "ayah_count": 64, "type": "Medinan"},
    {"id": 25, "name_ar": "الفرقان", "name_en": "Al-Furqan", "ayah_count": 77, "type": "Meccan"},
    {"id": 26, "name_ar": "الشعراء", "name_en": "Ash-Shuara", "ayah_count": 227, "type": "Meccan"},
    {"id": 27, "name_ar": "النمل", "name_en": "An-Naml", "ayah_count": 93, "type": "Meccan"},
    {"id": 28, "name_ar": "القصص", "name_en": "Al-Qasas", "ayah_count": 88, "type": "Meccan"},
    {"id": 29, "name_ar": "العنكبوت", "name_en": "Al-Ankaboot", "ayah_count": 69, "type": "Meccan"},
    {"id": 30, "name_ar": "الروم", "name_en": "Ar-Rum", "ayah_count": 60, "type": "Meccan"},
    {"id": 31, "name_ar": "لقمان", "name_en": "Luqman", "ayah_count": 34, "type": "Meccan"},
    {"id": 32, "name_ar": "السجدة", "name_en": "As-Sajdah", "ayah_count": 30, "type": "Meccan"},
    {"id": 33, "name_ar": "الأحزاب", "name_en": "Al-Ahzab", "ayah_count": 73, "type": "Medinan"},
    {"id": 34, "name_ar": "سبإ", "name_en": "Saba", "ayah_count": 54, "type": "Meccan"},
    {"id": 35, "name_ar": "فاطر", "name_en": "Fatir", "ayah_count": 45, "type": "Meccan"},
    {"id": 36, "name_ar": "يس", "name_en": "Ya-Sin", "ayah_count": 83, "type": "Meccan"},
    {"id": 37, "name_ar": "الصافات", "name_en": "As-Saffat", "ayah_count": 182, "type": "Meccan"},
    {"id": 38, "name_ar": "ص", "name_en": "Sad", "ayah_count": 88, "type": "Meccan"},
    {"id": 39, "name_ar": "الزمر", "name_en": "Az-Zumar", "ayah_count": 75, "type": "Meccan"},
    {"id": 40, "name_ar": "غافر", "name_en": "Ghafir", "ayah_count": 85, "type": "Meccan"},
    {"id": 41, "name_ar": "فصلت", "name_en": "Fussilat", "ayah_count": 54, "type": "Meccan"},
    {"id": 42, "name_ar": "الشورى", "name_en": "Ash-Shura", "ayah_count": 53, "type": "Meccan"},
    {"id": 43, "name_ar": "الزخرف", "name_en": "Az-Zukhruf", "ayah_count": 89, "type": "Meccan"},
    {"id": 44, "name_ar": "الدخان", "name_en": "Ad-Dukhan", "ayah_count": 59, "type": "Meccan"},
    {"id": 45, "name_ar": "الجاثية", "name_en": "Al-Jathiyah", "ayah_count": 37, "type": "Meccan"},
    {"id": 46, "name_ar": "الأحقاف", "name_en": "Al-Ahqaf", "ayah_count": 35, "type": "Meccan"},
    {"id": 47, "name_ar": "محمد", "name_en": "Muhammad", "ayah_count": 38, "type": "Medinan"},
    {"id": 48, "name_ar": "الفتح", "name_en": "Al-Fath", "ayah_count": 29, "type": "Medinan"},
    {"id": 49, "name_ar": "الحجرات", "name_en": "Al-Hujurat", "ayah_count": 18, "type": "Medinan"},
    {"id": 50, "name_ar": "ق", "name_en": "Qaf", "ayah_count": 45, "type": "Meccan"},
    {"id": 51, "name_ar": "الذاريات", "name_en": "Adh-Dhariyat", "ayah_count": 60, "type": "Meccan"},
    {"id": 52, "name_ar": "الطور", "name_en": "At-Tur", "ayah_count": 49, "type": "Meccan"},
    {"id": 53, "name_ar": "النجم", "name_en": "An-Najm", "ayah_count": 62, "type": "Meccan"},
    {"id": 54, "name_ar": "القمر", "name_en": "Al-Qamar", "ayah_count": 55, "type": "Meccan"},
    {"id": 55, "name_ar": "الرحمن", "name_en": "Ar-Rahman", "ayah_count": 78, "type": "Medinan"},
    {"id": 56, "name_ar": "الواقعة", "name_en": "Al-Waqiah", "ayah_count": 96, "type": "Meccan"},
    {"id": 57, "name_ar": "الحديد", "name_en": "Al-Hadid", "ayah_count": 29, "type": "Medinan"},
    {"id": 58, "name_ar": "المجادلة", "name_en": "Al-Mujadilah", "ayah_count": 22, "type": "Medinan"},
    {"id": 59, "name_ar": "الحشر", "name_en": "Al-Hashr", "ayah_count": 24, "type": "Medinan"},
    {"id": 60, "name_ar": "الممتحنة", "name_en": "Al-Mumtahanah", "ayah_count": 13, "type": "Medinan"},
    {"id": 61, "name_ar": "الصف", "name_en": "As-Saf", "ayah_count": 14, "type": "Medinan"},
    {"id": 62, "name_ar": "الجمعة", "name_en": "Al-Jumuah", "ayah_count": 11, "type": "Medinan"},
    {"id": 63, "name_ar": "المنافقون", "name_en": "Al-Munafiqun", "ayah_count": 11, "type": "Medinan"},
    {"id": 64, "name_ar": "التغابن", "name_en": "At-Taghabun", "ayah_count": 18, "type": "Medinan"},
    {"id": 65, "name_ar": "الطلاق", "name_en": "At-Talaq", "ayah_count": 12, "type": "Medinan"},
    {"id": 66, "name_ar": "التحريم", "name_en": "At-Tahrim", "ayah_count": 12, "type": "Medinan"},
    {"id": 67, "name_ar": "الملك", "name_en": "Al-Mulk", "ayah_count": 30, "type": "Meccan"},
    {"id": 68, "name_ar": "القلم", "name_en": "Al-Qalam", "ayah_count": 52, "type": "Meccan"},
    {"id": 69, "name_ar": "الحاقة", "name_en": "Al-Haqqah", "ayah_count": 52, "type": "Meccan"},
    {"id": 70, "name_ar": "المعارج", "name_en": "Al-Maarij", "ayah_count": 44, "type": "Meccan"},
    {"id": 71, "name_ar": "نوح", "name_en": "Nuh", "ayah_count": 28, "type": "Meccan"},
    {"id": 72, "name_ar": "الجن", "name_en": "Al-Jinn", "ayah_count": 28, "type": "Meccan"},
    {"id": 73, "name_ar": "المزمل", "name_en": "Al-Muzzammil", "ayah_count": 20, "type": "Meccan"},
    {"id": 74, "name_ar": "المدثر", "name_en": "Al-Muddaththir", "ayah_count": 56, "type": "Meccan"},
    {"id": 75, "name_ar": "القيامة", "name_en": "Al-Qiyamah", "ayah_count": 40, "type": "Meccan"},
    {"id": 76, "name_ar": "الإنسان", "name_en": "Al-Insan", "ayah_count": 31, "type": "Medinan"},
    {"id": 77, "name_ar": "المرسلات", "name_en": "Al-Mursalat", "ayah_count": 50, "type": "Meccan"},
    {"id": 78, "name_ar": "النبإ", "name_en": "An-Naba", "ayah_count": 40, "type": "Meccan"},
    {"id": 79, "name_ar": "النازعات", "name_en": "An-Naziat", "ayah_count": 46, "type": "Meccan"},
    {"id": 80, "name_ar": "عبس", "name_en": "Abasa", "ayah_count": 42, "type": "Meccan"},
    {"id": 81, "name_ar": "التكوير", "name_en": "At-Takwir", "ayah_count": 29, "type": "Meccan"},
    {"id": 82, "name_ar": "الإنفطار", "name_en": "Al-Infitar", "ayah_count": 19, "type": "Meccan"},
    {"id": 83, "name_ar": "المطففين", "name_en": "Al-Mutaffifin", "ayah_count": 36, "type": "Meccan"},
    {"id": 84, "name_ar": "الإنشقاق", "name_en": "Al-Inshiqaq", "ayah_count": 25, "type": "Meccan"},
    {"id": 85, "name_ar": "البروج", "name_en": "Al-Buruj", "ayah_count": 22, "type": "Meccan"},
    {"id": 86, "name_ar": "الطارق", "name_en": "At-Tariq", "ayah_count": 17, "type": "Meccan"},
    {"id": 87, "name_ar": "الأعلى", "name_en": "Al-Ala", "ayah_count": 19, "type": "Meccan"},
    {"id": 88, "name_ar": "الغاشية", "name_en": "Al-Ghashiyah", "ayah_count": 26, "type": "Meccan"},
    {"id": 89, "name_ar": "الفجر", "name_en": "Al-Fajr", "ayah_count": 30, "type": "Meccan"},
    {"id": 90, "name_ar": "البلد", "name_en": "Al-Balad", "ayah_count": 20, "type": "Meccan"},
    {"id": 91, "name_ar": "الشمس", "name_en": "Ash-Shams", "ayah_count": 15, "type": "Meccan"},
    {"id": 92, "name_ar": "الليل", "name_en": "Al-Layl", "ayah_count": 21, "type": "Meccan"},
    {"id": 93, "name_ar": "الضحى", "name_en": "Ad-Duha", "ayah_count": 11, "type": "Meccan"},
    {"id": 94, "name_ar": "الشرح", "name_en": "Ash-Sharh", "ayah_count": 8, "type": "Meccan"},
    {"id": 95, "name_ar": "التين", "name_en": "At-Tin", "ayah_count": 8, "type": "Meccan"},
    {"id": 96, "name_ar": "العلق", "name_en": "Al-Alaq", "ayah_count": 19, "type": "Meccan"},
    {"id": 97, "name_ar": "القدر", "name_en": "Al-Qadr", "ayah_count": 5, "type": "Meccan"},
    {"id": 98, "name_ar": "البينة", "name_en": "Al-Bayyinah", "ayah_count": 8, "type": "Medinan"},
    {"id": 99, "name_ar": "الزلزلة", "name_en": "Az-Zalzalah", "ayah_count": 8, "type": "Medinan"},
    {"id": 100, "name_ar": "العاديات", "name_en": "Al-Adiyat", "ayah_count": 11, "type": "Meccan"},
    {"id": 101, "name_ar": "القارعة", "name_en": "Al-Qariah", "ayah_count": 11, "type": "Meccan"},
    {"id": 102, "name_ar": "التكاثر", "name_en": "At-Takathur", "ayah_count": 8, "type": "Meccan"},
    {"id": 103, "name_ar": "العصر", "name_en": "Al-Asr", "ayah_count": 3, "type": "Meccan"},
    {"id": 104, "name_ar": "الهمزة", "name_en": "Al-Humazah", "ayah_count": 9, "type": "Meccan"},
    {"id": 105, "name_ar": "الفيل", "name_en": "Al-Fil", "ayah_count": 5, "type": "Meccan"},
    {"id": 106, "name_ar": "قريش", "name_en": "Quraysh", "ayah_count": 4, "type": "Meccan"},
    {"id": 107, "name_ar": "الماعون", "name_en": "Al-Maun", "ayah_count": 7, "type": "Meccan"},
    {"id": 108, "name_ar": "الكوثر", "name_en": "Al-Kawthar", "ayah_count": 3, "type": "Meccan"},
    {"id": 109, "name_ar": "الكافرون", "name_en": "Al-Kafirun", "ayah_count": 6, "type": "Meccan"},
    {"id": 110, "name_ar": "النصر", "name_en": "An-Nasr", "ayah_count": 3, "type": "Medinan"},
    {"id": 111, "name_ar": "المسد", "name_en": "Al-Masad", "ayah_count": 5, "type": "Meccan"},
    {"id": 112, "name_ar": "الإخلاص", "name_en": "Al-Ikhlas", "ayah_count": 4, "type": "Meccan"},
    {"id": 113, "name_ar": "الفلق", "name_en": "Al-Falaq", "ayah_count": 5, "type": "Meccan"},
    {"id": 114, "name_ar": "الناس", "name_en": "An-Nas", "ayah_count": 6, "type": "Meccan"},
]


_ayah_texts: dict[str, str] | None = None
_english_texts: dict[str, str] | None = None


def load_ayah_texts() -> dict[str, str]:
    global _ayah_texts
    if _ayah_texts is not None:
        return _ayah_texts

    filepath = os.path.join(os.path.dirname(__file__), "quran_uthmani.json")
    if os.path.exists(filepath):
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
        _ayah_texts = data
    else:
        _ayah_texts = _build_from_datasets()

    return _ayah_texts


def load_english_texts() -> dict[str, str]:
    global _english_texts
    if _english_texts is not None:
        return _english_texts

    filepath = os.path.join(os.path.dirname(__file__), "quran_en.json")
    if os.path.exists(filepath):
        with open(filepath, encoding="utf-8") as f:
            _english_texts = json.load(f)
    else:
        _english_texts = {}
    return _english_texts


def get_ayah_text(surah: int, ayah: int) -> str | None:
    texts = load_ayah_texts()
    return texts.get(f"{surah}:{ayah}")


def get_ayah_english(surah: int, ayah: int) -> str | None:
    texts = load_english_texts()
    return texts.get(f"{surah}:{ayah}")


def _build_from_datasets() -> dict[str, str]:
    try:
        from datasets import load_dataset
        ds = load_dataset("arbml/quran_uthmani", split="train", streaming=True)
        result = {}
        for row in ds:
            key = f"{row['surah']}:{row['ayah']}"
            result[key] = row["text"]
        return result
    except Exception as e:
        raise RuntimeError(
            f"Could not load Quran data: {e}. "
            "Place quran_uthmani.json in the data/ directory."
        )


def get_ayah_text(surah: int, ayah: int) -> str | None:
    texts = load_ayah_texts()
    return texts.get(f"{surah}:{ayah}")


def get_surah_text(surah: int, start_ayah: int = 1, end_ayah: int | None = None) -> list[dict]:
    texts = load_ayah_texts()
    surah_info = SURAH_NAMES[surah - 1]
    end = end_ayah or surah_info["ayah_count"]
    result = []
    for ayah in range(start_ayah, end + 1):
        text = texts.get(f"{surah}:{ayah}")
        if text:
            result.append({"surah": surah, "ayah": ayah, "text": text})
    return result


def build_index() -> dict[str, tuple[int, int, str]]:
    texts = load_ayah_texts()
    index: dict[str, tuple[int, int, str]] = {}
    for key, text in texts.items():
        parts = key.split(":")
        surah, ayah = int(parts[0]), int(parts[1])
        normalized = _normalize(text)
        index[normalized] = (surah, ayah, text)
    return index


def build_window_index(window_size: int = 3) -> dict[str, list[tuple[int, int, str]]]:
    texts = load_ayah_texts()
    all_ayahs = []
    for key, text in texts.items():
        parts = key.split(":")
        all_ayahs.append((int(parts[0]), int(parts[1]), text))

    all_ayahs.sort(key=lambda x: (x[0], x[1]))

    index: dict[str, list[tuple[int, int, str]]] = {}
    for i in range(len(all_ayahs) - window_size + 1):
        window = all_ayahs[i:i + window_size]
        combined = " ".join(_normalize(a[2]) for a in window)
        index[combined] = window

    return index


def _normalize(text: str) -> str:
    """Quranic-orthography normalization for exact matching.
    Strips all short vowels/tanwin/shadda/Quranic marks (incl. dagger alif
    U+0670 and sukun U+06E1), maps wasla/alefs to ا, ة->ه, ى->ي, hamza forms.
    Used by detection AND alignment canon; see _normalize_std for the
    standard-spelling variant."""
    text = text.replace("\u064b", "").replace("\u064c", "").replace("\u064d", "")
    text = text.replace("\u064e", "").replace("\u064f", "").replace("\u0650", "")
    text = text.replace("\u0651", "").replace("\u0652", "")
    text = text.replace("\u0653", "").replace("\u0654", "").replace("\u0655", "")
    text = text.replace("\u0656", "").replace("\u0657", "").replace("\u0658", "")
    text = text.replace("\u065c", "").replace("\u065e", "")
    # tatweel (used in اولـيك) and Quranic annotation marks U+06D6..U+06ED
    # (ۖ ۗ ۘ ۙ ۚ ۛ ۜ ۝ ۞ ۟ ۠ ۡ ۢ ۣ ۤ ۥ ۦ ۧ ۨ ۩ ۪ ۫ ۬ ۭ) plus thin space
    text = text.replace("\u0640", "")
    for cp in range(0x06D6, 0x06EE):
        text = text.replace(chr(cp), "")
    text = text.replace("\u2009", "")
    text = text.replace("\u0670", "").replace("\u06e1", "")
    text = text.replace("\u0621", "")
    text = text.replace("\u0671", "\u0627")
    text = text.replace("\u0629", "\u0647").replace("\u0626", "\u064a").replace("\u0624", "\u0648")
    text = text.replace("\u0622", "\u0627").replace("\u0623", "\u0627").replace("\u0625", "\u0627")
    text = text.replace("\u0649", "\u064a")
    return text.strip()


def _normalize_std(text: str) -> str:
    """Standard-spelling variant: dagger alif -> ا (الكتب -> الكتاب),
    وٰة -> اة (الصلوة -> الصلاه). Complements _normalize for detector
    word-index matching (whisper emits standard orthography)."""
    text = text.replace("\u0648\u0670\u0629", "\u0627\u0629")
    text = text.replace("\u0670", "\u0627")
    return _normalize(text)
