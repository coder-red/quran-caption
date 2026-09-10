"""Groq provider tests - mocked client, no API key / network required."""
import os
import sys
import tempfile
import types
import unittest
import unittest.mock
import warnings
from types import SimpleNamespace

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import soundfile as sf

import backend.providers.groq as groq_mod
from backend.detector import SurahDetector
from backend.segmenter import segments_from_provider_times
from data.quran_text import load_ayah_texts, _normalize

BISMILLAH = _normalize("بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ").split()


def _fake_words(offset: float = 0.0) -> list:
    texts = load_ayah_texts()
    canon = []
    canon += BISMILLAH
    for a in range(1, 5):
        canon += _normalize(texts[f"112:{a}"]).split()
    out = []
    t = offset
    for w in canon:
        out.append({"word": w, "start": t, "end": t + 0.4})
        t += 0.4
    return out


class FakeTranscription:
    text = "dummy"
    words: list = []


class FakeTranscriptions:
    def __init__(self, canned):
        self._canned = canned

    def create(self, **kwargs):
        return self._canned


class FakeAudio:
    def __init__(self, canned):
        self.transcriptions = FakeTranscriptions(canned)


class FakeClient:
    def __init__(self, canned):
        self.audio = FakeAudio(canned)


def _make_wav(path: str, seconds: float = 3.0):
    sf.write(path, np.zeros(int(seconds * 16000), dtype=np.float32), 16000)


class TestGroqProvider(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp()
        cls.wav = os.path.join(cls.tmpdir, "clip.wav")
        _make_wav(cls.wav)

    def _provider(self, **kw):
        return groq_mod.GroqProvider(api_key="test-key", **kw)

    def test_transcribe_keeps_bismillah_words_with_offsets(self):
        canned = FakeTranscription()
        canned.words = _fake_words(offset=0.0)
        fake = FakeClient(canned)
        with unittest.mock.patch.object(groq_mod, "Groq", return_value=fake):
            res = self._provider().transcribe(self.wav, slices=[(0.5, 2.8)])

        texts = load_ayah_texts()
        canon = BISMILLAH + [
            w for a in (1, 2, 3, 4) for w in _normalize(texts[f"112:{a}"]).split()
        ]
        self.assertEqual(res["transcript"], " ".join(canon))
        self.assertEqual(len(res["words"]), len(canon))
        self.assertEqual(_normalize(res["words"][0]["text"]), BISMILLAH[0])
        self.assertGreaterEqual(res["words"][0]["start"], 0.5)
        self.assertTrue(all(w["end"] > w["start"] for w in res["words"]))

    def test_no_api_key_raises_clear_error(self):
        with unittest.mock.patch.object(groq_mod, "GROQ_API_KEY", None):
            with self.assertRaises(RuntimeError) as ctx:
                groq_mod.GroqProvider(api_key=None).transcribe(self.wav)
            self.assertIn("GROQ_API_KEY", str(ctx.exception))

    def test_detector_chain_and_segments_end_to_end(self):
        canned = FakeTranscription()
        canned.words = _fake_words(offset=0.0)
        with unittest.mock.patch.object(groq_mod, "Groq", return_value=FakeClient(canned)):
            res = self._provider().transcribe(self.wav)

        d = SurahDetector()
        matches = d.detect_all(res["transcript"])
        self.assertEqual(len(matches), 1)
        m = matches[0]
        self.assertEqual((m.surah, m.start_ayah, m.end_ayah), (112, 1, 4))
        # Bismillah is stripped from matching but chain indices still point
        # into the full word list (which keeps it for captioning).
        self.assertTrue(m.chain and len(m.chain) == len(res["words"]) - len(BISMILLAH))
        self.assertEqual(m.chain[0][1], 112)

        segs = segments_from_provider_times(matches, res["words"])
        self.assertEqual([s["ayah"] for s in segs], [1, 1, 2, 3, 4])
        self.assertEqual(segs[0]["text_ar"], load_ayah_texts()["1:1"])
        self.assertGreaterEqual(segs[0]["end"], segs[0]["start"])
        self.assertEqual(segs[1]["text_ar"], load_ayah_texts()["112:1"])
        self.assertEqual(segs[4]["surah"], 112)
        self.assertLessEqual(segs[0]["start"], segs[1]["start"])

    def test_slice_packing_cuts_requests(self):
        canned = FakeTranscription()
        canned.words = _fake_words(offset=0.0)
        fake = FakeClient(canned)
        old_max = groq_mod.MAX_REQUEST_SECONDS
        groq_mod.MAX_REQUEST_SECONDS = 1.0
        try:
            with unittest.mock.patch.object(groq_mod, "Groq", return_value=fake):
                res = self._provider().transcribe(
                    self.wav, slices=[(0.0, 1.2), (1.5, 2.5)]
                )
        finally:
            groq_mod.MAX_REQUEST_SECONDS = old_max

        # Every request must be <= MAX_REQUEST_SECONDS even when a single
        # VAD slice is longer than the cap (long continuous recitations).
        self.assertEqual(res["bounds"], [(0.0, 1.0), (1.0, 1.2), (1.5, 2.5)])
        starts = [w["start"] for w in res["words"]]
        self.assertTrue(any(s >= 1.5 for s in starts), starts)


if __name__ == "__main__":
    unittest.main(verbosity=2)
