"""ElevenLabs Scribe provider tests - mocked client, no API key / network."""
import os
import sys
import tempfile
import unittest
import unittest.mock
import warnings
from types import SimpleNamespace

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import soundfile as sf

import backend.providers.elevenlabs as el_mod
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
        # Scribe word objects: text/start/end (+ characters we ignore)
        out.append(SimpleNamespace(text=w, start=t, end=t + 0.4,
                                   type="word", speaker_id=None,
                                   logprob=0.0, characters=[]))
        t += 0.4
    return out


class FakeSTT:
    def __init__(self, canned):
        self._canned = canned

    def convert(self, **kwargs):
        assert kwargs.get("model_id"), "model_id required"
        assert kwargs.get("timestamps_granularity") == "word"
        assert kwargs.get("diarize") is False
        return self._canned


class FakeClient:
    def __init__(self, canned):
        self.speech_to_text = FakeSTT(canned)


def _make_wav(path: str, seconds: float = 3.0):
    sf.write(path, np.zeros(int(seconds * 16000), dtype=np.float32), 16000)


class TestElevenLabsProvider(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp()
        cls.wav = os.path.join(cls.tmpdir, "clip.wav")
        _make_wav(cls.wav)

    def _provider(self, **kw):
        return el_mod.ElevenLabsProvider(api_key="test-key", **kw)

    def test_transcribe_keeps_bismillah_words_with_offsets(self):
        canned = SimpleNamespace(
            text="dummy", words=_fake_words(offset=0.0))
        fake = FakeClient(canned)
        with unittest.mock.patch.object(el_mod, "ElevenLabs",
                                        return_value=fake):
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

    def test_dict_words_and_no_key(self):
        canned = SimpleNamespace(text="x", words=[
            {"text": "بسم", "start": 0.0, "end": 0.4},
            {"text": "!", "start": 0.4, "end": 0.5},
        ])
        with unittest.mock.patch.object(el_mod, "ElevenLabs",
                                        return_value=FakeClient(canned)):
            res = self._provider().transcribe(self.wav)
        self.assertEqual([w["text"] for w in res["words"]], ["بسم"])

        with unittest.mock.patch.object(el_mod, "ELEVENLABS_API_KEY", None):
            with self.assertRaises(RuntimeError) as ctx:
                el_mod.ElevenLabsProvider(api_key=None).transcribe(self.wav)
            self.assertIn("ELEVENLABS_API_KEY", str(ctx.exception))

    def test_detector_chain_and_segments_end_to_end(self):
        canned = SimpleNamespace(text="dummy", words=_fake_words(offset=0.0))
        with unittest.mock.patch.object(el_mod, "ElevenLabs",
                                        return_value=FakeClient(canned)):
            res = self._provider().transcribe(self.wav)

        d = SurahDetector()
        matches = d.detect_all(res["transcript"])
        self.assertEqual(len(matches), 1)
        m = matches[0]
        self.assertEqual((m.surah, m.start_ayah, m.end_ayah), (112, 1, 4))

        segs = segments_from_provider_times(matches, res["words"])
        self.assertEqual([s["ayah"] for s in segs], [1, 1, 2, 3, 4])
        self.assertEqual(segs[0]["text_ar"], load_ayah_texts()["1:1"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
