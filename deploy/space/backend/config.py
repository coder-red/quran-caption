import os
import shutil

try:
    from dotenv import load_dotenv

    load_dotenv()  # relies on CWD (uvicorn runs from repo root)
    _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(os.path.join(_repo_root, ".env"), override=False)
except Exception:
    pass

_torch_configured = False


def configure_torch():
    """Import torch lazily and pin thread counts for CPU inference. The Groq
    / hosted-ASR path never needs torch, so importing it at module load would
    add ~30-180s of startup for nothing. Call from model load() sites only."""
    global _torch_configured
    import torch

    if not _torch_configured:
        torch.set_num_threads(max(1, os.cpu_count() or 1))
        torch.set_num_interop_threads(1)
        _torch_configured = True
    return torch

ASR_MODEL = os.getenv("ASR_MODEL", "tarteel-ai/whisper-base-ar-quran")
DEVICE = os.getenv("DEVICE", "cpu")
DTYPE = os.getenv("DTYPE", "float32")
ALIGN_MODEL = os.getenv("ALIGN_MODEL", "jonatasgrosman/wav2vec2-large-xlsr-53-arabic")
TEMP_DIR = os.getenv("TEMP_DIR", "temp")
MAX_AUDIO_SEC = int(os.getenv("MAX_AUDIO_SEC", "1800"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.85"))

# Hosted ASR backends: local (whisper fine-tuned + forced alignment) | groq |
# gemini. groq/gemini return real word timestamps so the heavy local
# forced-alignment stage is skipped. Default is groq (free tier, no card);
# local is the offline fallback.
ASR_BACKEND = os.getenv("ASR_BACKEND", "groq").strip().lower()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip() or None
GROQ_MODEL = os.getenv("GROQ_MODEL", "whisper-large-v3-turbo")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip() or None
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def _find_ffmpeg() -> str:
    env_path = os.getenv("FFMPEG_PATH")
    if env_path and os.path.isfile(env_path):
        return env_path
    which = shutil.which("ffmpeg")
    if which:
        return which
    for candidate in (
        r"C:\ffmpeg\ffmpeg-master-latest-win64-gpl\bin\ffmpeg.exe",
        r"C:\ffmpeg\bin\ffmpeg.exe",
        "/usr/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
    ):
        if os.path.isfile(candidate):
            return candidate
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    return "ffmpeg"


FFMPEG_PATH = _find_ffmpeg()
