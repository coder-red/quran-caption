from pydantic import BaseModel
from typing import Optional


class AlignRequest(BaseModel):
    duration: Optional[float] = None
    fps: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None


class WordTimestamp(BaseModel):
    text: str
    start: float
    end: float


class ChunkTimestamp(BaseModel):
    text: str
    start: float
    end: float


class SurahInfo(BaseModel):
    id: int
    name_ar: str
    name_en: str


class MatchInfo(BaseModel):
    surah: SurahInfo
    start_ayah: int
    end_ayah: int
    confidence: float


class Segment(BaseModel):
    surah: int
    ayah: int
    start: float
    end: float
    text_ar: str
    text_en: str


class AlignResponse(BaseModel):
    surah: SurahInfo
    start_ayah: int
    end_ayah: int
    confidence: float
    words: list[WordTimestamp]
    chunks: list[ChunkTimestamp]
    segments: list[Segment]
    srt: str
    vtt: str
    matches: list[MatchInfo] = []
    # Which speech detector produced the pause data the captions were cut
    # on: 'torch' | 'onnx' | 'energy'. The energy fallback finds different
    # pauses than Silero, so this is reported rather than hidden.
    vad_backend: str = "unknown"
    # Which alignment engine produced these captions ('qud' | 'groq' | ...).
    # QUD can silently fall back to Groq on quota/outage, so the client must
    # be able to tell which one actually ran.
    engine: str = ""


class RenderRequest(BaseModel):
    srt: str
    width: int = 1920
    height: int = 1080
    font_size: int = 48
    font_color: str = "white"
    font_bg: str = "black"


class RenderResponse(BaseModel):
    mp4_url: str


class ErrorResponse(BaseModel):
    error: str


class SyncRequest(BaseModel):
    """Captions to audit against the media file's own audio.

    The captions are sent by the client (not read from a stored align
    result) so the inspector reflects hand-edited timings too.
    """
    segments: list[Segment] = []
    words: list[WordTimestamp] = []
    # Upper bound on the number of timeline columns. Peaks/probs are
    # max-pooled to this, which is what keeps a 2h clip's payload small.
    max_points: int = 20000
