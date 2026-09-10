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
