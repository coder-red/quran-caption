import os
import re
import shutil
import subprocess
import tempfile
import threading
from backend.config import FFMPEG_PATH, TEMP_DIR

FONT_FILES = {
    "Segoe UI": ("segoeui.ttf", "segoeuib.ttf"),
    "Tahoma": ("tahoma.ttf", "tahomabd.ttf"),
    "Arial": ("arial.ttf", "arialbd.ttf"),
    "Times New Roman": ("times.ttf", "timesbd.ttf"),
    "Calibri": ("calibri.ttf", "calibrib.ttf"),
}

FONT_FALLBACK = ("C:\\Windows\\Fonts\\segoeui.ttf", "C:\\Windows\\Fonts\\segoeuib.ttf")

DEFAULT_STYLES = {
    "font_size": 48,
    "font_family": "Segoe UI",
    "font_color": "#FFFFFF",
    "outline_color": "#000000",
    "outline_size": 2,
    "shadow_size": 0,
    "bold": False,
    "box": False,
    "position": "bottom",
}

_encoder_lock = threading.Lock()
_cached_encoder = None


def write_srt_file(srt_content: str, output_path: str):
    with open(output_path, "w", encoding="utf-8-sig") as f:
        f.write(srt_content)


def _filter_escape(path: str) -> str:
    rel = os.path.relpath(path).replace("\\", "/")
    return (
        rel.replace(":", "\\:")
        .replace(",", "\\,")
        .replace("'", "\\'")
        .replace("[", "\\[")
        .replace("]", "\\]")
    )


def _find_font(family: str, bold: bool) -> str:
    fonts_dir = "C:\\Windows\\Fonts"
    pair = FONT_FILES.get(family) or FONT_FILES["Segoe UI"]
    file = pair[1] if bold else pair[0]
    path = os.path.join(fonts_dir, file)
    if os.path.exists(path):
        return path
    fallback = FONT_FALLBACK[1] if bold else FONT_FALLBACK[0]
    if os.path.exists(fallback):
        return fallback
    return "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def _hex_to_ass(color_hex: str, alpha: int = 0) -> str:
    color = (color_hex or "#FFFFFF").lstrip("#")
    if len(color) == 3:
        color = "".join(c * 2 for c in color)
    try:
        r, g, b = int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)
    except (ValueError, IndexError):
        r, g, b = 255, 255, 255
    return f"&H{alpha:02X}{b:02X}{g:02X}{r:02X}"


_POSITION_TAG = {"top": "{\\an8}", "middle": "{\\an5}", "bottom": "{\\an2}"}


def inject_position(srt_content: str, position: str) -> str:
    """Prefix each SRT cue's first text line with a \\anN override tag.

    The subtitles filter's force_style Alignment is unreliable (this libass
    build misplaces 5/8), but libass honors inline \\anN tags in SRT text.
    """
    tag = _POSITION_TAG.get(position or "bottom", "{\\an2}")
    out = []
    expect_text = False
    for line in srt_content.splitlines():
        s = line.strip()
        if "-->" in s:
            expect_text = True
            out.append(line)
        elif expect_text and s:
            out.append(tag + line)
            expect_text = False
        else:
            out.append(line)
            if not s:
                expect_text = False
    return "\n".join(out)


def _build_vfilter(srt_path: str, styles: dict, scale_filter: str = "") -> str:
    filter_path = _filter_escape(srt_path)
    font = _find_font(styles.get("font_family", "Segoe UI"), styles.get("bold", False))
    font_escaped = _filter_escape(font)

    font_size = int(styles.get("font_size", 48))
    text_color = _hex_to_ass(styles.get("font_color", "#FFFFFF"))
    outline_color = _hex_to_ass(styles.get("outline_color", "#000000"))
    outline = max(0, int(styles.get("outline_size", 2)))
    shadow = max(0, int(styles.get("shadow_size", 0)))

    if styles.get("box", False):
        border_style = 3
        back_color = _hex_to_ass("#000000", alpha=160)
    else:
        border_style = 1
        back_color = _hex_to_ass("#000000", alpha=255)

    position = styles.get("position", "bottom")
    alignment = 8 if position == "top" else (5 if position == "middle" else 2)

    style_opts = (
        f"FontName={font_escaped},FontSize={font_size},"
        f"PrimaryColour={text_color},OutlineColour={outline_color},"
        f"BackColour={back_color},BorderStyle={border_style},"
        f"Outline={outline},Shadow={shadow},Bold={1 if styles.get('bold') else 0},"
        f"Alignment={alignment}"
    )
    vfilter = f"subtitles={filter_path}:force_style='{style_opts}'"
    if scale_filter:
        vfilter = f"{scale_filter},{vfilter}"
    return vfilter


def _scale_filter(width: int, height: int) -> str:
    return (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2"
    )


def detect_encoder() -> str:
    """Pick a working H.264 encoder, preferring hardware (fast)."""
    global _cached_encoder
    with _encoder_lock:
        if _cached_encoder:
            return _cached_encoder

        candidates = ["h264_qsv", "h264_nvenc", "h264_amf", "libx264"]
        encoders = subprocess.run(
            [FFMPEG_PATH, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
        ).stdout

        for enc in candidates:
            if f" {enc} " not in f" {encoders} " and f" {enc}\n" not in f" {encoders} ":
                if enc not in encoders:
                    continue
            if enc == "libx264":
                _cached_encoder = enc
                return enc
            test = subprocess.run(
                [
                    FFMPEG_PATH, "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", "color=size=64x64:rate=1:duration=0.1",
                    "-c:v", enc, "-f", "null", "-",
                ],
                capture_output=True,
                text=True,
            )
            if test.returncode == 0:
                _cached_encoder = enc
                return enc

        _cached_encoder = "libx264"
        return _cached_encoder


def _encoder_args(encoder: str) -> list[str]:
    if encoder == "h264_qsv":
        return ["-c:v", "h264_qsv", "-preset", "fast", "-global_quality", "23"]
    if encoder == "h264_nvenc":
        return ["-c:v", "h264_nvenc", "-preset", "p5", "-cq", "23"]
    if encoder == "h264_amf":
        return ["-c:v", "h264_amf", "-quality", "quality", "-qp_i", "23", "-qp_p", "23"]
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23"]


def probe_duration(video_path: str) -> float:
    ffprobe = os.path.join(os.path.dirname(FFMPEG_PATH), "ffprobe.exe")
    if not os.path.isfile(ffprobe):
        ffprobe = os.path.join(os.path.dirname(FFMPEG_PATH), "ffprobe")
    if not os.path.isfile(ffprobe):
        ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return 0.0
    r = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        capture_output=True,
        text=True,
    )
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


def render_captions(
    video_path: str,
    srt_content: str,
    styles: dict | None = None,
    width: int | None = None,
    height: int | None = None,
    progress_cb=None,
) -> str:
    os.makedirs(TEMP_DIR, exist_ok=True)
    styles = {**DEFAULT_STYLES, **(styles or {})}

    srt_fd, srt_path = tempfile.mkstemp(suffix=".srt", dir=TEMP_DIR)
    os.close(srt_fd)
    write_srt_file(inject_position(srt_content, styles.get("position", "bottom")), srt_path)

    out_fd, out_path = tempfile.mkstemp(suffix=".mp4", dir=TEMP_DIR)
    os.close(out_fd)

    total = probe_duration(video_path)

    scale_filter = ""
    if width and height:
        scale_filter = _scale_filter(int(width), int(height))

    vfilter = _build_vfilter(srt_path, styles, scale_filter)
    encoder = detect_encoder()
    encoder_args = _encoder_args(encoder)

    cmd = [
        FFMPEG_PATH,
        "-hide_banner", "-nostats",
        "-i", video_path,
        "-vf", vfilter,
        *encoder_args,
        "-c:a", "aac", "-b:a", "128k",
        "-progress", "pipe:1",
        "-y",
        out_path,
    ]

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        encoding="utf-8", errors="replace",
    )

    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.strip()
        if line.startswith("out_time_ms="):
            try:
                ms = int(line.split("=", 1)[1])
                if progress_cb and total > 0:
                    progress_cb(min(99, ms / (total * 1000) * 100))
            except (ValueError, ZeroDivisionError):
                pass

    stderr = proc.stderr.read() if proc.stderr else ""
    proc.wait()

    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg error: {stderr[-3000:]}")

    return out_path


def render_preview(
    video_path: str,
    srt_content: str,
    styles: dict | None = None,
    frame_time: float = 1.0,
) -> str:
    """Render a single frame with styled captions (fast preview)."""
    os.makedirs(TEMP_DIR, exist_ok=True)
    styles = {**DEFAULT_STYLES, **(styles or {})}

    srt_fd, srt_path = tempfile.mkstemp(suffix=".srt", dir=TEMP_DIR)
    os.close(srt_fd)
    write_srt_file(inject_position(srt_content, styles.get("position", "bottom")), srt_path)

    out_fd, out_path = tempfile.mkstemp(suffix=".png", dir=TEMP_DIR)
    os.close(out_fd)

    vfilter = _build_vfilter(srt_path, styles)

    cmd = [
        FFMPEG_PATH,
        "-hide_banner", "-loglevel", "error", "-y",
        "-copyts",
        "-ss", f"{frame_time:.2f}",
        "-i", video_path,
        "-vf", f"{vfilter},scale=640:-1",
        "-frames:v", "1",
        out_path,
    ]

    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"FFmpeg error: {r.stderr[-2000:]}")
    return out_path
