"""Rendering needs a picture: an audio-only file must be refused up front.

The fast upload sends a video's soundtrack as "...mp4.audio.ogg" first. If
that file ever reaches /render (a session whose video upload did not land),
FFmpeg used to fail with a wall of filtergraph text; the API now says what
is actually wrong.
"""
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import backend.renderer as renderer


def _make_audio(path: str) -> None:
    subprocess.run(
        [renderer.FFMPEG_PATH, "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t", "0.5",
         "-y", path],
        check=True,
    )


def test_audio_only_file_has_no_video_stream(tmp_path):
    audio = str(tmp_path / "clip.mp3")
    _make_audio(audio)
    assert not renderer.has_video_stream(audio)


def test_render_refuses_a_soundtrack(tmp_path):
    audio = str(tmp_path / "clip.mp3")
    _make_audio(audio)
    with pytest.raises(RuntimeError, match="no picture"):
        renderer.render_captions(audio, "1\n00:00:00,000 --> 00:00:00,500\nx\n")


def test_video_file_is_recognized():
    clip = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "data", "test_video.mp4")
    if not os.path.exists(clip):
        pytest.skip("no sample video")
    assert renderer.has_video_stream(clip)


def test_throttled_host_skips_the_hardware_probe(monkeypatch):
    """On the 0.1-CPU deploy host the probes sat on 'detecting...' for
    minutes and starved the health check — a fractional core has no GPU."""
    monkeypatch.setattr(renderer, "_cached_encoder", None)
    monkeypatch.setattr(renderer, "_cpu_quota", lambda: 0.1)
    calls = []
    monkeypatch.setattr(renderer.subprocess, "run", lambda *a, **k: calls.append(a))
    assert renderer.detect_encoder() == "libx264"
    assert calls == []


def test_fractional_core_uses_the_cheap_preset(monkeypatch):
    monkeypatch.setattr(renderer, "_cpu_quota", lambda: 0.1)
    args = renderer._encoder_args("libx264")
    assert "ultrafast" in args
    assert args[args.index("-threads") + 1] == "1"


def test_full_core_keeps_the_quality_preset(monkeypatch):
    monkeypatch.setattr(renderer, "_cpu_quota", lambda: 4.0)
    assert "veryfast" in renderer._encoder_args("libx264")
