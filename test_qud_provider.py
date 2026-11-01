"""QUD³ provider: SSE parsing, stage progress, and the retry policy.

The aligner is a hosted service, so the provider's job is: stream a multipart
POST, turn `progress` events into the UI bar, return the `result` event, and
classify failures so a quota/outage falls back instead of hanging or retrying
forever. Everything here is offline (fake requests module).
"""
import json
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.providers import qud as qud_mod
from backend.providers.qud import QudProvider, QudUnavailable, validate_response


class FakeResp:
    def __init__(self, status=200, lines=None, body=None, text=None):
        self.status_code = status
        self._lines = list(lines or [])
        self._body = body if body is not None else {}
        self.text = text if text is not None else json.dumps(self._body)

    def iter_lines(self, decode_unicode=True):
        yield from self._lines

    def json(self):
        return self._body


class FakeRequests(types.SimpleNamespace):
    RequestException = ConnectionError

    def __init__(self, responses):
        super().__init__()
        self._responses = list(responses)
        self.calls = []
        self.post = self._post

    def _post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        resp = self._responses.pop(0)
        if isinstance(resp, Exception):
            raise resp
        return resp


@pytest.fixture
def audio(tmp_path):
    path = tmp_path / "audio.wav"
    path.write_bytes(b"RIFF....WAVEfmt ")
    return str(path)


def _provider(responses, monkeypatch, sleeps=None):
    fake = FakeRequests(responses)
    monkeypatch.setattr(qud_mod, "requests", fake)
    monkeypatch.setattr(qud_mod.time, "sleep", sleeps.append if sleeps is not None
                        else (lambda s: None))
    return fake, QudProvider(url="https://example.test/api/v1")


def _sse(*lines):
    return list(lines)


RESULT = {"segments": [{"kind": "quran", "ref_from": "59:1:1", "ref_to": "59:1:2",
                        "time_from": 0.3, "time_to": 1.9, "confidence": 0.93,
                        "words": []}]}


def test_stream_parses_stages_and_returns_result(monkeypatch, audio):
    lines = _sse(
        ": keepalive",
        "",
        "event: progress",
        'data: {"stage": "queued_gpu"}',
        "event: progress",
        'data: {"stage": "segmenting"}',
        "event: progress",
        'data: {"stage": "transcribing", "step": 3, "steps": 6}',
        "event: progress",
        'data: {"stage": "matching"}',
        "event: result",
        "data: " + json.dumps(RESULT),
    )
    fake, provider = _provider([FakeResp(lines=lines)], monkeypatch)
    seen = []
    out = provider.align(audio, progress_cb=lambda p, l: seen.append((p, l)))

    assert out == RESULT
    assert [(round(p), l) for p, l in seen][0][0] == 20
    pcts = [round(p) for p, _ in seen]
    assert pcts == [20, 30, 50, 68], pcts  # transcribing: 35 + 30*3/6
    url, kwargs = fake.calls[0]
    assert url == "https://example.test/api/v1/align/audio/stream"
    assert kwargs["data"] == {"device": provider.device,
                              "model_name": provider.model,
                              "riwayah": "hafs"}
    assert kwargs["stream"] is True


def test_503_is_retried_with_backoff(monkeypatch, audio):
    sleeps = []
    lines = ["event: result", "data: " + json.dumps(RESULT)]
    fake, provider = _provider(
        [FakeResp(status=503, body={"code": "busy", "message": "overloaded"}),
         FakeResp(lines=lines)],
        monkeypatch, sleeps=sleeps)
    out = provider.align(audio)
    assert out == RESULT
    assert len(fake.calls) == 2
    assert sleeps == [3.0], sleeps


def test_402_raises_immediately(monkeypatch, audio):
    sleeps = []
    fake, provider = _provider(
        [FakeResp(status=402, body={"code": "quota", "message": "no GPU quota"})],
        monkeypatch, sleeps=sleeps)
    with pytest.raises(QudUnavailable) as exc:
        provider.align(audio)
    assert "quota" in str(exc.value).lower()
    assert len(fake.calls) == 1
    assert sleeps == []


def test_429_short_retry_after_is_honoured(monkeypatch, audio):
    sleeps = []
    lines = ["event: result", "data: " + json.dumps(RESULT)]
    fake, provider = _provider(
        [FakeResp(status=429, body={"code": "rate_limited",
                                    "message": "slow down",
                                    "detail": {"retry_after_s": 4}}),
         FakeResp(lines=lines)],
        monkeypatch, sleeps=sleeps)
    assert provider.align(audio) == RESULT
    assert sleeps == [4.0], sleeps
    assert len(fake.calls) == 2


def test_429_long_retry_after_fails_fast(monkeypatch, audio):
    sleeps = []
    fake, provider = _provider(
        [FakeResp(status=429, body={"code": "rate_limited", "message": "later",
                                    "detail": {"retry_after_s": 120}})],
        monkeypatch, sleeps=sleeps)
    with pytest.raises(QudUnavailable):
        provider.align(audio)
    assert len(fake.calls) == 1, "a 120 s wait must fall back, not sleep"
    assert sleeps == []


def test_connection_error_retries_then_gives_up(monkeypatch, audio):
    sleeps = []
    fake, provider = _provider(
        [ConnectionError("connection reset")] * 3, monkeypatch, sleeps=sleeps)
    with pytest.raises(QudUnavailable):
        provider.align(audio)
    assert len(fake.calls) == 3
    assert sleeps == [3.0, 6.0], sleeps


def test_stream_without_result_is_transient(monkeypatch, audio):
    sleeps = []
    fake, provider = _provider(
        [FakeResp(lines=["event: progress", 'data: {"stage": "building"}'])] * 3,
        monkeypatch, sleeps=sleeps)
    with pytest.raises(QudUnavailable) as exc:
        provider.align(audio)
    assert exc.value.transient
    assert len(fake.calls) == 3


def test_error_event_raises(monkeypatch, audio):
    lines = ["event: error",
             'data: {"code": "alignment_failed", "message": "no speech", "status": 422}']
    fake, provider = _provider([FakeResp(lines=lines)], monkeypatch)
    with pytest.raises(QudUnavailable) as exc:
        provider.align(audio)
    assert "no speech" in str(exc.value)
    assert len(fake.calls) == 1


def test_validate_response_requires_a_quran_segment():
    with pytest.raises(QudUnavailable):
        validate_response({"segments": []})
    with pytest.raises(QudUnavailable):
        validate_response({"segments": [{"kind": "special", "special_type": "Basmala"}]})
    assert validate_response(RESULT) is RESULT


def test_transcribe_is_not_supported():
    with pytest.raises(QudUnavailable):
        QudProvider().transcribe(audio)
