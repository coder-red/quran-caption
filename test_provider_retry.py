"""Provider resilience: a dropped connection must not kill a whole align.

On a free tier with a shared, flaky uplink a single reset connection used
to abort a multi-minute align with "groq transcription failed: Connection
error" — even though the very next request would have worked. Transient
failures are now retried with backoff; permanent ones (4xx) still fail fast.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.providers.base import is_transient_error, retry_with_backoff


class _ConnErr(Exception):
    """Mimics httpx/groq connection errors."""


class _TimeoutErr(Exception):
    pass


class _BadRequest(Exception):
    status_code = 400


class _ServerErr(Exception):
    status_code = 503


class _RateLimited(Exception):
    status_code = 429


def test_connection_error_is_transient():
    assert is_transient_error(_ConnErr("Connection error."))


def test_timeout_is_transient():
    assert is_transient_error(_TimeoutErr("Request timed out"))


def test_5xx_is_transient():
    assert is_transient_error(_ServerErr("upstream down"))


def test_429_is_transient():
    assert is_transient_error(_RateLimited("slow down"))


def test_4xx_is_permanent():
    assert not is_transient_error(_BadRequest("bad request"))


def test_arbitrary_error_is_not_transient():
    assert not is_transient_error(ValueError("nope"))


def test_retry_recovers_after_transient_failures():
    calls = {"n": 0}

    def flaky(attempt):
        calls["n"] += 1
        if attempt < 2:
            raise _ConnErr("Connection error.")
        return "ok"

    t0 = time.time()
    out = retry_with_backoff(flaky, attempts=3, base_delay=0.01)
    assert out == "ok"
    assert calls["n"] == 3
    assert time.time() - t0 < 5


def test_retry_gives_up_after_attempts():
    calls = {"n": 0}

    def always_fail(attempt):
        calls["n"] += 1
        raise _ConnErr("Connection error.")

    try:
        retry_with_backoff(always_fail, attempts=2, base_delay=0.01)
        raise AssertionError("should have raised")
    except _ConnErr:
        pass
    assert calls["n"] == 3, calls


def test_retry_does_not_retry_permanent_errors():
    calls = {"n": 0}

    def bad(attempt):
        calls["n"] += 1
        raise _BadRequest("bad request")

    try:
        retry_with_backoff(bad, attempts=3, base_delay=0.01)
        raise AssertionError("should have raised")
    except _BadRequest:
        pass
    assert calls["n"] == 1, "a 4xx must fail fast, not burn retries"


def test_groq_provider_retries_a_dropped_connection():
    """The provider itself must survive one dropped connection."""
    from backend.providers.groq import GroqProvider

    prov = GroqProvider(api_key="test-key")
    calls = {"n": 0}

    class _FakeApi:
        class audio:  # noqa: N801
            class transcriptions:  # noqa: N801
                @staticmethod
                def create(**kwargs):
                    calls["n"] += 1
                    if calls["n"] < 3:
                        raise _ConnErr("Connection error.")
                    class R:
                        words = []
                    return R()

    prov._client = _FakeApi()
    import backend.providers.groq as gp
    orig_sleep = gp.time.sleep
    gp.time.sleep = lambda s: None
    try:
        import tempfile as tf
        import numpy as np
        import soundfile as sf
        fd, wav = tf.mkstemp(suffix=".wav")
        os.close(fd)
        sf.write(wav, np.zeros(16000, dtype="float32"), 16000)
        res = prov._create_transcription(wav, "ar")
        assert res is not None
        assert calls["n"] == 3, calls
    finally:
        gp.time.sleep = orig_sleep
        try:
            os.remove(wav)
        except OSError:
            pass


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
