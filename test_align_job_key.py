"""An align is a multi-minute job; starting it must survive a rerun.

Streamlit reruns (and aborts) the running script on ANY widget change, and the
burn-in video now lands mid-align on purpose (prefetch upload). The client
therefore starts the job with a key it keeps in session state: re-posting the
same key must return the job that is already running, not start a duplicate
or orphan the first one.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest
from fastapi.testclient import TestClient

import backend.main as main


@pytest.fixture()
def jobs(monkeypatch):
    monkeypatch.setattr(main, "ALIGN_JOBS", {})
    monkeypatch.setattr(main, "ALIGN_KEYS", {})
    # Never run a real align from here.
    monkeypatch.setattr(main, "_align_job_worker", lambda *a, **k: None)
    return main


def _post(client, key=None, name="tiny.wav"):
    data = {"job": "1"}
    if key is not None:
        data["key"] = key
    return client.post("/align", data=data,
                       files={"file": (name, b"RIFF....", "audio/wav")},
                       timeout=30)


def test_same_key_returns_the_same_job(jobs):
    client = TestClient(jobs.app)
    a = _post(client, "k1")
    b = _post(client, "k1")
    assert a.status_code == 200 and b.status_code == 200
    assert a.json()["job_id"] == b.json()["job_id"]
    assert len(jobs.ALIGN_JOBS) == 1
    assert jobs.ALIGN_KEYS == {"k1": a.json()["job_id"]}


def test_different_keys_start_different_jobs(jobs):
    client = TestClient(jobs.app)
    a = _post(client, "k1")
    b = _post(client, "k2")
    c = _post(client)  # no key: legacy behaviour, one job per request
    ids = {a.json()["job_id"], b.json()["job_id"], c.json()["job_id"]}
    assert len(ids) == 3


def test_second_post_reports_the_running_status(jobs):
    """The resumed start must not reset progress of the job it returns."""
    client = TestClient(jobs.app)
    jid = _post(client, "k1").json()["job_id"]
    jobs.ALIGN_JOBS[jid].update(status="running", progress=42.0)
    again = _post(client, "k1").json()
    assert again["job_id"] == jid
    assert again["status"] == "running"
    assert jobs.ALIGN_JOBS[jid]["progress"] == 42.0


def test_purge_drops_keys_whose_job_is_gone(jobs):
    jobs.ALIGN_KEYS["old"] = "gone"
    jobs.ALIGN_JOBS["fresh"] = {"status": "done", "created": time.time()}
    jobs.ALIGN_KEYS["live"] = "fresh"
    jobs.ALIGN_JOBS["stale"] = {"status": "done", "created": time.time() - 25 * 3600}
    jobs._purge_old_align_jobs()
    assert jobs.ALIGN_KEYS == {"live": "fresh"}
    assert set(jobs.ALIGN_JOBS) == {"fresh"}


def test_a_purged_key_starts_a_new_job(jobs):
    """After the 24h purge the old key must not resurrect a dead job."""
    client = TestClient(jobs.app)
    jid = _post(client, "k1").json()["job_id"]
    del jobs.ALIGN_JOBS[jid]
    fresh = _post(client, "k1").json()["job_id"]
    assert fresh != jid
    assert jobs.ALIGN_JOBS[fresh]["status"] == "queued"


def test_app_resumes_the_job_through_session_state():
    """Source guard: the frontend must keep the key + job id across reruns
    and start the engine job with that key (the prefetch video landing in
    the middle of an align is a rerun)."""
    root = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(root, "frontend", "app.py"), encoding="utf-8-sig") as f:
        src = f.read()
    assert 'ss.setdefault("align_key", None)' in src
    assert 'ss.setdefault("align_job_id", None)' in src
    assert '"key": ss["align_key"]' in src
    assert "def poll_align_job(" in src
    # The hidden step-1 uploader is the prefetch; it must stay hidden and
    # share the widget key of the render-step uploader.
    assert 'key=f"video_prefetch_{ss[\'media_id\']}"' in src
    assert 'key=f"video_uploader_{ss[\'media_id\']}"' in src
