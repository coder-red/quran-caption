import json
import os
import sys
import time
import uuid

import pandas as pd
import requests
import streamlit as st

# Sibling modules are imported by directory (not as a package): the app runs
# as `streamlit run frontend/app.py` both here and on the Space, so the
# script dir is the only stable anchor.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from sync_inspector import inspector_html as sync_inspector_html  # noqa: E402
import media_store  # noqa: E402

API_URL = os.environ.get("QCAP_API_URL", "http://localhost:8000")
PUBLIC_URL = os.environ.get("QCAP_PUBLIC_URL", API_URL)
DEPLOYED = bool(os.environ.get("SPACE_ID"))
if DEPLOYED:
    API_URL = "http://127.0.0.1:8000"
    PUBLIC_URL = ""

# Fast upload: the browser extracts a video's audio and uploads only that
# (see frontend/opus_proxy.js and frontend/auto_proxy.js).
def _read_asset(name: str) -> str:
    try:
        with open(os.path.join(_HERE, name), encoding="utf-8") as f:
            return f.read()
    except OSError:
        return ""


_PROXY_JS = _read_asset("opus_proxy.js") + "\n" + _read_asset("auto_proxy.js")

# Deployed browsers cannot reach the API port, so clips cross this process:
# frontend/media_store.py sends them in 4MB slices and publishes the previews
# into Streamlit's static dir (range-request streaming from disk). A single
# 40MB multipart body plus inlined base64 previews OOM-killed the 512MB host.
_TEMP_DIR = os.path.abspath(os.environ.get("TEMP_DIR") or "temp")
_STATIC_DIR = os.path.join(_HERE, "static")


def post_media(media_id: str, name: str, source, ctype: str = ""):
    return media_store.post_media(API_URL, media_id, name, source, ctype)


def publish_media(media_id: str):
    if not DEPLOYED:
        return None
    return media_store.publish_media(_TEMP_DIR, _STATIC_DIR, media_id)


def prune_static(media_ids):
    media_store.prune_static(_STATIC_DIR, media_ids)


def publish_render(job_id: str):
    # The rendered MP4 goes to disk and is streamed by Streamlit's static
    # server: bytes in the app (r.content + download_button copies) were what
    # OOM-killed the 512MB host after a render.
    return media_store.publish_render(API_URL, _STATIC_DIR, job_id)


def prune_renders(job_ids):
    media_store.prune_renders(_STATIC_DIR, job_ids)


def source_is_video() -> bool:
    """The picked clip was a video, whose soundtrack is uploaded first."""
    return media_store.is_audio_proxy((ss.get("file_meta") or {}).get("name") or "")


def source_video_name() -> str:
    name = (ss.get("file_meta") or {}).get("name") or ""
    return media_store.proxy_source_name(name) or name


def poll_align_job(jid: str, status):
    """Poll a running align job; the result dict, or None after an error.

    Lives on the (job id) instead of a single script run: the burn-in video
    landing mid-align stops the running script (Streamlit reruns on any
    widget change), and without this the finished job was orphaned and the
    page fell back to "Generate captions"."""
    prog = st.progress(0, text="Queued…")
    misses, started, last_pct = 0, time.time(), 0
    while True:
        time.sleep(0.8)
        try:
            job = requests.get(f"{API_URL}/align/status/{jid}", timeout=15).json()
            misses = 0
        except (requests.RequestException, ValueError):
            misses += 1
            elapsed = int(time.time() - started)
            prog.progress(
                last_pct, text=f"Reconnecting… ({misses}/5) · {elapsed}s elapsed",
            )
            if misses >= 5:
                prog.empty()
                status.update(label="Engine unreachable", state="error")
                st.error(
                    "Lost contact with the caption engine after "
                    f"{elapsed}s (it may have crashed). Press Generate captions "
                    "to retry once the engine is back up."
                )
                return None
            continue
        if not isinstance(job, dict) or "status" not in job:
            prog.empty()
            status.update(label="Job lost", state="error")
            st.error(
                "The engine no longer knows this align job (it may have "
                "restarted). Press Generate captions to run it again."
            )
            return None
        pct = job.get("progress", 0)
        last_pct = min(99, int(pct))
        prog.progress(
            last_pct,
            text=f"{job.get('stage', 'working…')} · {pct:.0f}% · "
                 f"{int(time.time() - started)}s elapsed",
        )
        status.update(label=job.get("stage", "working…"))
        if job.get("status") == "error":
            # /align/status carries no message — fetch it from the result
            # endpoint (now surfaces the real job error).
            msg = None
            try:
                msg = requests.get(f"{API_URL}/align/result/{jid}", timeout=30).json().get("error")
            except (requests.RequestException, ValueError):
                msg = None
            prog.empty()
            status.update(label="Failed", state="error")
            st.error(f"Align failed: {msg or 'unknown error'}")
            return None
        if job.get("status") == "done":
            break
    try:
        data = requests.get(f"{API_URL}/align/result/{jid}", timeout=60).json()
    except (requests.RequestException, ValueError) as e:
        prog.empty()
        status.update(label="Result fetch failed", state="error")
        st.error(f"Could not fetch the result: {e}")
        return None
    if "error" in data:
        prog.empty()
        status.update(label="Failed", state="error")
        st.error(f"Could not identify the surah: {data['error']}")
        return None
    prog.progress(100, text="Done · captions ready")
    status.update(label="Surah detected — captions ready", state="complete")
    return data


def ensure_render_video(vfile) -> str | None:
    """Put the picked video on the engine once; None on success.

    The transfer is ~0.4 MB/s on the free host, so it starts as soon as the
    file is picked (hidden uploader in step 1) instead of waiting for the
    align to finish — otherwise the render step sits on it for minutes."""
    if vfile is None:
        return None
    file_id = getattr(vfile, "file_id", None)
    if (ss.get("video_uploaded") and ss.get("video_media_id")
            and (ss.get("video_meta") or {}).get("file_id") == file_id):
        return None
    if ss.get("video_media_id") is None or (ss.get("video_meta") or {}).get("file_id") != file_id:
        ss["video_media_id"] = uuid.uuid4().hex[:10]
        ss["video_meta"] = {"name": vfile.name, "file_id": file_id}
        ss["preview_video_uri"] = None
        ss["preview_video_tried"] = False
        ss["video_uploaded"] = False
    try:
        with st.spinner("Sending the original video to the engine…"):
            up = post_media(ss["video_media_id"], vfile.name, vfile, vfile.type)
    except requests.RequestException as e:
        return f"Video upload failed: {e}"
    if up.status_code != 200:
        return f"Video upload failed: {api_error(up)}"
    ss["video_uploaded"] = True
    return None


def video_prefetch() -> None:
    """Hidden uploader that carries the burn-in video to the engine.

    It must be rendered on EVERY script run: Streamlit drops the state of a
    widget whose run does not render it, so skipping this block during the
    blocking align or in the result view deleted the file mid-transfer and
    the render step asked for the video all over again. The transfer also
    starts here, in parallel with the align — it takes ~2 minutes on the
    free host, longer than the align itself."""
    if not (ss.get("media_id") and source_is_video()
            and not ss.get("video_uploaded")):
        return
    with st.container(key=f"video_prefetch_{ss['media_id']}"):
        pfile = st.file_uploader(
            "Video for the burned-in MP4",
            type=["mp4", "webm", "mov", "avi", "mkv"],
            label_visibility="collapsed",
            key=f"video_uploader_{ss['media_id']}",
        )
    if pfile is not None:
        err = ensure_render_video(pfile)
        if err:
            st.caption(err + " You can pick the video again at the render step.")


@st.cache_resource
def _ensure_backend():
    """On Spaces: start the FastAPI child process. No model pre-downloads —
    the slim deploy uses the hosted ASR (ASR_BACKEND=groq)."""
    if not DEPLOYED:
        return
    import urllib.request

    try:
        urllib.request.urlopen(f"{API_URL}/health", timeout=3)
        return
    except Exception:
        pass

    import subprocess
    import sys

    os.makedirs("temp", exist_ok=True)
    subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.main:app",
         "--host", "127.0.0.1", "--port", "8000"],
        stdout=open(os.path.join("temp", "uvicorn.log"), "ab"),
        stderr=open(os.path.join("temp", "uvicorn.err.log"), "ab"),
    )
    deadline = time.time() + 180
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"{API_URL}/health", timeout=2)
            return
        except Exception:
            time.sleep(2)
    st.warning("Caption engine failed to start on this instance.")

st.set_page_config(
    page_title="Quran Caption",
    page_icon=":material/menu_book:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

_ensure_backend()

st.markdown("""
<style>
:root {
  --gold: #D4AF37;
  --emerald: #10B981;
  --emerald-2: #34D399;
  --emerald-soft: rgba(16, 185, 129, 0.12);
  --bg: #0B1220;
  --card: #131D31;
  --card-2: #17233B;
  --border: #24314D;
  --border-soft: #1C2842;
  --text: #E8ECF4;
  --text-dim: #8B95AB;
}

html, body, [data-testid="stAppViewContainer"] { background: var(--bg); }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stAppDeployButton"], [data-testid="stDecoration"] { display: none; }

.block-container { padding-top: 1.2rem; padding-bottom: 3.5rem; max-width: 1080px; }

/* ---------- Hero ---------- */
.hero { text-align: center; padding: 1.6rem 0 1.1rem; }
.hero .eyebrow {
  font-size: 0.68rem;
  font-weight: 700;
  letter-spacing: 0.24em;
  text-transform: uppercase;
  color: var(--gold);
  opacity: 0.85;
  margin-bottom: 0.5rem;
}
.hero .title {
  font-size: 2.9rem;
  font-weight: 800;
  letter-spacing: -0.03em;
  line-height: 1.05;
  background: linear-gradient(115deg, #34D399 0%, #D4AF37 62%, #F5C77E 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
.hero .lede {
  color: var(--text-dim);
  font-size: 1rem;
  margin: 0.55rem auto 0;
  max-width: 34rem;
}

/* ---------- Cards (st.container(key="card_*")) ---------- */
div[data-testid="stVerticalBlock"]:is([class*="st-key-card_"]) {
  background: linear-gradient(180deg, #131D31 0%, #111A2C 100%);
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 1.15rem 1.2rem 1.1rem;
  box-shadow: 0 18px 44px rgba(0, 0, 0, 0.26);
}

/* ---------- Card section headers ---------- */
.sec { display: flex; align-items: center; gap: 0.7rem; margin: 0 0 0.85rem; }
.sec .n {
  width: 1.85rem; height: 1.85rem; flex: none;
  display: grid; place-items: center;
  border-radius: 0.62rem;
  font-size: 0.85rem; font-weight: 800; color: #04160F;
  background: linear-gradient(150deg, #34D399, #10B981);
  box-shadow: 0 0 0 1px rgba(16,185,129,0.4), 0 6px 16px rgba(16,185,129,0.2);
}
.sec .txt { min-width: 0; }
.sec .txt .h { font-size: 1.06rem; font-weight: 700; letter-spacing: -0.01em; line-height: 1.25; }
.sec .txt .sub { color: var(--text-dim); font-size: 0.82rem; margin-top: 0.12rem; line-height: 1.35; }
.sec .tag {
  margin-left: auto; flex: none;
  font-size: 0.64rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase;
  color: var(--text-dim);
  border: 1px solid var(--border);
  border-radius: 999px;
  padding: 0.2rem 0.6rem;
  white-space: nowrap;
}

.subhead {
  font-size: 0.68rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase;
  color: var(--text-dim);
  margin: 1.1rem 0 0.45rem;
}

/* ---------- Detection summary ---------- */
.det-head {
  display: flex; align-items: flex-start; justify-content: space-between;
  gap: 1rem; flex-wrap: wrap;
}
.det-sura-ar { font-size: 2rem; font-weight: 700; line-height: 1.35; direction: rtl; }
.det-sura-en { color: var(--text-dim); font-size: 0.88rem; margin-top: 0.1rem; }
.det-side { display: flex; flex-direction: column; align-items: flex-end; gap: 0.35rem; }
.det-ok {
  font-size: 0.64rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase;
  color: var(--text-dim);
}
.engine-chip {
  font-size: 0.64rem; font-weight: 800; letter-spacing: 0.1em; text-transform: uppercase;
  color: var(--emerald-2);
  background: var(--emerald-soft);
  border: 1px solid rgba(16, 185, 129, 0.35);
  border-radius: 999px;
  padding: 0.22rem 0.6rem;
}
.stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.55rem; margin: 1rem 0 0.85rem; }
.stat {
  background: rgba(255, 255, 255, 0.02);
  border: 1px solid var(--border-soft);
  border-radius: 12px;
  padding: 0.6rem 0.4rem;
  text-align: center;
}
.stat .v { font-size: 1.22rem; font-weight: 750; line-height: 1.2; }
.stat .l {
  font-size: 0.6rem; letter-spacing: 0.09em; text-transform: uppercase;
  color: var(--text-dim); margin-top: 0.18rem;
}
.conf-bar {
  height: 6px; border-radius: 999px; background: var(--border);
  overflow: hidden;
}
.conf-fill {
  height: 100%; border-radius: 999px;
  background: linear-gradient(90deg, var(--emerald), var(--gold));
}

/* ---------- Uploader ---------- */
section[data-testid="stFileUploaderDropzone"] {
  background: rgba(255, 255, 255, 0.015);
  border: 1.5px dashed #33436199;
  border-radius: 14px;
  padding: 1.4rem 1.1rem;
  transition: border-color 120ms ease, background 120ms ease;
}
section[data-testid="stFileUploaderDropzone"]:hover {
  border-color: rgba(16, 185, 129, 0.65);
  background: rgba(16, 185, 129, 0.04);
}
section[data-testid="stFileUploaderDropzone"] button {
  background: var(--emerald);
  color: #04160F;
  font-weight: 700;
  border: none;
  border-radius: 9px;
  padding: 0.5rem 1rem;
}
section[data-testid="stFileUploaderDropzone"] button:hover { background: var(--emerald-2); }
div[data-testid="stFileUploaderDropzoneInstructions"] div { color: var(--text-dim); }
div[data-testid="stFileUploaderDropzoneInstructions"] span { color: var(--text); }

/* ---------- Buttons ---------- */
.stButton > button, .stDownloadButton > button {
  border-radius: 10px;
  font-weight: 650;
  border: 1px solid var(--border);
  background: rgba(255, 255, 255, 0.02);
  transition: border-color 120ms ease, background 120ms ease, transform 60ms ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  border-color: rgba(16, 185, 129, 0.6);
  background: rgba(16, 185, 129, 0.07);
}
.stButton > button:active, .stDownloadButton > button:active { transform: translateY(1px); }
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {
  background: linear-gradient(135deg, #34D399, #10B981);
  color: #04160F;
  font-weight: 800;
  border: none;
  padding: 0.55rem 1.2rem;
}
.stButton > button[kind="primary"]:hover, .stDownloadButton > button[kind="primary"]:hover {
  background: linear-gradient(135deg, #6EE7B7, #34D399);
  color: #04160F;
}

/* Download link for the burned MP4: Streamlit's download_button needs the
   whole video in RAM, which OOM-kills the 512MB host, so the file is served
   from the static dir and this is a plain link with a download attribute. */
a.dl-btn {
  display: block;
  margin-top: 0.5rem;
  text-align: center;
  text-decoration: none !important;
  font-weight: 800;
  padding: 0.62rem 1.2rem;
  border-radius: 10px;
  background: linear-gradient(135deg, #34D399, #10B981);
  color: #04160F !important;
}
a.dl-btn:hover { filter: brightness(1.07); }
a.dl-btn:active { transform: translateY(1px); }

/* The burn-in video starts transferring the moment a clip is picked, so it
   is already on the engine when the render step opens; the uploader itself
   is filled by auto_proxy.js and has no reason to be on screen. */
div[class*="st-key-video_prefetch"] { display: none !important; }

/* ---------- Selectable chips ---------- */
button[data-testid^="stBaseButton-segmented_control"],
button[data-testid^="stBaseButton-pills"] {
  border-radius: 999px !important;
  border: 1px solid var(--border) !important;
  background: transparent !important;
  color: var(--text) !important;
  font-weight: 600 !important;
}
button[data-testid="stBaseButton-segmented_controlActive"],
button[data-testid="stBaseButton-pillsActive"] {
  background: var(--emerald-soft) !important;
  border-color: rgba(16, 185, 129, 0.55) !important;
  color: var(--emerald-2) !important;
}
div[data-testid="stPills"] { flex-wrap: wrap; }

/* ---------- Progress / expanders / tables ---------- */
div[data-testid="stProgressBar"] > div > div {
  background: linear-gradient(90deg, var(--emerald), var(--gold));
}
div[data-testid="stExpander"] details {
  background: var(--card-2);
  border: 1px solid var(--border-soft);
  border-radius: 12px;
}
div[data-testid="stExpander"] summary { border-radius: 12px; }
div[data-testid="stDataFrame"] { border-radius: 10px; overflow-x: auto; }

/* ---------- Caption player ---------- */
#capwrap { box-shadow: 0 18px 44px rgba(0, 0, 0, 0.4); }
#fsbtn { min-height: 44px; min-width: 44px; }

/* ---------- Mobile ---------- */
div[data-testid="stHorizontalBlock"] { flex-wrap: wrap; }
div[data-testid="stHorizontalBlock"] > div[data-testid="column"] { min-width: 240px; flex: 1 1 240px; }

@media (max-width: 768px) {
  .block-container { padding: 0.8rem 0.8rem 3rem; }
  .hero { padding: 0.9rem 0 0.6rem; }
  .hero .title { font-size: 2.15rem; }
  .hero .lede { font-size: 0.9rem; }
  div[data-testid="stVerticalBlock"]:is([class*="st-key-card_"]) {
    padding: 0.95rem 0.9rem 0.95rem;
    border-radius: 15px;
  }
  .sec { gap: 0.55rem; }
  .sec .txt .h { font-size: 1rem; }
  .sec .tag { display: none; }
  .stats { gap: 0.4rem; }
  .stat { padding: 0.5rem 0.25rem; }
  .stat .v { font-size: 1.02rem; }
  .stat .l { font-size: 0.55rem; }
  .det-sura-ar { font-size: 1.6rem; }
  .det-side { align-items: flex-start; }
  div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
    min-width: 100%; flex-basis: 100%;
  }
  .stButton > button, .stDownloadButton > button { min-height: 46px; }
  section[data-testid="stFileUploaderDropzone"] { padding: 1rem 0.8rem; }
  /* Only on phones: pin the player while scrolling the controls below it.
     The element container is the right sticky box — a keyed card wrapper is
     exactly as tall as its content and gives sticky no scroll room. */
  div[data-testid="stElementContainer"]:has(#capwrap) {
    position: sticky;
    top: 0;
    z-index: 40;
    padding: 0.3rem 0;
    background: var(--bg);
    box-shadow: 0 12px 28px rgba(0, 0, 0, 0.55);
  }
  div[data-testid="stSlider"] { padding: 0.35rem 0; }
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="eyebrow">Quran Caption Studio</div>
  <div class="title">Quran Caption</div>
  <div class="lede">Upload a recitation and get ayah-accurate captions you can
  style, verify against the audio, and burn into the video.</div>
</div>
""", unsafe_allow_html=True)

MAX_SIZE = int(os.environ.get("QCAP_MAX_MB", "400")) * 1024 * 1024

PRESETS = {
    "Classic": {"font_size": 48, "font_family": "Segoe UI", "font_color": "#FFFFFF",
                "outline_color": "#000000", "outline_size": 2, "shadow_size": 0,
                "bold": False, "box": False, "position": "bottom"},
    "Elegant Gold": {"font_size": 52, "font_family": "Segoe UI", "font_color": "#FFD700",
                     "outline_color": "#1F2937", "outline_size": 3, "shadow_size": 1,
                     "bold": False, "box": False, "position": "bottom"},
    "Emerald": {"font_size": 48, "font_family": "Segoe UI", "font_color": "#A7F3D0",
                "outline_color": "#064E3B", "outline_size": 3, "shadow_size": 0,
                "bold": True, "box": False, "position": "bottom"},
    "Bold & Clear": {"font_size": 56, "font_family": "Segoe UI", "font_color": "#FFFFFF",
                     "outline_color": "#000000", "outline_size": 4, "shadow_size": 0,
                     "bold": True, "box": False, "position": "bottom"},
    "Minimal": {"font_size": 40, "font_family": "Segoe UI", "font_color": "#FFFFFF",
                "outline_color": "#000000", "outline_size": 1, "shadow_size": 0,
                "bold": False, "box": False, "position": "bottom"},
    "Boxed": {"font_size": 48, "font_family": "Segoe UI", "font_color": "#FFFFFF",
              "outline_color": "#000000", "outline_size": 1, "shadow_size": 0,
              "bold": False, "box": True, "position": "bottom"},
    "Top Aligned": {"font_size": 48, "font_family": "Segoe UI", "font_color": "#FFFFFF",
                    "outline_color": "#000000", "outline_size": 2, "shadow_size": 0,
                    "bold": False, "box": False, "position": "top"},
}

RESOLUTIONS = {
    "Original": (None, None),
    "1080p": (1920, 1080),
    "720p": (1280, 720),
    "480p": (854, 480),
}


def _host_cpu_quota() -> float:
    """CPU quota of this container (>=1.0 on a normal host).

    The free deploy host reports many host cores but allows 0.1 CPU; there a
    1080p x264 encode peaks at 327MB RSS (measured on the 8-min Hashr clip)
    before Streamlit and the API are counted — that is what OOM-killed it.
    The backend clamps to 720p on such hosts too, so the UI must not offer
    what cannot run."""
    try:
        with open("/sys/fs/cgroup/cpu.max", encoding="utf-8") as f:  # cgroup v2
            quota, period = f.read().split()[:2]
        if quota != "max":
            return float(quota) / float(period)
    except (OSError, ValueError):
        pass
    try:  # cgroup v1
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", encoding="utf-8") as f:
            quota = float(f.read().strip())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us", encoding="utf-8") as f:
            period = float(f.read().strip())
        if quota > 0 and period > 0:
            return quota / period
    except (OSError, ValueError):
        pass
    return float(os.cpu_count() or 1)


_LOW_CPU = _host_cpu_quota() < 1.0

LANGS = {
    "ar": "Arabic",
    "en": "English",
    "both": "Arabic + English",
}

ss = st.session_state
ss.setdefault("file_bytes", None)
ss.setdefault("file_meta", None)
ss.setdefault("align_result", None)
# An in-flight align. `align_key` is the idempotency key the engine started the
# job under, `align_job_id` the job to poll; both survive reruns so a widget
# change during the align (the burn-in video landing) resumes the job instead
# of orphaning it.
ss.setdefault("align_key", None)
ss.setdefault("align_job_id", None)
ss.setdefault("segments", None)
ss.setdefault("lang", "ar")
ss.setdefault("styles", dict(PRESETS["Classic"]))
ss.setdefault("preset", "Classic")
ss.setdefault("media_id", None)
ss.setdefault("media_url", None)
ss.setdefault("media_uploaded", False)
ss.setdefault("render_job_id", None)
ss.setdefault("render_started", None)
ss.setdefault("render_url", None)
ss.setdefault("render_name", None)
ss.setdefault("render_file", None)
# media_id of the video, uploaded separately at the burn-in step when the
# recitation itself was an audio file (captions still need something to burn
# into).
ss.setdefault("video_media_id", None)
ss.setdefault("video_uploaded", False)
ss.setdefault("video_meta", None)
# Preview of the burn-in video, fetched once it is on the server: the fast
# upload only sends the audio, so the picture arrives later.
ss.setdefault("preview_video_uri", None)
ss.setdefault("preview_video_tried", False)


def api_error(resp: requests.Response):
    try:
        detail = resp.json()
        if isinstance(detail, dict):
            return detail.get("detail") or detail.get("error") or str(detail)
        return str(detail)
    except Exception:
        return resp.text[:400]


def fmt_ts(seconds: float) -> str:
    ms = int(round(float(seconds) * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def srt_from_segments(segments: list[dict], language: str) -> str:
    lines = []
    for i, seg in enumerate(segments, 1):
        lines.append(str(i))
        lines.append(f"{fmt_ts(seg['start'])} --> {fmt_ts(seg['end'])}")
        if language == "en":
            lines.append(seg.get("text_en") or seg.get("text_ar") or "")
        elif language == "both":
            if seg.get("text_ar"):
                lines.append(seg["text_ar"])
            if seg.get("text_en"):
                lines.append(seg["text_en"])
        else:
            lines.append(seg.get("text_ar") or "")
        lines.append("")
    return "\n".join(lines)


def vtt_from_srt(srt: str) -> str:
    return "WEBVTT\n\n" + srt.replace(",", ".")


def style_payload(extra: dict | None = None) -> dict:
    s = ss["styles"]
    payload = {
        "font_size": str(s["font_size"]),
        "font_family": s["font_family"],
        "font_color": s["font_color"],
        "outline_color": s["outline_color"],
        "outline_size": str(s["outline_size"]),
        "shadow_size": str(s["shadow_size"]),
        "bold": "true" if s["bold"] else "false",
        "box": "true" if s["box"] else "false",
        "position": s["position"],
    }
    if extra:
        payload.update(extra)
    return payload


def player_html(media_url: str, segments: list[dict], lang: str, styles: dict, player_key: str = "player") -> str:
    import base64 as _b64
    seg_json = json.dumps(segments, ensure_ascii=True).replace("<", "\\u003c").replace("&", "\\u0026")
    style_json = json.dumps(styles, ensure_ascii=True).replace("<", "\\u003c").replace("&", "\\u0026")
    seg_b64 = _b64.b64encode(seg_json.encode()).decode()
    sty_b64 = _b64.b64encode(style_json.encode()).decode()
    return f"""
<div id="capwrap" data-segs="{seg_b64}" data-st="{sty_b64}" data-lang="{lang}" data-key="{player_key}"
     style="position:relative;width:100%;aspect-ratio:16/9;background:#000;border-radius:14px;overflow:hidden;">
  <video id="vid" src="{media_url}" controls preload="metadata" playsinline style="width:100%;height:100%;display:block;"></video>
  <div id="cap" style="position:absolute;left:3%;right:3%;bottom:7%;text-align:center;pointer-events:none;color:#fff;">
    <div id="ar" dir="rtl" style="line-height:1.5;"></div>
    <div id="en" dir="ltr" style="line-height:1.35;"></div>
  </div>
  <button id="fsbtn" title="Fullscreen" style="position:absolute;top:10px;right:12px;z-index:5;background:rgba(0,0,0,0.55);border:1px solid rgba(255,255,255,0.35);color:#fff;border-radius:8px;padding:4px 10px;font-size:13px;cursor:pointer;">Fullscreen</button>
</div>
<script>
// Streamlit st.html does NOT re-run this script when it re-renders the block
// (style/language edits rerun the app and emit a fresh player). So the whole
// caption engine binds ONCE at the document level (event delegation) and reads
// segs/styles/lang from data-* attributes set on each render. Reruns then just
// swap the data attributes — captions keep working without re-execution.
if (!window.__qcap) {{
  window.__qcap = {{}};

  // Cache keyed on the attribute value: Streamlit reruns swap data-segs /
  // data-st on the SAME DOM node when segments are re-aligned or styles
  // edited, so a parse-once cache would keep showing stale timings. Re-parse
  // whenever the attribute changes.
  function parseWrap(w) {{
    try {{
      if (w.__segsKey !== w.dataset.segs) {{
        w.__segs = JSON.parse(atob(w.dataset.segs || ''));
        w.__segsKey = w.dataset.segs;
      }}
      if (w.__stKey !== w.dataset.st) {{
        w.__st = JSON.parse(atob(w.dataset.st || ''));
        w.__stKey = w.dataset.st;
      }}
    }} catch (e) {{ w.__segs = w.__segs || []; w.__st = w.__st || {{}}; }}
    return w;
  }}

  // The video element uses object-fit:contain, so a source whose aspect differs
  // from the 16:9 player box is LETTERBOXED: the picture occupies only the
  // middle of the element and the rest is black bar. A caption sized to the
  // ELEMENT therefore runs out over the bars and gets clipped by the frame.
  // Fit the caption box to the actual PICTURE area instead.
  function pictureBox(w) {{
    var vid = w.querySelector('video');
    if (!vid) return null;
    var vw = vid.videoWidth, vh = vid.videoHeight;
    var ew = vid.clientWidth, eh = vid.clientHeight;
    if (!vw || !vh || !ew || !eh) return null;
    var scale = Math.min(ew / vw, eh / vh);
    var pw = vw * scale, ph = vh * scale;
    return {{ side: Math.max(0, (ew - pw) / 2), top: Math.max(0, (eh - ph) / 2),
             w: pw, h: ph }};
  }}

  function fitCaptionBox(w) {{
    var cap = w.querySelector('#cap');
    if (!cap) return;
    var pb = pictureBox(w);
    if (!pb) return;
    var st = w.__st || {{}};
    var pos = st.position || 'bottom';
    cap.style.left = pb.side + 'px';
    cap.style.right = pb.side + 'px';
    if (pos === 'bottom') {{
      cap.style.top = 'auto';
      cap.style.bottom = (pb.top + pb.h * 0.07) + 'px';
      cap.style.transform = 'none';
    }} else if (pos === 'middle') {{
      cap.style.top = (pb.top + pb.h / 2) + 'px';
      cap.style.bottom = 'auto';
      cap.style.transform = 'translateY(-50%)';
    }} else {{
      cap.style.top = (pb.top + pb.h * 0.05) + 'px';
      cap.style.bottom = 'auto';
      cap.style.transform = 'none';
    }}
    return pb;
  }}

  function applyStyleTo(w) {{
    if (!w) return;
    var cap = w.querySelector('#cap');
    var elAr = w.querySelector('#ar');
    if (!cap) return;
    parseWrap(w);
    var st = w.__st;
    var pb = fitCaptionBox(w);
    // Scale with the PICTURE width (not the element width) so the text is
    // sized against the part of the frame the viewer actually sees, and
    // clamp: on a 360px phone the raw scale (48 * 360/1280 = 13.5px) is
    // unreadable for Arabic, so floor at 17px.
    var picW = (pb && pb.w) || (cap.parentElement.clientWidth || 800);
    var fs = Math.max(17, (st.font_size || 48) * (picW / 1280));
    cap.style.fontSize = fs + 'px';
    cap.style.fontFamily = "'" + (st.font_family || 'Segoe UI') + "', sans-serif";
    cap.style.fontWeight = st.bold ? 700 : 400;
    cap.style.color = st.font_color || '#FFFFFF';
    var o = st.outline_size || 0;
    var shadows = [];
    for (var dx = -1; dx <= 1; dx++) for (var dy = -1; dy <= 1; dy++) {{
      if (!dx && !dy) continue;
      shadows.push((o*dx)+'px '+(o*dy)+'px 0 '+(st.outline_color || '#000000'));
    }}
    if (st.shadow_size > 0) shadows.push('0 0 '+(st.shadow_size*5)+'px rgba(0,0,0,0.85)');
    cap.style.textShadow = shadows.join(',');
    // position is handled by fitCaptionBox() (it is picture-relative)
    cap.style.background = st.box ? 'rgba(0,0,0,0.62)' : 'none';
    cap.style.padding = st.box ? '0.15em 0.6em' : '0';
    cap.style.borderRadius = st.box ? '0.35em' : '0';
    var elEn = w.querySelector('#en');
    if (elEn) elEn.style.fontSize = '0.72em';
  }}

  function upd(w, t) {{
    if (!w) return;
    parseWrap(w);
    var elAr = w.querySelector('#ar');
    var elEn = w.querySelector('#en');
    if (elAr) elAr.textContent = '';
    if (elEn) elEn.textContent = '';
    var lang = w.dataset.lang || 'ar';
    var segs = w.__segs || [];
    for (var i = 0; i < segs.length; i++) {{
      var s = segs[i];
      if (t >= s.start && t < s.end) {{
        if (lang !== 'en' && s.text_ar && elAr) elAr.textContent = s.text_ar;
        if (lang !== 'ar' && s.text_en && elEn) elEn.textContent = s.text_en;
        break;
      }}
    }}
  }}

  function curWrap() {{
    var w = document.getElementById('capwrap');
    return w || null;
  }}

  // The fast upload sends only the soundtrack, so the player would sit on a
  // black frame until the full video transfer lands. The browser already
  // holds the picked file (auto_proxy keeps it on window.__qcapSource), so
  // play THAT copy via a Blob URL: the picture appears with the captions and
  // the background upload keeps running for the burn-in render.
  function localBlob() {{
    var f = window.__qcapSource;
    if (!f) return null;
    try {{
      if (window.__qcapBlobSrc && window.__qcapBlobFor === f) return window.__qcapBlobSrc;
      if (window.__qcapBlobSrc) {{ try {{ URL.revokeObjectURL(window.__qcapBlobSrc); }} catch (e) {{}} }}
      window.__qcapBlobSrc = URL.createObjectURL(f);
      window.__qcapBlobFor = f;
      return window.__qcapBlobSrc;
    }} catch (e) {{ return null; }}
  }}

  function useLocalVideo() {{
    var w = curWrap();
    if (!w) return;
    var v = w.querySelector('video');
    if (!v) return;
    var b = localBlob();
    if (!b || v.getAttribute('src') === b) return;
    var t = v.currentTime || 0;
    var playing = !v.paused && t > 0;
    v.src = b;
    if (t > 0) {{
      v.addEventListener('loadedmetadata', function once() {{
        v.removeEventListener('loadedmetadata', once);
        try {{ v.currentTime = t; }} catch (e) {{}}
        if (playing) v.play().catch(function () {{}});
      }});
    }}
  }}
  window.__qcap.useLocalVideo = useLocalVideo;

  function syncPos() {{
    try {{
      var w = curWrap();
      if (!w) return;
      var v = w.querySelector('video');
      if (!v || v.readyState < 1) return;
      var t = v.currentTime || 0;
      var d = v.duration || 0;
      if (t <= 0 || (d > 0 && t >= d - 0.2)) return;
      sessionStorage.setItem('qcap_pos_' + (w.dataset.key || 'player'),
        JSON.stringify({{ t: t, playing: !v.paused }}));
    }} catch (e) {{}}
  }}

  function restorePos() {{
    try {{
      var w = curWrap();
      if (!w) return;
      var v = w.querySelector('video');
      if (!v) return;
      var saved = JSON.parse(sessionStorage.getItem('qcap_pos_' + (w.dataset.key || 'player')) || 'null');
      if (saved && saved.t > 0 && saved.t < (v.duration || 0) - 0.5) {{
        v.currentTime = saved.t;
        upd(w, saved.t);
        if (saved.playing) v.play().catch(function(){{}});
      }}
    }} catch (e) {{}}
  }}

  document.addEventListener('loadedmetadata', function (e) {{
    if (e.target && e.target.tagName === 'VIDEO' && e.target.closest('#capwrap')) {{
      applyStyleTo(e.target.closest('#capwrap'));
      restorePos();
    }}
  }}, true);
  // The picture area changes when the player is resized (and the video's
  // intrinsic size is only known after metadata loads), so the caption box
  // is re-fitted on both.
  var fitTimer = null;
  window.addEventListener('resize', function () {{
    if (fitTimer) clearTimeout(fitTimer);
    fitTimer = setTimeout(function () {{
      var w = curWrap();
      if (!w) return;
      fitCaptionBox(w);
      applyStyleTo(w);
    }}, 120);
  }});
  document.addEventListener('seeked', function (e) {{
    if (e.target && e.target.tagName === 'VIDEO' && e.target.closest('#capwrap')) {{
      upd(e.target.closest('#capwrap'), e.target.currentTime);
    }}
  }}, true);
  document.addEventListener('canplay', function (e) {{
    if (e.target && e.target.tagName === 'VIDEO' && e.target.closest('#capwrap')) {{
      var w = e.target.closest('#capwrap');
      applyStyleTo(w);
      upd(w, e.target.currentTime);
    }}
  }}, true);
  document.addEventListener('timeupdate', function (e) {{
    if (e.target && e.target.tagName === 'VIDEO' && e.target.closest('#capwrap')) {{
      var w = e.target.closest('#capwrap');
      upd(w, e.target.currentTime);
      syncPos();
    }}
  }}, true);
  document.addEventListener('ended', function (e) {{
    if (e.target && e.target.tagName === 'VIDEO' && e.target.closest('#capwrap')) {{
      try {{ sessionStorage.removeItem('qcap_pos_' + (e.target.closest('#capwrap').dataset.key || 'player')); }} catch (err) {{}}
    }}
  }}, true);
  // Frame-accurate caption sync: 'timeupdate' only fires ~4x/sec (up to
  // 250ms late blanking a caption when the reciter pauses), so a rAF loop
  // re-evaluates the overlay every frame while playing. timeupdate listeners
  // stay as the fallback (e.g. background tabs throttle rAF).
  function tick() {{
    try {{
      var w = curWrap();
      if (w) {{
        var v = w.querySelector('video');
        if (v && !v.paused && !v.seeking && v.readyState >= 1) upd(w, v.currentTime);
      }}
    }} catch (e) {{}}
    requestAnimationFrame(tick);
  }}
  requestAnimationFrame(tick);
  document.addEventListener('pause', function (e) {{
    if (e.target && e.target.tagName === 'VIDEO' && e.target.closest('#capwrap')) {{
      upd(e.target.closest('#capwrap'), e.target.currentTime);
    }}
  }}, true);
  window.addEventListener('resize', function () {{
    var w = curWrap();
    if (w) applyStyleTo(w);
  }});
  setInterval(function () {{ var w = curWrap(); if (w) syncPos(); }}, 250);
  // Streamlit reruns can re-emit the block with the server (audio) source, so
  // the local picture is re-applied whenever the element shows a different
  // source; the Blob itself is created once per picked file.
  setInterval(useLocalVideo, 250);

  document.addEventListener('click', function (e) {{
    var b = e.target.closest ? e.target.closest('#fsbtn') : null;
    if (!b) return;
    var w = curWrap();
    if (!w) return;
    var v = w.querySelector('video');
    if (!v) return;
    if (document.fullscreenElement) {{
      document.exitFullscreen();
      return;
    }}
    var tryNative = function () {{
      try {{
        if (v.webkitEnterFullscreen) {{ v.webkitEnterFullscreen(); return true; }}
      }} catch (err) {{}}
      return false;
    }};
    if (w.requestFullscreen) w.requestFullscreen().catch(function () {{ tryNative(); }});
    else tryNative();
  }}, true);
}}

// run on mount for the element streamlit just injected (document may already
// have the element; delegation covers subsequent events)
var wrap = document.getElementById('capwrap');
if (wrap) {{
  parseWrap(wrap);
  applyStyleTo(wrap);
  var v0 = wrap.querySelector('video');
  if (v0) {{
    if (!window.__qcapSource && v0.getAttribute('src') && v0.getAttribute('src').startsWith('data:')) {{
      (async function() {{
        try {{
          var blob = await (await fetch(v0.getAttribute('src'))).blob();
          v0.src = URL.createObjectURL(blob);
        }} catch (e) {{}}
      }})();
    }}
    if (window.__qcap && window.__qcap.useLocalVideo) window.__qcap.useLocalVideo();
    v0.addEventListener('loadedmetadata', restorePos);
    upd(wrap, v0.currentTime || 0);
  }}
}}
</script>
"""


def sync_payload(segments: list[dict], words: list[dict]) -> dict | None:
    """Fetch (and cache) the audio-vs-caption audit for the current media.

    Cached on a signature of the caption timing, so a style/language rerun is
    free and a timing edit refetches. The audio half of the analysis is
    cached server-side per media file, so an edit is cheap there too.
    """
    mid = ss.get("media_id")
    if not mid or not segments:
        return None
    sig = (mid, len(segments),
           round(sum(float(s["start"]) for s in segments), 2),
           round(sum(float(s["end"]) for s in segments), 2),
           len(words or []))
    if ss.get("sync_sig") == sig and ss.get("sync_payload"):
        return ss["sync_payload"]
    with st.spinner("Reading the audio (first time for this clip takes a moment)…"):
        try:
            resp = requests.post(
                f"{API_URL}/media/{mid}/sync",
                json={"segments": segments, "words": words or [],
                      "max_points": 20000},
                timeout=1800,
            )
        except Exception as e:
            st.warning(f"Sync inspector unavailable: {e}")
            return None
    if resp.status_code != 200:
        st.warning(f"Sync inspector unavailable: {api_error(resp)}")
        return None
    try:
        pay = resp.json()
    except Exception as e:
        st.warning(f"Sync inspector unavailable: {e}")
        return None
    ss["sync_sig"] = sig
    ss["sync_payload"] = pay
    return pay


def media_bytes():
    """File bytes for a (re-)upload to /media.

    Session bytes are dropped right after the first successful upload (they
    would otherwise pin the whole file in RAM forever — a 400MB clip plus
    Streamlit plus the API child does not fit a 512MB host). Re-reads from
    the uploader widget when needed (engine restart / legacy session)."""
    if ss.get("file_bytes") is not None:
        return ss["file_bytes"]
    up = st.session_state.get("file_uploader")
    if up is None:
        return None
    try:
        return up.getvalue()
    except Exception:
        return None


def drop_media_bytes():
    """Free the session file copy after /media accepted it."""
    ss["file_bytes"] = None


def ensure_media():
    """Resolve a playable preview for the server-side media copy.

    The file is uploaded ONCE (at Generate time); this only resolves the
    URL, falling back to a one-time upload for legacy sessions
    or after an engine restart wiped the server copy."""
    if ss["media_url"] is not None:
        return
    if ss["media_id"] is None:
        return
    if not ss.get("media_uploaded"):
        payload = media_bytes()
        if payload is None:
            st.warning("Video preview unavailable: re-select the file to upload it again.")
            return
        try:
            resp = post_media(
                ss["media_id"], ss["file_meta"]["name"], payload,
                ss["file_meta"]["type"],
            )
            if resp.status_code != 200:
                st.warning(f"Video preview unavailable: {api_error(resp)}")
                return
            ss["media_uploaded"] = True
            drop_media_bytes()
        except requests.RequestException as e:
            st.warning(f"Video preview unavailable: {e}")
            return
    if DEPLOYED:
        url = publish_media(ss["media_id"])
        if url:
            ss["media_url"] = url
        else:
            # Engine restarted and lost the copy — re-upload next run.
            ss["media_uploaded"] = False
            st.warning("Video preview unavailable: engine restarted, press Generate Captions again.")
    else:
        ss["media_url"] = PUBLIC_URL + f"/media/{ss['media_id']}"


def ensure_video_preview():
    """Swap in the real picture once the burn-in video is on the server.

    The fast upload only sends the audio, so the player starts on a black
    frame; the video the user picked is uploaded for the burn-in and its
    file replaces the audio one as soon as it has arrived."""
    if not ss.get("video_uploaded") or not ss.get("video_media_id"):
        return
    if ss.get("preview_video_uri") or ss.get("preview_video_tried"):
        return
    if not DEPLOYED:
        ss["preview_video_uri"] = PUBLIC_URL + f"/media/{ss['video_media_id']}"
    else:
        ss["preview_video_tried"] = True
        url = publish_media(ss["video_media_id"])
        if url:
            ss["preview_video_uri"] = url


def section_header(num: str, title: str, sub: str = "", tag: str = "") -> None:
    tag_html = f'<span class="tag">{tag}</span>' if tag else ""
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    st.markdown(
        f'<div class="sec"><span class="n">{num}</span>'
        f'<div class="txt"><div class="h">{title}</div>{sub_html}</div>'
        f'{tag_html}</div>',
        unsafe_allow_html=True,
    )


def show_detection(result: dict):
    surah = result["surah"]
    conf = float(result["confidence"])
    n_ayahs = max(0, result["end_ayah"] - result["start_ayah"] + 1)
    engine = (result.get("engine") or "").upper()
    chip = f'<span class="engine-chip">{engine}</span>' if engine else ""
    with st.container(border=True, key="card_detected"):
        st.markdown(
            f"""
            <div class="det">
              <div class="det-head">
                <div>
                  <div class="det-sura-ar" dir="rtl">{surah['name_ar']}</div>
                  <div class="det-sura-en">{surah['name_en']} · Surah {surah['id']}</div>
                </div>
                <div class="det-side">{chip}<div class="det-ok">Captions ready</div></div>
              </div>
              <div class="stats">
                <div class="stat"><div class="v">{result['start_ayah']}–{result['end_ayah']}</div><div class="l">{n_ayahs} ayahs</div></div>
                <div class="stat"><div class="v">{len(result['segments'])}</div><div class="l">Cues</div></div>
                <div class="stat"><div class="v">{conf:.0%}</div><div class="l">Confidence</div></div>
              </div>
              <div class="conf-bar"><div class="conf-fill" style="width:{max(0.0, min(100.0, conf * 100)):.0f}%"></div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if result.get("engine_note"):
            st.caption(f"Engine note — {result['engine_note']}")


# ------------------------------------------------- The picture's own transfer
# Rendered once, at a fixed spot outside every card and branch, so it exists
# in EVERY script run at the same element-tree position — including the run
# that blocks inside the align, and the result view that replaces step 1.
# Streamlit drops the state of a widget a run does not render, and moving the
# element between cards re-mounts the uploader and aborts the transfer in
# flight (both were measured: the file was lost and the render step asked for
# the video again). It has to start with the align, not after it: the transfer
# takes ~2 minutes on the free host, longer than the align itself.
video_prefetch()

# ---------------------------------------------------------------- Step 1: upload
if ss["align_result"] is None:
    with st.container(border=True, key="card_upload"):
        section_header(
            "1", "Upload recitation",
            "A video or audio recording — captions are aligned to the Quran text.",
            "step 1 of 3",
        )

        uploaded = st.file_uploader(
            "Choose a video — or an audio file",
            type=["mp4", "webm", "mov", "avi", "mp3", "wav", "m4a", "ogg"],
            label_visibility="collapsed",
            key="file_uploader",
        )
        if _PROXY_JS.strip():
            st.html(f"<script>{_PROXY_JS}</script>", unsafe_allow_javascript=True)

        if uploaded is not None:
            if uploaded.size > MAX_SIZE:
                st.error(
                    f"File too large. Maximum {MAX_SIZE // (1024 * 1024)}MB — "
                    "trim long recordings before uploading on mobile."
                )
                st.stop()
            is_new = (
                ss["file_meta"] is None
                or uploaded.name != ss["file_meta"].get("name")
                or getattr(uploaded, "file_id", None) != ss["file_meta"].get("file_id")
            )
            if is_new:
                prune_static([ss.get("media_id"), ss.get("video_media_id")])
                prune_renders([ss.get("render_file")])
                ss["render_file"] = None
                ss["file_bytes"] = uploaded.getvalue()
                ss["file_meta"] = {"name": uploaded.name, "type": uploaded.type, "file_id": getattr(uploaded, "file_id", None)}
                ss["align_result"] = None
                ss["align_key"] = None
                ss["align_job_id"] = None
                ss["segments"] = None
                ss["sync_live"] = False
                ss["sync_payload"] = None
                ss["sync_sig"] = None
                ss["lang"] = "ar"
                ss["render_job_id"] = None
                ss["render_url"] = None
                ss["render_name"] = None
                ss["media_id"] = uuid.uuid4().hex[:10]
                ss["media_url"] = None
                ss["media_uploaded"] = False
                ss["video_media_id"] = None
                ss["video_uploaded"] = False
                ss["video_meta"] = None
                ss["preview_video_uri"] = None
                ss["preview_video_tried"] = False
            if source_is_video():
                st.caption(
                    f"Video received: **{source_video_name()}** — only its soundtrack "
                    "was sent ahead so captions can start now. The picture plays "
                    "from your device straight away; the full video uploads in "
                    "the background for the burn-in, and the MP4 keeps your video."
                )

        has_file = ss["file_bytes"] is not None or ss.get("media_uploaded")
        if has_file:
            if ss["file_bytes"] is not None:
                _type = (ss.get("file_meta") or {}).get("type") or ""
                if _type.startswith("video/"):
                    st.video(ss["file_bytes"])
                elif not source_is_video():
                    st.audio(ss["file_bytes"])
            else:
                # Bytes were dropped after upload to save RAM; the server copy
                # is reused, so just offer Generate again (e.g. after a failure).
                st.info("This file is already on the caption engine — generate captions, or pick another file.")

            # Before the (blocking) align so its widget state survives the
            # rerun the finished job triggers.
            generate = st.button(
                "Generate captions", type="primary", width="stretch",
                icon=":material/auto_awesome:",
            )
            # An align is a multi-minute job, and ANY widget change (e.g. the
            # burn-in video landing mid-align) restarts this script. So the
            # job lives in session state: `align_key` makes the start
            # idempotent (the engine returns the job it already started) and
            # `align_job_id` resumes polling after a rerun.
            if generate and not ss.get("align_key"):
                if not ss.get("media_id"):
                    ss["media_id"] = uuid.uuid4().hex[:10]
                ss["align_key"] = uuid.uuid4().hex

            if ss.get("align_key"):
                with st.status("Running speech recognition + surah detection…", expanded=False) as status:
                    try:
                        # Upload ONCE: the file goes to /media, then /align
                        # reuses that server-side copy via media_id (no second
                        # upload).
                        if not ss.get("media_uploaded"):
                            status.update(label="Uploading file (once)…", state="running")
                            payload = media_bytes()
                            if payload is None:
                                st.error("Upload failed: file bytes unavailable — re-select the file.")
                                ss["align_key"] = None
                                st.stop()
                            up = post_media(
                                ss["media_id"], ss["file_meta"]["name"], payload,
                                ss["file_meta"]["type"],
                            )
                            if up.status_code != 200:
                                st.error(f"Upload failed: {api_error(up)}")
                                ss["align_key"] = None
                                st.stop()
                            ss["media_uploaded"] = True
                            drop_media_bytes()
                        if not ss.get("align_job_id"):
                            resp = requests.post(
                                f"{API_URL}/align",
                                # Engine is chosen server-side (ASR_BACKEND env)
                                # — no UI override, so that field is omitted.
                                data={"job": "1", "media_id": ss["media_id"],
                                      "key": ss["align_key"]},
                                timeout=120,
                            )
                            if resp.status_code != 200:
                                st.error(f"API error {resp.status_code}: {api_error(resp)}")
                                ss["align_key"] = None
                                st.stop()
                            ss["align_job_id"] = resp.json()["job_id"]
                    except requests.RequestException as e:
                        st.error(f"Could not reach the caption engine: {e}")
                        ss["align_key"] = None
                        st.stop()
                    data = poll_align_job(ss["align_job_id"], status)
                ss["align_job_id"] = None
                ss["align_key"] = None
                if data is None:
                    st.stop()
                ss["align_result"] = data
                ss["segments"] = [dict(s) for s in data["segments"]]
                # A new align means new evidence: drop the old inspector payload.
                ss["sync_live"] = False
                ss["sync_payload"] = None
                ss["sync_sig"] = None
                # The upload card belongs to the pre-align state; rerun so the
                # result view replaces it instead of stacking below it.
                st.rerun()



# ---------------------------------------------------------------- Step 2: preview & style
if ss["align_result"] is not None and ss["segments"]:
    show_detection(ss["align_result"])
    ensure_media()
    ensure_video_preview()

    with st.container(border=True, key="card_preview"):
        section_header(
            "2", "Preview & style",
            "The overlay is a preview — burn it into the video at the export step.",
            "step 2 of 3",
        )
        if source_is_video() and not ss.get("video_uploaded"):
            st.caption(
                "Your video is uploading in the background — the picture already "
                "plays from your device, and rendering unlocks when the upload lands."
            )

        lang = st.segmented_control(
            "Caption language",
            options=["ar", "en", "both"],
            format_func=lambda x: LANGS[x],
            default=ss["lang"] or "ar",
            key="lang_control",
            width="stretch",
        )
        if lang and lang != ss["lang"]:
            ss["lang"] = lang

        chosen = st.pills(
            "Style preset",
            options=list(PRESETS.keys()),
            default=ss["preset"],
            key="preset_pills",
            width="stretch",
        )
        if chosen and chosen != ss["preset"]:
            ss["preset"] = chosen
            ss["styles"] = dict(PRESETS[chosen])

        with st.expander("Fine-tune style", icon=":material/tune:", key="exp_style"):
            # Widget keys include the preset: picking a preset must remount
            # the controls with the preset's values (widget state otherwise
            # outlives a changed default).
            _pk = ss["preset"].replace(" ", "_").lower()
            c1, c2 = st.columns(2)
            with c1:
                ss["styles"]["font_size"] = st.slider(
                    "Font size", 24, 120, ss["styles"]["font_size"], 2,
                    key=f"style_fs_{_pk}",
                )
                ss["styles"]["font_color"] = st.color_picker(
                    "Text color", ss["styles"]["font_color"], key=f"style_fc_{_pk}",
                )
                ss["styles"]["outline_color"] = st.color_picker(
                    "Outline color", ss["styles"]["outline_color"], key=f"style_oc_{_pk}",
                )
                ss["styles"]["bold"] = st.toggle(
                    "Bold", ss["styles"]["bold"], key=f"style_bold_{_pk}",
                )
            with c2:
                ss["styles"]["font_family"] = st.selectbox(
                    "Font family",
                    ["Segoe UI", "Tahoma", "Arial", "Times New Roman", "Calibri"],
                    index=["Segoe UI", "Tahoma", "Arial", "Times New Roman", "Calibri"].index(
                        ss["styles"]["font_family"]
                    ),
                    key=f"style_ff_{_pk}",
                )
                ss["styles"]["outline_size"] = st.slider(
                    "Outline width", 0, 6, ss["styles"]["outline_size"],
                    key=f"style_ow_{_pk}",
                )
                ss["styles"]["shadow_size"] = st.slider(
                    "Shadow", 0, 5, ss["styles"]["shadow_size"],
                    key=f"style_sh_{_pk}",
                )
                ss["styles"]["box"] = st.toggle(
                    "Solid background box", ss["styles"]["box"], key=f"style_box_{_pk}",
                )
                ss["styles"]["position"] = st.segmented_control(
                    "Position", ["bottom", "middle", "top"],
                    default=ss["styles"]["position"],
                    key=f"style_pos_{_pk}",
                ) or ss["styles"]["position"]

    src = ss.get("preview_video_uri") or ss["media_url"]
    if src:
        st.html(
            player_html(
                src, ss["segments"], ss["lang"], ss["styles"],
                player_key=ss["media_id"] or "player",
            ),
            unsafe_allow_javascript=True,
        )

    # ---------------------------------------------------------------- Step 3: export
    with st.container(border=True, key="card_export"):
        section_header(
            "3", "Export",
            "Take the subtitles as files, or burn them into the video.",
            "step 3 of 3",
        )
        srt_content = srt_from_segments(ss["segments"], ss["lang"])

        st.markdown('<div class="subhead">Subtitle files</div>', unsafe_allow_html=True)
        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                "Download SRT",
                data=srt_content.encode("utf-8"),
                file_name=f"captions_{ss['lang']}.srt",
                mime="application/x-subrip",
                width="stretch",
                icon=":material/subtitles:",
            )
        with dl2:
            st.download_button(
                "Download VTT",
                data=vtt_from_srt(srt_content).encode("utf-8"),
                file_name=f"captions_{ss['lang']}.vtt",
                mime="text/vtt",
                width="stretch",
                icon=":material/closed_caption:",
            )
        st.caption("Files follow the caption language selected above.")

        st.markdown('<div class="subhead">Burn into the video</div>', unsafe_allow_html=True)
        _choices = list(RESOLUTIONS.keys())
        if _LOW_CPU:
            _choices = [k for k in _choices if (RESOLUTIONS[k][1] or 0) <= 720]
        res_label = st.selectbox(
            "Output resolution", _choices,
            index=_choices.index("720p") if _LOW_CPU and "720p" in _choices else 0,
        )
        if _LOW_CPU:
            st.caption("This free host renders up to 720p — 1080p exceeds its memory limit.")
        res_w, res_h = RESOLUTIONS[res_label]

        # An audio upload has no picture to burn the captions into, so the
        # matching video is asked for here (once, at render time). A video
        # picked in step 1 arrives as its soundtrack first
        # (frontend/auto_proxy.js) and is auto-filled here in the background —
        # that upload is the picture the MP4 must use, never the proxy.
        _meta = ss.get("file_meta") or {}
        _name = (_meta.get("name") or "").lower()
        _audio_only = (_meta.get("type") or "").startswith("audio/") or _name.endswith(
            (".mp3", ".wav", ".m4a", ".ogg", ".aac", ".flac", ".opus")
        )
        _from_video = source_is_video()
        _vup = bool(ss.get("video_uploaded"))
        if _audio_only:
            if _from_video:
                # The transfer itself is owned by the top-level prefetch
                # uploader (it outlives the step cards); here only its state
                # is shown, plus a manual fallback with its own widget key.
                if _vup:
                    st.caption(f"Video: {source_video_name()}")
                    st.success("Original video uploaded — ready to render.")
                else:
                    st.info(
                        "Sending your video to the caption engine in the background. "
                        "You can keep editing — rendering unlocks as soon as it lands."
                    )
                    _vfile = st.file_uploader(
                        "If the transfer stalled, pick the video file again",
                        type=["mp4", "webm", "mov", "avi", "mkv"],
                        label_visibility="collapsed",
                        key=f"video_uploader_fallback_{ss['media_id']}",
                    )
                    if _vfile is not None:
                        _verr = ensure_render_video(_vfile)
                        if _verr:
                            st.error(_verr)
                        else:
                            st.rerun()
            else:
                st.caption("The video the captions are burned into.")
                # The uploader is only rendered until the video is on the
                # engine: Streamlit keeps every uploaded file in memory
                # (BytesIO) for as long as its widget exists, and a 40MB+
                # video held for the rest of the session is part of what
                # OOM-killed the 512MB host.
                if not _vup:
                    _vfile = st.file_uploader(
                        "Video for the burned-in MP4",
                        type=["mp4", "webm", "mov", "avi", "mkv"],
                        label_visibility="collapsed",
                        # A new clip gets a fresh widget, so the previous
                        # clip's video can never be rendered under the new
                        # captions.
                        key=f"video_uploader_{ss['media_id']}",
                    )
                    if _vfile is not None:
                        _verr = ensure_render_video(_vfile)
                        if _verr:
                            st.error(_verr)
                        else:
                            # Drop the uploader (and with it the in-memory
                            # file) and swap the preview to the real picture.
                            st.rerun()
                    st.caption("Captions are burned into a picture — add the video to use.")
                else:
                    st.success("Original video uploaded — ready to render.")

        # Without the picture there is nothing to burn into, so the render
        # stays locked instead of failing in FFmpeg (it needs a video stream).
        _needs_video = _audio_only and not ss.get("video_uploaded")
        render_clicked = st.button(
            "Render MP4 with captions", type="primary", width="stretch",
            disabled=_needs_video, icon=":material/movie:",
        )
        if render_clicked:
            if _needs_video:
                st.error("The video has not finished uploading yet — the render needs the picture.")
                st.stop()
            # The burn-in uses the ORIGINAL video when one was supplied, else
            # the media the captions were made from.
            _render_media_id = ss.get("video_media_id") or ss["media_id"]
            # Render reuses the server-side copy (uploaded once at Generate
            # time) — no second upload. If the engine restarted since, upload
            # once more first.
            if not ss.get("media_uploaded") and not ss.get("video_media_id"):
                payload = media_bytes()
                if payload is None:
                    st.error("Upload failed: file bytes unavailable — re-select the file.")
                    st.stop()
                try:
                    up = post_media(
                        ss["media_id"], ss["file_meta"]["name"], payload,
                        ss["file_meta"]["type"],
                    )
                    if up.status_code != 200:
                        st.error(f"Upload failed: {api_error(up)}")
                        st.stop()
                    ss["media_uploaded"] = True
                    drop_media_bytes()
                except requests.RequestException as e:
                    st.error(f"Could not upload for render: {e}")
                    st.stop()
            try:
                _render_body = ({"media_id": _render_media_id, "srt": srt_content,
                                  "width": res_w, "height": res_h} if res_w
                                else {"media_id": _render_media_id, "srt": srt_content})
                resp = requests.post(
                    f"{API_URL}/render",
                    data=style_payload(_render_body),
                    timeout=180,
                )
            except requests.RequestException as e:
                st.error(f"Could not start render: {e}")
            else:
                if resp.status_code == 404:
                    ss["media_uploaded"] = False
                    st.error("Engine lost the uploaded copy (restarted) — press Render again to re-upload once.")
                elif resp.status_code != 200:
                    st.error(f"Render failed to start: {api_error(resp)}")
                else:
                    # A previous render from this session is replaced, not kept.
                    prune_renders([ss.get("render_file")])
                    ss["render_file"] = None
                    ss["render_job_id"] = resp.json()["job_id"]
                    ss["render_started"] = time.time()
                    ss["render_url"] = None
                    ss["render_name"] = None

        if ss["render_job_id"]:
            try:
                status_resp = requests.get(f"{API_URL}/render/status/{ss['render_job_id']}", timeout=30)
            except requests.RequestException:
                status_resp = None

            if status_resp is not None and status_resp.status_code == 200:
                job = status_resp.json()
                prog = st.progress(0, text="Starting…")
                _elapsed = int(time.time() - ss.get("render_started", time.time()))
                prog.progress(min(100, int(job["progress"])),
                              text=f"{job['status'].title()} · {job['progress']:.0f}%"
                                   + (f" · {job['encoder']}" if job.get("encoder") else "")
                                   + (f" · {_elapsed // 60} min {_elapsed % 60:02d}s" if _elapsed >= 60 else ""))
                st.caption("Rendering on a shared CPU — a long clip takes several minutes. "
                           "Keep this tab open to download the result.")

                if job["status"] == "error":
                    st.error(f"Render failed: {job['error']}")
                    ss["render_job_id"] = None
                elif job["status"] == "done":
                    _job = ss["render_job_id"]
                    ss["render_name"] = f"quran_captions_{_job}.mp4"
                    url = publish_render(_job)
                    ss["render_job_id"] = None
                    if url:
                        ss["render_url"] = url
                        ss["render_file"] = _job
                    else:
                        st.error("Could not fetch the rendered video.")
                else:
                    # Polling re-executes this whole script, and on the free
                    # tier that rerun competes with FFmpeg for the same 0.1
                    # CPU — a 1.5s poll starved the encoder and a 70s clip
                    # took >12 min.
                    time.sleep(6)
                    st.rerun()

        # The finished render stays available: a download click reruns the
        # script, so the result must outlive the single poll that produced it.
        # It lives in the static dir (streamed from disk with range support),
        # never in this process's memory.
        if ss.get("render_url") and not ss["render_job_id"]:
            st.success("Video rendered.")
            st.video(ss["render_url"])
            st.markdown(
                f'<a class="dl-btn" href="{ss["render_url"]}" '
                f'download="{ss.get("render_name") or "quran_captions.mp4"}">'
                '<span>Download MP4</span></a>',
                unsafe_allow_html=True,
            )

    # ---------------------------------------------------------------- Sync inspector
    _words = (ss.get("align_result") or {}).get("words") or []
    with st.expander("Sync inspector", icon=":material/graphic_eq:", key="exp_inspector"):
        st.caption(
            "Waveform, speech and captions on one timeline. "
            "Wheel = zoom, drag = pan, click = seek."
        )
        # Building the chart reads the whole audio, so it only runs on demand.
        if st.button("Build the sync chart", key="sync_build", icon=":material/analytics:"):
            ss["sync_live"] = True

    # The chart itself lives OUTSIDE the expander: clicking Build reruns the
    # app, and Streamlit re-renders an expander collapsed (its `expanded` flag
    # only applies when the element first mounts), so a chart inside would be
    # hidden the moment it finished building.
    if ss.get("sync_live"):
        _pay = sync_payload(ss["segments"], _words)
        if _pay:
            st.html(
                sync_inspector_html(_pay, ss["segments"]),
                unsafe_allow_javascript=True,
            )

    # ---------------------------------------------------------------- Timing editor
    def on_segment_edit():
        edits = st.session_state.get("seg_editor", {})
        edited = edits.get("edited_rows", {})
        if not edited:
            return
        segs = ss["segments"]
        for row_idx, changes in edited.items():
            seg = segs[int(row_idx)]
            for k, v in changes.items():
                if k in ("start", "end"):
                    seg[k] = max(0.0, float(v))
                else:
                    seg[k] = str(v)
        # keep chronological + non-overlapping after manual edits
        segs.sort(key=lambda s: s["start"])
        for a, b in zip(segs, segs[1:]):
            if b["start"] < a["end"]:
                b["start"] = a["end"]
        # no st.rerun() here: Streamlit reruns automatically after on_change
        # (an explicit rerun causes a double-rerun that can drop edits).

    with st.expander("Fine-tune timings & text", icon=":material/table_chart:", key="exp_timing"):
        st.caption("Edit any cue to fix a misalignment; changes apply to the player and both exports.")
        ayahs = sorted({s["ayah"] for s in ss["segments"]})
        jump = st.selectbox(
            "Jump to ayah",
            options=["All"] + [str(a) for a in ayahs],
            help="Filter the table to one ayah. Switch back to All to edit.",
        )
        full_df = pd.DataFrame(ss["segments"])[["ayah", "start", "end", "text_ar", "text_en"]]
        if jump == "All":
            df, editable, note = full_df, True, None
        else:
            df = full_df[full_df["ayah"] == int(jump)]
            editable, note = False, "Viewing one ayah — switch back to All to edit."
        if note:
            st.caption(note)
        st.data_editor(
            df,
            key="seg_editor",
            num_rows="fixed",
            hide_index=True,
            width="stretch",
            disabled=not editable,
            on_change=on_segment_edit if editable else None,
            column_config={
                "ayah": st.column_config.NumberColumn("Ayah", min_value=1, step=1, width="small"),
                "start": st.column_config.NumberColumn("Start (s)", min_value=0.0, format="%.2f", width="small"),
                "end": st.column_config.NumberColumn("End (s)", min_value=0.0, format="%.2f", width="small"),
                "text_ar": st.column_config.TextColumn("Arabic text", width="medium"),
                "text_en": st.column_config.TextColumn("English text", width="medium"),
            },
        )

    # ---------------------------------------------------------------- Start over
    st.write("")
    with st.container(horizontal=True, horizontal_alignment="center"):
        if st.button(
            "Start over with a different clip",
            icon=":material/restart_alt:",
            type="tertiary",
            key="start_over",
        ):
            prune_static([ss.get("media_id"), ss.get("video_media_id")])
            prune_renders([ss.get("render_file")])
            ss.clear()
            st.rerun()
