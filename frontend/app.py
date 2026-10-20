import base64
import json
import os
import time
import uuid

import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("QCAP_API_URL", "http://localhost:8000")
PUBLIC_URL = os.environ.get("QCAP_PUBLIC_URL", API_URL)
DEPLOYED = bool(os.environ.get("SPACE_ID"))
if DEPLOYED:
    API_URL = "http://127.0.0.1:8000"
    PUBLIC_URL = ""


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
    page_icon="🕌",
    layout="wide",
    initial_sidebar_state="collapsed",
)

_ensure_backend()

st.markdown("""
<style>
:root {
  --gold: #D4AF37;
  --emerald: #10B981;
  --bg: #0B1220;
  --card: #141E33;
  --border: #24314D;
  --text-dim: #8B95AB;
}

.block-container { padding-top: 1.5rem; max-width: 1200px; }

.hero {
  text-align: center;
  padding: 1.2rem 0 0.4rem;
  margin-bottom: 1rem;
}
.hero h1 {
  font-size: 2.6rem;
  font-weight: 800;
  letter-spacing: -0.02em;
  margin: 0;
  background: linear-gradient(120deg, #10B981 0%, #D4AF37 60%, #F5C77E 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
.hero p { color: var(--text-dim); font-size: 1.05rem; margin: 0.35rem 0 0; }

.card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 1.1rem 1.2rem;
  margin: 0.6rem 0;
}
.card h3 { margin: 0 0 0.4rem; font-size: 1.05rem; display: flex; align-items: center; gap: 0.5rem; }
.card .step-badge {
  background: var(--emerald);
  color: #04160F;
  font-weight: 800;
  font-size: 0.75rem;
  border-radius: 999px;
  padding: 0.1rem 0.6rem;
}
.dim { color: var(--text-dim); font-size: 0.85rem; }

.conf-bar {
  height: 8px;
  border-radius: 999px;
  background: #24314D;
  overflow: hidden;
  margin: 0.4rem 0 0.2rem;
}
.conf-fill {
  height: 100%;
  border-radius: 999px;
  background: linear-gradient(90deg, #10B981, #D4AF37);
}

.surah-name-ar {
  font-size: 1.9rem;
  font-weight: 700;
  direction: rtl;
  line-height: 1.5;
  margin: 0.15rem 0;
}
.surah-name-en {
  font-size: 1rem;
  color: var(--text-dim);
  margin-bottom: 0.4rem;
}
.metric-row { display: flex; gap: 2rem; margin: 0.6rem 0; }
.metric { text-align: center; }
.metric .v { font-size: 1.5rem; font-weight: 700; }
.metric .l { font-size: 0.75rem; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.06em; }

div[data-testid="stFileUploaderDropzone"] {
  background: var(--card);
  border: 1.5px dashed #3A4A6E;
  border-radius: 14px;
  padding: 1.6rem;
}
div[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--emerald); }
div[data-testid="stFileUploaderDropzone"] button {
  background: var(--emerald);
  color: #04160F;
  font-weight: 700;
  border-radius: 8px;
}

.stButton > button[kind="primary"] {
  background: var(--emerald);
  color: #04160F;
  font-weight: 700;
  border: none;
  border-radius: 10px;
  padding: 0.55rem 1.2rem;
}
.stButton > button[kind="primary"]:hover { background: #34D399; color: #04160F; }
.stButton > button {
  border-radius: 10px;
  font-weight: 600;
}

div[data-testid="stProgressBar"] > div > div { background: linear-gradient(90deg, #10B981, #D4AF37); }

.stExpander { border: 1px solid var(--border) !important; border-radius: 12px !important; background: var(--card); }

/* ---------- Mobile responsiveness ---------- */
/* Streamlit columns stay side-by-side and squeeze on phones by default;
   wrap them so language/style/download panels stack instead of crushing. */
div[data-testid="stHorizontalBlock"] { flex-wrap: wrap; }
div[data-testid="stHorizontalBlock"] > div[data-testid="column"] { min-width: 240px; flex: 1 1 240px; }
/* Pills / segmented controls: wrap instead of horizontal overflow */
div[data-testid="stPills"], div[data-baseweb="segmented-control"] { flex-wrap: wrap; }
/* Caption player: keep 16:9 but cap height on short screens so controls stay visible */
#capwrap { max-height: 72vh; }
#capwrap video { max-height: 72vh; object-fit: contain; background: #000; }
/* Sticky mini-player: stays pinned while scrolling the ayah table on long surahs */
div[data-testid="stHtml"] { position: sticky; top: 0.5rem; z-index: 50; }
div[data-testid="stHtml"] #capwrap { box-shadow: 0 8px 32px rgba(0,0,0,0.55); }
/* Fullscreen button needs a 44px touch target on mobile */
#fsbtn { min-height: 44px; min-width: 44px; }
/* Data editor: allow horizontal scroll on narrow screens */
div[data-testid="stDataFrame"] { overflow-x: auto; }

/* Fast upload (browser-side audio proxy) */
.qcap-fast { display: flex; flex-wrap: wrap; align-items: center; gap: 0.6rem;
             margin: 0.35rem 0 0.6rem; }
#qcap-fast-btn {
  background: linear-gradient(135deg, #10b981, #059669);
  color: #04110c; border: none; border-radius: 9px; padding: 0.5rem 0.9rem;
  font-weight: 700; font-size: 0.9rem; cursor: pointer;
  box-shadow: 0 2px 10px rgba(16, 185, 129, 0.25);
}
#qcap-fast-btn:hover { filter: brightness(1.08); }
#qcap-fast-btn:disabled { opacity: 0.6; cursor: progress; }
.qcap-fast-hint { font-size: 0.8rem; color: #8b93a7; flex: 1 1 260px; }
.qcap-fast-status { font-size: 0.82rem; color: #34d399; margin: 0.15rem 0 0.4rem; }
.qcap-fast-status.bad { color: #f87171; }

@media (max-width: 768px) {
  .block-container { padding: 0.8rem 0.8rem 2rem; }
  .hero { padding: 0.6rem 0 0.2rem; }
  .hero h1 { font-size: 1.7rem; }
  .hero p { font-size: 0.9rem; }
  .card { padding: 0.8rem 0.85rem; border-radius: 12px; }
  .metric-row { gap: 1rem; flex-wrap: wrap; justify-content: space-around; }
  .metric .v { font-size: 1.15rem; }
  .surah-name-ar { font-size: 1.5rem; }
  div[data-testid="stFileUploaderDropzone"] { padding: 1rem; }
  /* Single-column stacking for editor panels */
  div[data-testid="stHorizontalBlock"] > div[data-testid="column"] { min-width: 100%; flex-basis: 100%; }
  .stButton > button { width: 100%; min-height: 48px; font-size: 1rem; }
  #cap { left: 2%; right: 2%; }
  /* Sticky player sits closer to the viewport top on phones */
  div[data-testid="stHtml"] { top: 0.25rem; }
  #capwrap { border-radius: 10px; }
  /* Sliders / controls get full-width touch-friendly rows */
  div[data-testid="stSlider"] { padding: 0.4rem 0; }
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>📖 Quran Caption</h1>
  <p>Upload a recitation — we detect the surah, sync every ayah to the audio, and let you fine-tune timing, text, language, and style before rendering.</p>
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

LANGS = {
    "ar": "Arabic",
    "en": "English",
    "both": "Arabic + English",
}

ss = st.session_state
ss.setdefault("file_bytes", None)
ss.setdefault("file_meta", None)
ss.setdefault("align_result", None)
ss.setdefault("segments", None)
ss.setdefault("lang", "ar")
ss.setdefault("styles", dict(PRESETS["Classic"]))
ss.setdefault("preset", "Classic")
ss.setdefault("media_id", None)
ss.setdefault("media_url", None)
ss.setdefault("media_uploaded", False)
ss.setdefault("render_job_id", None)
# media_id of the ORIGINAL video, uploaded separately at the burn-in step
# when the align ran on a browser-made audio proxy.
ss.setdefault("video_media_id", None)


def api_error(resp: requests.Response):
    try:
        detail = resp.json()
        if isinstance(detail, dict):
            return detail.get("detail") or detail.get("error") or str(detail)
        return str(detail)
    except Exception:
        return resp.text[:400]


# ------------------------------------------------------- fast upload (proxy)
_APP_DIR = os.path.dirname(os.path.abspath(__file__))


def _read_asset(name: str) -> str:
    try:
        with open(os.path.join(_APP_DIR, name), "r", encoding="utf-8") as f:
            return f.read()
    except OSError:
        return ""


# The browser-side audio proxy encoder. Kept in its own file so it can be
# linted/tested on its own; inlined here because the page has no route to
# serve a local asset reliably on every host.
_PROXY_JS = _read_asset("opus_proxy.js")

# NOTE: a plain string, never an f-string — the JS below is full of braces.
_FAST_UPLOAD_JS = """
(function () {
  if (window.__qcapFastBound) { return; }
  window.__qcapFastBound = true;

  function statusEl() {
    var el = document.getElementById('qcap-fast-status');
    if (!el) {
      el = document.createElement('div');
      el.id = 'qcap-fast-status';
      el.className = 'qcap-fast-status';
      document.body.appendChild(el);
    }
    return el;
  }
  function setStatus(msg, bad) {
    var el = statusEl();
    el.textContent = msg || '';
    el.className = 'qcap-fast-status' + (bad ? ' bad' : '');
  }
  window.__qcapFastStatus = setStatus;

  function fmt(n) {
    if (n > 1048576) { return (n / 1048576).toFixed(1) + ' MB'; }
    return Math.round(n / 1024) + ' KB';
  }

  function injectFile(file) {
    var input = document.querySelector('input[data-testid="stFileUploaderDropzoneInput"]')
             || document.querySelector('input[type=file]');
    if (!input) {
      setStatus('Could not find the upload field — reload the page.', true);
      return;
    }
    try {
      var dt = new DataTransfer();
      dt.items.add(file);
      input.files = dt.files;
      input.dispatchEvent(new Event('change', { bubbles: true }));
    } catch (e) {
      setStatus('Could not hand the file to the uploader: ' + e, true);
    }
  }

  document.addEventListener('click', function (ev) {
    var btn = ev.target && ev.target.closest ? ev.target.closest('#qcap-fast-btn') : null;
    if (!btn) { return; }
    ev.preventDefault();

    if (!window.QcapProxy) {
      setStatus('This browser cannot build the audio proxy — use the normal upload.', true);
      return;
    }
    var picker = document.createElement('input');
    picker.type = 'file';
    picker.accept = 'video/*,audio/*';
    picker.style.display = 'none';
    document.body.appendChild(picker);
    picker.addEventListener('change', function () {
      var f = picker.files && picker.files[0];
      try { picker.remove(); } catch (e) {}
      if (!f) { return; }
      setStatus('Extracting audio…');
      window.QcapProxy.buildAudioProxy(f, { bitrate: 24000 })
        .then(function (out) {
          var base = (f.name || 'clip').replace(/\\.[^.]+$/, '');
          out.file.name = base + '_audio.ogg';
          window.__qcapOriginalFile = f;
          setStatus('Audio proxy ready: ' + fmt(out.bytes) + ' (was ' +
                    fmt(out.originalBytes) + ', ' +
                    Math.round((1 - out.ratio) * 100) + '% smaller). Uploading…');
          injectFile(out.file);
        })
        .catch(function (e) {
          setStatus('Could not build the audio proxy (' + (e && e.message || e) +
                    ') — use the normal upload instead.', true);
        });
    });
    picker.click();
  });
})();
"""

FAST_UPLOAD_HTML = (
    '<div class="qcap-fast">'
    '<button id="qcap-fast-btn" type="button">'
    '&#9889; Instant upload (audio only, ~20x smaller)'
    '</button>'
    '<span class="qcap-fast-hint">Extracts the audio in your browser and uploads a '
    'tiny proxy &mdash; captions in seconds instead of minutes. The burned-in MP4 '
    'still needs the original video (asked for at the end).</span>'
    '</div>'
    '<script>' + _PROXY_JS + _FAST_UPLOAD_JS + '</script>'
)


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

  function applyStyleTo(w) {{
    if (!w) return;
    var cap = w.querySelector('#cap');
    var elAr = w.querySelector('#ar');
    if (!cap) return;
    parseWrap(w);
    var st = w.__st;
    // Scale with player width but clamp: on a 360px phone the raw scale
    // (48 * 360/1280 = 13.5px) is unreadable for Arabic, so floor at 17px.
    var fs = Math.max(17, (st.font_size || 48) * ((cap.parentElement.clientWidth || 800) / 1280));
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
    var pos = st.position || 'bottom';
    cap.style.top = pos === 'middle' ? '50%' : (pos === 'top' ? '5%' : 'auto');
    cap.style.bottom = pos === 'bottom' ? '7%' : 'auto';
    cap.style.transform = pos === 'middle' ? 'translateY(-50%)' : 'none';
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
    if (v0.getAttribute('src') && v0.getAttribute('src').startsWith('data:')) {{
      (async function() {{
        try {{
          var blob = await (await fetch(v0.getAttribute('src'))).blob();
          v0.src = URL.createObjectURL(blob);
        }} catch (e) {{}}
      }})();
    }}
    v0.addEventListener('loadedmetadata', restorePos);
    upd(wrap, v0.currentTime || 0);
  }}
}}
</script>
"""


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
    URL / data-URI, falling back to a one-time upload for legacy sessions
    or after an engine restart wiped the server copy."""
    if ss["media_url"] is not None or ss.get("media_data_uri"):
        return
    if ss["media_id"] is None:
        return
    if not ss.get("media_uploaded"):
        payload = media_bytes()
        if payload is None:
            st.warning("Video preview unavailable: re-select the file to upload it again.")
            return
        try:
            resp = requests.post(
                f"{API_URL}/media/{ss['media_id']}",
                files={"file": (ss["file_meta"]["name"], payload, ss["file_meta"]["type"])},
                timeout=300,
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
        try:
            pv = requests.get(f"{API_URL}/media/{ss['media_id']}/preview", timeout=1200)
            # Inlining the whole preview as a base64 data URI costs ~33% extra
            # bytes, holds that copy in session_state for the life of the
            # session, and re-sends it over the websocket on EVERY rerun
            # (style/lang/segment edits all rerun). On a 512MB host that
            # growth is what tips the service into the OOM killer, and it
            # makes every interaction slow. Only inline while it is cheap.
            cap_mb = float(os.environ.get("QCAP_PREVIEW_INLINE_MB", "8") or 8)
            if pv.status_code == 200 and pv.content:
                if len(pv.content) <= cap_mb * 1024 * 1024:
                    ss["media_data_uri"] = (
                        "data:video/mp4;base64," + base64.b64encode(pv.content).decode()
                    )
                else:
                    ss["media_data_uri"] = None
                    st.warning(
                        f"Preview skipped: this clip's preview is "
                        f"{len(pv.content)/1e6:.0f} MB (over the {cap_mb:.0f} MB "
                        "inline limit) and inlining it would exhaust this host's "
                        "memory. Captions below are still correct — set "
                        "QCAP_PUBLIC_URL to a public API URL to restore the "
                        "video preview."
                    )
            elif pv.status_code == 404:
                # Engine restarted and lost the copy — re-upload next run.
                ss["media_uploaded"] = False
                st.warning("Video preview unavailable: engine restarted, press Generate Captions again.")
            else:
                st.warning("Video preview unavailable on this instance.")
        except requests.RequestException as e:
            st.warning(f"Video preview unavailable: {e}")
    else:
        ss["media_url"] = PUBLIC_URL + f"/media/{ss['media_id']}"


def show_detection(result: dict):
    surah = result["surah"]
    conf = result["confidence"]
    n_ayahs = max(0, result["end_ayah"] - result["start_ayah"] + 1)
    st.markdown(
        f"""
        <div class="card">
          <h3><span class="step-badge">✓</span> Detected Surah</h3>
          <div class="surah-name-ar">{surah['name_ar']}</div>
          <div class="surah-name-en">{surah['name_en']} (Surah {surah['id']})</div>
          <div class="metric-row">
            <div class="metric"><div class="v">{result['start_ayah']}–{result['end_ayah']}</div><div class="l">Ayahs · {n_ayahs} total</div></div>
            <div class="metric"><div class="v">{len(result['segments'])}</div><div class="l">Synced segments</div></div>
            <div class="metric"><div class="v">{conf:.0%}</div><div class="l">Confidence</div></div>
          </div>
          <div class="conf-bar"><div class="conf-fill" style="width:{min(100, conf*100):.0f}%"></div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------- Step 1: upload
st.markdown('<div class="card"><h3><span class="step-badge">1</span> Upload recitation</h3></div>', unsafe_allow_html=True)

# --- Fast upload (browser-side audio proxy) ---------------------------------
# A 8-minute recitation video is ~40 MB and the captioning step only ever
# needs the AUDIO. The browser can extract the audio and encode it to Opus
# (~3 KB/s) itself, so the upload drops from ~40 MB to ~1.5 MB — the align
# then starts in seconds instead of minutes on a slow link. The burned-in MP4
# still needs the original video, so that is offered separately at the end.
st.html(FAST_UPLOAD_HTML, unsafe_allow_javascript=True)

uploaded = st.file_uploader(
    "Choose a video or audio file",
    type=["mp4", "webm", "mov", "avi", "mp3", "wav", "m4a", "ogg"],
    label_visibility="collapsed",
    key="file_uploader",
)
st.caption(f"Big files upload slowly here (free tier, US server) — trim to the passage you want captioned. Max {MAX_SIZE // (1024 * 1024)}MB.")
if uploaded is not None:
    if uploaded.size > MAX_SIZE:
        st.error(f"File too large. Maximum {MAX_SIZE // (1024 * 1024)}MB — trim long recordings before uploading on mobile.")
        st.stop()
    is_new = (
        ss["file_meta"] is None
        or uploaded.name != ss["file_meta"].get("name")
        or getattr(uploaded, "file_id", None) != ss["file_meta"].get("file_id")
    )
    if is_new:
        ss["file_bytes"] = uploaded.getvalue()
        ss["file_meta"] = {"name": uploaded.name, "type": uploaded.type, "file_id": getattr(uploaded, "file_id", None)}
        ss["align_result"] = None
        ss["segments"] = None
        ss["lang"] = "ar"
        ss["render_job_id"] = None
        ss["media_id"] = uuid.uuid4().hex[:10]
        ss["media_url"] = None
        ss["media_data_uri"] = None
        ss["media_uploaded"] = False

if (ss["file_bytes"] is not None or ss.get("media_uploaded")) and ss["align_result"] is None:
    if ss["file_bytes"] is not None:
        st.video(ss["file_bytes"])
    else:
        # Bytes were dropped after upload to save RAM; the server copy is
        # reused, so just offer Generate again (e.g. after a failed run).
        st.info("File is already on the caption engine — press Generate Captions to (re)run.")

    generate = st.button("Generate Captions", type="primary", use_container_width=True)
    if generate:
        if not ss.get("media_id"):
            ss["media_id"] = uuid.uuid4().hex[:10]
        with st.status("Running speech recognition + surah detection…", expanded=True) as status:
            try:
                # Upload ONCE: the file goes to /media, then /align reuses
                # that server-side copy via media_id (no second upload).
                if not ss.get("media_uploaded"):
                    status.update(label="Uploading file (once)…", state="running")
                    payload = media_bytes()
                    if payload is None:
                        st.error("Upload failed: file bytes unavailable — re-select the file.")
                        st.stop()
                    up = requests.post(
                        f"{API_URL}/media/{ss['media_id']}",
                        files={"file": (ss["file_meta"]["name"], payload, ss["file_meta"]["type"])},
                        timeout=300,
                    )
                    if up.status_code != 200:
                        st.error(f"Upload failed: {api_error(up)}")
                        st.stop()
                    ss["media_uploaded"] = True
                    drop_media_bytes()
                resp = requests.post(
                    f"{API_URL}/align",
                    # Engine is chosen server-side (ASR_BACKEND env) — no UI
                    # override, so the form field is omitted entirely.
                    data={"job": "1", "media_id": ss["media_id"]},
                    timeout=120,
                )
            except requests.RequestException as e:
                st.error(f"Could not reach the caption engine: {e}")
                st.stop()
            if resp.status_code != 200:
                st.error(f"API error {resp.status_code}: {api_error(resp)}")
                st.stop()
            jid = resp.json()["job_id"]
            prog = st.progress(0, text="Queued…")
            # A dead engine used to spin here forever (every poll raised
            # RequestException -> continue) behind a frozen bar. Count
            # consecutive misses and bail out with a real message instead.
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
                        last_pct,
                        text=f"Reconnecting… ({misses}/5) · {elapsed}s elapsed",
                    )
                    if misses >= 5:
                        prog.empty()
                        status.update(label="Engine unreachable", state="error")
                        st.error(
                            "Lost contact with the caption engine after "
                            f"{elapsed}s (it may have crashed). Press Generate "
                            "Captions to retry once the engine is back up."
                        )
                        st.stop()
                    continue
                pct = job.get("progress", 0)
                last_pct = min(99, int(pct))
                prog.progress(
                    last_pct,
                    text=f"{job.get('stage', 'working…')} · {pct:.0f}% · "
                         f"{int(time.time() - started)}s elapsed",
                )
                status.update(label=job.get("stage", "working…"))
                if job.get("status") == "error":
                    # /align/status carries no message — fetch it from the
                    # result endpoint (now surfaces the real job error).
                    msg = None
                    try:
                        msg = requests.get(f"{API_URL}/align/result/{jid}", timeout=30).json().get("error")
                    except (requests.RequestException, ValueError):
                        msg = None
                    prog.empty()
                    status.update(label="Failed", state="error")
                    st.error(f"Align failed: {msg or 'unknown error'}")
                    st.stop()
                if job.get("status") == "done":
                    break
            try:
                data = requests.get(f"{API_URL}/align/result/{jid}", timeout=60).json()
            except requests.RequestException as e:
                st.error(f"Could not fetch the result: {e}")
                st.stop()
            if "error" in data:
                st.error(f"Could not identify the surah: {data['error']}")
                st.stop()
            ss["align_result"] = data
            ss["segments"] = [dict(s) for s in data["segments"]]
            prog.progress(100, text="Done · captions ready")
            status.update(label="Surah detected!", state="complete")

    if ss["align_result"] is not None:
        show_detection(ss["align_result"])

# ---------------------------------------------------------------- Step 2: sync & preview editor
if ss["align_result"] is not None and ss["segments"]:
    ensure_media()

    st.markdown('<div class="card"><h3><span class="step-badge">2</span> Sync & preview</h3></div>', unsafe_allow_html=True)
    st.caption("Press play — captions follow the audio. Adjust language, style, and ayah timing below; everything updates live.")
    st.caption("⚠️ This is a **live overlay preview** of your original file — captions are NOT saved into it. Use **“Render MP4 with captions”** below to get a video with captions permanently burned in.")

    # Full-width stacked controls: the old 3-column row ([2,1,2] with an
    # empty spacer) squeezed to unreadable widths on phones. One control
    # per row wraps cleanly on both desktop and mobile.
    lang = st.segmented_control(
        "Caption language",
        options=["ar", "en", "both"],
        format_func=lambda x: LANGS[x],
        default=ss["lang"],
        key="lang_control",
    )
    if lang != ss["lang"]:
        ss["lang"] = lang

    chosen = st.pills(
        "Style preset",
        options=list(PRESETS.keys()),
        default=ss["preset"],
        label_visibility="collapsed",
        key="preset_pills",
    )
    if chosen and chosen != ss["preset"]:
        ss["preset"] = chosen
        ss["styles"] = dict(PRESETS[chosen])

    with st.expander("Customize style", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            ss["styles"]["font_size"] = st.slider("Font size", 24, 120, ss["styles"]["font_size"], 2)
            ss["styles"]["font_color"] = st.color_picker("Text color", ss["styles"]["font_color"])
            ss["styles"]["outline_color"] = st.color_picker("Outline color", ss["styles"]["outline_color"])
            ss["styles"]["bold"] = st.toggle("Bold", ss["styles"]["bold"])
        with c2:
            ss["styles"]["font_family"] = st.selectbox(
                "Font family", ["Segoe UI", "Tahoma", "Arial", "Times New Roman", "Calibri"],
                index=["Segoe UI", "Tahoma", "Arial", "Times New Roman", "Calibri"].index(ss["styles"]["font_family"]),
            )
            ss["styles"]["outline_size"] = st.slider("Outline width", 0, 6, ss["styles"]["outline_size"])
            ss["styles"]["shadow_size"] = st.slider("Shadow", 0, 5, ss["styles"]["shadow_size"])
            ss["styles"]["box"] = st.toggle("Solid background box", ss["styles"]["box"])
            ss["styles"]["position"] = st.segmented_control(
                "Position", ["bottom", "middle", "top"], default=ss["styles"]["position"]
            )

    if ss["media_url"] or ss.get("media_data_uri"):
        src = ss.get("media_data_uri") or ss["media_url"]
        st.html(
            player_html(
                src, ss["segments"], ss["lang"], ss["styles"],
                player_key=ss["media_id"] or "player",
            ),
            unsafe_allow_javascript=True,
        )

    # ---------------------------------------------------------------- Burn-in & download
    st.markdown("#### Burn-in & download — renders the final MP4 once you like the look")
    st.caption("This is the **only** step that produces a video with captions saved inside it (burned into every frame). The preview above is a live overlay only.")
    srt_content = srt_from_segments(ss["segments"], ss["lang"])
    res_label = st.selectbox("Output resolution", list(RESOLUTIONS.keys()), index=0)
    res_w, res_h = RESOLUTIONS[res_label]

    # When the align ran on a browser-made audio proxy there is no video on
    # the server, so the burn-in needs the original file uploaded here.
    _media_name = (ss.get("file_meta") or {}).get("name", "")
    _audio_only = _media_name.lower().endswith((".ogg", ".opus"))
    if _audio_only:
        st.info(
            "The captions were made from an **audio-only** proxy (the fast upload). "
            "To burn them into a video, add the original file here — this is the "
            "one step that needs the full-size upload."
        )
        _vfile = st.file_uploader(
            "Original video (for the burned-in MP4)",
            type=["mp4", "webm", "mov", "avi", "mkv", "m4a", "mp3", "wav"],
            label_visibility="collapsed",
            key="video_uploader",
        )
        if _vfile is not None:
            if ss.get("video_media_id") is None or ss.get("video_meta", {}).get("file_id") != getattr(_vfile, "file_id", None):
                ss["video_media_id"] = uuid.uuid4().hex[:10]
                ss["video_meta"] = {"name": _vfile.name, "file_id": getattr(_vfile, "file_id", None)}
            if not ss.get("video_uploaded"):
                try:
                    up = requests.post(
                        f"{API_URL}/media/{ss['video_media_id']}",
                        files={"file": (_vfile.name, _vfile.getvalue(), _vfile.type)},
                        timeout=1800,
                    )
                    if up.status_code != 200:
                        st.error(f"Video upload failed: {api_error(up)}")
                    else:
                        ss["video_uploaded"] = True
                        st.success("Original video uploaded — ready to render.")
                except requests.RequestException as e:
                    st.error(f"Video upload failed: {e}")
            else:
                st.success("Original video uploaded — ready to render.")

    render_clicked = st.button("Render MP4 with captions", type="primary", use_container_width=True)
    if render_clicked:
        # The burn-in uses the ORIGINAL video when one was supplied, else the
        # media the captions were made from.
        _render_media_id = ss.get("video_media_id") or ss["media_id"]
        if ss.get("video_media_id") and not ss.get("video_uploaded"):
            st.error("The original video has not finished uploading yet.")
            st.stop()
        # Render reuses the server-side copy (uploaded once at Generate
        # time) — no second upload. If the engine restarted since, upload
        # once more first.
        if not ss.get("media_uploaded") and not ss.get("video_media_id"):
            payload = media_bytes()
            if payload is None:
                st.error("Upload failed: file bytes unavailable — re-select the file.")
                st.stop()
            try:
                up = requests.post(
                    f"{API_URL}/media/{ss['media_id']}",
                    files={"file": (ss["file_meta"]["name"], payload, ss["file_meta"]["type"])},
                    timeout=300,
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
                ss["render_job_id"] = resp.json()["job_id"]

    if ss["render_job_id"]:
        try:
            status_resp = requests.get(f"{API_URL}/render/status/{ss['render_job_id']}", timeout=30)
        except requests.RequestException:
            status_resp = None

        if status_resp is not None and status_resp.status_code == 200:
            job = status_resp.json()
            prog = st.progress(0, text="Starting…")
            prog.progress(min(100, int(job["progress"])),
                          text=f"{job['status'].title()} · {job['progress']:.0f}%"
                               + (f" · {job['encoder']}" if job.get("encoder") else ""))
            st.caption("This runs in the background — the page will refresh automatically. Close this tab anytime; the render continues.")

            if job["status"] == "error":
                st.error(f"Render failed: {job['error']}")
                ss["render_job_id"] = None
            elif job["status"] == "done":
                try:
                    r = requests.get(f"{API_URL}/render/result/{ss['render_job_id']}", timeout=1200)
                    if r.status_code == 200:
                        st.success("Video rendered.")
                        st.video(r.content)
                        st.caption("The player above is the **burned-in** result — captions are permanently in this MP4.")
                        st.download_button(
                            "Download MP4", data=r.content,
                            file_name=f"quran_captions_{ss['render_job_id']}.mp4",
                            mime="video/mp4", use_container_width=True,
                        )
                    else:
                        st.error("Could not download the rendered video.")
                except requests.RequestException as e:
                    st.error(f"Could not download the rendered video: {e}")
                ss["render_job_id"] = None
            else:
                time.sleep(1.5)
                st.rerun()

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

    st.markdown("#### Ayah timing & text — edit to fix any misalignment")
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
        editable, note = False, "Viewing one ayah — switch to All to edit."
    if note:
        st.caption(note)
    st.data_editor(
        df,
        key="seg_editor",
        num_rows="fixed",
        hide_index=True,
        disabled=not editable,
        on_change=on_segment_edit if editable else None,
        column_config={
            "ayah": st.column_config.NumberColumn("Ayah", min_value=1, step=1),
            "start": st.column_config.NumberColumn("Start (s)", min_value=0.0, format="%.2f"),
            "end": st.column_config.NumberColumn("End (s)", min_value=0.0, format="%.2f"),
            "text_ar": st.column_config.TextColumn("Arabic text", width="large"),
            "text_en": st.column_config.TextColumn("English text", width="large"),
        },
    )

    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            "Download SRT",
            data=srt_content.encode("utf-8"),
            file_name=f"captions_{ss['lang']}.srt",
            mime="application/x-subrip",
            use_container_width=True,
        )
    with dl2:
        st.download_button(
            "Download VTT",
            data=vtt_from_srt(srt_content).encode("utf-8"),
            file_name=f"captions_{ss['lang']}.vtt",
            mime="text/vtt",
            use_container_width=True,
        )
