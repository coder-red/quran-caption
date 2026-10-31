"""Markup for the sync inspector (pure functions, importable without
Streamlit so the block can be rendered and tested on its own).

The behaviour lives in sync_inspector.js; this module only builds the DOM
shell and inlines the payload. The payload travels as a base64 data
attribute because in the deployed Space the browser cannot reach the API
port — bytes must ride inside the page.
"""
import base64
import json
import os

_APP_DIR = os.path.dirname(os.path.abspath(__file__))


def _read_asset(name: str) -> str:
    try:
        with open(os.path.join(_APP_DIR, name), "r", encoding="utf-8") as f:
            return f.read()
    except OSError:
        return ""


JS = _read_asset("sync_inspector.js")

CSS = """
<style>
#qcap-sync { background:#0b0f14; border:1px solid #22303f; border-radius:12px;
             padding:8px; margin-top:4px; }
#qcap-sync canvas { display:block; width:100%; border-radius:6px;
                    cursor:crosshair; touch-action:none; }
#qcap-sync .qcap-sync-bar { display:flex; flex-wrap:wrap; gap:8px;
                            align-items:center; margin:8px 0 6px;
                            font-size:12px; color:#8fa3b6; }
#qcap-sync .qcap-sync-bar button { background:#1b2836; color:#cfdae6;
                                   border:1px solid #2c3f52; border-radius:6px;
                                   padding:3px 9px; font-size:12px;
                                   cursor:pointer; }
#qcap-sync .qcap-sync-issues { max-height:150px; overflow:auto; font-size:12px; }
#qcap-sync .qcap-sync-issue { padding:3px 6px; border-radius:5px;
                              cursor:pointer; }
#qcap-sync .qcap-sync-issue:hover { background:#182535; }
#qcap-sync .qcap-sync-issue.sev-warn { color:#ff8f92; }
#qcap-sync .qcap-sync-issue.sev-info { color:#ffc247; }
#qcap-sync .qcap-sync-issue.ok { color:#5fd08a; }
</style>
"""


def _b64_json(obj: dict) -> str:
    raw = json.dumps(obj, ensure_ascii=True)
    raw = raw.replace("<", "\\u003c").replace("&", "\\u0026")
    return base64.b64encode(raw.encode()).decode()


def summary_line(pay: dict) -> str:
    m = pay.get("metrics") or {}
    return (
        f'{m.get("n_segments", 0)} captions · '
        f'{m.get("n_speech_runs", 0)} speech runs · '
        f'warning(s) {m.get("n_warn", 0)} · clue(s) {m.get("n_info", 0)} · '
        f'over-breath {m.get("over_silence_s", 0)}s · '
        f'uncaptioned {m.get("empty_speech_s", 0)}s · '
        f'VAD {pay.get("vad_backend", "?")} · '
        f'curve {pay.get("curve_backend", "?")}'
    )


def inspector_html(pay: dict, segments: list[dict]) -> str:
    """Canvas timeline of the evidence behind the caption timings."""
    body = dict(pay)
    body["labels"] = [s.get("text_ar") or s.get("text_en") or ""
                      for s in segments]

    rows = []
    for it in pay.get("issues") or []:
        sev = "warn" if it.get("sev") == "warn" else "info"
        rows.append(
            f'<div class="qcap-sync-issue sev-{sev}" data-t="{it["t0"]}">'
            f'<b>{it["t0"]:.2f}s</b> {it["kind"]} — {it["text"]}</div>')
    if not rows:
        rows.append('<div class="qcap-sync-issue ok">no timing issues found — '
                    'every caption edge sits on the voice</div>')

    return "".join([
        CSS,
        f'<div id="qcap-sync" data-sync="{_b64_json(body)}">',
        '<canvas></canvas>',
        '<div class="qcap-sync-bar">',
        '<button class="qcap-sync-fit">Fit</button>',
        '<button class="qcap-sync-copy">Copy report</button>',
        f'<span>{summary_line(pay)}</span>',
        '</div>',
        '<div class="qcap-sync-issues">', "".join(rows), '</div>',
        '</div>',
        '<script>', JS, '</script>',
    ])
