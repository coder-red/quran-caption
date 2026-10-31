"""The Instant-upload button must actually hand a proxy to the uploader.

This is a runtime bug class a string scan cannot catch: the handler declared
`var fmt` (the proxy format string) which shadowed the `fmt()` byte formatter,
so the status line threw AFTER the proxy had been built and the whole fast
path silently fell back to the slow full upload ("Could not build the audio
proxy (fmt is not a function)").

The test lifts the shipped script out of `frontend/app.py`, runs it in Node
against a stubbed DOM, clicks the button, feeds a fake file, and asserts the
uploader received an `_audio.ogg` proxy.
"""
import ast
import json
import os
import shutil
import subprocess
import textwrap

import pytest

ROOT = os.path.dirname(os.path.abspath(__file__))

STUB = textwrap.dedent("""
    var __uploader = null;
    var __clickHandler = null;
    var __created = [];
    var __injected = [];
    var __opts = null;

    function __el(tag) {
      return {
        tagName: tag, style: {}, children: [], className: '', textContent: '',
        files: null,
        appendChild: function (c) { this.children.push(c); },
        remove: function () {},
        addEventListener: function (t, f) { this['on' + t] = f; },
        click: function () {},
        dispatchEvent: function (ev) { this.lastEvent = ev; }
      };
    }

    var document = {
      addEventListener: function (t, f) { if (t === 'click') { __clickHandler = f; } },
      createElement: function (tag) { var e = __el(tag); __created.push(e); return e; },
      querySelector: function (sel) {
        return String(sel).indexOf('stFileUploaderDropzoneInput') >= 0 ? __uploader : null;
      },
      getElementById: function (id) {
        for (var i = 0; i < __created.length; i++) {
          if (__created[i].id === id) { return __created[i]; }
        }
        return null;
      },
      body: { appendChild: function () {} }
    };
    __uploader = __el('input');

    var window = globalThis;
    function File(parts, name, opts) {
      this.name = name; this.type = (opts || {}).type; this.size = 1; this.parts = parts;
    }
    function DataTransfer() {
      this.__list = [];
      this.items = { add: function (f) { __injected.push(f); } };
      this.files = this.__list;
    }
    function Event(type) { this.type = type; }
""")

DRIVE = textwrap.dedent("""
    // ---- drive the shipped handler -------------------------------------
    var assertions = {};
    assertions.proxyApi = typeof window.QcapProxy === 'object'
      && typeof window.QcapProxy.buildAudioProxy === 'function';

    var FAKE = { name: 'recitation.mp4', type: 'video/mp4', size: 40000000 };

    window.QcapProxy = {
      buildAudioProxy: function (f, o) {
        __opts = o;
        return Promise.resolve({
          file: {}, bytes: 2000000, originalBytes: 40000000,
          ratio: 0.05, format: 'ogg_opus', durationSec: 480
        });
      }
    };

    __clickHandler({ target: { closest: function () { return {}; } },
                     preventDefault: function () {} });
    var picker = __created[__created.length - 1];
    assertions.pickerCreated = !!picker && typeof picker.onchange === 'function';
    picker.files = [FAKE];
    picker.onchange();

    setTimeout(function () {
      var statusEl = __created.filter(function (e) {
        return String(e.className).indexOf('qcap-fast-status') === 0;
      })[0];
      assertions.status = statusEl ? statusEl.textContent : null;
      assertions.injectedName = __injected[0] ? __injected[0].name : null;
      assertions.uploaderGotFile = __uploader.files !== null && !!__uploader.lastEvent;
      assertions.sameOriginal = window.__qcapOriginalFile === FAKE;
      assertions.format = __opts && __opts.format;
      console.log(JSON.stringify(assertions));
    }, 100);
""")


def _script_source() -> str:
    with open(os.path.join(ROOT, "frontend", "opus_proxy.js"), encoding="utf-8") as f:
        proxy_js = f.read()
    with open(os.path.join(ROOT, "frontend", "app.py"), encoding="utf-8-sig") as f:
        tree = ast.parse(f.read())
    fast = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "_FAST_UPLOAD_JS" for t in node.targets
        ):
            fast = ast.literal_eval(node.value)
            break
    assert fast, "could not lift _FAST_UPLOAD_JS from frontend/app.py"
    return proxy_js + "\n" + fast + "\n" + DRIVE


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_instant_upload_hands_the_uploader_a_proxy(tmp_path):
    path = tmp_path / "fast_upload.js"
    path.write_text(STUB + _script_source(), encoding="utf-8")
    out = subprocess.run(
        ["node", str(path)], capture_output=True, text=True, timeout=60,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    line = [l for l in out.stdout.strip().splitlines() if l.startswith("{")][-1]
    r = json.loads(line)

    assert r["proxyApi"], "QcapProxy must be defined by the shipped script"
    assert r["pickerCreated"]
    assert "Could not build" not in (r["status"] or ""), r["status"]
    assert "Audio proxy ready" in (r["status"] or ""), r["status"]
    assert r["injectedName"] == "recitation_audio.ogg", r["injectedName"]
    assert r["uploaderGotFile"], "the proxy must reach the Streamlit uploader"
    assert r["sameOriginal"], "the original file must stay reachable for burn-in"
    assert r["format"] == "ogg_opus"
