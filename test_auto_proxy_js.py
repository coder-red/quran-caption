"""The automatic fast upload must hand Streamlit an audio proxy, and later
the burn-in uploader must receive the original video.

This is a runtime bug class a string scan cannot catch. It lifts the shipped
`frontend/auto_proxy.js`, runs it in Node against a stubbed DOM, feeds it a
change event the way the browser would, and checks what actually reached the
two uploaders.
"""
import json
import os
import shutil
import subprocess
import textwrap

import pytest

ROOT = os.path.dirname(os.path.abspath(__file__))

STUB = textwrap.dedent("""
    var __inputs = [];
    var __changeHandler = null;
    var __interval = null;
    var __created = [];
    var __notes = [];
    var __videoBox = null;

    function __el(tag) {
      return {
        tagName: tag, style: {}, className: '', id: '', textContent: '',
        type: '', files: null, lastEvent: null, attrs: {},
        parentNode: null, children: [],
        appendChild: function (c) { this.children.push(c); },
        insertBefore: function (c) { this.children.push(c); },
        setAttribute: function (k, v) { this.attrs[k] = v; },
        getAttribute: function (k) { return this.attrs[k] || null; },
        dispatchEvent: function (ev) { this.lastEvent = ev; },
        querySelector: function () { return null; },
        contains: function () { return false; },
        closest: function () { return null; }
      };
    }

    function __input() {
      var e = __el('input');
      e.type = 'file';
      __inputs.push(e);
      return e;
    }

    var __host = __el('div');
    __host.parentNode = __el('div');

    var document = {
      addEventListener: function (t, f) { if (t === 'change') { __changeHandler = f; } },
      createElement: function (tag) {
        var e = __el(tag);
        if (tag === 'video') {
          e.removeAttribute = function (k) { delete this.src; };
          e.load = function () {};
          Object.defineProperty(e, 'src', {
            set: function () {
              setTimeout(function () {
                e.duration = 480;
                if (e.onloadedmetadata) { e.onloadedmetadata(); }
              }, 0);
            }
          });
        }
        __created.push(e);
        return e;
      },
      getElementById: function (id) {
        for (var i = 0; i < __created.length; i++) {
          if (__created[i].id === id) { return __created[i]; }
        }
        return null;
      },
      querySelector: function (sel) {
        var s = String(sel);
        if (s.indexOf('stFileUploader') >= 0) { return __host; }
        if (s.indexOf('st-key-video_uploader') >= 0) { return __videoBox; }
        return null;
      },
      querySelectorAll: function (sel) {
        if (String(sel).indexOf('stFileUploaderDropzoneInput') >= 0) { return __inputs; }
        return [];
      },
      body: __el('body')
    };

    var window = globalThis;
    var URL = {
      createObjectURL: function () { return 'blob:fake'; },
      revokeObjectURL: function () {}
    };
    var __timers = require('timers');
    var __intervals = [];
    function setInterval(f, ms) { __intervals.push(f); return __intervals.length; }
    function setTimeout(f, ms) { return __timers.setTimeout(f, Math.min(ms || 0, 5)); }
    function File(parts, name, opts) {
      this.name = name; this.type = (opts || {}).type; this.size = 2000000;
      this.__parts = parts;
    }
    function DataTransfer() {
      var self = this;
      this.__list = [];
      this.items = { add: function (f) { self.__list.push(f); } };
      Object.defineProperty(this, 'files', { get: function () { return self.__list; } });
    }
    function Event(type, opts) { this.type = type; this.bubbles = !!(opts && opts.bubbles); }
""")

DRIVE = textwrap.dedent("""
    function assert(c, msg) {
      if (!c) { console.error('FAIL: ' + msg); process.exit(1); }
    }

    var MAIN = __input();
    var FAKE = { name: 'recitation.mp4', type: 'video/mp4', size: 40000000 };
    MAIN.files = [FAKE];

    var proxied = null;
    window.QcapProxy = {
      buildAudioProxy: function (f, o) {
        proxied = { file: f, opts: o };
        return Promise.resolve({
          file: new File([], 'recitation_audio.ogg', { type: 'audio/ogg' }),
          bytes: 2100000, originalBytes: 40000000, ratio: 0.05,
          format: 'ogg_opus', durationSec: 480
        });
      }
    };

    __changeHandler({ target: MAIN, stopPropagation: function () {},
                      stopImmediatePropagation: function () {} });

    setTimeout(function () {
      assert(proxied, 'the video must go through the audio encoder');
      assert(proxied.opts && proxied.opts.format === 'ogg_opus', 'opus format');
      assert(window.__qcapSource === FAKE, 'the original stays in memory');
      assert(MAIN.files && MAIN.files[0] && MAIN.files[0].name === 'recitation.mp4.audio.ogg',
             'the uploader must receive the soundtrack, got ' +
             (MAIN.files && MAIN.files[0] && MAIN.files[0].name));
      assert(MAIN.lastEvent && MAIN.lastEvent.type === 'change',
             'Streamlit must be told the file changed');

      // ---- the burn-in step appears: the video fills it automatically ----
      var BURN = __input();
      __intervals.forEach(function (f) { f(); });
      assert(window.__qcapSourceUsed, 'auto-fill must run once');
      assert(BURN.files && BURN.files[0] === FAKE, 'the burn-in uploader gets the video');
      assert(BURN.lastEvent && BURN.lastEvent.type === 'change', 'burn-in change fired');
      assert(BURN.getAttribute('data-qcap-burnin'), 'burn-in input is marked');

      // ---- a replay of the same event must not re-enter the encoder ----
      var before = proxied;
      __changeHandler({ target: MAIN, stopPropagation: function () {},
                        stopImmediatePropagation: function () {} });
      assert(proxied === before, 'injected change must not be re-proxied');

      console.log(JSON.stringify({
        ok: true, source: window.__qcapSource.name,
        proxy: MAIN.files[0].name, burn: BURN.files[0].name
      }));
    }, 50);
""")

FALLBACK_DRIVE = textwrap.dedent("""
    var MAIN = __input();
    var FAKE = { name: 'broken.mp4', type: 'video/mp4', size: 40000000 };
    MAIN.files = [FAKE];
    window.QcapProxy = {
      buildAudioProxy: function () { return Promise.reject(new Error('no WebCodecs')); }
    };
    __changeHandler({ target: MAIN, stopPropagation: function () {},
                      stopImmediatePropagation: function () {} });
    setTimeout(function () {
      if (!(MAIN.lastEvent && MAIN.lastEvent.type === 'change')) {
        console.error('FAIL: fallback never reached the uploader'); process.exit(1);
      }
      if (MAIN.files[0] !== FAKE) {
        console.error('FAIL: fallback must upload the original'); process.exit(1);
      }
      console.log(JSON.stringify({ ok: true, fellBackTo: MAIN.files[0].name }));
    }, 50);
""")

# In the result view the step-1 uploader is gone, so the burn-in uploader is
# the only file input on the page — the background fill must still find it.
ONLY_BURNIN_FILL_DRIVE = textwrap.dedent("""
    function assert(c, msg) {
      if (!c) { console.error('FAIL: ' + msg); process.exit(1); }
    }

    var MAIN = __input();
    var FAKE = { name: 'recitation.mp4', type: 'video/mp4', size: 40000000 };
    MAIN.files = [FAKE];
    window.QcapProxy = {
      buildAudioProxy: function () {
        return Promise.resolve({
          file: new File([], 'x.ogg', { type: 'audio/ogg' }), bytes: 2000000
        });
      }
    };
    __changeHandler({ target: MAIN, stopPropagation: function () {},
                      stopImmediatePropagation: function () {} });
    setTimeout(function () {
      // The result view mounted: only the burn-in uploader exists now.
      __inputs.length = 0;
      var BURN = __input();
      __videoBox = __el('div');
      __videoBox.querySelector = function () { return BURN; };
      __intervals.forEach(function (f) { f(); });
      assert(window.__qcapSourceUsed, 'auto-fill must run once');
      assert(BURN.files && BURN.files[0] === FAKE,
             'the burn-in uploader must get the video');
      assert(BURN.getAttribute('data-qcap-burnin'), 'burn-in input is marked');
      console.log(JSON.stringify({ ok: true, burn: BURN.files[0].name }));
    }, 50);
""")

# A manual pick in the burn-in uploader is already the picture: proxying it
# would replace the video with an .ogg that uploader rejects.
MANUAL_BURNIN_DRIVE = textwrap.dedent("""
    function assert(c, msg) {
      if (!c) { console.error('FAIL: ' + msg); process.exit(1); }
    }

    var BURN = __input();
    var FAKE = { name: 'recitation.mp4', type: 'video/mp4', size: 40000000 };
    BURN.files = [FAKE];
    __videoBox = __el('div');
    __videoBox.contains = function (el) { return el === BURN; };
    var proxied = false, stopped = false;
    window.QcapProxy = {
      buildAudioProxy: function () {
        proxied = true;
        return Promise.resolve({ file: BURN.files[0] });
      }
    };
    __changeHandler({ target: BURN,
                      stopPropagation: function () { stopped = true; },
                      stopImmediatePropagation: function () { stopped = true; } });
    setTimeout(function () {
      assert(!proxied, 'a manual burn-in pick must not be re-encoded');
      assert(!stopped, 'the event must reach Streamlit untouched');
      assert(!BURN.lastEvent, 'nothing may be re-injected into the burn-in');
      console.log(JSON.stringify({ ok: true }));
    }, 50);
""")

PASSTHROUGH_DRIVE = textwrap.dedent("""
    var MAIN = __input();
    var AUDIO = { name: 'clip.mp3', type: 'audio/mpeg', size: 3000000 };
    MAIN.files = [AUDIO];
    var called = false;
    window.QcapProxy = { buildAudioProxy: function () { called = true; } };
    __changeHandler({ target: MAIN, stopPropagation: function () {},
                      stopImmediatePropagation: function () {} });
    setTimeout(function () {
      if (called) { console.error('FAIL: audio must not be proxied'); process.exit(1); }
      if (MAIN.lastEvent) { console.error('FAIL: audio change was swallowed'); process.exit(1); }
      console.log(JSON.stringify({ ok: true }));
    }, 50);
""")


def _script(drive: str) -> str:
    with open(os.path.join(ROOT, "frontend", "auto_proxy.js"), encoding="utf-8") as f:
        js = f.read()
    return STUB + js + "\n" + drive


def _run(tmp_path, name: str, drive: str) -> dict:
    path = tmp_path / name
    path.write_text(_script(drive), encoding="utf-8")
    out = subprocess.run(["node", str(path)], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, (out.stdout[-2000:], out.stderr[-2000:])
    line = [l for l in out.stdout.strip().splitlines() if l.startswith("{")][-1]
    return json.loads(line)


node_only = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


@node_only
def test_video_is_proxied_and_the_burnin_gets_the_original(tmp_path):
    r = _run(tmp_path, "auto_proxy.js", DRIVE)
    assert r == {"ok": True, "source": "recitation.mp4",
                 "proxy": "recitation.mp4.audio.ogg", "burn": "recitation.mp4"}


@node_only
def test_the_shipped_proxy_name_is_readable_by_the_app(tmp_path):
    """The JS names the proxy; frontend/media_store.py must recognize it.

    If these drift apart the app treats a picked video as an audio upload
    and can render (or refuse) the wrong thing."""
    from frontend import media_store

    name = _run(tmp_path, "auto_proxy_name.js", DRIVE)["proxy"]
    assert media_store.is_audio_proxy(name)
    assert media_store.proxy_source_name(name) == "recitation.mp4"


def test_only_video_soundtracks_count_as_proxies():
    from frontend import media_store

    for name in ("recitation.mp4.audio.ogg", "clip.MOV.audio.ogg"):
        assert media_store.is_audio_proxy(name), name
    for name in ("clip.mp3", "clip.ogg", "foo_audio.ogg", "recitation.mp4",
                 "notes.audio.ogg"):
        assert not media_store.is_audio_proxy(name), name


@node_only
def test_extraction_failure_falls_back_to_the_original(tmp_path):
    r = _run(tmp_path, "auto_proxy_fallback.js", FALLBACK_DRIVE)
    assert r["fellBackTo"] == "broken.mp4"


@node_only
def test_audio_files_are_never_touched(tmp_path):
    assert _run(tmp_path, "auto_proxy_audio.js", PASSTHROUGH_DRIVE)["ok"]


@node_only
def test_burnin_is_filled_when_it_is_the_only_uploader(tmp_path):
    """After aligning, step 1 is gone — the background upload must still run."""
    r = _run(tmp_path, "auto_proxy_only_burnin.js", ONLY_BURNIN_FILL_DRIVE)
    assert r == {"ok": True, "burn": "recitation.mp4"}


@node_only
def test_manual_burnin_pick_reaches_streamlit_untouched(tmp_path):
    assert _run(tmp_path, "auto_proxy_manual_burnin.js",
                MANUAL_BURNIN_DRIVE)["ok"]
