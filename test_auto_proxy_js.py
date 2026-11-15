"""The browser must hand Streamlit a tiny carrier and stream the real clip
straight to the engine.

This is a runtime bug class a string scan cannot catch. It lifts the shipped
frontend/auto_proxy.js, runs it in Node against a stubbed DOM/fetch, feeds it
a change event the way the browser would, and checks what reached the
uploader versus what went to /engine/media/....
"""
import json
import os
import shutil
import subprocess
import textwrap

import pytest

ROOT = os.path.dirname(os.path.abspath(__file__))

# The stub models: the step-1 uploader (st-key-file_uploader container),
# the fetch calls the script makes (ping + slice uploads), and the file
# objects the DataTransfer path carries.
STUB = textwrap.dedent("""
    var __inputs = [];
    var __changeHandler = null;
    var __created = [];
    var __notes = [];
    var __fetchCalls = [];
    var __fetchBehavior = function (url, opts) {
      return { ok: true, status: 200, json: { ok: true } };
    };
    var __seq = 0;

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

    function __makeFile(name, type, size, parts) {
      var body = parts || new Uint8Array(size);
      return {
        name: name, type: type || 'application/octet-stream', size: size,
        slice: function (a, b) {
          return { name: name, type: this.type, size: b - a, __range: [a, b] };
        },
        text: function () {
          return Promise.resolve(
            typeof body === 'string' ? body : Buffer.from(body).toString('utf8'));
        },
        __parts: body
      };
    }

    var __host = __el('div');
    __host.parentNode = __el('div');
    // The uploader container holds exactly the one step-1 input.
    var __uploaderHost = __el('div');
    __uploaderHost.querySelector = function (sel) {
      if (String(sel).indexOf('input') >= 0) { return __inputs[0] || null; }
      return null;
    };

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
        if (s.indexOf('st-key-file_uploader') >= 0) { return __uploaderHost; }
        if (s.indexOf('stFileUploader') >= 0) { return __host; }
        return null;
      },
      querySelectorAll: function () { return []; },
      body: __el('body')
    };

    var window = globalThis;
    var URL = {
      createObjectURL: function () { return 'blob:fake'; },
      revokeObjectURL: function () {}
    };
    var TextEncoder = require('util').TextEncoder;
    function File(parts, name, opts) {
      var body = (parts && parts[0]) || '';
      this.name = name;
      this.type = (opts || {}).type || 'application/octet-stream';
      if (body && body.byteLength != null) { this.size = body.byteLength; }
      else if (body && body.size != null) { this.size = body.size; }
      else { this.size = Buffer.byteLength(String(body)); }
      this.__body = body;
    }
    File.prototype.slice = function (a, b) {
      return { name: this.name, type: this.type, size: Math.max(0, b - a), __range: [a, b] };
    };
    File.prototype.text = function () {
      var b = this.__body;
      if (b && typeof b === 'object' && b.__range) { return Promise.resolve(''); }
      if (b instanceof Uint8Array) { return Promise.resolve(Buffer.from(b).toString('utf8')); }
      return Promise.resolve(String(b));
    };
    function DataTransfer() {
      var self = this;
      this.__list = [];
      this.items = { add: function (f) { self.__list.push(f); } };
      Object.defineProperty(this, 'files', { get: function () { return self.__list; } });
    }
    function Event(type, opts) { this.type = type; this.bubbles = !!(opts && opts.bubbles); }
    var setTimeout = require('timers').setTimeout;
    var setInterval = require('timers').setInterval;
    // Poll-until pattern for the async upload chain (fetch -> extraction ->
    // carrier handoff); a fixed sleep races it on slow machines.
    function waitFor(pred, cb, tries) {
      tries = tries === undefined ? 200 : tries;
      if (pred() || tries <= 0) { cb(); return; }
      setTimeout(function () { waitFor(pred, cb, tries - 1); }, 15);
    }
    var fetch = function (url, opts) {
      __fetchCalls.push({ url: String(url), opts: opts || {} });
      var r = __fetchBehavior(String(url), opts || {});
      if (r instanceof Error) { return Promise.reject(r); }
      return Promise.resolve({ ok: !!r.ok, status: r.status || 200,
                               json: function () {
                                 if (typeof r.json === 'function') { return r.json(); }
                                 return Promise.resolve(r.json || {});
                               } });
    };
    var crypto = { randomUUID: function () {
      return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function (c) {
        var r = Math.random() * 16 | 0;
        return (c === 'x' ? r : (r & 0x3 | 0x8)).toString(16);
      });
    } };
""")

VIDEO_DRIVE = textwrap.dedent("""
    function assert(c, msg) {
      if (!c) { console.error('FAIL: ' + msg); process.exit(1); }
    }

    var MAIN = __input();
    MAIN.files = [__makeFile('recitation.mp4', 'video/mp4', 40000000)];

    var encodeCalls = 0;
    window.QcapProxy = {
      buildAudioProxy: function (f, o) {
        encodeCalls++;
        assert(o && o.format === 'ogg_opus', 'opus format');
        return Promise.resolve({
          file: new File([new Uint8Array(2000000)], 'recitation_audio.ogg', { type: 'audio/ogg' }),
          bytes: 2000000, originalBytes: 40000000, ratio: 0.05,
          format: 'ogg_opus', durationSec: 480
        });
      }
    };
    __changeHandler({ target: MAIN, stopPropagation: function () {},
                      stopImmediatePropagation: function () {} });

    waitFor(function () {
      return MAIN.files && MAIN.files[0] && MAIN.files[0].name === 'caption_carrier.qcap';
    }, function () {
      var pings = __fetchCalls.filter(function (c) { return /\\/engine\\/media\\/ping/.test(c.url); });
      assert(pings.length === 1, 'exactly one reachability probe');
      var uploads = __fetchCalls.filter(function (c) { return /\\/engine\\/media\\/.+\\/upload/.test(c.url); });
      assert(uploads.length >= 2, 'soundtrack + raw slices must reach the engine, got ' + uploads.length);
      var carrier = MAIN.files && MAIN.files[0];
      assert(carrier && carrier.name === 'caption_carrier.qcap',
             'the uploader must receive the carrier, got ' + (carrier && carrier.name));
      carrier.text().then(function (txt) {
        var doc = JSON.parse(txt);
        assert(doc.name === 'recitation.mp4', 'carrier keeps the picked file name');
        assert(doc.is_video === true, 'carrier marks the pick as a video');
        assert(doc.size === 40000000, 'carrier keeps the size');
        assert(doc.media_id && doc.video_media_id, 'carrier carries both engine ids');
        assert(doc.media_id !== doc.video_media_id, 'soundtrack and raw copies are separate ids');
        // the soundtrack id (align) must be the one whose slices finished FIRST
        var aid = doc.media_id;
        var aidCalls = uploads.filter(function (c) { return c.url.indexOf('/media/' + aid + '/') >= 0; });
        var rawCalls = uploads.filter(function (c) { return c.url.indexOf('/media/' + doc.video_media_id + '/') >= 0; });
        assert(aidCalls.length >= 1, 'the soundtrack was streamed to the engine');
        assert(rawCalls.length >= 1, 'the raw video was streamed to the engine');
        assert(__fetchCalls.indexOf(aidCalls[aidCalls.length - 1]) < __fetchCalls.indexOf(rawCalls[0]),
               'the soundtrack stream finishes before the raw one starts');
        assert(window.__qcapSource && window.__qcapSource.name === 'recitation.mp4',
               'the picked file stays in memory for the Blob preview');
        console.log(JSON.stringify({ ok: true, align: doc.media_id, raw: doc.video_media_id,
                                     slices: uploads.length }));
      }).catch(function (e) { console.error('FAIL: ' + e); process.exit(1); });
    });
""")

EXTRACTION_FAILURE_DRIVE = textwrap.dedent("""
    function assert(c, msg) { if (!c) { console.error('FAIL: ' + msg); process.exit(1); } }
    var MAIN = __input();
    MAIN.files = [__makeFile('broken.mp4', 'video/mp4', 40000000)];
    window.QcapProxy = {
      buildAudioProxy: function () { return Promise.reject(new Error('no WebCodecs')); }
    };
    __changeHandler({ target: MAIN, stopPropagation: function () {},
                      stopImmediatePropagation: function () {} });
    waitFor(function () {
      return MAIN.files && MAIN.files[0] && MAIN.files[0].name === 'caption_carrier.qcap';
    }, function () {
      var carrier = MAIN.files && MAIN.files[0];
      assert(carrier && carrier.name === 'caption_carrier.qcap', 'carrier handed over');
      carrier.text().then(function (txt) {
        var doc = JSON.parse(txt);
        assert(doc.media_id === doc.video_media_id,
               'without extraction the captions align on the raw upload');
        var uploads = __fetchCalls.filter(function (c) { return /\\/upload/.test(c.url); });
        assert(uploads.length >= 1, 'raw bytes were streamed');
        console.log(JSON.stringify({ ok: true, align: doc.media_id }));
      });
    });
""")

ENGINE_DOWN_DRIVE = textwrap.dedent("""
    function assert(c, msg) { if (!c) { console.error('FAIL: ' + msg); process.exit(1); } }
    var MAIN = __input();
    MAIN.files = [__makeFile('recitation.mp4', 'video/mp4', 40000000)];
    __fetchBehavior = function (url) {
      if (/\\/engine\\/media\\/ping/.test(url)) { return new Error('connection refused'); }
      return { ok: true, status: 200 };
    };
    __changeHandler({ target: MAIN, stopPropagation: function () {},
                      stopImmediatePropagation: function () {} });
    waitFor(function () { return __fetchCalls.length >= 1 && MAIN.files[0]; }, function () {
      var handed = MAIN.files && MAIN.files[0];
      assert(handed && handed.name === 'recitation.mp4',
             'with no engine the ORIGINAL file goes through Streamlit');
      var uploads = __fetchCalls.filter(function (c) { return /\\/upload/.test(c.url); });
      assert(uploads.length === 0, 'no bytes may stream when the engine is down');
      console.log(JSON.stringify({ ok: true, handed: handed.name }));
    });
""")

ENGINE_HTML_DRIVE = textwrap.dedent("""
    // A host without the proxy answers /engine/media/ping with the app's
    // HTML page (HTTP 200). It must NOT pass the probe — that would send
    // uploads to a route that does not exist instead of falling back.
    function assert(c, msg) { if (!c) { console.error('FAIL: ' + msg); process.exit(1); } }
    var MAIN = __input();
    MAIN.files = [__makeFile('recitation.mp4', 'video/mp4', 40000000)];
    __fetchBehavior = function (url) {
      if (/\\/engine\\/media\\/ping/.test(url)) {
        return { ok: true, status: 200,
                 json: function () { return Promise.reject(new Error('Unexpected token <')); } };
      }
      return { ok: true, status: 200 };
    };
    __changeHandler({ target: MAIN, stopPropagation: function () {},
                      stopImmediatePropagation: function () {} });
    waitFor(function () { return __fetchCalls.length >= 1 && MAIN.files[0]; }, function () {
      var handed = MAIN.files && MAIN.files[0];
      assert(handed && handed.name === 'recitation.mp4',
             'an HTML 200 from a proxyless host must fall back to Streamlit');
      var uploads = __fetchCalls.filter(function (c) { return /\\/upload/.test(c.url); });
      assert(uploads.length === 0, 'no bytes may stream without the engine route');
      console.log(JSON.stringify({ ok: true, handed: handed.name }));
    });
""")

AUDIO_DRIVE = textwrap.dedent("""
    function assert(c, msg) { if (!c) { console.error('FAIL: ' + msg); process.exit(1); } }
    var MAIN = __input();
    MAIN.files = [__makeFile('lecture.mp3', 'audio/mpeg', 3000000)];
    var proxied = false;
    window.QcapProxy = { buildAudioProxy: function () { proxied = true;
      return Promise.resolve({ file: MAIN.files[0], bytes: 3000000 }); } };
    __changeHandler({ target: MAIN, stopPropagation: function () {},
                      stopImmediatePropagation: function () {} });
    waitFor(function () {
      return MAIN.files && MAIN.files[0] && MAIN.files[0].name === 'caption_carrier.qcap';
    }, function () {
      assert(!proxied, 'small audio skips the soundtrack detour');
      var carrier = MAIN.files && MAIN.files[0];
      assert(carrier && carrier.name === 'caption_carrier.qcap', 'carrier handed over');
      carrier.text().then(function (txt) {
        var doc = JSON.parse(txt);
        assert(doc.is_video === false, 'audio is not a video');
        assert(doc.video_media_id === null, 'audio has no picture for the burn-in');
        assert(doc.media_id, 'the raw stream id is the align id');
        var uploads = __fetchCalls.filter(function (c) { return /\\/upload/.test(c.url); });
        assert(uploads.length === 1, 'one slice for 3MB, got ' + uploads.length);
        console.log(JSON.stringify({ ok: true, align: doc.media_id }));
      });
    });
""")

TOO_LARGE_DRIVE = textwrap.dedent("""
    var MAIN = __input();
    MAIN.files = [__makeFile('huge.mp4', 'video/mp4', 900 * 1024 * 1024)];
    window.__qcapMaxBytes = 100 * 1024 * 1024;
    __changeHandler({ target: MAIN, stopPropagation: function () {},
                      stopImmediatePropagation: function () {} });
    waitFor(function () { return MAIN.files && MAIN.files[0]; }, function () {
      var carrier = MAIN.files && MAIN.files[0];
      var uploads = __fetchCalls.filter(function (c) { return /\\/upload/.test(c.url); });
      carrier.text().then(function (txt) {
        var doc = JSON.parse(txt);
        if (!doc.too_large) { console.error('FAIL: oversized pick must be flagged'); process.exit(1); }
        if (uploads.length !== 0) { console.error('FAIL: nothing may stream past the cap'); process.exit(1); }
        console.log(JSON.stringify({ ok: true }));
      });
    });
""")

RESTART_ON_409_DRIVE = textwrap.dedent("""
    function assert(c, msg) { if (!c) { console.error('FAIL: ' + msg); process.exit(1); } }
    var MAIN = __input();
    MAIN.files = [__makeFile('clip.mp3', 'audio/mpeg', 3000000)];
    var seen409 = 0;
    __fetchBehavior = function (url, opts) {
      if (/\\/engine\\/media\\/ping/.test(url)) { return { ok: true, status: 200, json: { ok: true } }; }
      if (/\\/upload/.test(url) && url.indexOf('offset=0') >= 0 && seen409 < 2) {
        seen409++;
        return { ok: false, status: 409 };
      }
      return { ok: true, status: 200 };
    };
    __changeHandler({ target: MAIN, stopPropagation: function () {},
                      stopImmediatePropagation: function () {} });
    waitFor(function () {
      return seen409 >= 2 && __fetchCalls.filter(function (c) { return /\\/upload/.test(c.url); }).length >= 3;
    }, function () {
      assert(seen409 >= 2, 'the two 409s were answered');
      var offsets = __fetchCalls
        .filter(function (c) { return /\\/upload/.test(c.url); })
        .map(function (c) { return (/offset=(\\d+)/.exec(c.url) || [])[1]; });
      assert(offsets[offsets.length - 1] === '0', 'after 409s the stream restarts at 0');
      console.log(JSON.stringify({ ok: true, calls: offsets.length }));
    });
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
def test_video_streams_to_engine_and_carrier_goes_to_streamlit(tmp_path):
    assert _run(tmp_path, "video.js", VIDEO_DRIVE)["ok"]


@node_only
def test_extraction_failure_aligns_on_the_raw_upload(tmp_path):
    assert _run(tmp_path, "extractfail.js", EXTRACTION_FAILURE_DRIVE)["ok"]


@node_only
def test_without_an_engine_the_original_file_is_handed_over(tmp_path):
    assert _run(tmp_path, "engine_down.js", ENGINE_DOWN_DRIVE)["ok"]


@node_only
def test_a_proxyless_host_degrades_to_streamlit(tmp_path):
    assert _run(tmp_path, "engine_html.js", ENGINE_HTML_DRIVE)["ok"]


@node_only
def test_small_audio_streams_raw_without_extraction(tmp_path):
    assert _run(tmp_path, "audio.js", AUDIO_DRIVE)["ok"]


@node_only
def test_oversized_pick_streams_nothing(tmp_path):
    assert _run(tmp_path, "toobig.js", TOO_LARGE_DRIVE)["ok"]


@node_only
def test_offset_mismatch_restarts_the_stream(tmp_path):
    assert _run(tmp_path, "restart.js", RESTART_ON_409_DRIVE)["ok"]
