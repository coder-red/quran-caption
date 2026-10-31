/* Sync inspector — draws the audio evidence the captions were timed against.
 *
 * One timeline, five layers, all on the same x-axis:
 *   ruler    issue markers (red = defect, amber = clue)
 *   captions the caption blocks the viewer actually sees
 *   words    provider word stamps (start/end ticks)
 *   speech   Silero speech runs (green) + the raw probability curve,
 *            measured breaths >= 0.4s (amber)
 *   wave     amplitude envelope
 *
 * Bound once at document level (Streamlit re-renders st.html WITHOUT
 * re-running its embedded script), and re-read from the element's data-sync
 * attribute whenever it changes — same pattern as the caption player.
 *
 * NOTE: never write an angle bracket followed by a letter, slash or bang in
 * this file, comments included. Streamlit sanitizes st.html with DOMPurify,
 * whose mXSS guard drops the entire script element when its text matches that
 * pattern (SAFE_FOR_XML), so a comment that quotes an HTML tag name in angle
 * brackets silently disables all of this code.
 */
(function () {
  'use strict';
  if (window.__qcapSync) { return; }
  window.__qcapSync = true;

  var LANE = { ruler: 0, cap: 12, capH: 30, word: 46, speech: 58,
               probBase: 80, run: 80, wave: 96, waveMid: 123, waveHalf: 25,
               axis: 154 };
  var H = 172;
  var MIN_WIN = 2.0;
  var COL = {
    bg: '#0b0f14', grid: '#1d2937', text: '#cfdae6', dim: '#6b7d8f',
    wave: '#3ec8d2', waveDim: '#1b6a72',
    speech: 'rgba(52, 200, 120, 0.16)', prob: 'rgba(110, 235, 175, 0.30)',
    probLine: 'rgba(150, 245, 200, 0.85)',
    silence: 'rgba(255, 176, 32, 0.13)',
    cap: 'rgba(70, 130, 230, 0.28)', capEdge: '#6ea8ff',
    capCur: 'rgba(150, 205, 255, 0.55)',
    word: 'rgba(220, 235, 255, 0.40)',
    play: '#ff5a5f', warn: '#ff5a5f', info: '#ffc247'
  };

  function fmt(t) {
    t = Math.max(0, t || 0);
    var m = Math.floor(t / 60);
    var s = t - m * 60;
    return m + ':' + (s < 10 ? '0' : '') + s.toFixed(1);
  }

  function decode(el) {
    if (el.__syncKey === el.dataset.sync) { return el.__sync; }
    try {
      el.__sync = JSON.parse(atob(el.dataset.sync || ''));
    } catch (e) {
      el.__sync = null;
    }
    el.__syncKey = el.dataset.sync;
    return el.__sync;
  }

  function player() {
    var w = document.getElementById('capwrap');
    return w ? w.querySelector('video') : null;
  }

  function playhead() {
    var fake = document.getElementById('qcap-sync-fake');
    if (fake && fake.offsetParent !== null) { return parseFloat(fake.value) || 0; }
    var v = player();
    return v ? (v.currentTime || 0) : 0;
  }

  function seek(t) {
    var v = player();
    if (v) {
      v.currentTime = Math.max(0, t);
      return;
    }
    var fake = document.getElementById('qcap-sync-fake');
    if (fake) { fake.value = String(t); }
  }

  function state(el) {
    if (!el.__st2) {
      el.__st2 = { t0: 0, t1: 0, base: document.createElement('canvas'),
                   key: '', cssW: 0 };
    }
    return el.__st2;
  }

  function layout(el) {
    var pay = decode(el);
    var st = state(el);
    var w = el.clientWidth || 800;
    var dpr = window.devicePixelRatio || 1;
    var cv = el.querySelector('canvas');
    if (!cv) { return null; }
    if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(H * dpr)) {
      cv.width = Math.round(w * dpr);
      cv.height = Math.round(H * dpr);
      cv.style.width = w + 'px';
      cv.style.height = H + 'px';
      st.key = '';
    }
    if (pay && (!st.t1 || st.t1 <= st.t0)) { st.t0 = 0; st.t1 = pay.duration || 1; }
    // data-win="t0,t1" opens the strip on a specific window (used by the
    // app to open at the playhead and by the harness to render a zoom).
    if (pay && el.dataset.win && el.__winKey !== el.dataset.win) {
      var parts = el.dataset.win.split(',');
      var a = parseFloat(parts[0]), b = parseFloat(parts[1]);
      if (isFinite(a) && isFinite(b) && b > a) { st.t0 = a; st.t1 = b; }
      el.__winKey = el.dataset.win;
      st.key = '';
    }
    return { cv: cv, ctx: cv.getContext('2d'), dpr: dpr, w: w, pay: pay, st: st };
  }

  function xOf(t, st, w, dur) {
    return (t - st.t0) / Math.max(1e-6, st.t1 - st.t0) * w;
  }

  function drawBase(L) {
    var pay = L.pay, st = L.st, ctx = st.base.getContext('2d');
    var w = L.w, dpr = L.dpr;
    st.base.width = Math.round(w * dpr);
    st.base.height = Math.round(H * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, H);
    ctx.fillStyle = COL.bg;
    ctx.fillRect(0, 0, w, H);
    if (!pay || !pay.duration) { return; }
    var dur = pay.duration;
    var X = function (t) { return xOf(t, st, w, dur); };
    var i, s, e, x0, x1;
    var span = st.t1 - st.t0;

    // One column per screen pixel, max-pooled: correct at every zoom level
    // (a fixed stride leaves the waveform empty when zoomed in).
    var pn = (pay.peaks || []).length;
    var qn = (pay.probs || []).length;
    var colPeak = new Float32Array(w);
    var colProb = new Float32Array(w);
    for (var px = 0; px < w; px++) {
      var a = st.t0 + span * px / w;
      var b = st.t0 + span * (px + 1) / w;
      var j0 = Math.max(0, Math.floor(a / pay.hop));
      var j1 = Math.min(qn - 1, Math.max(j0, Math.ceil(b / pay.hop) - 1));
      var mx = 0, mp = 0;
      for (var j = j0; j <= j1; j++) {
        if (j < pn && pay.peaks[j] > mx) { mx = pay.peaks[j]; }
        if (j < qn && pay.probs[j] > mp) { mp = pay.probs[j]; }
      }
      colPeak[px] = mx;
      colProb[px] = mp;
    }

    // ---- speech lane: measured breaths, speech runs, probability curve
    ctx.fillStyle = COL.silence;
    for (i = 0; i < (pay.silence || []).length; i++) {
      s = pay.silence[i][0]; e = pay.silence[i][1];
      x0 = X(s); x1 = X(e);
      if (x1 < 0 || x0 > w) { continue; }
      ctx.fillRect(x0, LANE.speech, Math.max(0.7, x1 - x0),
                   LANE.probBase - LANE.speech + 10);
    }
    ctx.fillStyle = COL.speech;
    for (i = 0; i < (pay.speech || []).length; i++) {
      s = pay.speech[i][0]; e = pay.speech[i][1];
      x0 = X(s); x1 = X(e);
      if (x1 < 0 || x0 > w) { continue; }
      ctx.fillRect(x0, LANE.run, Math.max(0.7, x1 - x0), 10);
    }
    // probability curve: soft fill + a visible edge
    ctx.beginPath();
    ctx.moveTo(0, LANE.probBase);
    for (px = 0; px < w; px++) {
      ctx.lineTo(px, LANE.probBase - (colProb[px] / 100) * 20);
    }
    ctx.lineTo(w, LANE.probBase);
    ctx.closePath();
    ctx.fillStyle = COL.prob;
    ctx.fill();
    ctx.beginPath();
    for (px = 0; px < w; px++) {
      var cy = LANE.probBase - (colProb[px] / 100) * 20;
      if (px === 0) { ctx.moveTo(px, cy); } else { ctx.lineTo(px, cy); }
    }
    ctx.strokeStyle = COL.probLine;
    ctx.lineWidth = 1;
    ctx.stroke();

    // ---- waveform (max envelope, mirrored)
    var mid = LANE.waveMid;
    ctx.fillStyle = COL.wave;
    for (px = 0; px < w; px++) {
      var h = (colPeak[px] / 127) * LANE.waveHalf;
      if (h < 0.5) { continue; }
      ctx.fillRect(px, mid - h, 1, h * 2);
    }
    ctx.fillStyle = COL.grid;
    ctx.fillRect(0, mid, w, 1);

    // ---- word ticks
    ctx.fillStyle = COL.word;
    for (i = 0; i < (pay.words || []).length; i++) {
      var ws = pay.words[i][0], we = pay.words[i][1];
      if (we < st.t0 || ws > st.t1) { continue; }
      ctx.fillRect(X(ws), LANE.word, 1, 7);
      ctx.fillRect(X(we), LANE.word + 3, 1, 4);
    }

    // ---- caption blocks
    ctx.font = '11px "Segoe UI", system-ui, sans-serif';
    ctx.textBaseline = 'middle';
    for (i = 0; i < (pay.segs || []).length; i++) {
      s = pay.segs[i][0]; e = pay.segs[i][1];
      if (e < st.t0 || s > st.t1) { continue; }
      x0 = X(s); x1 = Math.max(x0 + 1, X(e));
      ctx.fillStyle = COL.cap;
      ctx.fillRect(x0, LANE.cap, x1 - x0, LANE.capH);
      ctx.strokeStyle = COL.capEdge;
      ctx.lineWidth = 1;
      ctx.strokeRect(x0 + 0.5, LANE.cap + 0.5, Math.max(1, x1 - x0 - 1),
                     LANE.capH - 1);
      var label = (pay.labels || [])[i] || '';
      var xr = Math.min(x1, w);
      if (xr - x0 > 42 && label) {
        ctx.save();
        ctx.beginPath();
        ctx.rect(x0 + 2, LANE.cap, Math.max(4, xr - x0 - 4), LANE.capH);
        ctx.clip();
        ctx.fillStyle = COL.text;
        ctx.direction = 'rtl';
        ctx.textAlign = 'right';
        ctx.fillText(label.slice(0, 120), xr - 3, LANE.cap + LANE.capH / 2);
        ctx.restore();
      }
    }

    // ---- axis
    var span = st.t1 - st.t0;
    var steps = [0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600];
    var stepS = steps[steps.length - 1];
    for (i = 0; i < steps.length; i++) {
      if (span / steps[i] <= Math.max(3, w / 80)) { stepS = steps[i]; break; }
    }
    ctx.fillStyle = COL.dim;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    var t0 = Math.ceil(st.t0 / stepS) * stepS;
    for (var t = t0; t <= st.t1; t += stepS) {
      var x = X(t);
      ctx.fillRect(x, LANE.axis - 3, 1, 3);
      var label = fmt(t);
      var half = ctx.measureText(label).width / 2 + 2;
      ctx.fillText(label, Math.min(w - half, Math.max(half, x)), LANE.axis);
    }

    // ---- issue markers
    for (i = 0; i < (pay.issues || []).length; i++) {
      var it = pay.issues[i];
      if (it.t1 < st.t0 || it.t0 > st.t1) { continue; }
      var ix = X(it.t0);
      ctx.fillStyle = it.sev === 'warn' ? COL.warn : COL.info;
      ctx.beginPath();
      ctx.moveTo(ix - 3, LANE.ruler);
      ctx.lineTo(ix + 3, LANE.ruler);
      ctx.lineTo(ix, LANE.ruler + 6);
      ctx.closePath();
      ctx.fill();
    }
    st.key = fingerprint(pay, st.t0, st.t1, w);
  }

  function fingerprint(pay, t0, t1, w) {
    return [t0, t1, w, (pay.segs || []).length, (pay.issues || []).length].join('|');
  }

  function drawFrame(L) {
    var pay = L.pay, st = L.st, ctx = L.ctx, w = L.w, dpr = L.dpr;
    var key = pay ? fingerprint(pay, st.t0, st.t1, w) : 'empty';
    if (key !== st.key) { drawBase(L); }
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, L.cv.width, L.cv.height);
    ctx.drawImage(st.base, 0, 0);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (!pay || !pay.duration) { return; }
    var X = function (t) { return xOf(t, st, w, pay.duration); };
    var t = playhead();

    // current caption block highlighted
    for (var i = 0; i < (pay.segs || []).length; i++) {
      if (t >= pay.segs[i][0] && t < pay.segs[i][1]) {
        ctx.strokeStyle = COL.capCur;
        ctx.lineWidth = 2;
        ctx.strokeRect(X(pay.segs[i][0]) + 1, LANE.cap + 1,
                       Math.max(1, X(pay.segs[i][1]) - X(pay.segs[i][0]) - 2),
                       LANE.capH - 2);
        break;
      }
    }
    var px = X(t);
    if (px >= -1 && px <= w + 1) {
      ctx.fillStyle = COL.play;
      ctx.fillRect(px, LANE.ruler + 4, 1.5, H - LANE.ruler - 12);
      ctx.beginPath();
      ctx.moveTo(px - 4, LANE.ruler + 4);
      ctx.lineTo(px + 4, LANE.ruler + 4);
      ctx.lineTo(px, LANE.ruler + 10);
      ctx.closePath();
      ctx.fill();
    }
  }

  function fit(el) {
    var st = state(el), pay = decode(el);
    st.t0 = 0;
    st.t1 = pay && pay.duration ? pay.duration : 1;
    st.key = '';
  }

  function zoomAt(el, t, factor) {
    var st = state(el), pay = decode(el);
    if (!pay) { return; }
    var span = Math.max(MIN_WIN, Math.min(pay.duration, (st.t1 - st.t0) * factor));
    var frac = (t - st.t0) / Math.max(1e-6, st.t1 - st.t0);
    frac = Math.min(1, Math.max(0, frac));
    st.t0 = t - frac * span;
    st.t1 = st.t0 + span;
    if (st.t0 < 0) { st.t0 = 0; st.t1 = span; }
    if (st.t1 > pay.duration) { st.t1 = pay.duration; st.t0 = Math.max(0, st.t1 - span); }
    st.key = '';
  }

  function pan(el, seconds) {
    var st = state(el), pay = decode(el);
    if (!pay) { return; }
    var span = st.t1 - st.t0;
    st.t0 = Math.min(Math.max(0, st.t0 + seconds), Math.max(0, pay.duration - span));
    st.t1 = st.t0 + span;
    st.key = '';
  }

  function tAt(el, clientX) {
    var st = state(el), rect = el.querySelector('canvas').getBoundingClientRect();
    var frac = (clientX - rect.left) / Math.max(1, rect.width);
    return st.t0 + Math.min(1, Math.max(0, frac)) * (st.t1 - st.t0);
  }

  function attach(el) {
    if (el.__syncBound) { return; }
    el.__syncBound = true;
    var st = state(el);
    var drag = null;

    el.addEventListener('wheel', function (ev) {
      if (!decode(el)) { return; }
      ev.preventDefault();
      zoomAt(el, tAt(el, ev.clientX), ev.deltaY > 0 ? 1.25 : 0.8);
    }, { passive: false });

    el.addEventListener('mousedown', function (ev) {
      if (!decode(el)) { return; }
      drag = { x: ev.clientX, t0: st.t0, t1: st.t1, moved: false };
    });
    window.addEventListener('mousemove', function (ev) {
      if (!drag) { return; }
      var rect = el.querySelector('canvas').getBoundingClientRect();
      var dx = ev.clientX - drag.x;
      if (Math.abs(dx) > 3) { drag.moved = true; }
      if (!drag.moved) { return; }
      var span = drag.t1 - drag.t0;
      var pay = decode(el);
      var shift = -dx / Math.max(1, rect.width) * span;
      st.t0 = Math.min(Math.max(0, drag.t0 + shift),
                       Math.max(0, (pay ? pay.duration : 0) - span));
      st.t1 = st.t0 + span;
      st.key = '';
    });
    window.addEventListener('mouseup', function (ev) {
      if (!drag) { return; }
      var wasDrag = drag.moved;
      drag = null;
      if (!wasDrag) { seek(tAt(el, ev.clientX)); }
    });

    el.addEventListener('dblclick', function () { fit(el); });

    var copy = el.querySelector('.qcap-sync-copy');
    if (copy) {
      copy.addEventListener('click', function () {
        var pay = decode(el), m = pay.metrics || {};
        var lines = ['SYNC REPORT ' + new Date().toISOString()];
        lines.push('duration ' + pay.duration.toFixed(2) + 's  segments ' +
                   m.n_segments + '  speech runs ' + m.n_speech_runs +
                   '  vad ' + pay.vad_backend + '  curve ' + pay.curve_backend);
        lines.push('warnings ' + m.n_warn + ' (over-silence ' +
                   m.over_silence_s + 's, uncaptioned ' + m.empty_speech_s +
                   's)  clues ' + m.n_info);
        (pay.issues || []).forEach(function (i) {
          lines.push('  [' + i.sev + '] ' + i.t0.toFixed(2) + '-' +
                     i.t1.toFixed(2) + '  ' + i.kind + ': ' + i.text);
        });
        var text = lines.join('\n');
        try {
          navigator.clipboard.writeText(text);
          copy.textContent = 'Copied';
          setTimeout(function () { copy.textContent = 'Copy report'; }, 1200);
        } catch (e) {
          window.prompt('Copy sync report', text);
        }
      });
    }

    var fitBtn = el.querySelector('.qcap-sync-fit');
    if (fitBtn) { fitBtn.addEventListener('click', function () { fit(el); }); }
    el.querySelectorAll('.qcap-sync-issue').forEach(function (row) {
      row.addEventListener('click', function () {
        var t = parseFloat(row.dataset.t) || 0;
        var pad = 4;
        st.t1 = Math.min(decode(el).duration, t + pad);
        st.t0 = Math.max(0, t - pad);
        if (st.t1 - st.t0 < MIN_WIN) { st.t1 = st.t0 + MIN_WIN; }
        st.key = '';
        seek(t + 0.05);
      });
    });
  }

  function tick() {
    try {
      var els = document.querySelectorAll('[data-sync]');
      for (var i = 0; i < els.length; i++) {
        var el = els[i];
        if (!el.offsetParent) { continue; }   // collapsed expander
        attach(el);
        var L = layout(el);
        if (L && L.pay) { drawFrame(L); }
      }
    } catch (e) {}
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
})();
