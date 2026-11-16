/* Direct-to-engine upload.
 *
 * Browsers used to hand the whole clip to Streamlit's own file uploader,
 * which reads the entire body into RAM on the caption host and OOM-killed the
 * free 512MB instance mid-upload — "the upload takes forever and crashes".
 *
 * Now the picked file goes straight to the caption engine over the same
 * origin (/engine/media/..., a small reverse proxy forwards only those
 * routes), sliced into 4MB pieces addressed by absolute offset. Streamlit
 * only ever receives a tiny .qcap carrier (JSON) describing the file, so
 * neither process ever holds the clip in memory and the upload is limited
 * only by the user's connection.
 *
 * Soundtrack-first: for media files the browser also extracts a small Opus
 * soundtrack (opus_proxy.js) and uploads THAT first, so captions can start
 * while the picture is still on its way. If anything about extraction is
 * unavailable or fails, the engine aligns on the raw upload instead —
 * ffmpeg decodes it server-side, just later.
 *
 * Fallback: if the /engine route answers 404/fails (plain local dev without
 * the proxy), the picked file goes through Streamlit's uploader exactly as
 * before, so the app keeps working everywhere.
 */
(function () {
  'use strict';
  if (window.__qcapAutoBound) { return; }
  window.__qcapAutoBound = true;

  // Soundtrack extraction is only worth it above this size; below it the
  // decode would cost more than the upload saves.
  var MIN_PROXY_BYTES = 4 * 1024 * 1024;
  // Decoding holds the whole clip as PCM; 40 minutes stereo at 16 kHz is
  // ~300 MB, past which mobile browsers start to die.
  var MAX_PROXY_SECONDS = 2400;
  var CHUNK_BYTES = 4 * 1024 * 1024;
  // Server-side cap is authoritative (Python enforces it too, and the engine
  // answers 413); the app tells us its configured cap via this global.
  var DEFAULT_MAX_BYTES = 512 * 1024 * 1024;

  function fmtSize(n) {
    if (n >= 1048576) { return (n / 1048576).toFixed(1) + ' MB'; }
    return Math.max(1, Math.round(n / 1024)) + ' KB';
  }

  function maxBytes() {
    var v = Number(window.__qcapMaxBytes);
    return (v > 0 ? v : DEFAULT_MAX_BYTES);
  }

  function sleep(ms) { return new Promise(function (r) { setTimeout(r, ms); }); }

  function uploaderInput() {
    var host = document.querySelector('[class*="st-key-file_uploader"]');
    if (!host || !host.querySelector) { return null; }
    return host.querySelector('input[type=file]');
  }

  function note(msg, bad) {
    var el = document.getElementById('qcap-proxy-note');
    if (!el) {
      el = document.createElement('div');
      el.id = 'qcap-proxy-note';
      el.style.cssText = 'font-size:0.82rem;margin:-0.3rem 0 0.6rem;color:#34d399;';
      var host = document.querySelector('[data-testid="stFileUploader"]');
      if (host && host.parentNode) {
        host.parentNode.insertBefore(el, host.nextSibling);
      } else {
        document.body.appendChild(el);
      }
    }
    el.style.color = bad ? '#f87171' : '#34d399';
    el.textContent = msg;
  }

  function isVideo(f) {
    if (/^video\//.test(f.type || '')) { return true; }
    return /\.(mp4|webm|mov|avi|mkv|m4v|ogv)$/i.test(f.name || '');
  }

  function isAudio(f) {
    if (/^audio\//.test(f.type || '')) { return true; }
    return /\.(mp3|wav|m4a|ogg|aac|flac|opus)$/i.test(f.name || '');
  }

  function extOf(name) {
    var m = /\.([A-Za-z0-9]{1,8})$/.exec(name || '');
    return m ? '.' + m[1].toLowerCase() : '';
  }

  function baseOf(name) {
    return String(name || 'clip').replace(/\.[^.]+$/, '');
  }

  function uid() {
    if (window.crypto && crypto.randomUUID) { return crypto.randomUUID().replace(/-/g, ''); }
    return Date.now().toString(36) + Math.random().toString(36).slice(2, 10);
  }

  // Stream a file to the engine in CHUNK_BYTES slices addressed by absolute
  // offset. Each slice is retried a few times; a dropped connection just
  // re-sends that slice at the same offset, never corrupts the file.
  async function streamToEngine(file, mediaId, ext, onProgress) {
    var total = file.size;
    var offset = 0;
    var restarts = 0;
    while (offset < total) {
      var end = Math.min(total, offset + CHUNK_BYTES);
      var slice = file.slice(offset, end);
      var attempt = 0;
      for (;;) {
        try {
          var resp = await fetch('/engine/media/' + mediaId + '/upload?offset=' + offset
              + '&total=' + total + '&ext=' + encodeURIComponent(ext), {
            method: 'POST',
            headers: { 'Content-Type': 'application/octet-stream' },
            body: slice,
          });
          if (resp.ok) { break; }
          if (resp.status === 413) { throw new Error('file exceeds the server limit'); }
          // 409 means the engine's copy is behind or ahead of us;
          // restart the whole transfer from zero.
          if (resp.status === 409) {
            restarts += 1;
            if (restarts > 2) { throw new Error('the server rejected the upload'); }
            offset = -1;
            break;
          }
          attempt += 1;
          if (attempt >= 4) { throw new Error('upload failed (HTTP ' + resp.status + ')'); }
        } catch (e) {
          attempt += 1;
          if (attempt >= 4) { throw e; }
        }
        await sleep(1000 * attempt);
      }
      if (offset === -1) { offset = 0; continue; }
      offset = end;
      if (onProgress) { onProgress(offset, total); }
    }
  }

  function engineAvailable() {
    // The engine answers with JSON; a host without the embedded routes
    // answers with the app's HTML page (HTTP 200 as well), which must NOT
    // pass here — otherwise uploads would 404 instead of falling back to
    // the legacy path. A fresh instance spawns the engine lazily, so the
    // first pings may fail while it boots; allow a short retry window.
    var tries = Number(window.__qcapEnginePingRetries != null
                       ? window.__qcapEnginePingRetries : 4) + 1;
    function ping() {
      return fetch('/engine/media/ping', { method: 'GET' })
        .then(function (r) { return r.ok ? r.json() : null; })
        .then(function (d) { return !!(d && d.ok === true); })
        .catch(function () { return false; });
    }
    function attempt(left) {
      return ping().then(function (ok) {
        if (ok || left <= 1) { return ok; }
        return sleep(2500).then(function () { return attempt(left - 1); });
      });
    }
    return attempt(tries);
  }

  function hand(input, file) {
    var dt = new DataTransfer();
    dt.items.add(file);
    input.files = dt.files;
    try { window.__qcapInjecting = true; } catch (e) {}
    try {
      input.dispatchEvent(new Event('change', { bubbles: true }));
    } finally {
      setTimeout(function () { window.__qcapInjecting = false; }, 0);
    }
  }

  function makeCarrier(doc) {
    var bytes = new TextEncoder().encode(JSON.stringify(doc));
    return new File([bytes], 'caption_carrier.qcap', { type: 'application/qcap' });
  }

  function probeDuration(file) {
    return new Promise(function (resolve) {
      var url = URL.createObjectURL(file);
      var v = document.createElement('video');
      var done = false;
      function finish(sec) {
        if (done) { return; }
        done = true;
        try {
          v.removeAttribute('src');
          v.load();
          URL.revokeObjectURL(url);
        } catch (e) {}
        resolve(sec);
      }
      v.preload = 'metadata';
      v.onloadedmetadata = function () {
        finish(isFinite(v.duration) ? v.duration : 0);
      };
      v.onerror = function () { finish(0); };
      setTimeout(function () { finish(0); }, 8000);
      v.src = url;
    });
  }

  var pickSeq = 0;

  // Capture phase: Streamlit's own change handler sits on the React root, so
  // stopping propagation here keeps the raw clip out of Streamlit entirely.
  document.addEventListener('change', function (ev) {
    var input = ev.target;
    if (!input || input.type !== 'file' || window.__qcapInjecting) { return; }
    var uploader = uploaderInput();
    if (!uploader || input !== uploader) { return; }
    var f = input.files && input.files[0];
    if (!f) { return; }
    if (/\.qcap$/i.test(f.name || '')) { return; }

    ev.stopPropagation();
    ev.stopImmediatePropagation();

    var seq = ++pickSeq;
    var isVid = isVideo(f);
    window.__qcapSource = f;
    window.__qcapSourceIsVideo = isVid;

    if (f.size > maxBytes()) {
      window.__qcapSource = null;
      note('This file is too large — the limit is ' + fmtSize(maxBytes()) + '.', true);
      hand(input, makeCarrier({
        media_id: uid(), name: f.name || 'clip', type: f.type || '',
        size: f.size, is_video: isVid, too_large: true,
      }));
      return;
    }

    note('Sending your file to the caption engine …');

    engineAvailable().then(function (ok) {
      if (!ok) {
        // Plain local dev (no proxy in front): the only path that works is
        // Streamlit's own uploader, exactly as before.
        note('Direct upload unavailable here — sending the file through the app.', true);
        window.__qcapSource = f;
        hand(input, f);
        return;
      }

      var rawId = uid();
      var rawExt = extOf(f.name) || (isVid ? '.mp4' : '.mp3');
      var rawStarted = false;
      var carrierHanded = false;

      // The raw bytes are needed for the burn-in render, not for captions.
      // They go after the soundtrack so the soundtrack gets the whole pipe:
      // captions start seconds sooner. The exception is when there is no
      // soundtrack path at all — then the raw upload goes immediately.
      function startRaw() {
        if (rawStarted) { return; }
        rawStarted = true;
        streamToEngine(f, rawId, rawExt, function (o, t) {
          if (seq !== pickSeq) { return; }
          if (carrierHanded && o >= t) {
            note('Video uploaded — ready to render.');
            return;
          }
          note('The picture is still uploading … ' + Math.floor((o * 100) / t) + '%');
        }).catch(function () {
          if (seq === pickSeq) {
            note('The upload stopped — please pick the file again.', true);
          }
        });
      }

      function deliverCarrier(alignId) {
        if (seq !== pickSeq || carrierHanded) { return; }
        carrierHanded = true;
        var carrier = makeCarrier({
          name: f.name || 'clip', type: f.type || 'application/octet-stream',
          size: f.size, is_video: isVid, media_id: alignId,
          video_media_id: isVid ? rawId : null,
        });
        hand(input, carrier);
      }

      var extractable = (isVid || isAudio(f))
        && f.size >= MIN_PROXY_BYTES
        && window.QcapProxy;

      if (!extractable) {
        // No soundtrack detour: raw upload starts now and carries the captions.
        startRaw();
        deliverCarrier(rawId);
        return;
      }

      probeDuration(f).then(function (sec) {
        if (sec > MAX_PROXY_SECONDS) { throw new Error('clip too long to extract'); }
        return window.QcapProxy.buildAudioProxy(f, { bitrate: 32000, format: 'ogg_opus' });
      }).then(function (res) {
        var aid = uid();
        var pf = new File([res.file], baseOf(f.name) + '.audio.ogg', { type: 'audio/ogg' });
        return streamToEngine(pf, aid, '.ogg', function (o, t) {
          if (seq === pickSeq) {
            note('Sending the soundtrack … ' + Math.floor((o * 100) / t) + '%');
          }
        }).then(function () { return aid; });
      }).then(function (aid) {
        if (seq !== pickSeq) { return; }
        if (!aid) {
          // Extraction produced nothing: captions must use the raw file.
          note('Could not extract the audio — captions will use the file itself.', true);
          startRaw();
          deliverCarrier(rawId);
          return;
        }
        note('Soundtrack sent — captions can start. The rest uploads now.');
        deliverCarrier(aid);
        startRaw();
      }).catch(function () {
        if (seq !== pickSeq) { return; }
        note('Could not extract the audio — captions will use the file itself.', true);
        startRaw();
        deliverCarrier(rawId);
      });
    });
  }, true);
})();
