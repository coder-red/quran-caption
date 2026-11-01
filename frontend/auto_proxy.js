/* Automatic fast upload for video files.
 *
 * Captioning only needs the AUDIO of a video. When a video is picked, the
 * browser extracts its soundtrack (opus_proxy.js), uploads that small file
 * instead, and keeps the original video in memory: the burn-in step then
 * uploads it for real while the user is editing styles, so the heavy
 * transfer no longer stands between the user and their captions.
 *
 * Falls back to uploading the original file whenever anything is missing
 * (no WebCodecs, clip too long to decode in memory, extraction error).
 */
(function () {
  'use strict';
  if (window.__qcapAutoBound) { return; }
  window.__qcapAutoBound = true;

  var injecting = false;
  // Decoding holds the whole clip as PCM; 40 minutes stereo at 16 kHz is
  // ~300 MB, past which mobile browsers start to die.
  var MAX_PROXY_SECONDS = 2400;
  // Below this the video is no slower to upload than its audio.
  var MIN_PROXY_BYTES = 4 * 1024 * 1024;

  function fmtSize(n) {
    if (n >= 1048576) { return (n / 1048576).toFixed(1) + ' MB'; }
    return Math.max(1, Math.round(n / 1024)) + ' KB';
  }

  function uploaderInputs() {
    return Array.prototype.slice.call(
      document.querySelectorAll('[data-testid="stFileUploaderDropzoneInput"]'));
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

  function hideNoteSoon(ms) {
    setTimeout(function () {
      var el = document.getElementById('qcap-proxy-note');
      if (el && el.parentNode) { el.parentNode.removeChild(el); }
    }, ms || 6000);
  }

  function hand(input, file) {
    var dt = new DataTransfer();
    dt.items.add(file);
    input.files = dt.files;
    injecting = true;
    try {
      input.dispatchEvent(new Event('change', { bubbles: true }));
    } finally {
      setTimeout(function () { injecting = false; }, 0);
    }
    hideNoteSoon();
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
          v.load();  // stop buffering before the blob URL disappears
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

  // Capture phase on the document: Streamlit's own change handler sits on
  // the React root, so stopping propagation here keeps the raw video from
  // starting a full-size upload. The proxy goes over a moment later.
  document.addEventListener('change', function (ev) {
    var input = ev.target;
    if (!input || input.type !== 'file' || injecting) { return; }
    if (input.getAttribute('data-qcap-burnin')) { return; }
    var inputs = uploaderInputs();
    if (inputs.length === 0 || input !== inputs[0]) { return; }
    var f = input.files && input.files[0];
    if (!f || !isVideo(f) || !window.QcapProxy) { return; }
    if (f.size < MIN_PROXY_BYTES) { return; }

    ev.stopPropagation();
    ev.stopImmediatePropagation();
    window.__qcapSource = f;
    window.__qcapSourceUsed = false;
    note('Preparing the audio …');

    probeDuration(f).then(function (sec) {
      if (sec > MAX_PROXY_SECONDS) {
        throw new Error('clip too long to extract in the browser');
      }
      return window.QcapProxy.buildAudioProxy(f, { bitrate: 32000, format: 'ogg_opus' });
    }).then(function (res) {
      // Keep the video's own name and extension in the proxy name: the app
      // reads a name ending in ".audio.ogg" as a picked video's soundtrack
      // and refuses to render the MP4 from it (frontend/media_store.py).
      var base = f.name || 'clip.mp4';
      res.file = new File([res.file], base + '.audio.ogg', { type: 'audio/ogg' });
      window.__qcapProxyFile = res.file;
      note('Video accepted — sending its soundtrack (' + fmtSize(res.bytes)
           + ' of ' + fmtSize(f.size) + ') first');
      hand(input, res.file);
    }).catch(function () {
      note('Could not extract the audio — uploading the video itself.', true);
      hand(input, f);
    });
  }, true);

  // The burn-in step asks for the video the captions are burned into. That
  // file is already in memory, so fill the uploader automatically and let
  // the full-size upload run while the user edits styles.
  function fillBurnIn() {
    if (!window.__qcapSource || window.__qcapSourceUsed) { return; }
    var inputs = uploaderInputs();
    if (inputs.length < 2) { return; }
    var box = document.querySelector('[class*="st-key-video_uploader"]');
    var target = box ? box.querySelector('input[type=file]') : inputs[inputs.length - 1];
    if (!target || (target.files && target.files.length)) { return; }
    window.__qcapSourceUsed = true;
    target.setAttribute('data-qcap-burnin', '1');
    note('Video ' + fmtSize(window.__qcapSource.size) + ' uploading in the background');
    hand(target, window.__qcapSource);
  }
  setInterval(fillBurnIn, 1500);
})();
