/* Browser-side audio proxy encoder.
 *
 * Problem: a 8-minute recitation video is ~40 MB. Uploading it over a slow
 * link to the caption server is the slowest part of the whole flow, and the
 * captioning step only ever needs the AUDIO.
 *
 * This file extracts the audio track in the browser, encodes it to Opus
 * (~24 kbps mono = ~3 KB/s) and wraps it in an Ogg container, so the same
 * 8 minutes becomes ~1.4 MB instead of ~40 MB. The server decodes it with
 * ffmpeg exactly like any other audio file.
 *
 * Pure browser APIs (decodeAudioData + WebCodecs AudioEncoder). No libraries.
 * If anything is unsupported the caller falls back to uploading the
 * original file unchanged.
 */
(function (global) {
  'use strict';

  // ---------------------------------------------------------------- Ogg CRC
  // Ogg uses CRC-32 with poly 0x04c11db7, no reflection, init 0, no final xor.
  var CRC_TABLE = (function () {
    var t = new Uint32Array(256);
    for (var i = 0; i < 256; i++) {
      var c = i << 24;
      for (var j = 0; j < 8; j++) {
        c = (c & 0x80000000) ? ((c << 1) ^ 0x04c11db7) >>> 0 : (c << 1) >>> 0;
      }
      t[i] = c >>> 0;
    }
    return t;
  })();

  function oggCrc(bytes) {
    var crc = 0;
    for (var i = 0; i < bytes.length; i++) {
      crc = ((crc << 8) ^ CRC_TABLE[((crc >>> 24) & 0xff) ^ bytes[i]]) >>> 0;
    }
    return crc >>> 0;
  }

  // ------------------------------------------------------------- Ogg pages
  function buildPage(serial, seq, granule, headerType, payload) {
    // lacing: each packet is split into 255-byte segments; a trailing 0
    // segment marks a packet whose length is a multiple of 255.
    var segs = [];
    var n = payload.length;
    var i = 0;
    while (i < n) {
      var take = Math.min(255, n - i);
      segs.push(take);
      i += take;
    }
    if (segs.length === 0) segs.push(0);
    if (segs.length > 255) {
      throw new Error('packet too large for a single Ogg page');
    }
    var pageLen = 27 + segs.length + payload.length;
    var page = new Uint8Array(pageLen);
    var dv = new DataView(page.buffer);
    page[0] = 0x4f; page[1] = 0x67; page[2] = 0x67; page[3] = 0x53; // 'OggS'
    page[4] = 0;                       // stream structure version
    page[5] = headerType;              // 0x02 BOS, 0x04 EOS, 0x00 data
    dv.setBigInt64(6, BigInt(granule), true);
    dv.setUint32(14, serial >>> 0, true);
    dv.setUint32(18, seq >>> 0, true);
    dv.setUint32(22, 0, true);         // CRC placeholder
    page[26] = segs.length;
    for (var s = 0; s < segs.length; s++) page[27 + s] = segs[s];
    page.set(payload, 27 + segs.length);
    dv.setUint32(22, oggCrc(page), true);
    return page;
  }

  function asciiBytes(str) {
    var out = new Uint8Array(str.length);
    for (var i = 0; i < str.length; i++) out[i] = str.charCodeAt(i) & 0xff;
    return out;
  }

  function concatBytes(list) {
    var total = 0, i;
    for (i = 0; i < list.length; i++) total += list[i].length;
    var out = new Uint8Array(total);
    var off = 0;
    for (i = 0; i < list.length; i++) { out.set(list[i], off); off += list[i].length; }
    return out;
  }

  /* OpusHead + OpusTags */
  function opusHeaders(channels, inputRate, preSkip) {
    var vendor = asciiBytes('quran-caption-proxy');
    var head = new Uint8Array(19);
    var dv = new DataView(head.buffer);
    head.set(asciiBytes('OpusHead'), 0);
    head[8] = 1;                 // version
    head[9] = channels;
    dv.setUint16(10, preSkip, true);
    dv.setUint32(12, inputRate, true);
    dv.setUint16(16, 0, true);   // output gain
    head[18] = 0;                // channel mapping family
    var tags = new Uint8Array(8 + 4 + vendor.length + 4);
    var tv = new DataView(tags.buffer);
    tags.set(asciiBytes('OpusTags'), 0);
    tv.setUint32(8, vendor.length, true);
    tags.set(vendor, 12);
    tv.setUint32(12 + vendor.length, 0, true); // 0 user comments
    return [head, tags];
  }

  // --------------------------------------------------------------- encoding
  function encodeOpusOgg(audioBuffer, opts) {
    opts = opts || {};
    var targetRate = opts.sampleRate || 48000;
    var bitrate = opts.bitrate || 24000;
    var channels = 1;
    var preSkip = opts.preSkip != null ? opts.preSkip : 312;

    if (typeof AudioEncoder === 'undefined') {
      return Promise.reject(new Error('WebCodecs AudioEncoder unavailable'));
    }

    // Resample to the encoder's rate with an OfflineAudioContext.
    var frames = Math.max(1, Math.round(audioBuffer.duration * targetRate));
    var off = new OfflineAudioContext(channels, frames, targetRate);
    var src = off.createBufferSource();
    src.buffer = audioBuffer;
    src.connect(off.destination);
    src.start(0);

    return off.startRendering().then(function (rendered) {
      var mono = rendered.getChannelData(0);
      var chunkFrames = 960;              // 20 ms at 48 kHz
      var packets = [];
      var enc = null;
      var encErr = null;

      return new Promise(function (resolve, reject) {
        try {
          enc = new AudioEncoder({
            output: function (chunk) {
              var buf = new Uint8Array(chunk.byteLength);
              chunk.copyTo(buf);
              packets.push(buf);
            },
            error: function (e) { encErr = e; }
          });
        } catch (e) { reject(e); return; }

        enc.configure({
          codec: 'opus',
          sampleRate: targetRate,
          numberOfChannels: channels,
          bitrate: bitrate
        });

        var offset = 0;
        var ts = 0;
        // Feed the encoder in ~1 second batches. AudioEncoder queues work
        // internally, so we only need to yield often enough to keep the UI
        // responsive — yielding per 20 ms frame cost 13 s for 30 s of audio.
        var BATCH = targetRate;   // 1 second of frames
        function pump() {
          if (encErr) { reject(encErr); return; }
          if (offset >= mono.length) {
            enc.flush().then(function () { resolve(packets); })
               .catch(function (e) { reject(e); });
            return;
          }
          var batchEnd = Math.min(mono.length, offset + BATCH);
          try {
            while (offset < batchEnd) {
              var n = Math.min(chunkFrames, batchEnd - offset);
              var planar = new Float32Array(n);
              planar.set(mono.subarray(offset, offset + n));
              var ad = new AudioData({
                format: 'f32-planar',
                sampleRate: targetRate,
                numberOfFrames: n,
                numberOfChannels: channels,
                timestamp: ts,
                data: planar
              });
              enc.encode(ad);
              ad.close();
              offset += n;
              ts += Math.round((n / targetRate) * 1e6);
            }
          } catch (e) { reject(e); return; }
          setTimeout(pump, 0);
        }
        pump();
      }).then(function (packets) {
        if (!packets.length) throw new Error('opus encoder produced no packets');
        var serial = (Math.floor(Math.random() * 0x7fffffff) | 1) >>> 0;
        var pages = [];
        var seq = 0;
        var hdrs = opusHeaders(channels, targetRate, preSkip);
        pages.push(buildPage(serial, seq++, 0, 0x02, hdrs[0]));   // BOS
        pages.push(buildPage(serial, seq++, 0, 0x00, hdrs[1]));   // OpusTags

        var granule = 0;
        var samplesPerPacket = 960;   // 20 ms at 48 kHz
        for (var i = 0; i < packets.length; i++) {
          granule += samplesPerPacket;
          var last = (i === packets.length - 1);
          pages.push(buildPage(serial, seq++, granule, last ? 0x04 : 0x00, packets[i]));
        }
        return { bytes: concatBytes(pages), packets: packets.length,
                 durationSec: packets.length * 0.02 };
      });
    });
  }

  /* Decode any media file the browser can read, then proxy-encode it. */
  function buildAudioProxy(file, opts) {
    opts = opts || {};
    var Ctx = global.AudioContext || global.webkitAudioContext;
    if (!Ctx) return Promise.reject(new Error('AudioContext unavailable'));
    var ctx = new Ctx();
    return file.arrayBuffer()
      .then(function (buf) { return ctx.decodeAudioData(buf); })
      .then(function (ab) {
        return encodeOpusOgg(ab, opts).then(function (res) {
          try { ctx.close(); } catch (e) {}
          var name = (file.name || 'clip').replace(/\.[^.]+$/, '') + '.ogg';
          var out = new File([res.bytes], name, { type: 'audio/ogg' });
          return {
            file: out,
            bytes: res.bytes.length,
            originalBytes: file.size,
            durationSec: res.durationSec,
            ratio: file.size ? res.bytes.length / file.size : 1
          };
        });
      })
      .catch(function (e) {
        try { ctx.close(); } catch (e2) {}
        throw e;
      });
  }

  global.QcapProxy = {
    encodeOpusOgg: encodeOpusOgg,
    buildAudioProxy: buildAudioProxy
  };
})(window);
