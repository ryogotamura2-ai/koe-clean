/* WAV（リニアPCM 24bit）への書き出し。
   channels: Float32Array の配列（左右など）。同じ長さであること。 */
(function (global) {
  "use strict";

  function encodeWav24(channels, sampleRate) {
    var numCh = channels.length;
    var frames = channels[0].length;
    var bytesPerSample = 3;
    var blockAlign = numCh * bytesPerSample;
    var dataSize = frames * blockAlign;
    var pad = dataSize % 2; // RIFF の決まり: 奇数長のチャンクは 1 バイト埋める
    if (44 + dataSize + pad > 0xFFFFFFFF) throw new Error("ファイルが大きすぎて WAV に書き出せません（4GB 超）");

    var buf = new ArrayBuffer(44 + dataSize + pad);
    var v = new DataView(buf);
    writeStr(v, 0, "RIFF");
    v.setUint32(4, 36 + dataSize + pad, true);
    writeStr(v, 8, "WAVE");
    writeStr(v, 12, "fmt ");
    v.setUint32(16, 16, true);
    v.setUint16(20, 1, true); // PCM
    v.setUint16(22, numCh, true);
    v.setUint32(24, sampleRate, true);
    v.setUint32(28, sampleRate * blockAlign, true);
    v.setUint16(32, blockAlign, true);
    v.setUint16(34, 24, true);
    writeStr(v, 36, "data");
    v.setUint32(40, dataSize, true);

    var out = new Uint8Array(buf, 44, dataSize);
    var o = 0;
    for (var i = 0; i < frames; i++) {
      for (var c = 0; c < numCh; c++) {
        var s = channels[c][i];
        if (s > 1) s = 1; else if (s < -1) s = -1; else if (s !== s) s = 0;
        var n = Math.round(s * 8388607);
        out[o++] = n & 0xFF;
        out[o++] = (n >> 8) & 0xFF;
        out[o++] = (n >> 16) & 0xFF;
      }
    }
    return new Blob([buf], { type: "audio/wav" });
  }

  function writeStr(v, off, s) {
    for (var i = 0; i < s.length; i++) v.setUint8(off + i, s.charCodeAt(i));
  }

  global.KoeWav = { encodeWav24: encodeWav24 };
})(self);
