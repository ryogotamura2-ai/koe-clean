/* 無圧縮（STORE）の ZIP を作る。WAV はほとんど縮まないので圧縮はしない。
   4GB を超える ZIP（ZIP64）には対応しない。 */
(function (global) {
  "use strict";

  var CRC_TABLE = (function () {
    var t = new Uint32Array(256);
    for (var n = 0; n < 256; n++) {
      var c = n;
      for (var k = 0; k < 8; k++) c = c & 1 ? 0xEDB88320 ^ (c >>> 1) : c >>> 1;
      t[n] = c >>> 0;
    }
    return t;
  })();

  function crc32(bytes) {
    var c = 0xFFFFFFFF;
    for (var i = 0; i < bytes.length; i++) c = CRC_TABLE[(c ^ bytes[i]) & 0xFF] ^ (c >>> 8);
    return (c ^ 0xFFFFFFFF) >>> 0;
  }

  function dosDateTime(d) {
    var time = (d.getHours() << 11) | (d.getMinutes() << 5) | Math.floor(d.getSeconds() / 2);
    var date = ((d.getFullYear() - 1980) << 9) | ((d.getMonth() + 1) << 5) | d.getDate();
    return { time: time, date: date };
  }

  /* files: [{ name: string, blob: Blob }] → Promise<Blob> */
  async function makeZip(files) {
    var enc = new TextEncoder();
    var parts = [];
    var central = [];
    var offset = 0;
    var dt = dosDateTime(new Date());

    for (var i = 0; i < files.length; i++) {
      var name = enc.encode(files[i].name);
      var data = new Uint8Array(await files[i].blob.arrayBuffer());
      var crc = crc32(data);
      var size = data.length;
      if (offset + 30 + name.length + size > 0xFFFFFFFF) throw new Error("合計が 4GB を超えるため ZIP にできません。保存先フォルダを選べるブラウザ（Chrome など）を使うか、数回に分けて保存してください");

      var lh = new DataView(new ArrayBuffer(30));
      lh.setUint32(0, 0x04034B50, true);
      lh.setUint16(4, 20, true);
      lh.setUint16(6, 0x0800, true); // ファイル名は UTF-8
      lh.setUint16(8, 0, true);      // STORE
      lh.setUint16(10, dt.time, true);
      lh.setUint16(12, dt.date, true);
      lh.setUint32(14, crc, true);
      lh.setUint32(18, size, true);
      lh.setUint32(22, size, true);
      lh.setUint16(26, name.length, true);
      lh.setUint16(28, 0, true);
      parts.push(lh.buffer, name, data);

      var ch = new DataView(new ArrayBuffer(46));
      ch.setUint32(0, 0x02014B50, true);
      ch.setUint16(4, 20, true);
      ch.setUint16(6, 20, true);
      ch.setUint16(8, 0x0800, true);
      ch.setUint16(10, 0, true);
      ch.setUint16(12, dt.time, true);
      ch.setUint16(14, dt.date, true);
      ch.setUint32(16, crc, true);
      ch.setUint32(20, size, true);
      ch.setUint32(24, size, true);
      ch.setUint16(28, name.length, true);
      ch.setUint16(30, 0, true);
      ch.setUint16(32, 0, true);
      ch.setUint16(34, 0, true);
      ch.setUint16(36, 0, true);
      ch.setUint32(38, 0, true);
      ch.setUint32(42, offset, true);
      central.push(ch.buffer, name);

      offset += 30 + name.length + size;
    }

    var cdSize = 0;
    for (var j = 0; j < central.length; j++) cdSize += central[j].byteLength;
    var end = new DataView(new ArrayBuffer(22));
    end.setUint32(0, 0x06054B50, true);
    end.setUint16(8, files.length, true);
    end.setUint16(10, files.length, true);
    end.setUint32(12, cdSize, true);
    end.setUint32(16, offset, true);
    end.setUint16(20, 0, true);

    return new Blob(parts.concat(central, [end.buffer]), { type: "application/zip" });
  }

  global.KoeZip = { makeZip: makeZip, crc32: crc32 };
})(self);
