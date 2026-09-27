/* koe-clean — 画面側の処理
   流れ: ファイル受け付け → 48kHz に変換して読み込み → ノイズ除去（Web Worker で並列）→ 試聴・保存
   「効きの強さ」は処理のやり直しではなく、保存・試聴のときに処理前の音を混ぜ戻して作る
   （DeepFilterNet の atten_lim_db と同じ計算）。 */
(function () {
  "use strict";

  var SR = 48000;
  var WORKER_URL = "dfn3/dfn3-worker.js";
  var STRENGTHS = [
    { id: "max", label: "最大", db: null, hint: "ノイズを消せるだけ消します（聞き比べの C と同じ）。" },
    { id: "strong", label: "強め", db: 20, hint: "ノイズを最大でも約 1/10（-20dB）までしか下げません。" },
    { id: "normal", label: "ふつう", db: 12, hint: "ノイズを最大でも約 1/4（-12dB）までしか下げません。声がこもるときに。" },
    { id: "light", label: "弱め", db: 6, hint: "ノイズを約 1/2（-6dB）まで下げます。いちばん自然ですが、ノイズは残ります。" }
  ];
  var AUDIO_EXT = /\.(m4a|aac|wav|wave|mp3|flac|ogg|oga|opus|aif|aiff|caf|webm|mp4|m4b)$/i;

  var $ = function (id) { return document.getElementById(id); };
  var listEl = $("list"), barEl = $("bar"), dropEl = $("drop");
  var tpl = $("rowTpl");

  var items = [];
  var seq = 0;
  var running = 0;
  var saving = false; // まとめて保存の最中か（二重に押されないように）
  var strength = loadStrength();

  /* ================= 強さ ================= */
  function loadStrength() {
    try {
      var s = localStorage.getItem("koe-clean.strength");
      if (STRENGTHS.some(function (x) { return x.id === s; })) return s;
    } catch (e) {}
    return "max";
  }
  function strengthDef() {
    return STRENGTHS.filter(function (x) { return x.id === strength; })[0];
  }
  /* 処理前の音を混ぜ戻す割合。null（最大）なら 0 */
  function mixBack() {
    var db = strengthDef().db;
    return db == null ? 0 : Math.pow(10, -db / 20);
  }
  function renderStrength() {
    var seg = $("strength");
    seg.innerHTML = "";
    STRENGTHS.forEach(function (s) {
      var b = document.createElement("button");
      b.type = "button";
      b.setAttribute("role", "radio");
      b.setAttribute("aria-checked", s.id === strength ? "true" : "false");
      b.textContent = s.label;
      b.addEventListener("click", function () {
        strength = s.id;
        try { localStorage.setItem("koe-clean.strength", strength); } catch (e) {}
        renderStrength();
        player.applyGains();
      });
      seg.appendChild(b);
    });
    $("strengthHint").textContent = strengthDef().hint + " 強さはあとから変えても、処理し直さずにすぐ反映されます。";
  }

  /* ================= ノイズ除去の部品（初回だけダウンロード） =================
     実行部品（wasm）とモデルは画面側で1回だけ取り、並列に動かすワーカー全員に渡す。
     ワーカーごとに取りに行かせると、初回は同じ 23MB を人数分ダウンロードしてしまう。
     size は圧縮前の大きさ（配信側が gzip しても進み具合がずれないよう、ここを分母にする）。 */
  var ASSETS = [
    { key: "wasmBinary", url: "dfn3/vendor/ort/ort-wasm-simd-threaded.wasm", size: 14239897, loaded: 0 },
    { key: "model", url: "dfn3/models/dfn3_stateful.onnx", size: 8576292, loaded: 0 }
  ];
  var assets = { promise: null, ready: false, loading: false };

  function loadAssets() {
    if (!assets.promise) {
      assets.loading = true;
      setTimeout(refresh, 0); // 読み込み中の表示をすぐ出す（最初のデータが届くまで待たない）
      // 片方が失敗したら、もう片方のダウンロードも止める（やり直し時に二重に取りに行かないように）
      var attempt = { failed: false, ctrl: typeof AbortController === "function" ? new AbortController() : null };
      assets.promise = Promise.all(ASSETS.map(function (a) { return fetchAsset(a, attempt); })).then(function (bufs) {
        var out = {};
        ASSETS.forEach(function (a, i) { out[a.key] = bufs[i]; });
        assets.ready = true;
        assets.loading = false;
        refresh();
        return out;
      });
      assets.promise.catch(function () {
        attempt.failed = true;
        if (attempt.ctrl) attempt.ctrl.abort();
        assets.promise = null; // 次に処理するときにやり直せるように
        assets.loading = false;
        ASSETS.forEach(function (a) { a.loaded = 0; });
        refresh();
      });
    }
    return assets.promise;
  }
  function fetchAsset(a, attempt) {
    var netErr = function () {
      throw new Error("ノイズ除去の部品を取得できませんでした。ネットワークにつながっているか確認してください");
    };
    return fetch(a.url, attempt.ctrl ? { signal: attempt.ctrl.signal } : undefined).then(function (res) {
      if (!res.ok) throw new Error("ノイズ除去の部品を取得できませんでした（" + res.status + "）");
      if (!res.body || !res.body.getReader) {
        return res.arrayBuffer().then(function (b) {
          if (!attempt.failed) { a.loaded = a.size; refresh(); }
          return b;
        }, netErr);
      }
      var reader = res.body.getReader();
      var chunks = [], got = 0, lastPaint = 0;
      function pump() {
        return reader.read().then(function (r) {
          if (attempt.failed) { reader.cancel().catch(function () {}); throw new Error("canceled"); }
          if (r.done) {
            var buf = new Uint8Array(got), o = 0;
            chunks.forEach(function (c) { buf.set(c, o); o += c.length; });
            a.loaded = a.size;
            refresh();
            return buf.buffer;
          }
          chunks.push(r.value);
          got += r.value.length;
          a.loaded = Math.min(got, a.size);
          var now = performance.now();
          if (now - lastPaint > 150) { lastPaint = now; refresh(); }
          return pump();
        }, netErr);
      }
      return pump();
    }, netErr);
  }
  function assetProgressText() {
    var got = 0, total = 0;
    ASSETS.forEach(function (a) { got += a.loaded; total += a.size; });
    return (got / 1e6).toFixed(1) + " / " + (total / 1e6).toFixed(1) + " MB";
  }

  /* ================= ワーカーの並列実行 ================= */
  function poolSize() {
    var n = navigator.hardwareConcurrency || 2;
    return Math.max(1, Math.min(4, n - 1));
  }

  function WorkerPool(size) {
    this.size = size;
    this.workers = [];
    this.idle = [];
    this.queue = [];
    this.jobSeq = 0;
    this.assets = null;
    this.waiting = false;
  }
  WorkerPool.prototype.run = function (samples, onProgress) {
    var self = this;
    var job = { samples: samples, onProgress: onProgress, worker: null, done: false };
    job.promise = new Promise(function (resolve, reject) { job.resolve = resolve; job.reject = reject; });
    job.cancel = function () {
      if (job.done) return;
      job.done = true;
      var qi = self.queue.indexOf(job);
      if (qi >= 0) self.queue.splice(qi, 1);
      if (job.worker) self._kill(job.worker);
      job.reject(new Error("canceled"));
      self._next();
    };
    this.queue.push(job);
    this._next();
    return job;
  };
  WorkerPool.prototype._spawn = function () {
    var w = new Worker(WORKER_URL, { type: "module" });
    // 部品は複製して渡る（転送すると次のワーカーに渡せなくなるため）
    w.postMessage({ type: "init", model: this.assets.model, wasmBinary: this.assets.wasmBinary });
    this.workers.push(w);
    return w;
  };
  WorkerPool.prototype._kill = function (w) {
    w.terminate();
    this.workers = this.workers.filter(function (x) { return x !== w; });
    this.idle = this.idle.filter(function (x) { return x !== w; });
  };
  WorkerPool.prototype._next = function () {
    var self = this;
    if (this.queue.length && !this.assets) {
      if (!this.waiting) {
        this.waiting = true;
        loadAssets().then(function (a) {
          self.assets = a;
          self.waiting = false;
          self._next();
        }, function (err) {
          self.waiting = false;
          self.queue.splice(0).forEach(function (j) { j.done = true; j.reject(err); });
        });
      }
      return;
    }
    while (this.queue.length && (this.idle.length || this.workers.length < this.size)) {
      var w = this.idle.pop() || this._spawn();
      this._exec(w, this.queue.shift());
    }
  };
  WorkerPool.prototype._exec = function (w, job) {
    var self = this;
    var id = ++this.jobSeq;
    job.worker = w;
    function finish() {
      w.onmessage = null;
      w.onerror = null;
      job.worker = null;
      if (self.workers.indexOf(w) >= 0) self.idle.push(w);
      self._next();
    }
    w.onmessage = function (e) {
      var m = e.data;
      if (!m || job.done) return;
      if (m.type === "error" && m.id == null) {
        // 準備（init）の失敗。このワーカーは使えないので捨てる
        job.done = true;
        self._kill(w);
        job.reject(new Error(m.message || "ノイズ除去の部品を準備できませんでした"));
        self._next();
        return;
      }
      if (m.id !== id) return;
      if (m.type === "progress") {
        job.onProgress(m.value);
      } else if (m.type === "result") {
        job.done = true;
        finish();
        job.resolve(m.samples);
      } else if (m.type === "error") {
        job.done = true;
        finish();
        job.reject(new Error(m.message || "ノイズ除去に失敗しました"));
      }
    };
    w.onerror = function (ev) {
      if (job.done) return;
      job.done = true;
      self._kill(w);
      job.reject(new Error(ev && ev.message ? ev.message : "ノイズ除去の部品を読み込めませんでした"));
      self._next();
    };
    w.postMessage({ type: "process", id: id, samples: job.samples, attenLimDb: null }, [job.samples.buffer]);
  };

  var pool = new WorkerPool(poolSize());

  /* ================= 読み込み（48kHz に変換） ================= */
  function decodeFile(file) {
    return file.arrayBuffer().then(function (ab) {
      var OAC = window.OfflineAudioContext || window.webkitOfflineAudioContext;
      var ctx = new OAC(1, 1, SR);
      return new Promise(function (resolve, reject) {
        var ret = ctx.decodeAudioData(ab, resolve, function (err) { reject(err || new Error("decode")); });
        // 新しいブラウザは Promise も返す。失敗がコールバックと二重に届くので、こちらは握りつぶす
        if (ret && typeof ret.catch === "function") ret.catch(function () {});
      });
    }).then(function (audio) {
      var n = audio.numberOfChannels;
      var chs = [];
      if (n === 1) {
        chs.push(new Float32Array(audio.getChannelData(0)));
      } else if (n === 2) {
        var l = audio.getChannelData(0), r = audio.getChannelData(1);
        if (nearlySame(l, r)) chs.push(new Float32Array(l));
        else chs.push(new Float32Array(l), new Float32Array(r));
      } else {
        var mono = new Float32Array(audio.length);
        for (var c = 0; c < n; c++) {
          var d = audio.getChannelData(c);
          for (var i = 0; i < d.length; i++) mono[i] += d[i] / n;
        }
        chs.push(mono);
      }
      return chs;
    });
  }
  /* 左右の差が -60dB 未満なら、中身はモノラルとみなす */
  function nearlySame(a, b) {
    var diff = 0, pow = 0;
    for (var i = 0; i < a.length; i++) {
      var d = a[i] - b[i];
      diff += d * d;
      pow += a[i] * a[i];
    }
    return pow === 0 ? diff === 0 : diff / pow < 1e-6;
  }

  /* ================= 一覧 ================= */
  function addFiles(files) {
    var added = 0, skipped = 0;
    files = files.filter(function (f) {
      var ok = AUDIO_EXT.test(f.name) || /^audio\//.test(f.type);
      if (!ok) skipped++;
      return ok;
    });
    files.sort(function (a, b) {
      return (a.webkitRelativePath || a.name).localeCompare(b.webkitRelativePath || b.name, "ja", { numeric: true });
    });
    files.forEach(function (f) {
      var dup = items.some(function (it) {
        return it.file.name === f.name && it.file.size === f.size && it.file.lastModified === f.lastModified;
      });
      if (dup) { skipped++; return; }
      var it = makeItem(f);
      items.push(it);
      listEl.appendChild(it.el.row);
      added++;
    });
    if (skipped && !added) flash(skipped + " 件は音声ファイルではないか、すでに一覧にあるため追加しませんでした。");
    else if (skipped) flash(added + " 件を追加しました（" + skipped + " 件は対象外か重複のため除外）。");
    refresh();
    pump();
  }

  function makeItem(file) {
    var frag = tpl.content.cloneNode(true);
    var row = frag.querySelector(".row");
    var it = {
      id: ++seq,
      file: file,
      name: file.name,
      status: "queued",
      progress: 0,
      chProgress: [],
      duration: null,
      noisy: null,
      enhanced: null,
      error: null,
      saved: false,
      ab: "after",
      jobs: [],
      removed: false,
      buffers: null,
      el: {
        row: row,
        name: row.querySelector(".r-name"),
        dur: row.querySelector(".r-dur"),
        state: row.querySelector(".r-state"),
        fill: row.querySelector(".r-fill"),
        play: row.querySelector(".r-play"),
        abBtns: row.querySelectorAll(".ab-btn"),
        seek: row.querySelector(".r-seek"),
        save: row.querySelector(".r-save"),
        retry: row.querySelector(".r-retry"),
        remove: row.querySelector(".r-remove")
      }
    };
    it.el.name.textContent = file.webkitRelativePath || file.name;
    it.el.play.addEventListener("click", function () { player.toggle(it); });
    Array.prototype.forEach.call(it.el.abBtns, function (b) {
      b.addEventListener("click", function () {
        it.ab = b.dataset.ab;
        syncAb(it);
        if (player.item === it) player.applyGains();
      });
    });
    it.el.seek.addEventListener("input", function () {
      var t = it.el.seek.value / 1000 * (it.duration || 0);
      if (player.item === it) player.seek(t); else it.offset = t;
    });
    it.el.save.addEventListener("click", function () { saveOne(it); });
    it.el.retry.addEventListener("click", function () {
      if (it.status !== "error") return;
      it.status = "queued";
      it.error = null;
      it.progress = 0;
      it.jobs = [];
      renderRow(it); refresh();
      pump();
    });
    it.el.remove.addEventListener("click", function () { removeItem(it); });
    renderRow(it);
    return it;
  }

  function syncAb(it) {
    Array.prototype.forEach.call(it.el.abBtns, function (b) {
      b.setAttribute("aria-pressed", b.dataset.ab === it.ab ? "true" : "false");
    });
  }

  function renderRow(it) {
    var r = it.el;
    r.row.dataset.status = it.status;
    r.dur.textContent = it.duration != null ? fmtDur(it.duration) : "--:--";
    var text;
    switch (it.status) {
      case "queued": text = "順番待ち"; break;
      case "decoding": text = "読み込み中"; break;
      case "processing": text = it.progress > 0 ? "ノイズ除去中 " + Math.floor(it.progress * 100) + "%" : "準備中"; break;
      case "done": text = it.saved ? "完了・保存済み" : "完了"; break;
      case "error": text = it.error || "失敗しました"; break;
    }
    r.state.textContent = text;
    r.fill.style.width = (it.status === "processing" ? it.progress * 100 : it.status === "decoding" ? 2 : 0) + "%";
    var ready = it.status === "done";
    r.play.disabled = !ready;
    r.save.disabled = !ready;
    r.seek.disabled = !ready;
    Array.prototype.forEach.call(r.abBtns, function (b) { b.disabled = !ready; });
    syncAb(it);
  }

  function removeItem(it) {
    if (player.item === it) player.stop();
    it.removed = true;
    it.jobs.forEach(function (j) { j.cancel(); });
    items = items.filter(function (x) { return x !== it; });
    it.el.row.remove();
    refresh();
  }

  $("clearAll").addEventListener("click", function () {
    var unsaved = items.filter(function (it) { return it.status === "done" && !it.saved; }).length;
    var btn = $("clearAll");
    if (unsaved && btn.dataset.confirm !== "1") {
      btn.dataset.confirm = "1";
      btn.textContent = "保存していない " + unsaved + " 件も消えます。もう一度押すと空にします";
      setTimeout(function () { btn.dataset.confirm = ""; btn.textContent = "一覧を空にする"; }, 4000);
      return;
    }
    btn.dataset.confirm = "";
    btn.textContent = "一覧を空にする";
    items.slice().forEach(removeItem);
  });

  /* ================= 順番に処理 ================= */
  function pump() {
    while (running < pool.size) {
      var next = items.filter(function (it) { return it.status === "queued"; })[0];
      if (!next) break;
      running++;
      processItem(next).then(done, done);
    }
    function done() { running--; pump(); }
  }

  function processItem(it) {
    it.status = "decoding";
    it.startedAt = performance.now();
    renderRow(it); refresh();
    return decodeFile(it.file).then(function (chs) {
      if (it.removed) return;
      it.noisy = chs;
      it.duration = chs[0].length / SR;
      it.status = "processing";
      it.chProgress = chs.map(function () { return 0; });
      renderRow(it); refresh();
      it.jobs = chs.map(function (ch, i) {
        return pool.run(new Float32Array(ch), function (p) {
          it.chProgress[i] = p;
          it.progress = it.chProgress.reduce(function (a, b) { return a + b; }, 0) / it.chProgress.length;
          renderRow(it); refresh();
        });
      });
      return Promise.all(it.jobs.map(function (j) { return j.promise; })).then(function (outs) {
        if (it.removed) return;
        it.enhanced = outs;
        it.status = "done";
        it.progress = 1;
        it.elapsed = (performance.now() - it.startedAt) / 1000;
        renderRow(it); refresh();
      });
    }).catch(function (err) {
      // 片方の音声が失敗したら、もう片方の処理も止めて枠を空ける
      it.jobs.forEach(function (j) { j.cancel(); });
      if (it.removed || (err && err.message === "canceled")) return;
      it.status = "error";
      it.error = describeError(err);
      renderRow(it); refresh();
    });
  }

  function describeError(err) {
    var msg = err && err.message ? err.message : String(err);
    if (!msg || msg === "decode" || /decod|EncodingError|Unable to decode/i.test(msg) || (err && err.name === "EncodingError")) {
      return "読み込めませんでした。このブラウザがこの形式に対応していない可能性があります。Chrome で開き直すか、WAV に変換して試してください";
    }
    return "失敗しました：" + msg;
  }

  /* ================= 集計 ================= */
  function refresh() {
    document.body.classList.toggle("has-items", items.length > 0);
    barEl.hidden = items.length === 0 && $("sumSub").dataset.flash !== "1";
    var total = items.length;
    var cnt = { queued: 0, decoding: 0, processing: 0, done: 0, error: 0 };
    var prog = 0;
    items.forEach(function (it) {
      cnt[it.status]++;
      prog += it.status === "done" || it.status === "error" ? 1 : it.status === "processing" ? it.progress : 0;
    });
    var active = cnt.queued + cnt.decoding + cnt.processing;
    var main = total + " 件";
    var parts = [];
    if (cnt.done) parts.push("完了 " + cnt.done);
    if (cnt.decoding + cnt.processing) parts.push("処理中 " + (cnt.decoding + cnt.processing));
    if (cnt.queued) parts.push("待ち " + cnt.queued);
    if (cnt.error) parts.push("失敗 " + cnt.error);
    $("sumMain").textContent = main + (parts.length ? " · " + parts.join(" · ") : "");
    var sub;
    if (active && assets.loading) {
      sub = "ノイズ除去の部品を読み込んでいます " + assetProgressText() + "（初回のみ。次からは速く始まります）";
    } else if (active) {
      var eta = estimate();
      sub = eta != null ? "残り 約" + fmtEta(eta) + "（目安）" : "処理しています。このタブは開いたままにしてください";
    } else if (cnt.done) {
      var unsaved = items.filter(function (it) { return it.status === "done" && !it.saved; }).length;
      sub = unsaved ? "すべて終わりました。「まとめて保存」で書き出せます" : "すべて保存しました";
      if (cnt.error) sub += "。失敗したものは「やり直す」で再挑戦できます";
    } else if (cnt.error) {
      sub = "失敗したものは「やり直す」で再挑戦できます";
    } else {
      sub = "";
    }
    if ($("sumSub").dataset.flash !== "1") $("sumSub").textContent = sub;
    $("sumFill").style.width = (total ? prog / total * 100 : 0) + "%";
    $("saveAll").disabled = saving || cnt.done === 0;
    document.title = active ? "(" + cnt.done + "/" + total + ") koe-clean" : "koe-clean — ナレーションのノイズをまとめて除去";
  }

  /* 終わったファイルの「音声1秒あたりの処理時間」から、残りを見積もる */
  function estimate() {
    var doneItems = items.filter(function (it) { return it.status === "done" && it.elapsed && it.duration; });
    if (!doneItems.length) return null;
    var secPerAudio = doneItems.reduce(function (a, it) { return a + it.elapsed; }, 0) /
      doneItems.reduce(function (a, it) { return a + it.duration; }, 0);
    var known = items.filter(function (it) { return it.duration; });
    var avgDur = known.reduce(function (a, it) { return a + it.duration; }, 0) / known.length;
    var remaining = 0;
    items.forEach(function (it) {
      if (it.status === "queued" || it.status === "decoding") remaining += it.duration || avgDur;
      else if (it.status === "processing") remaining += it.duration * (1 - it.progress);
    });
    return remaining * secPerAudio / pool.size;
  }

  /* ================= 書き出し ================= */
  function renderChannels(it) {
    var k = mixBack();
    if (k === 0) return it.enhanced;
    return it.enhanced.map(function (enh, c) {
      var noisy = it.noisy[c];
      var out = new Float32Array(enh.length);
      for (var i = 0; i < enh.length; i++) out[i] = (1 - k) * enh[i] + k * noisy[i];
      return out;
    });
  }
  function outName(it) {
    var base = it.name.replace(/\.[^.]+$/, "");
    return base + "_clean.wav";
  }
  function wavBlob(it) {
    return KoeWav.encodeWav24(renderChannels(it), SR);
  }
  function download(blob, name) {
    var url = URL.createObjectURL(blob);
    var a = document.createElement("a");
    a.href = url;
    a.download = name;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(function () { URL.revokeObjectURL(url); }, 60000);
  }
  function saveOne(it) {
    download(wavBlob(it), outName(it));
    it.saved = true;
    renderRow(it); refresh();
  }

  $("saveAll").addEventListener("click", function () {
    var targets = items.filter(function (it) { return it.status === "done"; });
    if (!targets.length || saving) return;
    var btn = $("saveAll");
    saving = true;
    btn.disabled = true;
    var label = btn.textContent;
    btn.textContent = "保存しています…";
    var job = window.showDirectoryPicker ? saveToFolder(targets) : saveZip(targets);
    job.then(function (ok) {
      if (ok) {
        targets.forEach(function (it) { it.saved = true; renderRow(it); });
        flash(targets.length + " 件を保存しました。");
      }
    }).catch(function (err) {
      flash("保存できませんでした：" + (err && err.message ? err.message : err));
    }).then(function () {
      saving = false;
      btn.textContent = label;
      refresh();
    });
  });

  function saveToFolder(targets) {
    return window.showDirectoryPicker({ id: "koe-clean", mode: "readwrite" }).then(function (dir) {
      var chain = Promise.resolve();
      targets.forEach(function (it) {
        chain = chain.then(function () {
          return uniqueName(dir, outName(it)).then(function (name) {
            return dir.getFileHandle(name, { create: true });
          }).then(function (fh) {
            return fh.createWritable();
          }).then(function (w) {
            return w.write(wavBlob(it)).then(function () { return w.close(); });
          });
        });
      });
      return chain.then(function () { return true; });
    }, function (err) {
      if (err && err.name === "AbortError") return false; // 選ぶのをやめた
      return saveZip(targets); // フォルダを選べない環境
    });
  }
  function uniqueName(dir, name) {
    var base = name.replace(/\.wav$/, "");
    function tryN(n) {
      var cand = n === 1 ? name : base + " (" + n + ").wav";
      return dir.getFileHandle(cand).then(function () { return tryN(n + 1); }, function () { return cand; });
    }
    return tryN(1);
  }
  function saveZip(targets) {
    var used = {};
    var files = targets.map(function (it) {
      var name = outName(it);
      var base = name.replace(/\.wav$/, "");
      for (var n = 2; used[name]; n++) name = base + " (" + n + ").wav";
      used[name] = true;
      return { name: name, blob: wavBlob(it) };
    });
    return KoeZip.makeZip(files).then(function (zip) {
      var d = new Date();
      var stamp = d.getFullYear() + pad(d.getMonth() + 1) + pad(d.getDate()) + "-" + pad(d.getHours()) + pad(d.getMinutes());
      download(zip, "koe-clean_" + stamp + ".zip");
      return true;
    });
  }

  window.addEventListener("beforeunload", function (e) {
    var busy = items.some(function (it) {
      return (it.status === "done" && !it.saved) || it.status === "processing" || it.status === "decoding" || it.status === "queued";
    });
    if (busy) { e.preventDefault(); e.returnValue = ""; }
  });

  /* ================= 試聴（処理前と処理後を同時に鳴らし、音量で切り替える） ================= */
  var player = {
    ctx: null, item: null, srcs: [], gBefore: null, gAfter: null,
    playing: false, startAt: 0, offset: 0,

    ensure: function () {
      if (this.ctx) return;
      var AC = window.AudioContext || window.webkitAudioContext;
      try { this.ctx = new AC({ sampleRate: SR }); } catch (e) { this.ctx = new AC(); }
      this.gBefore = this.ctx.createGain();
      this.gAfter = this.ctx.createGain();
      this.gBefore.connect(this.ctx.destination);
      this.gAfter.connect(this.ctx.destination);
    },
    buffers: function (it) {
      if (it.buffers) return it.buffers;
      var ctx = this.ctx;
      function mk(chs) {
        var b = ctx.createBuffer(chs.length, chs[0].length, SR);
        chs.forEach(function (d, c) { b.getChannelData(c).set(d); });
        return b;
      }
      it.buffers = { before: mk(it.noisy), after: mk(it.enhanced) };
      return it.buffers;
    },
    applyGains: function () {
      if (!this.ctx || !this.item) return;
      var now = this.ctx.currentTime;
      var k = mixBack();
      var before = this.item.ab === "before" ? 1 : k;
      var after = this.item.ab === "before" ? 0 : 1 - k;
      [[this.gBefore, before], [this.gAfter, after]].forEach(function (p) {
        p[0].gain.cancelScheduledValues(now);
        p[0].gain.setTargetAtTime(p[1], now, 0.006);
      });
    },
    toggle: function (it) {
      this.ensure();
      if (this.ctx.state === "suspended") this.ctx.resume();
      if (this.item === it && this.playing) { this.pause(); return; }
      if (this.item !== it) {
        this.stop();
        this.item = it;
        this.offset = it.offset || 0;
      }
      this.start(this.offset);
    },
    start: function (at) {
      var self = this, it = this.item;
      this.killSources();
      var bufs = this.buffers(it);
      if (at >= it.duration - 0.05) at = 0;
      var when = this.ctx.currentTime + 0.03;
      this.srcs = [[bufs.before, this.gBefore], [bufs.after, this.gAfter]].map(function (p) {
        var s = self.ctx.createBufferSource();
        s.buffer = p[0];
        s.connect(p[1]);
        s.start(when, at);
        return s;
      });
      this.srcs[1].onended = function () {
        if (self.srcs[1] && self.srcs[1].buffer && self.playing && self.pos() >= it.duration - 0.1) {
          self.playing = false;
          self.offset = 0;
          self.mark();
        }
      };
      this.applyGains();
      this.startAt = when;
      this.offset = at;
      this.playing = true;
      this.mark();
    },
    pos: function () {
      if (!this.item) return 0;
      if (!this.playing) return this.offset;
      return Math.min(this.item.duration, this.offset + Math.max(0, this.ctx.currentTime - this.startAt));
    },
    pause: function () {
      this.offset = this.pos();
      this.killSources();
      this.playing = false;
      this.mark();
    },
    stop: function () {
      if (this.item) { this.item.offset = this.pos(); }
      this.killSources();
      this.playing = false;
      this.mark();
      this.item = null;
    },
    seek: function (t) {
      if (this.playing) this.start(t); else { this.offset = t; this.mark(); }
    },
    killSources: function () {
      this.srcs.forEach(function (s) { s.onended = null; try { s.stop(); } catch (e) {} s.disconnect(); });
      this.srcs = [];
    },
    mark: function () {
      var self = this;
      items.forEach(function (it) {
        var on = it === self.item && self.playing;
        it.el.row.classList.toggle("playing", on);
        it.el.play.setAttribute("aria-label", on ? "一時停止" : "再生");
        it.el.play.innerHTML = on
          ? '<svg viewBox="0 0 12 12" aria-hidden="true"><path d="M2.5 1.5h2.5v9H2.5zM7 1.5h2.5v9H7z"/></svg>'
          : '<svg viewBox="0 0 12 12" aria-hidden="true"><path d="M2.5 1.5l8 4.5-8 4.5z"/></svg>';
      });
    }
  };

  function tick() {
    if (player.item && document.activeElement !== player.item.el.seek) {
      var it = player.item;
      it.el.seek.value = String(Math.round(player.pos() / it.duration * 1000));
    }
    requestAnimationFrame(tick);
  }

  /* ================= ドロップとファイル選択 ================= */
  ["dragenter", "dragover"].forEach(function (t) {
    document.addEventListener(t, function (e) {
      if (!hasFiles(e)) return;
      e.preventDefault();
      dropEl.classList.add("over");
    });
  });
  ["dragleave", "drop"].forEach(function (t) {
    document.addEventListener(t, function (e) {
      if (t === "dragleave" && e.relatedTarget) return;
      dropEl.classList.remove("over");
    });
  });
  document.addEventListener("drop", function (e) {
    if (!hasFiles(e)) return;
    e.preventDefault();
    collectDropped(e.dataTransfer).then(addFiles);
  });
  function hasFiles(e) {
    return e.dataTransfer && Array.prototype.indexOf.call(e.dataTransfer.types || [], "Files") >= 0;
  }
  function collectDropped(dt) {
    var entries = [];
    if (dt.items) {
      Array.prototype.forEach.call(dt.items, function (item) {
        var en = item.webkitGetAsEntry && item.webkitGetAsEntry();
        if (en) entries.push(en);
      });
    }
    if (!entries.length) return Promise.resolve(Array.prototype.slice.call(dt.files || []));
    return Promise.all(entries.map(walk)).then(function (lists) {
      return [].concat.apply([], lists);
    });
  }
  function walk(entry) {
    if (entry.isFile) {
      return new Promise(function (res) { entry.file(function (f) { res([f]); }, function () { res([]); }); });
    }
    if (entry.isDirectory) {
      var reader = entry.createReader();
      var all = [];
      return new Promise(function (res) {
        (function readMore() {
          reader.readEntries(function (batch) {
            if (!batch.length) {
              Promise.all(all.map(walk)).then(function (l) { res([].concat.apply([], l)); });
              return;
            }
            all = all.concat(Array.prototype.slice.call(batch));
            readMore();
          }, function () { res([]); });
        })();
      });
    }
    return Promise.resolve([]);
  }
  ["pickFiles", "pickFolder"].forEach(function (id) {
    var input = $(id);
    input.addEventListener("change", function () {
      addFiles(Array.prototype.slice.call(input.files || []));
      input.value = "";
    });
  });
  dropEl.addEventListener("keydown", function (e) {
    if (e.target !== dropEl) return; // 中の「ファイル/フォルダを選ぶ」はそれ自身に任せる
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("pickFiles").click(); }
  });

  /* ================= 小物 ================= */
  var flashTimer = null;
  function flash(text) {
    var el = $("sumSub");
    barEl.hidden = false;
    el.textContent = text;
    el.dataset.flash = "1";
    clearTimeout(flashTimer);
    flashTimer = setTimeout(function () { el.dataset.flash = ""; refresh(); }, 5000);
  }
  function fmtDur(t) {
    var m = Math.floor(t / 60), s = Math.floor(t % 60);
    return m + ":" + pad(s);
  }
  function fmtEta(t) {
    if (t < 60) return Math.max(1, Math.round(t)) + "秒";
    var m = Math.floor(t / 60), s = Math.round(t % 60);
    return m + "分" + (s ? s + "秒" : "");
  }
  function pad(n) { return (n < 10 ? "0" : "") + n; }

  renderStrength();
  refresh();
  requestAnimationFrame(tick);

  /* 動作確認用（開発者ツールから状態を見るため） */
  window.__koe = { items: function () { return items; }, pool: pool };
})();
