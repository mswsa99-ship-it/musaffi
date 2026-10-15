// مُصفّي — نسخة المتصفح: كل المعالجة على جهاز المستخدم، ولا يُرفع أي ملف
"use strict";
(() => {
  const $ = (s) => document.querySelector(s);
  const $$ = (s) => Array.from(document.querySelectorAll(s));
  const D = window.MUSAFFI_DSP, SEP = window.MUSAFFI_SEP, V = window.MUSAFFI_VISION;
  const MANIFEST = window.MUSAFFI_MANIFEST;
  const ORT_CDN = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0/dist/";
  const SR = 44100, SAMPLE_FPS = 6, MAX_SEC = 600;
  const STEMS = [
    { key: "speech", name: "الكلام", color: "var(--speech)" },
    { key: "music", name: "الموسيقى والأصوات المحيطة", color: "var(--music)" },
    { key: "noise", name: "الضوضاء الثابتة", color: "var(--noise)", hint: "طنين، مكيف، هواء" },
  ];
  const PRESETS = { speech: { speech: 1, music: 0, noise: 0 }, clean: { speech: 1, music: 0, noise: 0.15 }, original: { speech: 1, music: 1, noise: 1 } };

  const S = { file: null, url: null, hasVideo: false, audio: null, stems: null, analysis: null, tracker: null, decisions: {}, gains: { ...PRESETS.speech }, muted: {} };
  let ortReady = null, sessions = {};

  // ---------- أدوات ----------
  const show = (id) => ["#step-upload", "#step-progress", "#step-studio"].forEach((s) => ($(s).hidden = s !== id));
  const fmtTime = (t) => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`;
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  function setProg(p, msg, sub) {
    $("#prog-bar").style.width = `${Math.max(1, Math.min(100, p))}%`;
    if (msg !== undefined) $("#prog-msg").textContent = msg;
    if (sub !== undefined) $("#prog-sub").textContent = sub;
  }
  function fail(msg) {
    show("#step-upload");
    $("#upload-error").textContent = msg;
  }

  // ---------- تحميل النماذج (مرة واحدة، وتُحفظ في المتصفح) ----------
  async function fetchModel(name, onBytes) {
    const m = MANIFEST[name];
    const parts = [];
    let cache = null;
    try { cache = await caches.open("musaffi-models-v1"); } catch (_) {}
    for (let i = 0; i < m.parts; i++) {
      const url = `models/${name}.${i}.bin`;
      let resp = null;
      try { resp = cache && (await cache.match(url)); } catch (_) {}
      if (!resp) {
        resp = await fetch(url);
        if (!resp.ok) throw new Error(`تعذّر تحميل ${name}`);
        try { if (cache) await cache.put(url, resp.clone()); } catch (_) {}
      }
      const buf = new Uint8Array(await resp.arrayBuffer());
      onBytes(buf.length);
      parts.push(buf);
    }
    const total = parts.reduce((s, p) => s + p.length, 0);
    const out = new Uint8Array(total);
    let o = 0;
    for (const p of parts) { out.set(p, o); o += p.length; }
    return out;
  }

  async function loadOrt(onBytes) {
    if (!window.ort) throw new Error("تعذّر تحميل محرك الذكاء الاصطناعي. تأكد من اتصال الإنترنت.");
    const wasm = await fetchModel("ortwasm", onBytes);
    ort.env.wasm.wasmBinary = wasm.buffer;
    ort.env.wasm.wasmPaths = { mjs: ORT_CDN + "ort-wasm-simd-threaded.jsep.mjs" };
    ort.env.wasm.numThreads = (self.crossOriginIsolated && navigator.hardwareConcurrency) ? Math.min(4, navigator.hardwareConcurrency) : 1;
    ort.env.wasm.proxy = false;
    ort.env.logLevel = "error";
  }

  async function session(name, bytes, preferGpu) {
    if (sessions[name]) return sessions[name];
    let s = null;
    const timeout = (p, ms) => Promise.race([p, new Promise((_, rej) => setTimeout(() => rej(new Error("timeout")), ms))]);
    if (preferGpu && navigator.gpu) {
      // كرت الشاشة أسرع بكثير، لكن بعض الأجهزة تتعثر فيه؛ نجربه بمهلة ثم نرجع للمعالج
      try {
        const adapter = await timeout(navigator.gpu.requestAdapter(), 5000);
        if (adapter && !adapter.isFallbackAdapter) {
          s = await timeout(ort.InferenceSession.create(bytes, { executionProviders: ["webgpu"] }), 60000);
          s._gpu = true;
        }
      } catch (_) { s = null; }
    }
    if (!s) s = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
    sessions[name] = s;
    return s;
  }

  async function ensureModels(needAudio, needVideo) {
    const names = ["ortwasm"];
    if (needAudio) names.push("vocft");
    if (needVideo) names.push("yolo", "det", "ga");
    const total = names.filter((n) => !sessions[n]).reduce((s, n) => s + MANIFEST[n].size, 0);
    let got = 0;
    const onBytes = (b) => {
      got += b;
      setProg(got / total * 100, "تحميل نماذج الذكاء الاصطناعي إلى جهازك",
        `${(got / 1048576).toFixed(0)} من ${(total / 1048576).toFixed(0)} ميجابايت — مرة واحدة فقط، وتُحفظ في متصفحك للمرات القادمة`);
    };
    if (!ortReady) ortReady = loadOrt(onBytes);
    await ortReady;
    if (needAudio && !sessions.vocft) await session("vocft", await fetchModel("vocft", onBytes), true);
    if (needVideo) {
      if (!sessions.yolo) await session("yolo", await fetchModel("yolo", onBytes), true);
      if (!sessions.det) await session("det", await fetchModel("det", onBytes), false);
      if (!sessions.ga) await session("ga", await fetchModel("ga", onBytes), false);
    }
  }

  // ---------- الرفع والخيارات ----------
  const drop = $("#drop");
  const setFile = (f) => {
    S.file = f;
    $("#drop-text").textContent = f ? `${f.name} — ${(f.size / 1048576).toFixed(1)} ميجابايت` : "اسحب الملف هنا أو اضغط للاختيار";
    drop.classList.toggle("has-file", !!f);
    validate();
  };
  $("#file").addEventListener("change", (e) => setFile(e.target.files[0]));
  drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") $("#file").click(); });
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => e.dataTransfer.files[0] && setFile(e.dataTransfer.files[0]));
  $$("input[name=blur]").forEach((r) => r.addEventListener("change", () => {
    $("#sens-wrap").hidden = $("input[name=blur]:checked").value !== "women"; validate();
  }));
  $("#opt-audio").addEventListener("change", validate);
  function validate() {
    const something = $("#opt-audio").checked || $("input[name=blur]:checked").value !== "none";
    $("#btn-start").disabled = !(S.file && something);
  }
  $("#btn-start").addEventListener("click", () => start().catch((e) => { console.error(e); fail(e.message || "حدث خطأ أثناء المعالجة"); }));

  async function start() {
    $("#upload-error").textContent = "";
    const opts = {
      audio: $("#opt-audio").checked,
      blur: $("input[name=blur]:checked").value,
      sens: $("input[name=sens]:checked").value,
    };
    S.opts = opts;
    show("#step-progress");
    setProg(1, "قراءة الملف على جهازك", "");
    if (S.url) URL.revokeObjectURL(S.url);
    S.url = URL.createObjectURL(S.file);
    const video = $("#video");
    S.hasVideo = (S.file.type || "").startsWith("video") || /\.(mp4|mov|webm|mkv|m4v|3gp)$/i.test(S.file.name);
    video.src = S.url;
    if (S.hasVideo) {
      await new Promise((res, rej) => {
        video.onloadedmetadata = res;
        video.onerror = () => rej(new Error("المتصفح لا يدعم صيغة هذا الفيديو. جرّب MP4 من الجوال أو واتساب."));
      });
      S.hasVideo = video.videoWidth > 0;
    }
    S.audio = await decodeAudio(S.file).catch(() => null);
    const dur = S.hasVideo ? video.duration : (S.audio ? S.audio.length / SR : 0);
    if (!dur) throw new Error("تعذّر قراءة الملف");
    if (dur > MAX_SEC) throw new Error(`مدة المقطع أطول من ${MAX_SEC / 60} دقائق. قصّ المقطع وحاول مرة ثانية.`);
    const needAudio = opts.audio && !!S.audio;
    const needVideo = opts.blur !== "none" && S.hasVideo;
    if (!needAudio && !needVideo) throw new Error(S.audio ? "اختر عملية واحدة على الأقل" : "الملف لا يحتوي على صوت");

    await ensureModels(needAudio, needVideo);
    const t0 = performance.now();
    let base = 0;
    if (needAudio) {
      await runAudio((p) => progressEta(p * (needVideo ? 60 : 100), t0, "فصل الصوت البشري عن الموسيقى"));
      base = needVideo ? 60 : 100;
    }
    if (needVideo) {
      await runVideo((p, n, total) => progressEta(base + p * (100 - base), t0, `تحليل الصورة: اكتشاف الأشخاص والوجوه (${n} من ${total})`));
    }
    openStudio();
  }

  function progressEta(p, t0, msg) {
    const el = (performance.now() - t0) / 1000;
    const eta = p > 2 ? el * (100 - p) / p : null;
    setProg(p, msg, eta === null ? "جارٍ الحساب..." : `الوقت المتبقي تقريبًا: ${eta > 90 ? Math.round(eta / 60) + " دقيقة" : Math.max(5, Math.round(eta)) + " ثانية"} — أبقِ الصفحة مفتوحة`);
  }

  async function decodeAudio(file) {
    const buf = await file.arrayBuffer();
    const Ctx = window.OfflineAudioContext || window.webkitOfflineAudioContext;
    const ctx = new Ctx(2, SR, SR);
    const ab = await ctx.decodeAudioData(buf);
    // إعادة التشكيل إلى 44.1 كيلوهرتز وقناتين
    const off = new Ctx(2, Math.ceil(ab.duration * SR), SR);
    const src = off.createBufferSource();
    src.buffer = ab; src.connect(off.destination); src.start();
    const r = await off.startRendering();
    const L = r.getChannelData(0), R = r.numberOfChannels > 1 ? r.getChannelData(1) : r.getChannelData(0);
    return { L: new Float32Array(L), R: new Float32Array(R), length: L.length };
  }

  // ---------- الصوت ----------
  async function runAudio(onP) {
    const { L, R } = S.audio;
    const [vL, vR] = await SEP.separateVocals(ort, sessions.vocft, L, R, { dsp: D, overlap: 0.15, onProgress: (p) => onP(p * 0.9) });
    onP(0.92);
    await sleep(10);
    const { clean, noise } = SEP.splitStationaryNoise(vL, vR, D, SR);
    const n = L.length;
    const mL = new Float32Array(n), mR = new Float32Array(n);
    for (let i = 0; i < n; i++) { mL[i] = L[i] - vL[i]; mR[i] = R[i] - vR[i]; }
    S.stems = { speech: clean, music: [mL, mR], noise };
    S.analysisAudio = analyzeStems(S.stems);
    onP(1);
  }

  function analyzeStems(stems) {
    const energy = {}, env = {};
    let peak = 0;
    for (const [k, [L, R]] of Object.entries(stems)) {
      let e = 0;
      for (let i = 0; i < L.length; i++) e += L[i] * L[i] + R[i] * R[i];
      energy[k] = e;
      const bins = 300, n = Math.max(1, Math.floor(L.length / bins)), arr = [];
      for (let b = 0; b < bins; b++) {
        let s = 0;
        for (let i = b * n; i < (b + 1) * n && i < L.length; i++) s += Math.abs(L[i]) + Math.abs(R[i]);
        arr.push(s / (2 * n)); peak = Math.max(peak, s / (2 * n));
      }
      env[k] = arr;
    }
    const total = Object.values(energy).reduce((a, b) => a + b, 0) || 1;
    const out = {};
    for (const k of Object.keys(stems)) {
      const rms = Math.sqrt(energy[k] / (2 * stems[k][0].length) + 1e-12);
      out[k] = { share: Math.round(1000 * energy[k] / total) / 10, db: Math.round(200 * Math.log10(rms + 1e-12)) / 10, envelope: env[k].map((v) => v / (peak || 1)) };
    }
    return out;
  }

  // ---------- الصورة ----------
  async function runVideo(onP) {
    const video = $("#video");
    const W = video.videoWidth, H = video.videoHeight, dur = video.duration;
    const yc = document.createElement("canvas"); yc.width = yc.height = V.YOLO_SIZE;
    const yctx = yc.getContext("2d", { willReadFrequently: true });
    const fc = document.createElement("canvas"); fc.width = fc.height = V.FACE_SIZE;
    const fctx = fc.getContext("2d", { willReadFrequently: true });
    const gc = document.createElement("canvas"); gc.width = gc.height = 96;
    const gctx = gc.getContext("2d", { willReadFrequently: true });
    const ys = V.YOLO_SIZE / Math.max(W, H), fs = V.FACE_SIZE / Math.max(W, H);
    const tracker = new V.Tracker();
    const thumbs = {}, thumbArea = {};
    const steps = Math.max(1, Math.floor(dur * SAMPLE_FPS));
    video.muted = true; video.pause();

    const toTensor = (ctx, size, norm) => {
      const d = ctx.getImageData(0, 0, size, size).data, n = size * size;
      const arr = new Float32Array(3 * n);
      for (let i = 0; i < n; i++) {
        arr[i] = norm(d[4 * i]); arr[n + i] = norm(d[4 * i + 1]); arr[2 * n + i] = norm(d[4 * i + 2]);
      }
      return arr;
    };
    const seek = (t) => new Promise((res) => {
      const done = () => { video.removeEventListener("seeked", done); res(); };
      video.addEventListener("seeked", done);
      video.currentTime = t;
    });

    for (let k = 0; k <= steps; k++) {
      const t = Math.min(dur - 0.01, k / SAMPLE_FPS);
      await seek(t);
      yctx.fillStyle = "rgb(114,114,114)"; yctx.fillRect(0, 0, V.YOLO_SIZE, V.YOLO_SIZE);
      yctx.drawImage(video, 0, 0, Math.round(W * ys), Math.round(H * ys));
      const yIn = toTensor(yctx, V.YOLO_SIZE, (v) => v / 255);
      const yo = await sessions.yolo.run({ [sessions.yolo.inputNames[0]]: new ort.Tensor("float32", yIn, [1, 3, V.YOLO_SIZE, V.YOLO_SIZE]) });
      const o0 = yo[sessions.yolo.outputNames[0]], o1 = yo[sessions.yolo.outputNames[1]];
      const dets = V.yoloDecode(await o0.getData(), await o1.getData(), o0.dims[2], o1.dims[2], ys, W, H);
      tracker.update(t, dets);

      if (k % 2 === 0 && dets.length) { // الوجوه كل ثلث ثانية
        fctx.fillStyle = "#000"; fctx.fillRect(0, 0, V.FACE_SIZE, V.FACE_SIZE);
        fctx.drawImage(video, 0, 0, Math.round(W * fs), Math.round(H * fs));
        const fIn = toTensor(fctx, V.FACE_SIZE, (v) => (v - 127.5) / 128);
        const fo = await sessions.det.run({ [sessions.det.inputNames[0]]: new ort.Tensor("float32", fIn, [1, 3, V.FACE_SIZE, V.FACE_SIZE]) });
        const faces = V.scrfdDecode(sessions.det.outputNames.map((n) => fo[n].data), V.FACE_SIZE, fs);
        for (const f of faces) {
          const [x1, y1, x2, y2] = f.box, size = Math.min(x2 - x1, y2 - y1);
          let pf = null;
          if (size >= 20) {
            const side = Math.max(x2 - x1, y2 - y1) * 1.5, cx = (x1 + x2) / 2, cy = (y1 + y2) / 2;
            gctx.fillStyle = "#000"; gctx.fillRect(0, 0, 96, 96);
            gctx.drawImage(video, cx - side / 2, cy - side / 2, side, side, 0, 0, 96, 96);
            const gIn = toTensor(gctx, 96, (v) => v);
            const go = await sessions.ga.run({ [sessions.ga.inputNames[0]]: new ort.Tensor("float32", gIn, [1, 3, 96, 96]) });
            pf = V.genderPFemale(go[sessions.ga.outputNames[0]].data);
          }
          tracker.addFace(t, f.box, pf, f.score, dets);
        }
      }
      for (const d of dets) { // صورة مصغرة لكل شخص
        const a = (d.box[2] - d.box[0]) * (d.box[3] - d.box[1]);
        if (a > (thumbArea[d.tid] || 0) * 1.2) {
          thumbArea[d.tid] = a;
          const c = document.createElement("canvas"), bw = d.box[2] - d.box[0], bh = d.box[3] - d.box[1], s = 160 / Math.max(bw, bh);
          c.width = Math.max(1, Math.round(bw * s)); c.height = Math.max(1, Math.round(bh * s));
          c.getContext("2d").drawImage(video, d.box[0], d.box[1], bw, bh, 0, 0, c.width, c.height);
          thumbs[d.tid] = c.toDataURL("image/jpeg", 0.8);
        }
      }
      onP(k / steps, k, steps);
    }
    S.tracker = tracker;
    S.summary = tracker.summary();
    S.decisions = V.decide(S.summary, S.opts.blur, S.opts.sens);
    S.thumbs = thumbs;
    await seek(0);
  }

  // ---------- الاستوديو ----------
  const video = $("#video"), canvas = $("#stage"), ctx = canvas.getContext("2d");
  let actx = null, nodes = {}, buffers = {}, sources = [], playing = false;

  function openStudio() {
    show("#step-studio");
    $("#audio-only").hidden = S.hasVideo;
    canvas.hidden = !S.hasVideo;
    $("#fmt-video").hidden = !S.hasVideo;
    $("#fmt-stems").hidden = !S.stems;
    $("#fmt-audio").hidden = !S.audio;
    if (!S.hasVideo) $("input[name=fmt][value=audio]").checked = true;
    if (S.hasVideo) { canvas.width = video.videoWidth; canvas.height = video.videoHeight; }
    actx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: SR });
    if (S.stems) setupMixer(); else $("#mixer").hidden = true;
    if (S.tracker) setupPeople(); else $("#people").hidden = true;
    requestAnimationFrame(drawLoop);
  }

  function makeBuffer([L, R]) {
    const b = actx.createBuffer(2, L.length, SR);
    b.copyToChannel(L, 0); b.copyToChannel(R, 1);
    return b;
  }

  function setupMixer() {
    $("#mixer").hidden = false;
    const wrap = $("#stems");
    wrap.innerHTML = "";
    for (const s of STEMS) {
      const a = S.analysisAudio[s.key];
      buffers[s.key] = makeBuffer(S.stems[s.key]);
      const g = actx.createGain(); g.connect(actx.destination); nodes[s.key] = g;
      const el = document.createElement("div");
      el.className = "stem";
      el.innerHTML = `
        <div class="stem-head">
          <span class="stem-name"><span class="dot" style="background:${s.color}"></span>${s.name}</span>
          <span class="stem-meta">${a.share}% من الصوت · ${a.db} dB</span>
        </div>
        ${s.hint ? `<div class="stem-meta">${s.hint}</div>` : ""}
        <div class="meter"><div style="width:${Math.min(100, a.share)}%;background:${s.color}"></div></div>
        <canvas class="env" width="300" height="34" aria-hidden="true"></canvas>
        <div class="stem-ctl">
          <button class="mute" type="button">كتم</button>
          <input type="range" min="0" max="150" step="1" id="gain-${s.key}" aria-label="قوة ${s.name}">
          <output class="mono"></output>
        </div>`;
      wrap.appendChild(el);
      drawEnvelope(el.querySelector("canvas"), a.envelope, s.color);
      const range = el.querySelector("input"), out = el.querySelector("output"), mute = el.querySelector(".mute");
      range.value = Math.round(S.gains[s.key] * 100); out.textContent = `${range.value}%`;
      range.addEventListener("input", () => { S.gains[s.key] = range.value / 100; out.textContent = `${range.value}%`; applyGains(); });
      mute.addEventListener("click", () => { S.muted[s.key] = !S.muted[s.key]; mute.classList.toggle("on", S.muted[s.key]); applyGains(); });
      s.el = { range, out, mute };
    }
    $$("[data-preset]").forEach((b) => b.onclick = () => {
      Object.assign(S.gains, PRESETS[b.dataset.preset]);
      STEMS.forEach((s) => { s.el.range.value = Math.round(S.gains[s.key] * 100); s.el.out.textContent = `${s.el.range.value}%`; S.muted[s.key] = false; s.el.mute.classList.remove("on"); });
      applyGains();
    });
    applyGains();
  }
  const effGain = (k) => (S.muted[k] ? 0 : S.gains[k]);
  function applyGains() { for (const s of STEMS) if (nodes[s.key]) nodes[s.key].gain.value = effGain(s.key); }

  function drawEnvelope(cv, env, color) {
    const c = cv.getContext("2d"), w = cv.width, h = cv.height;
    c.fillStyle = getComputedStyle(document.documentElement).getPropertyValue(color.slice(4, -1)).trim() || "#888";
    const bw = w / env.length;
    env.forEach((v, i) => { const bh = Math.max(1, v * h); c.fillRect(i * bw, (h - bh) / 2, Math.max(1, bw - 0.3), bh); });
  }

  // الصوت الأصلي إذا لم تُفصل المسارات
  function originalBuffer() {
    if (!buffers.original && S.audio) buffers.original = makeBuffer([S.audio.L, S.audio.R]);
    return buffers.original;
  }

  let clockStart = 0, clockOffset = 0;
  const duration = () => (S.hasVideo ? video.duration : (S.audio ? S.audio.length / SR : 0)) || 0;
  const now = () => (S.hasVideo ? video.currentTime : playing ? clockOffset + actx.currentTime - clockStart : clockOffset);

  function startAudio(at) {
    stopAudio();
    actx.resume();
    const list = S.stems ? STEMS.map((s) => [buffers[s.key], nodes[s.key]]) : (originalBuffer() ? [[originalBuffer(), actx.destination]] : []);
    sources = list.map(([buf, dest]) => {
      const src = actx.createBufferSource(); src.buffer = buf; src.connect(dest);
      src.start(0, Math.min(at, buf.duration)); return src;
    });
  }
  function stopAudio() { sources.forEach((s) => { try { s.stop(); } catch (_) {} }); sources = []; }
  async function play() {
    if (S.hasVideo) { video.muted = true; await video.play(); startAudio(video.currentTime); }
    else { clockStart = actx.currentTime; startAudio(clockOffset); }
    playing = true; $("#btn-play").textContent = "❚❚";
  }
  function pause() {
    if (S.hasVideo) video.pause(); else clockOffset = now();
    stopAudio(); playing = false; $("#btn-play").textContent = "▶";
  }
  $("#btn-play").addEventListener("click", () => (playing ? pause() : play()));
  video.addEventListener("ended", pause);
  $("#seek").addEventListener("input", (e) => {
    const t = e.target.value / 1000 * duration();
    if (S.hasVideo) video.currentTime = t; else clockOffset = t;
    if (playing) { if (S.hasVideo) startAudio(t); else { clockStart = actx.currentTime; startAudio(t); } }
  });

  function setupPeople() {
    $("#people").hidden = false;
    const list = $("#people-list");
    list.innerHTML = "";
    const ids = Object.keys(S.summary).filter((t) => S.summary[t].count >= 3).sort((a, b) => S.summary[b].count - S.summary[a].count);
    if (!ids.length) list.innerHTML = `<p class="muted small">لم يُكتشف أي شخص في المقطع.</p>`;
    for (const t of ids) {
      const info = S.summary[t], pf = info.p_female;
      const guess = pf === null || info.votes < 2 ? "غير معروف" : pf >= 0.5 ? `امرأة غالبًا (${Math.round(pf * 100)}%)` : `رجل غالبًا (${Math.round((1 - pf) * 100)}%)`;
      const el = document.createElement("button");
      el.type = "button"; el.className = "person";
      el.innerHTML = `<img alt="" src="${S.thumbs[t] || ""}"><div class="lbl">${guess}</div><div class="state"></div>`;
      const sync = () => { el.classList.toggle("blurred", !!S.decisions[t]); el.querySelector(".state").textContent = S.decisions[t] ? "مُموَّه" : "ظاهر"; el.setAttribute("aria-pressed", String(!!S.decisions[t])); };
      el.addEventListener("click", () => { S.decisions[t] = !S.decisions[t]; sync(); });
      el.addEventListener("mouseenter", () => { if (!playing) video.currentTime = (info.first + info.last) / 2; });
      sync();
      list.appendChild(el);
    }
  }

  // ---------- رسم الإطار مع التمويه (للمعاينة والتصدير) ----------
  const maskC = document.createElement("canvas"), maskX = maskC.getContext("2d");
  const blurC = document.createElement("canvas"), blurX = blurC.getContext("2d");
  const pixC = document.createElement("canvas"), pixX = pixC.getContext("2d");
  const m32 = document.createElement("canvas"); m32.width = m32.height = V.MASK_RES;
  const m32x = m32.getContext("2d");
  const maskImgCache = new WeakMap();
  function maskImage(mask) {
    let img = maskImgCache.get(mask);
    if (!img) {
      img = m32x.createImageData(V.MASK_RES, V.MASK_RES);
      for (let i = 0; i < mask.length; i++) { img.data[4 * i] = img.data[4 * i + 1] = img.data[4 * i + 2] = 255; img.data[4 * i + 3] = mask[i]; }
      maskImgCache.set(mask, img);
    }
    return img;
  }

  function renderFrame(target, t, withBlur) {
    const W = target.canvas.width, H = target.canvas.height;
    target.drawImage(video, 0, 0, W, H);
    if (!withBlur || !S.tracker) return;
    const region = $("input[name=region]:checked").value;
    if (maskC.width !== W) { maskC.width = blurC.width = W; maskC.height = blurC.height = H; }
    maskX.clearRect(0, 0, W, H);
    let any = false;
    const pad = 0.04 * Math.max(W, H);
    for (const [id, tr] of S.tracker.tracks) {
      if (!S.decisions[id]) continue;
      const at = V.trackAt(tr, t);
      if (!at) continue;
      any = true;
      const [x1, y1, x2, y2] = at.box;
      if (region === "face") {
        const fb = V.faceBoxAt(tr, t, at.box);
        const cx = (fb[0] + fb[2]) / 2, cy = (fb[1] + fb[3]) / 2;
        maskX.fillStyle = "#fff"; maskX.beginPath();
        maskX.ellipse(cx, cy, (fb[2] - fb[0]) * 0.8 + pad / 2, (fb[3] - fb[1]) * 0.85 + pad / 2, 0, 0, Math.PI * 2);
        maskX.fill();
      } else {
        m32x.putImageData(maskImage(at.mask), 0, 0);
        maskX.imageSmoothingEnabled = true;
        // نوسّع القناع قليلًا ونرسمه عدة مرات بإزاحات لتغطية الحركة بين العينات
        for (const [dx, dy] of [[0, 0], [-pad, 0], [pad, 0], [0, -pad], [0, pad]]) {
          maskX.drawImage(m32, x1 - pad / 2 + dx, y1 - pad / 2 + dy, x2 - x1 + pad, y2 - y1 + pad);
        }
      }
    }
    if (!any) return;
    const bw = Math.max(14, Math.round(Math.min(W, H) / 11));
    pixC.width = Math.ceil(W / bw); pixC.height = Math.ceil(H / bw);
    pixX.drawImage(video, 0, 0, pixC.width, pixC.height);
    blurX.globalCompositeOperation = "source-over";
    blurX.clearRect(0, 0, W, H);
    blurX.imageSmoothingEnabled = false;
    blurX.filter = `blur(${Math.round(bw / 2)}px)`;
    blurX.drawImage(pixC, 0, 0, W, H);
    blurX.filter = "none";
    blurX.globalCompositeOperation = "destination-in";
    blurX.drawImage(maskC, 0, 0);
    target.drawImage(blurC, 0, 0);
  }

  function drawLoop() {
    requestAnimationFrame(drawLoop);
    if ($("#step-studio").hidden || exporting) return;
    const d = duration();
    if (d && document.activeElement !== $("#seek")) $("#seek").value = Math.round(now() / d * 1000);
    $("#time").textContent = `${fmtTime(now())} / ${fmtTime(d)}`;
    if (!S.hasVideo) { if (playing && now() >= d) { pause(); clockOffset = 0; } return; }
    if (video.readyState >= 2) renderFrame(ctx, video.currentTime, $("#preview-blur").checked);
  }

  // ---------- التصدير ----------
  let exporting = false;
  const downloadsP = (window.claude && window.claude.use) ? window.claude.use("downloads").catch(() => null) : Promise.resolve(null);

  async function saveFile(filename, blob) {
    const dl = await downloadsP;
    if (dl) {
      try { await dl.save({ filename, data: blob }); return true; }
      catch (e) {
        if (e && e.code === "declined") return false;
        throw new Error("تعذّر حفظ الملف في هذا العرض. افتح الصفحة في المتصفح على الكمبيوتر وحاول مرة ثانية.");
      }
    }
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob); a.download = filename;
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 60000);
    return true;
  }

  async function mixOffline() {
    const n = S.audio.length;
    const Ctx = window.OfflineAudioContext || window.webkitOfflineAudioContext;
    const off = new Ctx(2, n, SR);
    const list = S.stems ? STEMS.map((s) => [S.stems[s.key], effGain(s.key)]) : [[[S.audio.L, S.audio.R], 1]];
    for (const [[L, R], g] of list) {
      if (g <= 0) continue;
      const b = off.createBuffer(2, n, SR); b.copyToChannel(L, 0); b.copyToChannel(R, 1);
      const src = off.createBufferSource(); src.buffer = b;
      const gn = off.createGain(); gn.gain.value = g;
      src.connect(gn); gn.connect(off.destination); src.start();
    }
    const r = await off.startRendering();
    const L = r.getChannelData(0), R = r.getChannelData(1);
    let peak = 0;
    for (let i = 0; i < n; i++) peak = Math.max(peak, Math.abs(L[i]), Math.abs(R[i]));
    if (peak > 0.99) { const k = 0.99 / peak; for (let i = 0; i < n; i++) { L[i] *= k; R[i] *= k; } }
    return [L, R];
  }

  function wav([L, R]) {
    const n = L.length, buf = new ArrayBuffer(44 + n * 4), v = new DataView(buf);
    const w = (o, s) => [...s].forEach((c, i) => v.setUint8(o + i, c.charCodeAt(0)));
    w(0, "RIFF"); v.setUint32(4, 36 + n * 4, true); w(8, "WAVE"); w(12, "fmt ");
    v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 2, true); v.setUint32(24, SR, true);
    v.setUint32(28, SR * 4, true); v.setUint16(32, 4, true); v.setUint16(34, 16, true); w(36, "data"); v.setUint32(40, n * 4, true);
    let o = 44;
    for (let i = 0; i < n; i++) {
      v.setInt16(o, Math.max(-1, Math.min(1, L[i])) * 32767, true);
      v.setInt16(o + 2, Math.max(-1, Math.min(1, R[i])) * 32767, true); o += 4;
    }
    return new Uint8Array(buf);
  }

  // ملف ZIP بسيط بدون ضغط
  function zip(files) {
    const crcT = new Uint32Array(256);
    for (let i = 0; i < 256; i++) { let c = i; for (let k = 0; k < 8; k++) c = c & 1 ? 0xEDB88320 ^ (c >>> 1) : c >>> 1; crcT[i] = c >>> 0; }
    const crc = (d) => { let c = 0xFFFFFFFF; for (let i = 0; i < d.length; i++) c = crcT[(c ^ d[i]) & 255] ^ (c >>> 8); return (c ^ 0xFFFFFFFF) >>> 0; };
    const enc = new TextEncoder(), chunks = [], central = [];
    let off = 0;
    for (const [name, data] of files) {
      const nm = enc.encode(name), c = crc(data);
      const h = new DataView(new ArrayBuffer(30));
      h.setUint32(0, 0x04034b50, true); h.setUint16(4, 20, true); h.setUint16(6, 0x0800, true);
      h.setUint32(14, c, true); h.setUint32(18, data.length, true); h.setUint32(22, data.length, true); h.setUint16(26, nm.length, true);
      chunks.push(new Uint8Array(h.buffer), nm, data);
      const cd = new DataView(new ArrayBuffer(46));
      cd.setUint32(0, 0x02014b50, true); cd.setUint16(4, 20, true); cd.setUint16(6, 20, true); cd.setUint16(8, 0x0800, true);
      cd.setUint32(16, c, true); cd.setUint32(20, data.length, true); cd.setUint32(24, data.length, true); cd.setUint16(28, nm.length, true);
      cd.setUint32(42, off, true);
      central.push(new Uint8Array(cd.buffer), nm);
      off += 30 + nm.length + data.length;
    }
    const cdSize = central.reduce((s, x) => s + x.length, 0);
    const end = new DataView(new ArrayBuffer(22));
    end.setUint32(0, 0x06054b50, true); end.setUint16(8, files.length, true); end.setUint16(10, files.length, true);
    end.setUint32(12, cdSize, true); end.setUint32(16, off, true);
    return new Blob([...chunks, ...central, new Uint8Array(end.buffer)], { type: "application/zip" });
  }

  async function recordVideo(onP) {
    const mime = ["video/mp4;codecs=avc1.42E01F,mp4a.40.2", "video/mp4", "video/webm;codecs=vp9,opus", "video/webm;codecs=vp8,opus", "video/webm"]
      .find((m) => window.MediaRecorder && MediaRecorder.isTypeSupported(m));
    if (!mime) throw new Error("متصفحك لا يدعم تصدير الفيديو. استخدم Chrome أو Edge على الكمبيوتر.");
    const ext = mime.startsWith("video/mp4") ? "mp4" : "webm";
    const out = document.createElement("canvas"); out.width = video.videoWidth; out.height = video.videoHeight;
    const octx = out.getContext("2d");
    const stream = out.captureStream(30);
    const dest = actx.createMediaStreamDestination();
    if (S.audio) {
      const [L, R] = await mixOffline();
      const b = actx.createBuffer(2, L.length, SR); b.copyToChannel(L, 0); b.copyToChannel(R, 1);
      var src = actx.createBufferSource(); src.buffer = b; src.connect(dest);
      dest.stream.getAudioTracks().forEach((tr) => stream.addTrack(tr));
    }
    const rec = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: 6_000_000, audioBitsPerSecond: 192_000 });
    const parts = [];
    rec.ondataavailable = (e) => e.data.size && parts.push(e.data);
    const stopped = new Promise((r) => (rec.onstop = r));
    video.pause(); video.currentTime = 0;
    await new Promise((r) => video.addEventListener("seeked", r, { once: true }));
    renderFrame(octx, 0, true);
    await actx.resume();
    rec.start(1000);
    if (src) src.start();
    video.muted = true;
    await video.play();
    await new Promise((resolve) => {
      const step = () => {
        if (video.ended || video.currentTime >= video.duration - 0.02) return resolve();
        renderFrame(octx, video.currentTime, true);
        onP(video.currentTime / video.duration);
        if (video.requestVideoFrameCallback) video.requestVideoFrameCallback(step); else requestAnimationFrame(step);
      };
      step();
      video.addEventListener("ended", resolve, { once: true });
    });
    renderFrame(octx, video.duration, true);
    await sleep(300);
    rec.stop();
    try { src && src.stop(); } catch (_) {}
    await stopped;
    return { blob: new Blob(parts, { type: mime.split(";")[0] }), ext };
  }

  $("#btn-export").addEventListener("click", async () => {
    pause();
    const fmt = $("input[name=fmt]:checked").value;
    const base = (S.file.name.replace(/\.[^.]+$/, "") || "musaffi").slice(0, 60);
    $("#btn-export").disabled = true;
    $("#export-progress").hidden = false;
    const setE = (p, m) => { $("#exp-bar").style.width = `${Math.max(2, p * 100)}%`; $("#exp-msg").textContent = m; };
    exporting = true;
    try {
      let ok;
      if (fmt === "stems") {
        setE(0.2, "تجهيز المسارات");
        const files = STEMS.map((s) => [`${s.key}.wav`, wav(S.stems[s.key])]);
        ok = await saveFile(`${base}-stems.zip`, zip(files));
      } else if (fmt === "audio") {
        setE(0.3, "مزج الصوت");
        ok = await saveFile(`${base}-clean.zip`, zip([[`${base}-clean.wav`, wav(await mixOffline())]]));
      } else {
        setE(0, "تسجيل الفيديو — يستغرق نفس مدة المقطع، أبقِ هذه الصفحة ظاهرة");
        const { blob, ext } = await recordVideo((p) => setE(p, `تسجيل الفيديو ${Math.round(p * 100)}% — أبقِ هذه الصفحة ظاهرة على الشاشة`));
        setE(1, "حفظ الملف");
        ok = await saveFile(`${base}-clean.${ext}`, blob);
      }
      setE(1, ok ? "تم حفظ الملف على جهازك ✓" : "لم يُحفظ الملف");
    } catch (e) {
      console.error(e);
      $("#exp-msg").textContent = e.message || "تعذّر التصدير";
    } finally {
      exporting = false;
      $("#btn-export").disabled = false;
    }
  });

  $("#btn-new").addEventListener("click", () => location.reload());
})();
