"use strict";
(() => {
  const $ = (s) => document.querySelector(s);
  const $$ = (s) => Array.from(document.querySelectorAll(s));
  const STEMS = [
    { key: "speech", name: "الكلام", color: "var(--speech)" },
    { key: "music", name: "الموسيقى", color: "var(--music)" },
    { key: "noise", name: "الضوضاء الثابتة", color: "var(--noise)", hint: "طنين، مكيف، رياح" },
    { key: "ambient", name: "أصوات محيطة وأخرى", color: "var(--ambient)", hint: "جمهور، تصفيق، شارع، بقايا غناء" },
  ];
  const PRESETS = {
    speech: { speech: 1, music: 0, noise: 0, ambient: 0 },
    nomusic: { speech: 1, music: 0, noise: 0.3, ambient: 0.8 },
    original: { speech: 1, music: 1, noise: 1, ambient: 1 },
  };

  let jobId = null, file = null, status = null, pollTimer = null;
  const gains = { ...PRESETS.speech };
  const muted = {};
  let decisions = {}, tracksData = null;

  // ---------- أدوات ----------
  const show = (id) => ["#step-upload", "#step-progress", "#step-studio", "#step-done"]
    .forEach((s) => ($(s).hidden = s !== id));
  const fmtTime = (t) => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`;
  async function api(path, opts = {}) {
    const r = await fetch(path, opts);
    if (!r.ok) {
      let msg = "حدث خطأ";
      try { msg = (await r.json()).detail || msg; } catch (_) {}
      const e = new Error(msg); e.status = r.status; throw e;
    }
    return r.json();
  }
  function finish(title, msg) {
    clearInterval(pollTimer);
    stopAudio();
    $("#done-title").textContent = title;
    $("#done-msg").textContent = msg;
    show("#step-done");
  }

  // ---------- الرفع ----------
  api("/api/limits").then((l) => {
    $("#limits").textContent = `فيديو أو صوت — حتى ${l.max_upload_mb} ميجابايت و${l.max_duration_min} دقيقة`;
    $$(".ttl").forEach((e) => (e.textContent = l.ttl_min));
  }).catch(() => {});

  const drop = $("#drop");
  const setFile = (f) => {
    file = f;
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
  ["#consent", "#opt-audio"].forEach((s) => $(s).addEventListener("change", validate));
  function validate() {
    const something = $("#opt-audio").checked || $("input[name=blur]:checked").value !== "none";
    $("#btn-start").disabled = !(file && $("#consent").checked && something);
  }

  $("#btn-start").addEventListener("click", () => {
    $("#upload-error").textContent = "";
    const fd = new FormData();
    fd.append("file", file);
    fd.append("separate_audio", $("#opt-audio").checked);
    fd.append("blur_mode", $("input[name=blur]:checked").value);
    fd.append("sensitivity", $("input[name=sens]:checked").value);
    fd.append("quality", $("input[name=quality]:checked").value);
    show("#step-progress");
    $("#prog-title").textContent = "جارٍ رفع المقطع";
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/jobs");
    xhr.upload.onprogress = (e) => e.lengthComputable && setProg(e.loaded / e.total * 100, `تم رفع ${Math.round(e.loaded / e.total * 100)}%`);
    xhr.onload = () => {
      let body = {};
      try { body = JSON.parse(xhr.responseText); } catch (_) {}
      if (xhr.status !== 200) { show("#step-upload"); $("#upload-error").textContent = body.detail || "تعذّر الرفع"; return; }
      jobId = body.id;
      $("#prog-title").textContent = "جارٍ المعالجة";
      poll();
    };
    xhr.onerror = () => { show("#step-upload"); $("#upload-error").textContent = "انقطع الاتصال أثناء الرفع"; };
    xhr.send(fd);
  });

  function setProg(p, msg, bar = "#prog-bar", msgEl = "#prog-msg") {
    $(bar).style.width = `${Math.max(2, Math.min(100, p))}%`;
    $(msgEl).textContent = msg || "";
  }

  // ---------- متابعة الحالة ----------
  function poll() {
    clearInterval(pollTimer);
    const tick = async () => {
      try {
        status = await api(`/api/jobs/${jobId}`);
      } catch (e) {
        if (e.status === 404) finish("تم حذف المقطع", "انتهت صلاحية المقطع وحُذف من خوادمنا.");
        return;
      }
      const s = status.state;
      const inStudio = !$("#step-studio").hidden;
      const waitMsg = status.queue_position ? `في الانتظار — دورك ${status.queue_position}` : "في الانتظار";
      if (inStudio) {
        if (s === "queued") setProg(1, waitMsg, "#exp-bar", "#exp-msg");
        else if (s === "exporting") setProg(status.progress, status.message, "#exp-bar", "#exp-msg");
        else if (s === "exported") onExported();
        else if (s === "error") {
          clearInterval(pollTimer);
          $("#exp-msg").textContent = status.message;
          $("#btn-export").disabled = false;
        }
      } else if (s === "queued") setProg(1, waitMsg);
      else if (s === "processing") setProg(status.progress, status.message);
      else if (s === "error") { clearInterval(pollTimer); setProg(100, status.message); $("#prog-title").textContent = "تعذّرت المعالجة"; }
      else if (s === "ready") openStudio();
    };
    tick();
    pollTimer = setInterval(tick, 1500);
  }

  // ---------- الاستوديو ----------
  const video = $("#video");
  const canvas = $("#overlay");
  const ctx = canvas.getContext("2d");
  let actx = null, buffers = {}, nodes = {}, sources = [], playing = false, startAt = 0, offset = 0;
  let hasStems = false, hasVideo = false;

  async function openStudio() {
    clearInterval(pollTimer);
    pollTimer = setInterval(() => api(`/api/jobs/${jobId}`).catch(() => {}), 60000); // إبقاء المهلة
    show("#step-studio");
    const meta = status.meta, result = status.result || {};
    hasVideo = meta.has_video;
    hasStems = !!result.audio;
    $("#audio-only").hidden = hasVideo;
    video.hidden = !hasVideo;
    $("#fmt-video").hidden = !hasVideo;
    $("#fmt-audio").hidden = !meta.has_audio;
    $("#fmt-stems").hidden = !hasStems;
    if (!hasVideo) $("input[name=fmt][value=audio]").checked = true;

    video.src = `/api/jobs/${jobId}/source`;
    video.muted = hasStems;
    video.addEventListener("loadedmetadata", sizeCanvas);
    window.addEventListener("resize", sizeCanvas);

    if (hasStems) await setupMixer(result.audio);
    if (result.video) await setupPeople(result.video);
    requestAnimationFrame(drawLoop);
  }

  async function setupMixer(analysis) {
    $("#mixer").hidden = false;
    const wrap = $("#stems");
    wrap.innerHTML = "";
    for (const s of STEMS) {
      const a = analysis[s.key];
      const el = document.createElement("div");
      el.className = "stem";
      el.innerHTML = `
        <div class="stem-head">
          <span class="stem-name"><span class="dot" style="background:${s.color}"></span>${s.name}</span>
          <span class="stem-meta">${a.share}% من الصوت · ${a.db} dB</span>
        </div>
        ${s.hint ? `<div class="stem-meta">${s.hint}</div>` : ""}
        <div class="meter" title="نسبة هذا المسار من طاقة الصوت"><div style="width:${Math.min(100, a.share)}%;background:${s.color}"></div></div>
        <canvas class="env" width="400" height="34" aria-hidden="true"></canvas>
        <div class="stem-ctl">
          <button class="mute" type="button">كتم</button>
          <input type="range" min="0" max="150" step="1" aria-label="قوة ${s.name}">
          <output class="mono"></output>
        </div>`;
      wrap.appendChild(el);
      drawEnvelope(el.querySelector("canvas"), a.envelope, s.color);
      const range = el.querySelector("input"), out = el.querySelector("output"), mute = el.querySelector(".mute");
      range.value = Math.round(gains[s.key] * 100);
      out.textContent = `${range.value}%`;
      range.addEventListener("input", () => { gains[s.key] = range.value / 100; out.textContent = `${range.value}%`; applyGains(); });
      mute.addEventListener("click", () => { muted[s.key] = !muted[s.key]; mute.classList.toggle("on", muted[s.key]); applyGains(); });
      s.el = { range, out, mute };
    }
    $$("[data-preset]").forEach((b) => b.addEventListener("click", () => {
      Object.assign(gains, PRESETS[b.dataset.preset]);
      STEMS.forEach((s) => { s.el.range.value = Math.round(gains[s.key] * 100); s.el.out.textContent = `${s.el.range.value}%`; muted[s.key] = false; s.el.mute.classList.remove("on"); });
      applyGains();
    }));
    // تحميل المسارات وفك ترميزها للمعاينة الفورية
    actx = new (window.AudioContext || window.webkitAudioContext)();
    await Promise.all(STEMS.map(async (s) => {
      const r = await fetch(`/api/jobs/${jobId}/stem/${s.key}`);
      buffers[s.key] = await actx.decodeAudioData(await r.arrayBuffer());
      const g = actx.createGain(); g.connect(actx.destination); nodes[s.key] = g;
    }));
    applyGains();
  }

  function drawEnvelope(cv, env, color) {
    const c = cv.getContext("2d"), w = cv.width, h = cv.height;
    c.clearRect(0, 0, w, h);
    c.fillStyle = getComputedStyle(document.documentElement).getPropertyValue(color.slice(4, -1)).trim() || "#888";
    const bw = w / env.length;
    env.forEach((v, i) => { const bh = Math.max(1, v * h); c.fillRect(i * bw, (h - bh) / 2, Math.max(1, bw - 0.3), bh); });
  }

  function applyGains() {
    for (const s of STEMS) if (nodes[s.key]) nodes[s.key].gain.value = muted[s.key] ? 0 : gains[s.key];
  }
  const effGains = () => Object.fromEntries(STEMS.map((s) => [s.key, muted[s.key] ? 0 : gains[s.key]]));

  // التشغيل المتزامن: الفيديو هو الساعة، والمسارات الصوتية تتبعه
  function startAudio(at) {
    stopAudio();
    if (!hasStems || !actx) return;
    actx.resume();
    sources = STEMS.map((s) => {
      const src = actx.createBufferSource();
      src.buffer = buffers[s.key]; src.connect(nodes[s.key]);
      src.start(0, Math.min(at, src.buffer.duration));
      return src;
    });
  }
  function stopAudio() { sources.forEach((s) => { try { s.stop(); } catch (_) {} }); sources = []; }
  const duration = () => (hasVideo ? video.duration : (buffers.speech ? buffers.speech.duration : 0)) || 0;
  const now = () => (hasVideo ? video.currentTime : playing ? offset + actx.currentTime - startAt : offset);

  async function play() {
    if (hasVideo) { await video.play(); startAudio(video.currentTime); }
    else { startAt = actx.currentTime; startAudio(offset); }
    playing = true; $("#btn-play").textContent = "❚❚";
  }
  function pause() {
    if (hasVideo) video.pause(); else offset = now();
    stopAudio(); playing = false; $("#btn-play").textContent = "▶";
  }
  $("#btn-play").addEventListener("click", () => (playing ? pause() : play()));
  video.addEventListener("ended", pause);
  $("#seek").addEventListener("input", (e) => {
    const t = e.target.value / 1000 * duration();
    if (hasVideo) video.currentTime = t; else offset = t;
    if (playing) { if (hasVideo) startAudio(t); else { startAt = actx.currentTime; startAudio(t); } }
  });

  // ---------- التمويه: معاينة داخل المتصفح ----------
  async function setupPeople(res) {
    $("#people").hidden = false;
    $("#preview-blur-wrap").hidden = false;
    decisions = { ...res.decisions };
    tracksData = await api(`/api/jobs/${jobId}/tracks`);
    const list = $("#people-list");
    list.innerHTML = "";
    const ids = Object.keys(res.tracks).filter((t) => res.tracks[t].count >= 3)
      .sort((a, b) => res.tracks[b].count - res.tracks[a].count);
    // الأشخاص الذين ظهروا أقل من 3 إطارات يُموَّهون حسب القرار الآلي ولا يُعرضون
    if (!ids.length) list.innerHTML = `<p class="muted small">لم يُكتشف أي شخص في المقطع.</p>`;
    for (const t of ids) {
      const info = res.tracks[t];
      const el = document.createElement("button");
      el.type = "button";
      el.className = "person";
      const pf = info.p_female;
      const guess = pf === null || info.votes < 2 ? "غير معروف" : pf >= 0.5 ? `امرأة غالبًا (${Math.round(pf * 100)}%)` : `رجل غالبًا (${Math.round((1 - pf) * 100)}%)`;
      el.innerHTML = `<img src="/api/jobs/${jobId}/thumb/${t}" alt="" loading="lazy"><div class="lbl">${guess}</div><div class="state"></div>`;
      const sync = () => { el.classList.toggle("blurred", !!decisions[t]); el.querySelector(".state").textContent = decisions[t] ? "مُموَّه" : "ظاهر"; el.setAttribute("aria-pressed", !!decisions[t]); };
      el.addEventListener("click", () => { decisions[t] = !decisions[t]; sync(); });
      el.addEventListener("mouseenter", () => { if (!playing && hasVideo) video.currentTime = (info.first + info.count / 2) / tracksData.info.fps; });
      sync();
      list.appendChild(el);
    }
  }

  let pixel = document.createElement("canvas"), pctx = pixel.getContext("2d");
  function sizeCanvas() {
    const r = video.getBoundingClientRect(), pr = video.parentElement.getBoundingClientRect();
    canvas.style.left = `${r.left - pr.left}px`; canvas.style.top = `${r.top - pr.top}px`;
    canvas.style.width = `${r.width}px`; canvas.style.height = `${r.height}px`;
    canvas.width = video.videoWidth || 1; canvas.height = video.videoHeight || 1;
  }
  function drawLoop() {
    requestAnimationFrame(drawLoop);
    const d = duration();
    if (d && document.activeElement !== $("#seek")) $("#seek").value = Math.round(now() / d * 1000);
    $("#time").textContent = `${fmtTime(now())} / ${fmtTime(d)}`;
    if (!hasVideo) { if (playing && now() >= d) { pause(); offset = 0; } return; }
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!tracksData || !$("#preview-blur").checked || !video.videoWidth) return;
    const fi = Math.min(tracksData.frames.length - 1, Math.round(video.currentTime * tracksData.info.fps));
    const entries = tracksData.frames[fi] || [];
    const region = $("input[name=region]:checked").value;
    const sx = canvas.width / tracksData.info.width, sy = canvas.height / tracksData.info.height;
    ctx.save(); ctx.beginPath();
    let any = false;
    for (const [tid, x1, y1, x2, y2, fb, poly] of entries) {
      if (!decisions[String(tid)]) continue;
      any = true;
      if (region === "face") {
        const cx = (fb[0] + fb[2]) / 2 * sx, cy = (fb[1] + fb[3]) / 2 * sy;
        ctx.moveTo(cx + (fb[2] - fb[0]) * 0.8 * sx, cy);
        ctx.ellipse(cx, cy, (fb[2] - fb[0]) * 0.8 * sx, (fb[3] - fb[1]) * 0.85 * sy, 0, 0, Math.PI * 2);
      } else if (poly.length >= 6) {
        ctx.moveTo(poly[0] * sx, poly[1] * sy);
        for (let i = 2; i < poly.length; i += 2) ctx.lineTo(poly[i] * sx, poly[i + 1] * sy);
        ctx.closePath();
      } else ctx.rect(x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy);
    }
    if (any) {
      ctx.lineWidth = Math.max(8, canvas.width * 0.025); ctx.lineJoin = "round";
      ctx.clip();
      const bw = Math.max(14, Math.round(Math.min(canvas.width, canvas.height) / 11));
      pixel.width = Math.ceil(canvas.width / bw); pixel.height = Math.ceil(canvas.height / bw);
      pctx.drawImage(video, 0, 0, pixel.width, pixel.height);
      ctx.imageSmoothingEnabled = false;
      ctx.filter = `blur(${Math.round(bw / 2)}px)`;
      ctx.drawImage(pixel, 0, 0, canvas.width, canvas.height);
    }
    ctx.restore();
  }

  // ---------- التصدير والتحميل ----------
  $("#btn-export").addEventListener("click", async () => {
    pause();
    $("#btn-export").disabled = true;
    $("#export-progress").hidden = false;
    setProg(1, "في الانتظار", "#exp-bar", "#exp-msg");
    try {
      await api(`/api/jobs/${jobId}/export`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ format: $("input[name=fmt]:checked").value, gains: effGains(), decisions,
          region: $("input[name=region]:checked").value }),
      });
      poll();
    } catch (e) { $("#exp-msg").textContent = e.message; $("#btn-export").disabled = false; }
  });

  function onExported() {
    clearInterval(pollTimer);
    setProg(100, "الملف جاهز", "#exp-bar", "#exp-msg");
    const a = $("#btn-download");
    a.href = `/api/jobs/${jobId}/download`;
    a.hidden = false;
    $("#btn-export").hidden = true;
    a.addEventListener("click", () => {
      setTimeout(() => finish("تم التحميل وحذف المقطع", "بدأ تحميل ملفك. حُذف المقطع وكل نتائجه نهائيًا من خوادمنا فور انتهاء التحميل."), 800);
    }, { once: true });
  }

  $$("[data-action=delete]").forEach((b) => b.addEventListener("click", async () => {
    if (!jobId) return location.reload();
    if (!confirm("سيُحذف المقطع وكل نتائجه نهائيًا. هل أنت متأكد؟")) return;
    try { await api(`/api/jobs/${jobId}`, { method: "DELETE" }); } catch (_) {}
    finish("تم حذف المقطع", "حُذف المقطع وكل نتائجه نهائيًا من خوادمنا.");
  }));
})();
