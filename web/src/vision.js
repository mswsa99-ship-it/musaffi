// الرؤية: فك مخرجات YOLO11-seg وSCRFD وgenderage، وتتبع الأشخاص عبر الزمن
"use strict";

const YOLO_SIZE = 384, FACE_SIZE = 640, MASK_RES = 32;

function iou(a, b) {
  const x1 = Math.max(a[0], b[0]), y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[2], b[2]), y2 = Math.min(a[3], b[3]);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
  return u > 0 ? inter / u : 0;
}

function nms(dets, thr) {
  dets.sort((a, b) => b.score - a.score);
  const keep = [];
  for (const d of dets) if (keep.every((k) => iou(k.box, d.box) < thr)) keep.push(d);
  return keep;
}

// out0: [1, 116, N] (4 مربع + 80 فئة + 32 معامل قناع)، out1: [1, 32, P, P]
// scale: نسبة تصغير الصورة الأصلية إلى 384 (الصورة موضوعة أعلى اليسار)
function yoloDecode(out0, out1, n, P, scale, imgW, imgH, conf = 0.3) {
  const dets = [];
  for (let i = 0; i < n; i++) {
    const score = out0[4 * n + i]; // الفئة 0 = شخص
    if (score < conf) continue;
    const cx = out0[i], cy = out0[n + i], w = out0[2 * n + i], h = out0[3 * n + i];
    const box = [(cx - w / 2) / scale, (cy - h / 2) / scale, (cx + w / 2) / scale, (cy + h / 2) / scale];
    box[0] = Math.max(0, box[0]); box[1] = Math.max(0, box[1]);
    box[2] = Math.min(imgW, box[2]); box[3] = Math.min(imgH, box[3]);
    if (box[2] - box[0] < 4 || box[3] - box[1] < 4) continue;
    const coef = new Float32Array(32);
    for (let k = 0; k < 32; k++) coef[k] = out0[(84 + k) * n + i];
    dets.push({ box, score, coef });
  }
  const kept = nms(dets, 0.5);
  // قناع الشخص بدقة 32×32 داخل مربعه
  const PP = P * P, ratio = P / YOLO_SIZE;
  for (const d of kept) {
    const m = new Uint8Array(MASK_RES * MASK_RES);
    const [x1, y1, x2, y2] = d.box;
    for (let my = 0; my < MASK_RES; my++) {
      const py = Math.min(P - 1, Math.max(0, Math.floor((y1 + (my + 0.5) * (y2 - y1) / MASK_RES) * scale * ratio)));
      for (let mx = 0; mx < MASK_RES; mx++) {
        const px = Math.min(P - 1, Math.max(0, Math.floor((x1 + (mx + 0.5) * (x2 - x1) / MASK_RES) * scale * ratio)));
        let s = 0;
        const o = py * P + px;
        for (let k = 0; k < 32; k++) s += d.coef[k] * out1[k * PP + o];
        m[my * MASK_RES + mx] = s > 0 ? 255 : 0; // sigmoid(s) > 0.5
      }
    }
    d.mask = m;
    delete d.coef;
  }
  return kept;
}

// SCRFD (det_500m): مخرجات 9 بالترتيب: درجات (8،16،32)، مربعات، نقاط الوجه
function scrfdDecode(outs, inputSize, scale, thr = 0.45) {
  const strides = [8, 16, 32], dets = [];
  strides.forEach((s, si) => {
    const scores = outs[si], boxes = outs[3 + si];
    const W = inputSize / s;
    const count = scores.length;
    for (let i = 0; i < count; i++) {
      const sc = scores[i];
      if (sc < thr) continue;
      const loc = Math.floor(i / 2);
      const cx = (loc % W) * s, cy = Math.floor(loc / W) * s;
      const b = [cx - boxes[4 * i] * s, cy - boxes[4 * i + 1] * s, cx + boxes[4 * i + 2] * s, cy + boxes[4 * i + 3] * s].map((v) => v / scale);
      dets.push({ box: b, score: sc });
    }
  });
  return nms(dets, 0.4);
}

function genderPFemale(logits) {
  const a = logits[0], b = logits[1], m = Math.max(a, b);
  const ea = Math.exp(a - m), eb = Math.exp(b - m);
  return ea / (ea + eb); // 0 = أنثى ، 1 = ذكر
}

// متتبع بسيط بالتداخل (IoU) مناسب لعينات كل 1/6 ثانية
class Tracker {
  constructor() { this.next = 1; this.active = []; this.tracks = new Map(); }
  update(t, dets) {
    const pairs = [];
    this.active.forEach((tr, ti) => dets.forEach((d, di) => {
      const v = iou(tr.box, d.box);
      if (v > 0.2) pairs.push([v, ti, di]);
    }));
    pairs.sort((a, b) => b[0] - a[0]);
    const usedT = new Set(), usedD = new Set();
    for (const [, ti, di] of pairs) {
      if (usedT.has(ti) || usedD.has(di)) continue;
      usedT.add(ti); usedD.add(di);
      const tr = this.active[ti];
      tr.box = dets[di].box; tr.last = t;
      tr.samples.push({ t, box: dets[di].box, mask: dets[di].mask });
      dets[di].tid = tr.id;
    }
    dets.forEach((d, di) => {
      if (usedD.has(di)) return;
      const tr = { id: this.next++, box: d.box, last: t, samples: [{ t, box: d.box, mask: d.mask }], faces: [], votes: [] };
      this.active.push(tr); this.tracks.set(tr.id, tr);
      d.tid = tr.id;
    });
    this.active = this.active.filter((tr) => t - tr.last <= 1.5);
  }
  addFace(t, faceBox, pf, detScore, dets) {
    const cx = (faceBox[0] + faceBox[2]) / 2, cy = (faceBox[1] + faceBox[3]) / 2;
    let best = null, area = Infinity;
    for (const d of dets) {
      const [x1, y1, x2, y2] = d.box;
      if (cx >= x1 && cx <= x2 && cy >= y1 && cy <= y1 + 0.6 * (y2 - y1)) {
        const a = (x2 - x1) * (y2 - y1);
        if (a < area) { area = a; best = d; }
      }
    }
    if (!best) return false;
    const tr = this.tracks.get(best.tid);
    const [x1, y1, x2, y2] = best.box, w = x2 - x1, h = y2 - y1;
    tr.faces.push({ t, rel: [(faceBox[0] - x1) / w, (faceBox[1] - y1) / h, (faceBox[2] - x1) / w, (faceBox[3] - y1) / h] });
    const size = Math.min(faceBox[2] - faceBox[0], faceBox[3] - faceBox[1]);
    if (pf !== null && size >= 20) tr.votes.push([pf, detScore * Math.min(1, size / 60)]);
    return true;
  }
  summary() {
    const out = {};
    for (const [id, tr] of this.tracks) {
      const ws = tr.votes.reduce((s, v) => s + v[1], 0);
      out[id] = {
        first: tr.samples[0].t, last: tr.samples[tr.samples.length - 1].t, count: tr.samples.length,
        votes: tr.votes.length, p_female: ws > 0 ? tr.votes.reduce((s, v) => s + v[0] * v[1], 0) / ws : null,
      };
    }
    return out;
  }
}

function decide(summary, mode, sensitivity) {
  const out = {};
  const thr = sensitivity === "cautious" ? 0.35 : 0.5;
  for (const [id, s] of Object.entries(summary)) {
    if (mode === "all") out[id] = true;
    else if (mode === "none") out[id] = false;
    else if (s.p_female === null || s.votes < 2) out[id] = sensitivity === "cautious";
    else out[id] = s.p_female >= thr;
  }
  return out;
}

// موقع الشخص في لحظة t: استيفاء بين أقرب عينتين، مع إبقائه 0.6 ثانية بعد آخر ظهور
function trackAt(tr, t, hold = 0.6) {
  const s = tr.samples;
  if (t < s[0].t - hold || t > s[s.length - 1].t + hold) return null;
  let lo = 0, hi = s.length - 1;
  while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (s[mid].t <= t) lo = mid; else hi = mid; }
  const a = s[lo], b = s[hi];
  let box, mask;
  if (t <= a.t || a === b) { box = a.box; mask = a.mask; }
  else if (t >= b.t) { box = b.box; mask = b.mask; }
  else {
    if (b.t - a.t > 1.5) { // فجوة طويلة: نستخدم الأقرب
      const near = t - a.t < b.t - t ? a : b;
      if (Math.abs(t - near.t) > hold) return null;
      return { box: near.box, mask: near.mask };
    }
    const k = (t - a.t) / (b.t - a.t);
    box = a.box.map((v, i) => v + (b.box[i] - v) * k);
    mask = k < 0.5 ? a.mask : b.mask;
  }
  return { box, mask };
}

function faceBoxAt(tr, t, box) {
  const [x1, y1, x2, y2] = box, w = x2 - x1, h = y2 - y1;
  if (tr.faces.length) {
    let best = tr.faces[0];
    for (const f of tr.faces) if (Math.abs(f.t - t) < Math.abs(best.t - t)) best = f;
    const r = best.rel;
    return [x1 + r[0] * w, y1 + r[1] * h, x1 + r[2] * w, y1 + r[3] * h];
  }
  const hw = Math.min(w, h * 0.3) * 0.8, cx = (x1 + x2) / 2;
  return [cx - hw / 2, y1, cx + hw / 2, y1 + hw * 1.25];
}

if (typeof module !== "undefined") module.exports = { YOLO_SIZE, FACE_SIZE, MASK_RES, iou, nms, yoloDecode, scrfdDecode, genderPFemale, Tracker, decide, trackAt, faceBoxAt };
