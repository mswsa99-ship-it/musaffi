// فصل الصوت البشري عن الموسيقى بنموذج UVR-MDX-NET-Voc_FT — نسخة مطابقة لما في audio-separator
"use strict";

const MDX = { nfft: 7680, hop: 1024, dimF: 3072, dimT: 256, compensate: 1.021 };

async function separateVocals(ort, session, L, R, opts = {}) {
  const { makeFFT, hannPeriodic, stft2, istft2 } = opts.dsp;
  const overlap = opts.overlap ?? 0.2;
  const onProgress = opts.onProgress || (() => {});
  const { nfft, hop, dimF, dimT } = MDX;
  const fft = makeFFT(nfft), win = hannPeriodic(nfft);
  const trim = nfft / 2;
  const chunk = hop * (dimT - 1);
  const gen = chunk - 2 * trim;
  const n = L.length;
  const pad = gen + trim - (n % gen);
  const total = trim + n + pad;
  const mL = new Float32Array(total), mR = new Float32Array(total);
  mL.set(L, trim); mR.set(R, trim);
  const resL = new Float32Array(total), resR = new Float32Array(total), div = new Float32Array(total);
  const step = Math.max(1, Math.floor((1 - overlap) * chunk));
  const starts = [];
  for (let i = 0; i < total; i += step) starts.push(i);
  const inName = session.inputNames[0], outName = session.outputNames[0];

  for (let c = 0; c < starts.length; c++) {
    const start = starts[c];
    const end = Math.min(start + chunk, total);
    const actual = end - start;
    const pL = new Float32Array(chunk), pR = new Float32Array(chunk);
    pL.set(mL.subarray(start, end)); pR.set(mR.subarray(start, end));
    const { data } = stft2(pL, pR, nfft, hop, dimF, fft, win);
    // تصفير أول 3 ترددات (كما في UVR)
    const FT = dimF * dimT;
    for (let ch = 0; ch < 4; ch++) data.fill(0, ch * FT, ch * FT + 3 * dimT);
    const out = await session.run({ [inName]: new ort.Tensor("float32", data, [1, 4, dimF, dimT]) });
    const spec = out[outName].data;
    const [vL, vR] = istft2(spec, dimF, dimT, nfft, hop, fft, win);
    for (let i = 0; i < actual; i++) {
      // نافذة hanning (numpy) لدمج الأجزاء المتداخلة بنعومة
      const w = overlap > 0 ? (actual > 1 ? 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (actual - 1)) : 1) : 1;
      resL[start + i] += vL[i] * w; resR[start + i] += vR[i] * w; div[start + i] += w;
    }
    onProgress((c + 1) / starts.length);
    await new Promise((r) => setTimeout(r, 0));
  }
  const oL = new Float32Array(n), oR = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const d = div[i + trim] || 1;
    oL[i] = resL[i + trim] / d; oR[i] = resR[i + trim] / d;
  }
  return [oL, oR];
}

// فصل الضوضاء الثابتة (طنين، مكيف) عن الكلام بتقدير أرضية الضوضاء لكل تردد
function splitStationaryNoise(L, R, dsp, sr) {
  const { makeFFT, hannPeriodic } = dsp;
  const nfft = 2048, hop = 512, fft = makeFFT(nfft), win = hannPeriodic(nfft);
  const n = L.length, half = nfft / 2;
  const T = Math.max(1, Math.ceil(n / hop));
  const re = new Float64Array(nfft), im = new Float64Array(nfft);
  const mags = new Float32Array(T * (half + 1));
  const frame = (t) => {
    const s0 = t * hop - half;
    for (let i = 0; i < nfft; i++) {
      const s = s0 + i, ok = s >= 0 && s < n;
      re[i] = ok ? L[s] * win[i] : 0; im[i] = ok ? R[s] * win[i] : 0;
    }
    fft.forward(re, im);
  };
  for (let t = 0; t < T; t++) {
    frame(t);
    for (let f = 0; f <= half; f++) {
      const nf = f === 0 ? 0 : nfft - f;
      const lr = (re[f] + re[nf]) / 2, li = (im[f] - im[nf]) / 2;
      const rr = (im[f] + im[nf]) / 2, ri = -(re[f] - re[nf]) / 2;
      mags[t * (half + 1) + f] = Math.sqrt(lr * lr + li * li + rr * rr + ri * ri);
    }
  }
  // أرضية الضوضاء = النسبة المئوية 15 لكل تردد عبر الزمن
  const floor = new Float32Array(half + 1), col = new Float32Array(T);
  for (let f = 0; f <= half; f++) {
    for (let t = 0; t < T; t++) col[t] = mags[t * (half + 1) + f];
    const s = Array.from(col).sort((a, b) => a - b);
    floor[f] = s[Math.floor(0.15 * (T - 1))];
  }
  const nL = new Float64Array(n + nfft), nR = new Float64Array(n + nfft), env = new Float64Array(n + nfft);
  for (let t = 0; t < T; t++) {
    frame(t);
    for (let f = 0; f <= half; f++) {
      const m = mags[t * (half + 1) + f];
      const g = m > 0 ? Math.min(1, (1.5 * floor[f]) / m) : 0; // نسبة الضوضاء في هذه الخلية
      re[f] *= g; im[f] *= g;
      if (f > 0 && f < half) { re[nfft - f] *= g; im[nfft - f] *= g; }
    }
    fft.inverse(re, im);
    const s0 = t * hop - half;
    for (let i = 0; i < nfft; i++) {
      const s = s0 + i;
      if (s < 0 || s >= n) continue;
      nL[s] += re[i] / nfft * win[i]; nR[s] += im[i] / nfft * win[i]; env[s] += win[i] * win[i];
    }
  }
  const noiseL = new Float32Array(n), noiseR = new Float32Array(n), cleanL = new Float32Array(n), cleanR = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const e = env[i] > 1e-8 ? env[i] : 1;
    noiseL[i] = nL[i] / e; noiseR[i] = nR[i] / e;
    cleanL[i] = L[i] - noiseL[i]; cleanR[i] = R[i] - noiseR[i];
  }
  return { clean: [cleanL, cleanR], noise: [noiseL, noiseR] };
}

if (typeof module !== "undefined") module.exports = { MDX, separateVocals, splitStationaryNoise };
