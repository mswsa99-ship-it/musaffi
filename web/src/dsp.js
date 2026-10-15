// معالجة الإشارة: FFT بأي طول (عوامل 2، 3، 5) + STFT/iSTFT مطابق لـ torch.stft (center=True)
"use strict";

function makeFFT(n) {
  const factors = [];
  let m = n;
  for (const p of [4, 2, 3, 5]) while (m % p === 0) { factors.push(p); m /= p; }
  if (m !== 1) throw new Error("FFT size must factor into 2,3,5: " + n);
  const cos = new Float64Array(n), sin = new Float64Array(n);
  for (let i = 0; i < n; i++) { cos[i] = Math.cos(-2 * Math.PI * i / n); sin[i] = Math.sin(-2 * Math.PI * i / n); }
  const bufRe = new Float64Array(n), bufIm = new Float64Array(n);

  // Stockham autosort: ping-pong بين مصفوفتين، بدون إعادة ترتيب البتات
  function run(re, im, inverse) {
    let aRe = re, aIm = im, bRe = bufRe, bIm = bufIm;
    let l = 1, mm = n;
    const sgn = inverse ? -1 : 1;
    for (const p of factors) {
      mm /= p;
      // aRe[j*?]: input viewed as [p][mm][l] ... output [mm][p][l] (Stockham)
      for (let j = 0; j < l; j++) {
        for (let k = 0; k < mm; k++) {
          // twiddle w^(j*... ) using global table: angle = 2πi * j * q / (l*p)
          for (let q = 0; q < p; q++) {
            let sr = 0, si = 0;
            for (let r = 0; r < p; r++) {
              const idx = j + l * (k + mm * r); // input element r of butterfly
              const xr = aRe[idx], xi = aIm[idx];
              // total twiddle exponent: (j + l*q) * r * (n/(l*p))  mod n
              const e = (((j + l * q) * r) % (l * p)) * (n / (l * p));
              const c = cos[e], s = sgn * sin[e];
              sr += xr * c - xi * s; si += xr * s + xi * c;
            }
            const o = j + l * (q + p * k);
            bRe[o] = sr; bIm[o] = si;
          }
        }
      }
      [aRe, bRe] = [bRe, aRe]; [aIm, bIm] = [bIm, aIm];
      l *= p;
    }
    if (aRe !== re) { re.set(aRe); im.set(aIm); }
  }
  return { n, forward: (re, im) => run(re, im, false), inverse: (re, im) => run(re, im, true) };
}

function hannPeriodic(n) {
  const w = new Float64Array(n);
  for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / n);
  return w;
}

// STFT لقناتين معًا (L + iR في FFT واحد). يرجع مصفوفة [4][F][T] بالترتيب: L.re, L.im, R.re, R.im
function stft2(L, R, nfft, hop, dimF, fft, win) {
  const len = L.length, half = nfft / 2;
  const T = 1 + Math.floor(len / hop);
  const out = new Float32Array(4 * dimF * T);
  const re = new Float64Array(nfft), im = new Float64Array(nfft);
  const reflect = (x, i) => { // padding بالانعكاس مثل torch
    if (i < 0) i = -i; if (i >= len) i = 2 * (len - 1) - i; return x[i];
  };
  const FT = dimF * T;
  for (let t = 0; t < T; t++) {
    const start = t * hop - half;
    for (let i = 0; i < nfft; i++) {
      const s = start + i;
      const w = win[i];
      if (s >= 0 && s < len) { re[i] = L[s] * w; im[i] = R[s] * w; }
      else { re[i] = reflect(L, s) * w; im[i] = reflect(R, s) * w; }
    }
    fft.forward(re, im);
    for (let f = 0; f < dimF; f++) {
      const nf = f === 0 ? 0 : nfft - f;
      const zr = re[f], zi = im[f], cr = re[nf], ci = -im[nf];
      // X_L = (Z + conj Z[-k]) / 2 ، X_R = (Z - conj Z[-k]) / 2i
      out[0 * FT + f * T + t] = (zr + cr) / 2;
      out[1 * FT + f * T + t] = (zi + ci) / 2;
      out[2 * FT + f * T + t] = (zi - ci) / 2;
      out[3 * FT + f * T + t] = -(zr - cr) / 2;
    }
  }
  return { data: out, T };
}

// iSTFT لقناتين: يستقبل [4][F][T] (F <= nfft/2+1، الباقي أصفار) ويرجع [L, R] بطول hop*(T-1)
function istft2(spec, dimF, T, nfft, hop, fft, win) {
  const half = nfft / 2, FT = dimF * T;
  const full = nfft + hop * (T - 1);
  const yL = new Float64Array(full), yR = new Float64Array(full), env = new Float64Array(full);
  const re = new Float64Array(nfft), im = new Float64Array(nfft);
  for (let t = 0; t < T; t++) {
    re.fill(0); im.fill(0);
    for (let f = 0; f < dimF && f <= half; f++) {
      const lr = spec[f * T + t], li = spec[FT + f * T + t];
      const rr = spec[2 * FT + f * T + t], ri = spec[3 * FT + f * T + t];
      // Z[k] = X_L[k] + i X_R[k]
      re[f] = lr - ri; im[f] = li + rr;
      if (f > 0 && f < half) { // الجزء المرآة: conj(X_L) + i conj(X_R)
        re[nfft - f] = lr + ri; im[nfft - f] = -li + rr;
      }
    }
    // torch.istft يتجاهل الجزء التخيلي لـ DC وNyquist — نصفّره للمطابقة
    fft.inverse(re, im);
    const off = t * hop;
    for (let i = 0; i < nfft; i++) {
      const w = win[i];
      yL[off + i] += re[i] / nfft * w;
      yR[off + i] += im[i] / nfft * w;
      env[off + i] += w * w;
    }
  }
  const n = hop * (T - 1);
  const L = new Float32Array(n), R = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const e = env[i + half];
    const d = e > 1e-11 ? e : 1;
    L[i] = yL[i + half] / d; R[i] = yR[i + half] / d;
  }
  return [L, R];
}

if (typeof module !== "undefined") module.exports = { makeFFT, hannPeriodic, stft2, istft2 };
