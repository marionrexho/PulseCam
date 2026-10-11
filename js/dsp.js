// Signal processing: resampling, FFT, filtering, spectral analysis, peak finding and HRV.

export const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
export function mean(x) { let s = 0; for (let i = 0; i < x.length; i++) s += x[i]; return s / (x.length || 1); }
export function std(x) { const m = mean(x); let s = 0; for (let i = 0; i < x.length; i++) s += (x[i] - m) ** 2; return Math.sqrt(s / (x.length || 1)); }
export function median(a) { const s = [...a].sort((p, q) => p - q); const n = s.length; return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2; }
export const nextPow2 = n => 2 ** Math.ceil(Math.log2(Math.max(2, n)));

// Linear interpolation of irregular camera timestamps onto a uniform FS grid
export function resampleSeries(ts, vs, tStart, tEnd, fs) {
  const N = Math.floor((tEnd - tStart) * fs) + 1, out = new Float64Array(N);
  let j = 0;
  for (let i = 0; i < N; i++) {
    const t = tStart + i / fs;
    while (j < ts.length - 2 && ts[j + 1] < t) j++;
    const t0 = ts[j], t1 = ts[j + 1];
    const a = t1 > t0 ? clamp((t - t0) / (t1 - t0), 0, 1) : 0;
    out[i] = vs[j] + (vs[j + 1] - vs[j]) * a;
  }
  return out;
}

// In-place radix-2 complex FFT
export function fft(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = -2 * Math.PI / len, wr = Math.cos(ang), wi = Math.sin(ang), half = len >> 1;
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < half; k++) {
        const a = i + k, b = a + half;
        const xr = re[b] * cr - im[b] * ci, xi = re[b] * ci + im[b] * cr;
        re[b] = re[a] - xr; im[b] = im[a] - xi; re[a] += xr; im[a] += xi;
        const t = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = t;
      }
    }
  }
}
export function ifft(re, im) {
  const n = re.length;
  for (let i = 0; i < n; i++) im[i] = -im[i];
  fft(re, im);
  for (let i = 0; i < n; i++) { re[i] /= n; im[i] = -im[i] / n; }
}

export function detrend(x) {
  const n = x.length; let sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (let i = 0; i < n; i++) { sx += i; sy += x[i]; sxx += i * i; sxy += i * x[i]; }
  const d = n * sxx - sx * sx, b = d ? (n * sxy - sx * sy) / d : 0, a = (sy - b * sx) / n;
  const o = new Float64Array(n);
  for (let i = 0; i < n; i++) o[i] = x[i] - (a + b * i);
  return o;
}

export function bandGain(f, lo, hi, w) {
  if (f < lo - w || f > hi + w) return 0;
  if (f < lo) return 0.5 - 0.5 * Math.cos(Math.PI * (f - (lo - w)) / w);
  if (f > hi) return 0.5 + 0.5 * Math.cos(Math.PI * (f - hi) / w);
  return 1;
}

// Zero-phase FFT band-pass with raised-cosine edges and mirror padding
export function bandpass(x, fs, lo, hi, w = 0.15) {
  const n = x.length, N = nextPow2(2 * n);
  const re = new Float64Array(N), im = new Float64Array(N), d = detrend(x);
  for (let i = 0; i < n; i++) { re[i] = d[i]; re[2 * n - 1 - i] = d[i]; }
  fft(re, im);
  for (let k = 0; k < N; k++) { const g = bandGain(Math.min(k, N - k) * fs / N, lo, hi, w); re[k] *= g; im[k] *= g; }
  ifft(re, im);
  return re.slice(0, n);
}

export function powerSpectrum(x, nfft) {
  const re = new Float64Array(nfft), im = new Float64Array(nfft);
  const n = Math.min(x.length, nfft), off = x.length - n, m = mean(x.slice(off));
  for (let i = 0; i < n; i++) re[i] = (x[off + i] - m) * (0.5 - 0.5 * Math.cos(2 * Math.PI * i / (n - 1)));
  fft(re, im);
  const P = new Float64Array(nfft / 2);
  for (let k = 0; k < nfft / 2; k++) P[k] = re[k] * re[k] + im[k] * im[k];
  return P;
}

export function argmaxRange(P, a, b) { let k = a; for (let i = a; i <= b; i++) if (P[i] > P[k]) k = i; return k; }
export function parabolic(P, k) {
  if (k <= 0 || k >= P.length - 1) return 0;
  const a = P[k - 1], b = P[k], c = P[k + 1], den = a - 2 * b + c;
  return den ? clamp(0.5 * (a - c) / den, -0.5, 0.5) : 0;
}
// SNR (de Haan 2013): power near f0 and its harmonic vs. the rest of the band
export function snrAt(P, f0, df, kLo, kHi) {
  let s = 0, nz = 0;
  for (let k = kLo; k <= kHi; k++) {
    const f = k * df;
    if (Math.abs(f - f0) <= 0.1 || Math.abs(f - 2 * f0) <= 0.2) s += P[k]; else nz += P[k];
  }
  return 10 * Math.log10((s || 1e-12) / (nz || 1e-12));
}

export function findPeaks(x, fs, hr) {
  const minDist = Math.round(0.6 * 60 / hr * fs), idx = [];
  for (let i = 1; i < x.length - 1; i++) {
    if (x[i] > x[i - 1] && x[i] >= x[i + 1] && x[i] > 0) {
      const last = idx[idx.length - 1];
      if (idx.length && i - last < minDist) { if (x[i] > x[last]) idx[idx.length - 1] = i; }
      else idx.push(i);
    }
  }
  return idx.map(i => i + parabolic(x, i)); // sub-sample peak timing
}

export function hrvFromPeaks(pk, fs) {
  if (pk.length < 6) return null;
  const ibi = [];
  for (let i = 1; i < pk.length; i++) ibi.push((pk[i] - pk[i - 1]) / fs * 1000);
  const med = median(ibi), ok = ibi.map(v => Math.abs(v - med) < 0.25 * med);
  const good = ibi.filter((_, i) => ok[i]);
  if (good.length < 5) return null;
  let ss = 0, c = 0;
  for (let i = 1; i < ibi.length; i++) if (ok[i] && ok[i - 1]) { const d = ibi[i] - ibi[i - 1]; ss += d * d; c++; }
  if (c < 3) return null;
  return { rmssd: Math.sqrt(ss / c), sdnn: std(good) };
}
