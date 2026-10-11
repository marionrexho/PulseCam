// Neural pulse model (EfficientPhys from rPPG-Toolbox) running in the browser through ONNX Runtime Web.

import { FS, HR_LO, HR_HI, ORT_URL, NN_URL } from "./config.js";
import { bandpass, detrend, fft, ifft, nextPow2 } from "./dsp.js";
import { setNN } from "./dom.js";

// The network sees 72x72 face crops at 30 fps, 10 frames at a time (+1 earlier frame for the frame difference).
// Groups of 10 are independent, so each new group is run as soon as it is complete.
export const NN_SIZE = 72, NN_PX = NN_SIZE * NN_SIZE;
const NN_GROUP = 10;
const NN_STAT_FRAMES = 150, NN_MIN_FRAMES = 90, NN_KEEP = 320, NN_OUT_KEEP = 1200;

export function nnReset(nn) {
  Object.assign(nn, { frames: [], base: 0, t0: 0, nextT: null, outs: [], outBase: 1, nextGroup: 0, epoch: (nn.epoch || 0) + 1 });
}

// Camera frames arrive at irregular times; the model expects a steady 30 fps clip, so every 1/FS slot gets the newest frame.
export function nnPushFrame(nn, t, fr) {
  // short stalls are bridged by repeating the last frame; a long gap (face lost) starts a fresh clip
  if (nn.nextT == null || t - nn.nextT > 1.0) { nnReset(nn); nn.t0 = t; nn.nextT = t; }
  while (nn.nextT <= t + 1e-6) { nn.frames.push(fr); nn.nextT += 1 / FS; }
  while (nn.frames.length > NN_KEEP) { nn.frames.shift(); nn.base++; }
}

// Z-score over the recent clip (the toolbox standardises each clip the same way); subsampled for speed
function nnStats(nn) {
  const from = Math.max(0, nn.frames.length - NN_STAT_FRAMES);
  let s = 0, c = 0;
  for (let f = from; f < nn.frames.length; f++) { const fr = nn.frames[f]; for (let i = 0; i < fr.length; i += 5) { s += fr[i]; c++; } }
  const m = s / (c || 1); let v = 0;
  for (let f = from; f < nn.frames.length; f++) { const fr = nn.frames[f]; for (let i = 0; i < fr.length; i += 5) v += (fr[i] - m) ** 2; }
  return { mean: m, sd: Math.sqrt(v / (c || 1)) || 1 };
}

export async function nnRun(nn, maxGroups = 8) {
  if (!nn.session || nn.busy || nn.frames.length < NN_MIN_FRAMES) return 0;
  nn.busy = true; const epoch = nn.epoch; let done = 0;
  try {
    const { mean, sd } = nnStats(nn), L = 3 * NN_PX;
    while (done < maxGroups) {
      const a = nn.nextGroup * NN_GROUP;
      if (a < nn.base) { nnReset(nn); break; }                       // inference fell too far behind; start over
      if (a + NN_GROUP - nn.base >= nn.frames.length) break;         // group not complete yet
      const x = new Float32Array((NN_GROUP + 1) * L);
      for (let j = 0; j <= NN_GROUP; j++) { const fr = nn.frames[a + j - nn.base]; for (let i = 0; i < L; i++) x[j * L + i] = (fr[i] - mean) / sd; }
      const t0 = performance.now();
      const r = await nn.session.run({ frames: new nn.Tensor("float32", x, [NN_GROUP + 1, 3, NN_SIZE, NN_SIZE]) });
      if (nn.epoch !== epoch) break;                                 // stream was reset while waiting
      nn.ms = nn.ms ? 0.8 * nn.ms + 0.2 * (performance.now() - t0) : performance.now() - t0;
      const y = r.pulse.data;
      for (let j = 0; j < NN_GROUP; j++) nn.outs.push(y[j]);
      nn.nextGroup++; done++;
      while (nn.outs.length > NN_OUT_KEEP) { nn.outs.shift(); nn.outBase++; }
    }
  } finally { nn.busy = false; }
  return done;
}

// The model outputs the pulse's first difference. Fully integrating it (cumulative sum) tilts the spectrum so that
// plain noise looks like a strong slow peak (~45 bpm), and a weak signal then drags the estimate down to the bottom
// of the band. Dividing by sqrt(frequency) instead (a "half integration") keeps real pulses intact but removes that bias.
function halfIntegrate(x, fs) {
  const n = x.length, N = nextPow2(2 * n), re = new Float64Array(N), im = new Float64Array(N), d = detrend(x);
  for (let i = 0; i < n; i++) { re[i] = d[i]; re[2 * n - 1 - i] = d[i]; }
  fft(re, im);
  for (let k = 0; k < N; k++) { const g = 1 / Math.sqrt(Math.max(Math.min(k, N - k) * fs / N, 0.3)); re[k] *= g; im[k] *= g; }
  ifft(re, im);
  return re.slice(0, n);
}
export function nnWave(nn, winSec) {
  const n = Math.min(nn.outs.length, Math.round(winSec * FS));
  if (n < 4 * FS) return null;
  const seg = Float64Array.from(nn.outs.slice(-n));
  const first = nn.outBase + nn.outs.length - n;
  return { pulse: bandpass(halfIntegrate(seg, FS), FS, HR_LO, HR_HI), tStart: nn.t0 + first / FS, tEnd: nn.t0 + (first + n - 1) / FS };
}

// Shared model state: the ONNX session plus the rolling frame/output buffers (see nnReset)
export const NN = { session: null, Tensor: null, loading: null, box: null, ms: 0, luma: null, error: null };
nnReset(NN);

// Load onnxruntime-web and the model (starts as soon as the page opens)
export function ensureNeural() {
  if (NN.session) return Promise.resolve(true);
  if (NN.loading) return NN.loading;
  NN.loading = (async () => {
    try {
      NN.error = null; setNN("Loading neural model…");
      const ort = await import(ORT_URL + "ort.wasm.min.mjs");
      ort.env.wasm.wasmPaths = ORT_URL;
      ort.env.wasm.proxy = true;   // run the model in a background worker so it never freezes the camera loop
      const res = await fetch(NN_URL);
      if (!res.ok) throw new Error("model file not found (HTTP " + res.status + ")");
      NN.session = await ort.InferenceSession.create(await res.arrayBuffer(), { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
      NN.Tensor = ort.Tensor;
      setNN("Neural model ready");
      return true;
    } catch (e) {
      NN.error = location.protocol === "file:" ? "open this page through the local server (start-server.bat)" : e.message;
      setNN("Neural model unavailable: " + NN.error);
      return false;
    } finally { NN.loading = null; }
  })();
  return NN.loading;
}
