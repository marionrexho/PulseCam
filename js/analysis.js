// Turns the neural model's output into heart rate, signal quality, HRV and respiration.

import { FS, HR_LO, HR_HI, HR_WIN, HRV_WIN, RESP_WIN, MIN_SEC, MIN_KEEP, MOTION_OK, MOTION_WARN } from "./config.js";
import { $, setStatus, setNN, parseRef } from "./dom.js";
import { bandpass, clamp, mean, resampleSeries, powerSpectrum, parabolic, argmaxRange, snrAt, findPeaks, hrvFromPeaks } from "./dsp.js";
import { NN, nnRun, nnWave } from "./neural.js";
import { session, samples, history, state } from "./state.js";
import { updateUI, drawPlots } from "./ui.js";

function series(sub, fn, tStart, tEnd) { return resampleSeries(sub.map(s => s.t), sub.map(fn), tStart, tEnd, FS); }

// Pulse waveform from the neural model, plus the fraction of the window without head motion
function computePulse(winSec) {
  const nw = NN.session && nnWave(NN, winSec); if (!nw) return null;
  const win = samples.filter(s => s.t >= nw.tStart && s.t <= nw.tEnd);
  return { tStart: nw.tStart, tEnd: nw.tEnd, pulse: nw.pulse, keep: win.length ? win.filter(s => !s.bad).length / win.length : 0 };
}

// Runs every 100 ms while the camera is on
export async function analyze() {
  if (!session.running) return;
  const now = performance.now() / 1000;
  if (now - session.lastFace > 1) {
    samples.length = 0; state.conf = 0;
    setStatus("No face detected — center your face in good light", "warn"); updateUI(); return;
  }
  if (samples.length < 2) return;
  const span = samples[samples.length - 1].t - samples[0].t;
  const recent = samples.filter(s => s.t > now - 2);
  state.motion = recent.length ? recent.reduce((a, s) => a + s.motion, 0) / recent.length : 0;
  $("motion").style.display = state.motion > MOTION_WARN ? "block" : "none";
  if (!NN.session) { setStatus(NN.error ? "Neural model unavailable — see the message below the video" : "Loading neural model…", NN.error ? "warn" : ""); return; }
  if (span < MIN_SEC) { setStatus(`Collecting signal… ${Math.floor(span)} / ${MIN_SEC} s — stay still`); return; }

  await nnRun(NN);
  if (NN.ms) setNN(`Neural model · ${NN.ms.toFixed(0)} ms per 10 frames`);
  const res = computePulse(HR_WIN);
  if (!res) { setStatus("Collecting frames for the neural model…"); updateUI(); return; }
  state.keep = res.keep;
  const NFFT = 2048, df = FS / NFFT, kLo = Math.ceil(HR_LO / df), kHi = Math.floor(HR_HI / df);

  // Spectrum of the pulse, normalized to unit power in-band
  const fused = powerSpectrum(res.pulse, NFFT); let tot = 0;
  for (let k = kLo; k <= kHi; k++) tot += fused[k];
  for (let k = 0; k < fused.length; k++) fused[k] /= (tot || 1);

  // Peak pick. A true pulse has energy at 2×f0 too, so each bin earns a bonus from its harmonic;
  // that stops a noise bump or the 2nd harmonic from beating the real fundamental.
  // Resting heart rates are mostly 50–120, so bins near the ends of the band are down-weighted.
  // A soft tracking prior around the current estimate resists spurious jumps.
  const physio = bpm => Math.min(clamp((bpm - 40) / 15, 0.4, 1), clamp((200 - bpm) / 40, 0.5, 1));
  let best = kLo, bv = -1;
  for (let k = kLo; k <= kHi; k++) {
    let v = (fused[k] + 0.4 * (2 * k < fused.length ? fused[2 * k] : 0)) * physio(k * df * 60);
    if (state.hr && state.conf > 0.3) { const d = (k * df * 60 - state.hr) / 12; v *= 0.25 + 0.75 * Math.exp(-0.5 * d * d); }
    if (v > bv) { bv = v; best = k; }
  }
  const f0 = (best + parabolic(fused, best)) * df, hrRaw = f0 * 60;
  const snr = snrAt(fused, f0, df, kLo, kHi);
  const motionFactor = clamp(1 - (state.motion - MOTION_OK) / (2 * MOTION_WARN - MOTION_OK), 0, 1);
  const keepFactor = res.keep < MIN_KEEP ? 0 : clamp((res.keep - MIN_KEEP) / (0.9 - MIN_KEEP), 0, 1);
  // A "peak" sitting at the very edge of the band is almost always drift or noise, not a pulse
  const edgeFactor = clamp((hrRaw - 46) / 8, 0, 1) * clamp((205 - hrRaw) / 10, 0, 1);
  const conf = clamp((snr + 5) / 8, 0, 1) * motionFactor * keepFactor * edgeFactor;

  trackHR(hrRaw, conf, now);

  const wave = res.pulse;
  Object.assign(state, { hrRaw, snr, conf, fused, wave, df });

  // Fit the beat phase (least-squares sinusoid at f0) so the overlay and heart beat in sync
  const w0 = 2 * Math.PI * f0, n = wave.length, i0 = Math.max(0, n - 4 * FS), i1 = n - FS;
  let a = 0, b = 0;
  for (let i = i0; i < i1; i++) { const t = res.tStart + i / FS; a += wave[i] * Math.cos(w0 * t); b += wave[i] * Math.sin(w0 * t); }
  state.omega = w0; state.phi = Math.atan2(b, a);

  if (now - session.lastSlow >= 1) { session.lastSlow = now; slowMetrics(span); pushHistory(); }
  if (res.keep < MIN_KEEP) setStatus("Too much movement — hold still", "warn");
  else if (state.hr && conf < 0.3) setStatus("Weak signal — holding last value", "warn");
  else if (!state.hr && state.motion > MOTION_WARN) setStatus("Head is moving — hold still", "warn");
  else if (!state.hr && snr < -1) setStatus("Weak pulse signal — add light, sit still, face the camera", "warn");
  else if (!state.hr && edgeFactor < 0.5) setStatus("No clear pulse yet — keep still", "warn");
  else setStatus(state.hr ? "Tracking" : "Locking on…", state.hr ? "live" : "");
  updateUI(); drawPlots();
}

// Steadies the displayed heart rate: only good estimates count, small changes are followed slowly, and a big change
// (more than 6 bpm) must keep showing up for 5 s before it is believed. The first lock needs 3 s of agreement.
function trackHR(hrRaw, conf, now) {
  if (conf < (state.hr == null ? 0.25 : 0.2)) { state.cand = null; return; }
  const near = state.cand && Math.abs(hrRaw - state.cand.v) <= 5;
  if (state.hr == null) {
    if (!near) state.cand = { v: hrRaw, t: now }; else state.cand.v = hrRaw;
    if (now - state.cand.t >= 3) { state.hr = hrRaw; state.cand = null; }
    return;
  }
  const d = hrRaw - state.hr;
  if (Math.abs(d) <= 6) { state.hr += (0.015 + 0.035 * conf) * d; state.cand = null; return; }
  if (!near) state.cand = { v: hrRaw, t: now }; else state.cand.v = hrRaw;
  if (now - state.cand.t >= 5) { state.hr = hrRaw; state.cand = null; }
}

// HRV and respiration change slowly, so they are only recomputed once per second
function slowMetrics(span) {
  if (span >= 15 && state.hr && state.conf > 0.3) {
    const r = computePulse(HRV_WIN);
    if (r && r.keep > 0.85) {
      // Tight band-pass around the known heart rate makes beat timing far cleaner than the wide 42–210 BPM band
      const f = state.hr / 60;
      const wave = bandpass(r.pulse, FS, Math.max(HR_LO, f - 0.4), Math.min(HR_HI, f + 0.4), 0.1), n = wave.length;
      const pk = findPeaks(wave, FS, state.hr).filter(p => p > FS * 0.5 && p < n - FS * 0.5);
      const hv = hrvFromPeaks(pk, FS);
      if (hv) { state.rmssd = hv.rmssd; state.sdnn = hv.sdnn; }
    }
  }
  if (span >= 20) {
    // Breathing from subtle vertical head motion, band 0.1–0.5 Hz (6–30 breaths/min)
    const tEnd = samples[samples.length - 1].t, tStart = Math.max(samples[0].t, tEnd - RESP_WIN);
    const sub = samples.filter(s => s.t >= tStart - 0.2);
    const y = bandpass(series(sub, s => s.y, tStart, tEnd), FS, 0.1, 0.5, 0.05);
    const NF = 4096, P = powerSpectrum(y, NF), d = FS / NF;
    const k = argmaxRange(P, Math.ceil(0.1 / d), Math.floor(0.5 / d));
    state.resp = (k + parabolic(P, k)) * d * 60;
  }
}

function pushHistory() {
  if (session.startT == null) session.startT = samples[0].t;
  const ref = parseRef();
  history.push({ t: samples[samples.length - 1].t - session.startT, hr: state.hr, ref, conf: state.conf, snr: state.snr,
                 rmssd: state.rmssd, sdnn: state.sdnn, resp: state.resp, keep: state.keep });
  if (history.length > 3600) history.shift();
  const errs = history.filter(r => r.ref != null && r.hr != null && r.conf > 0.3).map(r => Math.abs(r.hr - r.ref));
  state.maeN = errs.length; state.mae = errs.length ? mean(errs) : null;
}
