// Rendering: live video overlay, readouts and the three plots.

import { FS, COLORS as C } from "./config.js";
import { $, view, vctx, video, parseRef } from "./dom.js";
import { findPeaks } from "./dsp.js";
import { NN } from "./neural.js";
import { history, state } from "./state.js";

export function updateUI() {
  const c = state.conf;
  $("bpm").textContent = state.hr ? Math.round(state.hr) : "--";
  $("bpm").classList.toggle("stale", state.hr != null && c < 0.3);
  $("qualBar").style.width = (c * 100).toFixed(0) + "%";
  $("qualBar").style.background = c > 0.6 ? "var(--good)" : c > 0.3 ? "var(--warn)" : "var(--bad)";
  const qb = $("qualBadge"), q = c > 0.6 ? "good" : c > 0.3 ? "fair" : "poor";
  qb.className = "badge" + (state.omega ? " " + q : "");
  qb.textContent = state.omega ? q[0].toUpperCase() + q.slice(1) : "—";
  $("snrLabel").textContent = isFinite(state.snr) ? `SNR ${state.snr.toFixed(1)} dB` : "";
  $("rmssd").textContent = state.rmssd ? Math.round(state.rmssd) : "--";
  $("sdnn").textContent = state.sdnn ? Math.round(state.sdnn) : "--";
  $("resp").textContent = state.resp ? state.resp.toFixed(0) : "--";
  const ref = parseRef();
  $("refErr").textContent = ref != null && state.hr ? (state.hr - ref >= 0 ? "+" : "−") + Math.abs(state.hr - ref).toFixed(1) : "--";
  $("refMae").textContent = state.mae != null ? `${state.mae.toFixed(1)} (n=${state.maeN})` : "--";
}

// Mirrored camera frame with the face mesh, the analysed face box and a pulse-synchronised flash
export function render(nowMs) {
  const W = view.width, H = view.height;
  vctx.save(); vctx.translate(W, 0); vctx.scale(-1, 1); vctx.drawImage(video, 0, 0, W, H); vctx.restore();
  const live = state.omega && state.conf > 0.2;
  const v = live ? Math.cos(state.omega * nowMs / 1000 - state.phi) : 0;
  const lm = state.face;
  if (lm) {
    vctx.fillStyle = "rgba(255,255,255,.45)";                       // face mesh (always on)
    for (const p of lm) vctx.fillRect((1 - p.x) * W - 1, p.y * H - 1, 2, 2);
    // outline of the face the model is analysing (the video is mirrored on screen)
    const B = NN.box;
    if (B) {
      const w = B.side / 1.5, h = w * 1.25, x = W - (B.cx + w / 2), y = B.cy - h / 2;
      if (live) { vctx.fillStyle = `rgba(229,72,77,${((0.04 + 0.22 * Math.max(0, v)) * state.conf).toFixed(3)})`; vctx.fillRect(x, y, w, h); }
      vctx.strokeStyle = "rgba(255,255,255,.85)"; vctx.lineWidth = 1.25; vctx.strokeRect(x, y, w, h);
    }
  }
  $("heart").style.transform = `scale(${(1 + 0.2 * Math.max(0, v) ** 6).toFixed(3)})`;
}

// Size a plot canvas for the device pixel ratio, clear it and draw the baseline
function prep(c) {
  const r = c.getBoundingClientRect(), d = devicePixelRatio || 1;
  if (c.width !== Math.round(r.width * d) || c.height !== Math.round(r.height * d)) { c.width = Math.round(r.width * d); c.height = Math.round(r.height * d); }
  const x = c.getContext("2d"); x.setTransform(d, 0, 0, d, 0, 0); x.clearRect(0, 0, r.width, r.height);
  x.font = "10px ui-monospace, Consolas, monospace"; x.fillStyle = C.faint;
  x.strokeStyle = C.line; x.lineWidth = 1; x.beginPath(); x.moveTo(0, r.height - .5); x.lineTo(r.width, r.height - .5); x.stroke();
  return [x, r.width, r.height];
}

export function drawPlots() { drawWave(); drawSpec(); drawHist(); }

function drawWave() {
  const [x, W, H] = prep($("wavePlot")), w = state.wave; if (!w) return;
  let m = 0; for (const v of w) m = Math.max(m, Math.abs(v)); m = m || 1;
  const Y = v => H / 2 - v / m * H * 0.42, X = i => i / (w.length - 1) * W;
  x.strokeStyle = C.grid; x.beginPath(); x.moveTo(0, H / 2); x.lineTo(W, H / 2); x.stroke();
  x.strokeStyle = C.brand; x.lineWidth = 1.75; x.lineJoin = "round"; x.beginPath();
  for (let i = 0; i < w.length; i++) i ? x.lineTo(X(i), Y(w[i])) : x.moveTo(X(i), Y(w[i]));
  x.stroke();
  if (state.hr) {
    x.fillStyle = C.pulse;
    for (const p of findPeaks(w, FS, state.hr)) { x.beginPath(); x.arc(X(p), Y(w[Math.round(p)]), 3, 0, 7); x.fill(); }
  }
}

function drawSpec() {
  const [x, W, H] = prep($("specPlot")); if (!state.fused) return;
  const P = state.fused, df = state.df, bLo = 40, bHi = 210, kA = Math.ceil(bLo / 60 / df), kB = Math.floor(bHi / 60 / df);
  const X = bpm => (bpm - bLo) / (bHi - bLo) * W;
  x.strokeStyle = C.grid; x.lineWidth = 1;
  for (const b of [60, 90, 120, 150, 180]) { x.beginPath(); x.moveTo(X(b), 0); x.lineTo(X(b), H - 12); x.stroke(); x.fillStyle = C.faint; x.fillText(b, X(b) - 8, H - 2); }
  let m = 0; for (let k = kA; k <= kB; k++) m = Math.max(m, P[k]); m = m || 1;
  x.strokeStyle = C.brand; x.lineWidth = 1.75; x.beginPath();
  for (let k = kA; k <= kB; k++) { const px = X(k * df * 60), py = H - 14 - P[k] / m * (H - 26); k === kA ? x.moveTo(px, py) : x.lineTo(px, py); }
  x.stroke();
  const px = X(state.hrRaw);
  x.setLineDash([2, 3]); x.strokeStyle = C.muted; x.beginPath(); x.moveTo(px, 0); x.lineTo(px, H - 12); x.stroke(); x.setLineDash([]);
  x.fillStyle = C.ink; x.fillText(`${Math.round(state.hrRaw)} bpm`, Math.min(px + 4, W - 50), 11);
}

function drawHist() {
  const [x, W, H] = prep($("histPlot"));
  const h = history.filter(r => r.hr).slice(-120); if (h.length < 2) return;
  const ref = parseRef();
  const vals = h.map(r => r.hr).concat(ref != null ? [ref] : []);
  const lo = Math.min(...vals) - 8, hi = Math.max(...vals) + 8;
  const t0 = h[0].t, t1 = h[h.length - 1].t;
  const X = t => 28 + (t - t0) / ((t1 - t0) || 1) * (W - 34), Y = v => 6 + (hi - v) / (hi - lo) * (H - 18);
  x.fillStyle = C.faint; x.fillText(Math.round(hi), 2, 12); x.fillText(Math.round(lo), 2, H - 8);
  if (ref != null) {
    x.setLineDash([3, 3]); x.strokeStyle = C.pulse; x.lineWidth = 1; x.beginPath(); x.moveTo(28, Y(ref)); x.lineTo(W - 6, Y(ref)); x.stroke(); x.setLineDash([]);
    x.fillStyle = C.pulse; x.fillText("ref " + Math.round(ref), W - 46, Y(ref) - 4);
  }
  x.strokeStyle = C.brand; x.lineWidth = 1.75; x.lineJoin = "round"; x.beginPath();
  h.forEach((r, i) => i ? x.lineTo(X(r.t), Y(r.hr)) : x.moveTo(X(r.t), Y(r.hr))); x.stroke();
}
