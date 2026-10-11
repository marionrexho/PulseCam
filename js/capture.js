// Per-frame capture: face detection, motion tracking, and cropping the face for the neural model.

import { BUF_SEC, MOTION_REJECT, REJECT_HOLD } from "./config.js";
import { video } from "./dom.js";
import { NN, NN_SIZE, NN_PX, nnPushFrame } from "./neural.js";
import { session, samples, state, cropCtx } from "./state.js";
import { render } from "./ui.js";

const HAS_RVFC = "requestVideoFrameCallback" in HTMLVideoElement.prototype;

// Crop a face box (1.5x the face width, as in the toolbox) to 72x72 and queue it for the model.
// The box only moves when the face has clearly moved, which mimics the toolbox's fixed crop and avoids jitter.
function nnCapture(t, lm) {
  const vw = video.videoWidth, vh = video.videoHeight;
  const cx = (lm[234].x + lm[454].x) / 2 * vw, cy = (lm[10].y + lm[152].y) / 2 * vh;
  const side = Math.hypot((lm[454].x - lm[234].x) * vw, (lm[454].y - lm[234].y) * vh) * 1.5;
  const b = NN.box;
  if (!b || Math.abs(cx - b.cx) > .12 * b.side || Math.abs(cy - b.cy) > .12 * b.side || Math.abs(side / b.side - 1) > .15) NN.box = { cx, cy, side };
  const B = NN.box;
  cropCtx.drawImage(video, B.cx - B.side / 2, B.cy - B.side / 2, B.side, B.side, 0, 0, NN_SIZE, NN_SIZE);
  const d = cropCtx.getImageData(0, 0, NN_SIZE, NN_SIZE).data, fr = new Uint8Array(3 * NN_PX);
  let lum = 0;
  for (let i = 0; i < NN_PX; i++) {
    const r = d[4 * i], g = d[4 * i + 1], bl = d[4 * i + 2];
    fr[i] = r; fr[NN_PX + i] = g; fr[2 * NN_PX + i] = bl; lum += 0.299 * r + 0.587 * g + 0.114 * bl;
  }
  NN.luma = lum / NN_PX;
  nnPushFrame(NN, t, fr);
}

// tSec is the time the camera captured the frame, in the performance.now() clock (seconds)
function processFrame(tSec) {
  const dm = Math.max(performance.now(), session.lastDetect + 1); session.lastDetect = dm;   // MediaPipe needs strictly increasing times
  const res = session.landmarker.detectForVideo(video, dm);
  const lm = res.faceLandmarks && res.faceLandmarks[0];
  state.face = lm || null;
  if (!lm) { session.prevPts = null; return; }
  session.lastFace = performance.now() / 1000;
  const prev = samples[samples.length - 1];
  if (prev && tSec <= prev.t) return;                                         // keep sample times strictly increasing
  const vw = video.videoWidth, vh = video.videoHeight;
  const faceH = Math.max(1, (lm[152].y - lm[10].y) * vh);
  const key = [1, 10, 152, 234, 454, 33, 263], cur = key.map(i => [lm[i].x * vw, lm[i].y * vh]);
  let motion = 0;
  if (session.prevPts) {
    for (let k = 0; k < cur.length; k++) motion += Math.hypot(cur[k][0] - session.prevPts[k][0], cur[k][1] - session.prevPts[k][1]);
    motion /= cur.length * faceH;
  }
  session.prevPts = cur;
  if (motion > MOTION_REJECT) session.badUntil = tSec + REJECT_HOLD;
  samples.push({ t: tSec, y: ((lm[10].y + lm[152].y) / 2 * vh) / faceH, motion, bad: tSec < session.badUntil });
  while (samples.length && tSec - samples[0].t > BUF_SEC) samples.shift();
  if (NN.session) nnCapture(tSec, lm);
}

// Preferred path: one callback per decoded camera frame, with the true capture timestamp
function onFrame(now, meta) {
  if (!session.running) return;
  let t = now;
  if (meta && meta.captureTime && Math.abs(meta.captureTime - now) < 1000) t = meta.captureTime;
  processFrame(t / 1000);
  video.requestVideoFrameCallback(onFrame);
}

function loop() {
  if (!session.running) return;
  const now = performance.now();
  if (!HAS_RVFC && video.readyState >= 2 && video.currentTime !== session.lastVideoTime) { session.lastVideoTime = video.currentTime; processFrame(now / 1000); }
  render(now);
  requestAnimationFrame(loop);
}

// Begin capturing frames and drawing the live view
export function startCapture() {
  if (HAS_RVFC) video.requestVideoFrameCallback(onFrame);
  requestAnimationFrame(loop);
}
