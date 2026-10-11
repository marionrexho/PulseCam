// Mutable application state shared between the capture, analysis and UI modules.

import { NN_SIZE } from "./neural.js";

// Runtime flags and timestamps for the camera / detection loop
export const session = {
  landmarker: null, running: false, lastVideoTime: -1, prevPts: null,
  lastFace: 0, lastSlow: 0, startT: null, lastDetect: 0, badUntil: 0,
};

export const samples = [];   // per-frame face samples (time, vertical position, motion)
export const history = [];   // one row per second for the trend plot

// Latest estimates, as shown in the UI
export const state = {
  hr: null, hrRaw: null, snr: -Infinity, conf: 0, rmssd: null, sdnn: null, resp: null, omega: 0, phi: 0,
  motion: 0, keep: 1, face: null, fused: null, wave: null, df: 1, cand: null, mae: null, maeN: 0,
};

// Offscreen canvas that receives the 72x72 face crop fed to the network
export const cropCanvas = document.createElement("canvas");
cropCanvas.width = cropCanvas.height = NN_SIZE;
export const cropCtx = cropCanvas.getContext("2d", { willReadFrequently: true });
cropCtx.imageSmoothingQuality = "high";
