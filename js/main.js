// Entry point: loads the models and wires the Start button.

import { FaceLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";
import { FACE_MODEL_URL, MEDIAPIPE_WASM_URL } from "./config.js";
import { $, setStatus } from "./dom.js";
import { NN, ensureNeural } from "./neural.js";
import { session } from "./state.js";
import { openCamera, listCameras, scheduleLock, bindCameraControls } from "./camera.js";
import { startCapture } from "./capture.js";
import { analyze } from "./analysis.js";

async function initLandmarker() {
  setStatus("Loading face tracker…");
  const files = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM_URL);
  const make = delegate => FaceLandmarker.createFromOptions(files, {
    baseOptions: { modelAssetPath: FACE_MODEL_URL, delegate }, runningMode: "VIDEO", numFaces: 1 });
  try { session.landmarker = await make("GPU"); } catch { session.landmarker = await make("CPU"); }
}

async function start() {
  const btn = $("startBtn"); btn.disabled = true;
  try {
    if (!session.landmarker) await initLandmarker();
    if (!NN.session) ensureNeural();   // retry if the first attempt failed
    setStatus("Requesting camera…");
    let saved = null; try { saved = localStorage.getItem("pulsecam.camera"); } catch {}
    try { await openCamera(saved); } catch (e) { if (saved) await openCamera(null); else throw e; }   // saved camera unplugged? use the default
    await listCameras();
    session.running = true; btn.textContent = "Running"; $("intro").style.display = "none";
    startCapture();
    setInterval(analyze, 100);
    scheduleLock();
  } catch (e) {
    setStatus("Error: " + e.message, "warn"); btn.disabled = false;
  }
}

ensureNeural();   // start downloading the heart-rate model as soon as the page opens
bindCameraControls();
$("startBtn").onclick = start;
