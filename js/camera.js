// Camera access: stream selection, device list, and exposure / white-balance locking.

import { LOCK_DELAY } from "./config.js";
import { $, video, view, lockTog, setStatus } from "./dom.js";
import { NN, nnReset } from "./neural.js";
import { session, samples, state } from "./state.js";

const videoTrack = () => video.srcObject && video.srcObject.getVideoTracks()[0];

// Auto-exposure and auto-white-balance change pixel values on their own, which looks like noise
// on top of a ~0.1% pulse signal. Freeze them at whatever the camera settled on.
export async function lockCamera() {
  const track = videoTrack();
  if (!track || !track.getCapabilities) { $("lockState").textContent = "Lock not supported"; return; }
  const before = NN.luma;
  const caps = track.getCapabilities(), s = track.getSettings(), adv = {}, locked = [];
  if (caps.exposureMode && caps.exposureMode.includes("manual") && s.exposureTime != null) {
    adv.exposureMode = "manual"; adv.exposureTime = s.exposureTime; locked.push("exposure");
  }
  if (caps.whiteBalanceMode && caps.whiteBalanceMode.includes("manual") && s.colorTemperature != null) {
    adv.whiteBalanceMode = "manual"; adv.colorTemperature = s.colorTemperature; locked.push("white balance");
  }
  if (!locked.length) { $("lockState").textContent = "Camera doesn't allow locking"; return; }
  try { await track.applyConstraints({ advanced: [adv] }); $("lockState").textContent = "Locked: " + locked.join(" + "); }
  catch { $("lockState").textContent = "Lock failed"; return; }
  // Safety net: some cameras go dark when exposure is frozen. If the face got noticeably darker, undo it.
  await new Promise(r => setTimeout(r, 800));
  const after = NN.luma;
  if (before && after && after < before * 0.8) {
    lockTog.checked = false; await unlockCamera();
    $("lockState").textContent = "Lock undone (image got darker)";
  }
}

export async function unlockCamera() {
  const track = videoTrack();
  if (!track || !track.getCapabilities) return;
  const caps = track.getCapabilities(), adv = {};
  if (caps.exposureMode && caps.exposureMode.includes("continuous")) adv.exposureMode = "continuous";
  if (caps.whiteBalanceMode && caps.whiteBalanceMode.includes("continuous")) adv.whiteBalanceMode = "continuous";
  try { if (Object.keys(adv).length) await track.applyConstraints({ advanced: [adv] }); } catch {}
  $("lockState").textContent = "Auto";
}

// Show "Settling…" and lock once auto-exposure has had time to converge
export function scheduleLock() {
  $("lockState").textContent = lockTog.checked ? "Settling…" : "Auto";
  setTimeout(() => { if (lockTog.checked) lockCamera(); }, LOCK_DELAY);
}

export async function openCamera(deviceId) {
  if (video.srcObject) video.srcObject.getTracks().forEach(t => t.stop());
  const v = { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } };
  if (deviceId) v.deviceId = { exact: deviceId }; else v.facingMode = "user";
  const stream = await navigator.mediaDevices.getUserMedia({ video: v, audio: false });
  video.srcObject = stream; await video.play();
  await unlockCamera();   // a previous manual exposure can persist in the camera driver; always start in auto
  view.width = video.videoWidth; view.height = video.videoHeight;
  $("wrap").style.aspectRatio = `${video.videoWidth}/${video.videoHeight}`;
}

// Camera names are only available once the browser has given camera permission
export async function listCameras() {
  const sel = $("camSel");
  const cams = (await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === "videoinput");
  const track = videoTrack();
  const cur = track && track.getSettings().deviceId;
  sel.innerHTML = "";
  cams.forEach((d, i) => { const o = document.createElement("option"); o.value = d.deviceId; o.textContent = d.label || `Camera ${i + 1}`; sel.appendChild(o); });
  if (cur) sel.value = cur;
  sel.disabled = cams.length < 1;
}

export function bindCameraControls() {
  lockTog.onchange = () => { if (session.running) (lockTog.checked ? lockCamera : unlockCamera)(); };

  $("camSel").onchange = async () => {
    const id = $("camSel").value;
    try { localStorage.setItem("pulsecam.camera", id); } catch {}
    if (!session.running) return;
    try {
      samples.length = 0; session.prevPts = null; nnReset(NN); NN.box = null; NN.luma = null;
      Object.assign(state, { hr: null, cand: null, conf: 0, omega: 0, rmssd: null, sdnn: null, resp: null });
      await openCamera(id);
      scheduleLock();
    } catch (e) { setStatus("Camera error: " + e.message, "warn"); }
  };

  // plug or unplug while the page is open
  if (navigator.mediaDevices) navigator.mediaDevices.ondevicechange = () => { if (session.running) listCameras(); };
}
