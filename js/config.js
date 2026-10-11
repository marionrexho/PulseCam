// Tunable constants and external resource locations.

export const FS = 30;               // the neural model expects a steady 30 fps clip
export const HR_WIN = 15;           // seconds of signal used for heart rate
export const HRV_WIN = 20;          // seconds used for HRV
export const RESP_WIN = 30;         // seconds used for breathing rate
export const BUF_SEC = 32;          // rolling buffer length
export const MIN_SEC = 8;           // seconds before first estimate
export const HR_LO = 0.7, HR_HI = 3.5;          // 42–210 BPM
export const MOTION_OK = 0.004, MOTION_WARN = 0.01; // per-frame motion, as fraction of face height
export const MOTION_REJECT = 0.02;  // frames moving more than this count as unusable
export const REJECT_HOLD = 0.15;    // also for this many seconds after a jerk
export const MIN_KEEP = 0.6;        // below this fraction of usable frames, estimates are paused
export const LOCK_DELAY = 2000;     // ms to let auto-exposure settle before locking it

export const ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/";
export const NN_URL = "model/efficientphys_pure.onnx";
export const FACE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
export const MEDIAPIPE_WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";

// Colours used on canvases (kept in sync with the CSS palette)
export const COLORS = { brand: "#1d4e75", pulse: "#c8333a", ink: "#16202c", muted: "#566272", faint: "#8792a0", line: "#dde1e6", grid: "#eceff2" };
