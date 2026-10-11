// DOM handles and small helpers shared by every module.

export const $ = id => document.getElementById(id);

export const video = $("video");
export const view = $("view");
export const vctx = view.getContext("2d");
export const lockTog = $("lockTog");

export const setStatus = (s, kind = "") => { const el = $("status"); el.textContent = s; el.className = kind; };
export const setNN = s => { $("nnState").textContent = s; };

// Reference BPM typed into the validation box, or null if empty / out of range
export const parseRef = () => { const r = parseFloat($("refBpm").value); return r >= 30 && r <= 220 ? r : null; };
