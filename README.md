# PulseCam

Camera-based pulse estimation that runs entirely in the browser. Nothing is uploaded; video is processed on the device.

The pulse is estimated by a pretrained neural network (EfficientPhys, see *Credits and licence*). The face mesh and pulse overlay are always shown on the video.

PulseCam is a school project and **not a medical device**.

## Run it on your computer

The page loads `model/efficientphys_pure.onnx`, and browsers block that for files opened directly from disk, so use the local server:

1. Double-click `start-server.bat` (needs Python installed). It opens <http://localhost:8000/>. Keep the black window open while you use the page.
2. Click **Start camera**.

The first time, it downloads about 11 MB of runtime (cached afterwards).

## Put it on phones with a QR code

Phone browsers only allow the camera on **https** pages, so host the folder somewhere free, then make a QR code from its address.

GitHub Pages (free):
1. Create a GitHub repository and upload everything in this folder (keep the `model` folder, including its licence files).
2. Repository **Settings → Pages →** deploy from the `main` branch, root folder.
3. After a minute your page is at `https://<your-username>.github.io/<repo-name>/`.
4. Paste that address into any QR code generator and show the code.

Cloudflare Pages and Netlify also work (drag and drop the folder).

Tips for phones: prop the phone up (do not hold it), face a bright steady light, stay still for about 15 seconds. The exposure lock does not work on iPhones; leave it off.

## Checking accuracy

Type the reading from a pulse oximeter or chest strap into **Validation → Reference BPM**. The page shows the error and mean absolute error, and **Export CSV** saves every sample with the reference value. 

## How it works

1. The face tracker finds your face; a box 1.5x the face width is cropped and resized to 72x72.
2. Frames are placed on a steady 30 fps clock and standardised.
3. EfficientPhys takes 10 frames (plus one earlier frame) at a time and outputs the change in pulse signal.
4. The page converts that output into a pulse waveform (a "half integration" that avoids favouring slow frequencies), band-pass filters it, and finds the strongest rhythm in its spectrum.
5. The displayed heart rate is smoothed and only accepts a big change after it persists for several seconds. If the signal is weak the number turns grey and holds its last value.

`tools/export_model.py` shows how the PyTorch checkpoint was converted to ONNX. The converted model matches the original to within about 1e-6.

## Credits and licence

- Model and weights: rPPG-Toolbox, <https://github.com/ubicomplab/rPPG-Toolbox>, Responsible AI Source Code License v1.1 (copy in `model/LICENSE-rPPG-Toolbox.txt`). Section 3 restricts some uses, for example surveillance and diagnosing a medical condition without human oversight. Keep the licence with the model if you share it.
- Please cite: Liu et al., *EfficientPhys*, WACV 2023, and Liu et al., *rPPG-Toolbox: Deep Remote PPG Toolbox*, NeurIPS 2023. The checkpoint was trained on the PURE dataset (Stricker et al., 2014); check the toolbox README for exact citations.
- Face tracking: MediaPipe Face Landmarker. In-browser inference: ONNX Runtime Web.
