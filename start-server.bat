@echo off
rem Serves this folder at http://localhost:8000 so the browser can load the neural model file.
rem (Browsers block loading model files when index.html is opened by double-clicking it.)
cd /d "%~dp0"
start "" http://localhost:8000/
python -m http.server 8000
