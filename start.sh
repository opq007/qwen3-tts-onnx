#!/usr/bin/env bash
# Qwen3-TTS ONNX - Linux / macOS One-Click Startup
set -uo pipefail
cd "$(dirname "$0")" || exit 1

echo "============================================================"
echo "  Qwen3-TTS ONNX  -  Linux/macOS One-Click Startup"
echo "============================================================"

# ---- 1. Python check ----
PY_CMD=""
for c in python3 python; do
    if command -v "$c" >/dev/null 2>&1; then PY_CMD="$c"; break; fi
done
if [ -z "$PY_CMD" ]; then
    echo "[ERROR] Python 3.10+ not found. Please install Python and add it to PATH."
    exit 1
fi
if ! "$PY_CMD" -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" 2>/dev/null; then
    echo "[ERROR] Python 3.10+ required (found older version)."
    exit 1
fi

# ---- 2. Virtualenv ----
if [ ! -x ".venv/bin/python" ]; then
    echo "[..] Creating virtualenv .venv ..."
    "$PY_CMD" -m venv .venv || { echo "[ERROR] Failed to create venv."; exit 1; }
fi
VENV_PY=".venv/bin/python"

# ---- 3. Install deps (auto) ----
if ! "$VENV_PY" -c "import fastapi, onnxruntime, uvicorn" >/dev/null 2>&1; then
    echo "[..] Installing dependencies (first run, may take a while)..."
    "$VENV_PY" -m pip install --upgrade pip >/dev/null
    "$VENV_PY" -m pip install -r requirements.txt || { echo "[ERROR] pip install failed."; exit 1; }
fi

# ---- 4. Config ----
if [ ! -f "config.yaml" ]; then
    echo "[..] Creating config.yaml from config.example.yaml ..."
    cp config.example.yaml config.yaml
fi

# ---- 5. Model check ----
NEED_MODELS=0
[ ! -f "models/0.6B-Base/cpu_int4/manifest.json" ] && NEED_MODELS=1
[ ! -f "models/1.7B-VoiceDesign/cpu_int4/manifest.json" ] && NEED_MODELS=1
if [ "$NEED_MODELS" = "1" ]; then
    echo
    echo "[WARN] ONNX models not found under ./models/"
    echo "       You can download them now (0.6B ~1.6GB + 1.7B ~2.9GB int4)."
    read -r -p "Download models now? [y/N] " DL
    if [ "$DL" = "y" ] || [ "$DL" = "Y" ]; then
        "$VENV_PY" scripts/download_models.py || { echo "[ERROR] Model download failed."; exit 1; }
    else
        echo "[WARN] Skipping download. Server will start, but synthesis will fail until models exist."
    fi
fi

# ---- 6. ffmpeg check (optional, mp3 only) ----
if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "[WARN] ffmpeg not found in PATH - mp3 encode/decode disabled (wav still works)."
    echo "       Install ffmpeg or set audio.ffmpeg_path in config.yaml."
fi

# ---- 7. Launch ----
PORT="${Q3TTS_PORT:-8000}"
echo
echo "Starting server on http://localhost:${PORT} ..."
echo "  API docs  : http://localhost:${PORT}/docs"
echo "  Web UI    : http://localhost:${PORT}/"
echo "  Press Ctrl+C to stop."
echo

( sleep 2 && (command -v xdg-open >/dev/null 2>&1 && xdg-open "http://localhost:${PORT}/") ) &
"$VENV_PY" -m uvicorn app.main:app --host 0.0.0.0 --port "$PORT"