@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
title Qwen3-TTS ONNX - One-Click Start
cd /d "%~dp0"

echo ============================================================
echo   Qwen3-TTS ONNX  -  Windows One-Click Startup
echo ============================================================

REM ---- 1. Python check ----
set "PY_CMD="
where py >nul 2>nul && set "PY_CMD=py -3"
if not defined PY_CMD ( where python >nul 2>nul && set "PY_CMD=python" )
if not defined PY_CMD (
    echo [ERROR] Python 3.10+ not found. Please install Python and add it to PATH.
    pause
    exit /b 1
)

%PY_CMD% -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python 3.10+ required (found older version).
    pause
    exit /b 1
)

REM ---- 2. Virtualenv ----
if not exist ".venv\Scripts\python.exe" (
    echo [..] Creating virtualenv .venv ...
    %PY_CMD% -m venv .venv
    if errorlevel 1 ( echo [ERROR] Failed to create venv. & pause & exit /b 1 )
)
set "VENV_PY=.venv\Scripts\python.exe"

REM ---- 3. Install deps (auto) ----
"%VENV_PY%" -c "import fastapi, onnxruntime, uvicorn" >nul 2>nul
if errorlevel 1 (
    echo [..] Installing dependencies (first run, may take a while)...
    "%VENV_PY%" -m pip install --upgrade pip >nul
    "%VENV_PY%" -m pip install -r requirements.txt
    if errorlevel 1 ( echo [ERROR] pip install failed. & pause & exit /b 1 )
)

REM ---- 4. Config ----
if not exist "config.yaml" (
    echo [..] Creating config.yaml from config.example.yaml ...
    copy /y config.example.yaml config.yaml >nul
)

REM ---- 5. Model check ----
set "NEED_MODELS=0"
if not exist "models\0.6B-Base\cpu_int4\manifest.json" set "NEED_MODELS=1"
if not exist "models\1.7B-VoiceDesign\cpu_int4\manifest.json" set "NEED_MODELS=1"
if "!NEED_MODELS!"=="1" (
    echo.
    echo [WARN] ONNX models not found under .\models\
    echo        You can download them now (0.6B ~1.6GB + 1.7B ~2.9GB int4).
    set /p DL="Download models now? [y/N] "
    if /i "!DL!"=="y" (
        "%VENV_PY%" scripts\download_models.py
        if errorlevel 1 ( echo [ERROR] Model download failed. & pause & exit /b 1 )
    ) else (
        echo [WARN] Skipping download. Server will start, but synthesis will fail until models exist.
    )
)

REM ---- 6. ffmpeg check (optional, mp3 only) ----
where ffmpeg >nul 2>nul
if errorlevel 1 (
    echo [WARN] ffmpeg not found in PATH - mp3 encode/decode disabled (wav still works).
    echo        Install ffmpeg or set audio.ffmpeg_path in config.yaml.
)

REM ---- 7. Launch ----
set "PORT=8000"
if defined Q3TTS_PORT set "PORT=%Q3TTS_PORT%"

echo.
echo Starting server on http://localhost:%PORT% ...
echo   API docs  : http://localhost:%PORT%/docs
echo   Web UI    : http://localhost:%PORT%/
echo   Press Ctrl+C to stop.
echo.

start "" cmd /c "timeout /t 2 /nobreak >nul & start http://localhost:%PORT%/"
"%VENV_PY%" -m uvicorn app.main:app --host 0.0.0.0 --port %PORT%

endlocal
