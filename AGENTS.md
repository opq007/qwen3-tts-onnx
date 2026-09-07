# AGENTS.md

Qwen3-TTS ONNX: FastAPI service running Qwen3-TTS on pure CPU via ONNX Runtime.
0.6B Base (voice clone) + 1.7B VoiceDesign (voice design). No GPU, no PyTorch inference.

## Run / Verify

```bash
# One-click startup (creates .venv, installs deps, generates config.yaml,
# checks/downloads models, checks ffmpeg, opens browser)
./start.sh          # Linux/macOS (chmod +x first)
start.bat           # Windows

# Manual
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**Tests (no real models or ffmpeg needed):**
```bash
.\.venv\Scripts\python -m pytest tests\test_api.py -q   # 15 tests, mock engines
```
- `tests/conftest_mock.py` replaces `Qwen3TtsEngine` with `MockEngine` (sine-wave output) — API, auth, voice store, and task lifecycle all run without ONNX weights.
- Tests must run from repo root with the `.venv` (Python 3.11). Always re-run this suite after touching app/. No pytest config file exists.
- Repo is developed on **Windows / PowerShell**: use `.venv\Scripts\python` paths and PowerShell syntax (`;` + `if ($?)` instead of `&&`).

## Config

- `config.yaml` (gitignored) is generated from `config.example.yaml`. Precedence: **env var > config.yaml > config.example.yaml**.
- Env vars: `Q3TTS_API_TOKEN` (empty = no auth), `PORT`/`Q3TTS_PORT`, `DATA_DIR` (default `./data`), `MODELS_DIR` (default `./models`), `FFMPEG_PATH`, `HF_TOKEN` (gated model download).
- `PORT`, `DATA_DIR` env vars are read in `app/config.py::load_config`; `Q3TTS_PORT` is read by the startup scripts.

## Model layout (non-obvious)

Each model needs TWO sibling dirs under `models/<Name>/`:
- `cpu_int4/` — ONNX submodels + `manifest.json` (gitignored; must be downloaded, ~1.6GB/2.9GB)
- `original/` — HF config + tokenizer files (tracked in git, no weights)

Loads manifest via `app/inference/pipeline.py::Qwen3TtsOnnxPipeline` — it reads `manifest.json` for submodel filenames and execution provider. A fresh clone is missing `cpu_int4/`; run `python scripts/download_models.py` (or the start script) to fetch it. Startup logs a `_warn_missing_model_dirs` warning when config points at missing ONNX dirs.

`capability` in config must match the ONNX manifest's `tts_model_type` (`base` → `voice_clone`, `voice_design` → `voice_design`); `engine.py::check_capability` raises on mismatch.

## Architecture

- `app/main.py::create_app()` wires everything and `create_app()` runs at module import. App state lives on `app.state.*` (`config`, `db`, `voice_store`, `task_store`, `orchestrator`) — API deps in `app/api/deps.py` read them from the Request.
- Lifecycle is a **lifespan context** in `create_app` (was `on_event`; deprecated — don't reintroduce). Preloads `load_on_start` models in a daemon thread, cleans expired tasks, shuts down the executor.
- Routes: `/v1/audio/speech` (OpenAI-compat; JSON or multipart), `/v1/voices`, `/api/tasks`, `/api/models/*`, `/healthz`, `/v1/models`. Auth = static token middleware (`app/api/auth.py`), Bearer or `x-api-key`; `/`, `/static`, `/healthz`, `/docs` exempt.
- **Concurrency**: `Orchestrator._enqueue` serializes per-model via `asyncio.Lock`, runs CPU inference via `asyncio.to_thread`, and enforces `max_queue_length` with a per-model `asyncio.Semaphore` → raises `ModelBusy` → HTTP **429** (`openai.py`) or task failure code `model_busy` (`tasks.py`).
- Web UI is static (`web/`) — plain JS/CSS, no build step; served at `/static` and `/`.

## Gotchas / Conventions

- **`speed` is validated but NOT implemented** — VoiceDesign pace control happens through `instruct`. Don't wire speed through to the pipeline.
- **ffmpeg is only needed for mp3** — wav path uses soundfile + librosa. Tests' mock never calls it.
- **If you add fields to `Orchestrator.__init__`, mirror them in `tests/conftest_mock.py::MockOrchestrator.__init__`** (it duplicates init independently). `_enqueue` reads `_sems` via `getattr(..., {})` defensively because of this.
- All API errors use the envelope `{"error": {"code", "message"}}` — raise `HTTPException(..., detail={"code": ..., "message": ...})`.
- `docs/01..06-*.md` are the design specs (requirements, architecture, API spec, data model, UI, constraints) — authoritative for intent; `docs/03-api-spec.md` governs API behavior.
- Runtime artifacts gitignored: `data/`, `models/*/cpu_int4/`, `config.yaml`, `test_out_*.wav`, `e2e_*.log`.