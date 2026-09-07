"""Qwen3-TTS ONNX 部署服务 — 配置加载。

配置优先级（高→低）：
1. 环境变量（Q3TTS_* / PORT / DATA_DIR / MODELS_DIR）
2. config.yaml
3. 内置默认值
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

# 模型 ID 常量
MODEL_BASE = "qwen3-tts-0.6b-base"
MODEL_DESIGN = "qwen3-tts-1.7b-voicedesign"

LANGUAGES = ["Chinese", "English", "Japanese", "Korean", "German",
             "French", "Russian", "Portuguese", "Spanish", "Italian"]

B64_VOICE_ID = "[A-Za-z0-9_\u4e00-\u9fa5-]{1,64}"


@dataclass
class ModelConfig:
    id: str
    name: str
    capability: str                # voice_clone | voice_design
    onnx_dir: str
    tts_dir: str                   # 原版 checkpoint（config + tokenizer）
    load_on_start: bool = True
    unload_after_idle_seconds: int = 0
    max_new_tokens: int = 2048
    top_k: int = 50
    top_p: float = 1.0
    temperature: float = 0.9
    seed: int = 0


@dataclass
class Config:
    host: str = "0.0.0.0"
    port: int = 8000
    api_token: str = ""
    docs_public: bool = True
    data_dir: str = "./data"
    models_dir: str = ""
    log_level: str = "info"
    max_queue_length: int = 16
    default_voice: str = ""
    ffmpeg_path: str = "ffmpeg"
    mp3_bitrate: str = "128k"
    max_ref_audio_seconds: int = 30
    max_upload_mb: int = 50
    task_ttl_seconds: int = 7 * 24 * 3600
    models: list[ModelConfig] = field(default_factory=list)
    provider: str = "CPUExecutionProvider"
    torch_threads: int = 0          # 0=自动

    # ── 派生路径（随 data_dir / models_dir 变化） ──
    @property
    def db_path(self) -> str:
        return os.path.join(self.data_dir, "app.db")

    @property
    def voices_dir(self) -> str:
        return os.path.join(self.data_dir, "voices")

    @property
    def tasks_dir(self) -> str:
        return os.path.join(self.data_dir, "tasks")

    @property
    def auth_enabled(self) -> bool:
        return bool(self.api_token)


def _env(name: str, default: Any = None) -> Any:
    return os.environ.get(name, default)


def _resolve_models_dir(cfg: dict) -> str:
    md = _env("MODELS_DIR") or cfg.get("data_dir", "./data")
    return os.path.join(md, "models") if md != "./models" else md


def load_config(path: Optional[str] = None) -> Config:
    c: dict = {}
    # 1) 默认 yaml
    if path and Path(path).exists():
        c = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    elif Path("config.yaml").exists():
        c = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8")) or {}
    elif Path("config.example.yaml").exists():
        c = yaml.safe_load(Path("config.example.yaml").read_text(encoding="utf-8")) or {}

    server = c.get("server", {}) or {}
    storage = c.get("storage", {}) or {}
    voice = c.get("voice", {}) or {}
    audio = c.get("audio", {}) or {}

    data_dir = _env("DATA_DIR") or str(storage.get("data_dir", "./data"))
    models_cfg = c.get("models", []) or []
    if not models_cfg:
        # 默认双模型
        md = _env("MODELS_DIR") or data_dir
        base_dir = os.path.join(md, "models")
        models_cfg = [
            {"id": MODEL_BASE, "name": "0.6B Base", "capability": "voice_clone",
             "onnx_dir": os.path.join(base_dir, "0.6B-Base", "cpu_int4"),
             "tts_dir": os.path.join(base_dir, "0.6B-Base", "original")},
            {"id": MODEL_DESIGN, "name": "1.7B VoiceDesign", "capability": "voice_design",
             "onnx_dir": os.path.join(base_dir, "1.7B-VoiceDesign", "cpu_int4"),
             "tts_dir": os.path.join(base_dir, "1.7B-VoiceDesign", "original")},
        ]

    models: list[ModelConfig] = []
    for m in models_cfg:
        m = dict(m)
        models.append(ModelConfig(
            id=m["id"],
            name=m.get("name", m["id"]),
            capability=m.get("capability", "voice_design"),
            onnx_dir=m.get("onnx_dir", ""),
            tts_dir=m.get("tts_dir", ""),
            load_on_start=bool(m.get("load_on_start", True)),
            unload_after_idle_seconds=int(m.get("unload_after_idle_seconds", 0)),
            max_new_tokens=int(m.get("max_new_tokens", 2048)),
            top_k=int(m.get("top_k", 50)),
            top_p=float(m.get("top_p", 1.0)),
            temperature=float(m.get("temperature", 0.9)),
            seed=int(m.get("seed", 0)),
        ))

    port = int(os.environ.get("PORT", server.get("port", 8000)))
    token = _env("Q3TTS_API_TOKEN", server.get("api_token", "")) or ""

    cfg_obj = Config(
        host=str(server.get("host", "0.0.0.0")),
        port=port,
        api_token=str(token),
        docs_public=bool(server.get("docs_public", True)),
        data_dir=str(data_dir),
        models_dir=_env("MODELS_DIR", ""),
        log_level=str(server.get("log_level", "info")),
        max_queue_length=int(server.get("max_queue_length", 16)),
        default_voice=str(voice.get("default_voice", "")),
        ffmpeg_path=str(audio.get("ffmpeg_path", _env("FFMPEG_PATH", "ffmpeg"))),
        mp3_bitrate=str(audio.get("mp3_bitrate", "128k")),
        max_ref_audio_seconds=int(audio.get("max_ref_audio_seconds", 30)),
        max_upload_mb=int(server.get("max_upload_mb", 50)),
        task_ttl_seconds=int(storage.get("task_ttl_seconds", 7 * 24 * 3600)),
        models=models,
    )
    return cfg_obj