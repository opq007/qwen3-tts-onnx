"""自测辅助 — 创建测试应用实例（MOCK 引擎，不依赖真实模型/ffmpeg）。

MOCK 引擎：synthesize 返回一个 0.2s 的正弦波（模拟 24kHz wav），
让 API 全链路（鉴权/语音库/任务/编排队列）无真实模型也能跑通。
"""
from __future__ import annotations

import io
import os
import tempfile
import wave
from pathlib import Path

import numpy as np

from app.config import Config, ModelConfig, MODEL_BASE, MODEL_DESIGN
from app.services.orchestrator import Orchestrator, SynthesisRequest
from app.store.schema import init_db
from app.store.task_store import TaskStore
from app.store.voice_store import VoiceStore


class MockEngine:
    """替代 Qwen3TtsEngine 的假引擎：替身 synthesize / load / unload / to_wav_bytes。"""

    def __init__(self, mc: ModelConfig):
        self.cfg = mc
        self.status = type("S", (), {"state": "unloaded", "capability": mc.capability,
                                     "load_time_ms": 5, "infer_count": 0,
                                     "last_used_at": None, "error": "",
                                     "estimated_ram_mb": 1024,
                                     "model": mc.id, "loaded_at": None})()
        self.pipeline = None
        self.infer_count = 0
        self.generated: list[dict] = []

    def load(self):
        self.status.state = "loaded"
        self.status.last_used_at = __import__("time").time()
        self.pipeline = object()

    def unload(self):
        self.status.state = "unloaded"
        self.pipeline = None

    @property
    def is_loaded(self):
        return self.pipeline is not None

    def synthesize(self, text, language, voice_type, instruct=None,
                   ref_audio_path=None, ref_text=None, max_new_tokens=None,
                   do_sample=True):
        self.infer_count += 1
        self.status.infer_count = self.infer_count
        self.status.last_used_at = __import__("time").time()
        self.generated.append({
            "text": text, "language": language, "instruct": instruct,
            "ref_audio_path": ref_audio_path, "ref_text": ref_text,
        })
        sr = 24000
        t = np.arange(int(sr * 0.2)) / sr
        wav = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        return wav, sr

    def to_wav_bytes(self, wav: np.ndarray) -> bytes:
        buf = io.BytesIO()
        import soundfile as sf
        sf.write(buf, wav, 24000, format="WAV", subtype="PCM_16")
        return buf.getvalue()

    def partial_json(self):
        j = {
            "id": self.cfg.id, "name": self.cfg.name,
            "capability": self.cfg.capability, "status": self.status.state,
            "onnx_dir": self.cfg.onnx_dir, "tts_dir": self.cfg.tts_dir,
            "estimated_ram_mb": self.status.estimated_ram_mb,
            "load_time_ms": self.status.load_time_ms,
            "infer_count": self.infer_count,
            "last_used_at": self.status.last_used_at,
            "idle_seconds": 0, "error": self.status.error,
        }
        return j


class MockOrchestrator(Orchestrator):
    """用 MockEngine 替换 Qwen3TtsEngine。"""

    def __init__(self, cfg: Config, voice_store: VoiceStore):
        self.cfg = cfg
        self.voice_store = voice_store
        self.engines: dict[str, MockEngine] = {}
        for m in cfg.models:
            self.engines[m.id] = MockEngine(m)
        self._locks = {m.id: __import__("asyncio").Lock() for m in cfg.models}
        self._sems = {}   # 测试环境不启用队列限制（getattr 兼容真实 Orchestrator）
        self._executor = None
        self.startup_time = __import__("time").time()


def make_test_config(tmp: str, with_auth: bool = True, token: str = "test-token") -> Config:
    """构建测试配置（模型目录指向不存在路径——Mock 引擎不真实加载）。"""
    md = os.path.join(tmp, "models")
    cfg = Config(
        host="127.0.0.1", port=8000,
        api_token=token if with_auth else "",
        docs_public=True,
        data_dir=tmp,
        models_dir=md,
        max_queue_length=16,
        default_voice="",
        ffmpeg_path="ffmpeg",   # mock 路径下不会真实调用
        mp3_bitrate="128k",
        max_ref_audio_seconds=30,
        max_upload_mb=5,
        task_ttl_seconds=86400,
        models=[
            ModelConfig(id=MODEL_BASE, name="0.6B Base", capability="voice_clone",
                        onnx_dir=os.path.join(md, "0.6B-Base", "cpu_int4"),
                        tts_dir=os.path.join(md, "0.6B-Base", "original"),
                        load_on_start=False, max_new_tokens=64),
            ModelConfig(id=MODEL_DESIGN, name="1.7B VoiceDesign", capability="voice_design",
                        onnx_dir=os.path.join(md, "1.7B-VoiceDesign", "cpu_int4"),
                        tts_dir=os.path.join(md, "1.7B-VoiceDesign", "original"),
                        load_on_start=False, max_new_tokens=64),
        ],
    )
    return cfg


def build_test_app(tmp: str, with_auth: bool = True, token: str = "test-token"):
    """返回 (app, orchestrator)。"""
    from app.main import create_app
    # 直接构造 app 状态
    import app.main as main_mod
    app = main_mod.app

    cfg = make_test_config(tmp, with_auth=with_auth, token=token)
    Path(cfg.data_dir).mkdir(parents=True, exist_ok=True)
    conn = init_db(cfg.db_path)
    vs = VoiceStore(conn, cfg.voices_dir)
    ts = TaskStore(conn, cfg.tasks_dir)
    orch = MockOrchestrator(cfg, vs)

    app.state.config = cfg
    app.state.db = conn
    app.state.voice_store = vs
    app.state.task_store = ts
    app.state.orchestrator = orch

    from app.api.auth import AuthMiddleware
    # 中间件已在模块级添加；此处仅确保状态就绪
    return app, orch