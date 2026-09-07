"""推理引擎封装：为编排层提供加载/卸载/合成都统一接口。

引擎持有 Qwen3TtsOnnxPipeline + 配置，负责：
- 加载（惰性）与卸载（释放 onnxruntime session）
- 将文本→codes→波形→wav/mp3 字节的完整流程
- 能力门控校验
"""
from __future__ import annotations

import io
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..config import ModelConfig
from .pipeline import Qwen3TtsOnnxPipeline, SR

# 状态机
UNLOADED = "unloaded"
LOADING = "loading"
LOADED = "loaded"
UNLOADING = "unloading"


@dataclass
class EngineStatus:
    model: str = ""
    state: str = UNLOADED
    capability: str = ""
    load_time_ms: float = 0
    infer_count: int = 0
    last_used_at: Optional[float] = None
    error: str = ""
    estimated_ram_mb: int = 0
    loaded_at: Optional[float] = None


class Qwen3TtsEngine:
    """单模型推理引擎（线程安全：编排层通过 asyncio.Lock 串行化外部调用）。"""

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.pipeline: Optional[Qwen3TtsOnnxPipeline] = None
        self.status = EngineStatus(model=self.cfg.id)

    # ── 加载/卸载 ────────────────────────────────────────────────────────
    def load(self) -> None:
        if self.pipeline is not None:
            return
        self.status.state = LOADING
        self.status.error = ""
        t0 = time.perf_counter()
        try:
            self.pipeline = Qwen3TtsOnnxPipeline(
                model_path=self.cfg.onnx_dir, tts_dir=self.cfg.tts_dir)
            self.pipeline.check_capability(self.cfg.capability)
            self.status.load_time_ms = (time.perf_counter() - t0) * 1000
            self.status.state = LOADED
            self.status.last_used_at = time.time()
            # 粗略估算驻内存（汇总 onnx 文件大小，乘以膨胀系数）
            try:
                from pathlib import Path
                total = 0.0
                for f in Path(self.cfg.onnx_dir).rglob("*.onnx*"):
                    total += f.stat().st_size
                self.status.estimated_ram_mb = int(total / (1024 ** 2) * 1.15)
            except OSError:
                self.status.estimated_ram_mb = 0
        except Exception as e:  # noqa: BLE001
            self.pipeline = None
            self.status.state = UNLOADED
            self.status.error = f"{type(e).__name__}: {e}"
            raise

    def unload(self) -> None:
        self.status.state = UNLOADING
        self.pipeline = None
        self.status.state = UNLOADED
        self.status.load_time_ms = 0

    # ── 生成（供编排层在锁内调用；阻塞 CPU）──────────────────────────────
    def synthesize(self, text: str, language: str, voice_type: str,
                   instruct: Optional[str] = None,
                   ref_audio_path: Optional[str] = None,
                   ref_text: Optional[str] = None,
                   max_new_tokens: Optional[int] = None,
                   do_sample: bool = True) -> tuple[np.ndarray, int]:
        """返回 (wav float32 mono @24k, sr=24000)。"""
        if self.pipeline is None:
            self.load()
        cfg = self.cfg
        mn = max_new_tokens or cfg.max_new_tokens

        codes = self.pipeline.generate(
            text, language=language,
            instruct=instruct,
            ref_audio=ref_audio_path,
            ref_text=ref_text,
            max_new_tokens=mn,
            do_sample=do_sample,
            top_k=cfg.top_k,
            top_p=cfg.top_p,
            temperature=cfg.temperature,
            seed=cfg.seed,
            verbose=False,
        )
        self.status.infer_count += 1
        self.status.last_used_at = time.time()
        if codes.shape[0] == 0:
            raise RuntimeError("无输出帧生成（立即 EOS）——文本过短或生成参数问题。")
        wav = self.pipeline.decode_chunked(codes[None]).reshape(-1)
        return wav.astype(np.float32), SR

    # ── 工具 ────────────────────────────────────────────────────────────
    def to_wav_bytes(self, wav: np.ndarray) -> bytes:
        import soundfile as sf
        buf = io.BytesIO()
        sf.write(buf, wav, SR, format="WAV", subtype="PCM_16")
        return buf.getvalue()

    @property
    def is_loaded(self) -> bool:
        return self.pipeline is not None

    def partial_json(self) -> dict:
        s = self.status
        return {
            "id": self.cfg.id,
            "name": self.cfg.name,
            "capability": self.cfg.capability,
            "status": s.state,
            "onnx_dir": self.cfg.onnx_dir,
            "tts_dir": self.cfg.tts_dir,
            "estimated_ram_mb": s.estimated_ram_mb,
            "load_time_ms": s.load_time_ms,
            "infer_count": s.infer_count,
            "last_used_at": s.last_used_at,
            "idle_seconds": (time.time() - s.last_used_at) if (s.state == LOADED and s.last_used_at) else 0,
            "error": s.error,
        }