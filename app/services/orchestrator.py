"""编排层 — 模型注册表 + 串行推理队列 + 生命周期状态机。

并发模型（见 docs/02-architecture.md §4）：
- 每模型一把 asyncio.Lock = 串行队列；并发请求 await 排锁
- CPU 密集推理在线程池执行（run_in_executor），不阻塞事件循环
- 生命周期：unloaded → loading → loaded → unloading → unloaded；空闲自动卸载
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Callable, Optional

import numpy as np

from ..config import Config, ModelConfig
from ..inference.engine import Qwen3TtsEngine, LOADED, UNLOADED
from ..services.audio import decode_to_wav_bytes, encode_mp3
from ..store.voice_store import VoiceStore, VoiceNotFound

logger = logging.getLogger("q3tts.orchestrator")


class ModelNotFound(Exception):
    pass


class ModelBusy(Exception):
    pass


class ModelUnavailable(Exception):
    pass


class ModelBusy(Exception):
    """请求排队超过 max_queue_length 时抛出的重载信号（API 层映射为 429）。"""
    pass


class SynthesisRequest:
    """一次合成请求的参数（服务层已解析 voice）。"""

    def __init__(self, text: str, language: str, model_id: str,
                 instruct: Optional[str] = None,
                 ref_audio_path: Optional[str] = None,
                 ref_text: Optional[str] = None,
                 response_format: str = "wav",
                 do_sample: bool = True):
        self.text = text
        self.language = language
        self.model_id = model_id
        self.instruct = instruct
        self.ref_audio_path = ref_audio_path
        self.ref_text = ref_text
        self.response_format = response_format
        self.do_sample = do_sample


class Orchestrator:
    def __init__(self, cfg: Config, voice_store: VoiceStore):
        self.cfg = cfg
        self.voice_store = voice_store
        self.engines: dict[str, Qwen3TtsEngine] = {}
        for m in cfg.models:
            self.engines[m.id] = Qwen3TtsEngine(m)
        self._locks: dict[str, asyncio.Lock] = {
            m.id: asyncio.Lock() for m in cfg.models}
        # 队列长度限制：每模型一把信号量。max_queue_length <=0 表示不限制。
        self._sems: dict[str, asyncio.Semaphore] = {}
        if cfg.max_queue_length > 0:
            self._sems = {
                m.id: asyncio.Semaphore(cfg.max_queue_length)
                for m in cfg.models}
        self._executor = None  # 默认 loop 线程池
        self._loaded_by_capability: dict[str, str] = {}
        for e in self.engines.values():
            self._loaded_by_capability.setdefault(e.cfg.capability, e.cfg.id)
        self._idle_task: Optional[asyncio.Task] = None
        self.startup_time = time.time()

    # ── 生命周期 ────────────────────────────────────────────────────────
    async def load_model(self, model_id: str, wait: bool = True) -> str:
        eng = self._get(model_id)
        async with self._locks[model_id]:
            if eng.pipeline is None:
                await asyncio.to_thread(eng.load)
            return eng.status.state

    async def unload_model(self, model_id: str) -> str:
        eng = self._get(model_id)
        async with self._locks[model_id]:
            if eng.pipeline is not None:
                await asyncio.to_thread(eng.unload)
            return eng.status.state

    def load_all_on_start(self) -> None:
        for m in self.cfg.models:
            if m.load_on_start and self.engines[m.id].pipeline is None:
                self.engines[m.id].load()
                logger.info("模型 %s 已常驻加载", m.id)

    # ── 同步合成（编排锁内执行，供 /v1/audio/speech）───────────────────
    async def synthesize_sync(self, req: SynthesisRequest) -> tuple[bytes, str]:
        """返回 (音频字节, content_type)。"""
        done = await self._enqueue(
            req.model_id,
            lambda m, r=req: self._do_synthesize(m, r),
        )
        return done

    # ── 供异步任务 worker 使用 ─────────────────────────────────────────
    async def synthesize_task(self, req: SynthesisRequest,
                              progress_cb: Optional[Callable[[str, Optional[int], Optional[int]], None]] = None) -> tuple[bytes, str]:
        return await self._enqueue(
            req.model_id,
            lambda m, r=req: self._do_synthesize(m, r),
            progress_cb=progress_cb,
        )

    # ── 内部 ───────────────────────────────────────────────────────────
    def _get(self, model_id: str) -> Qwen3TtsEngine:
        if model_id not in self.engines:
            raise ModelNotFound(model_id)
        return self.engines[model_id]

    async def _enqueue(self, model_id: str, worker,
                       progress_cb: Optional[Callable] = None) -> tuple[bytes, str]:
        eng = self._get(model_id)
        lock = self._locks[model_id]
        sem = (getattr(self, "_sems", {}) or {}).get(model_id)
        # 队列满 → 立即 429，避免无限排队拖垮内存
        if sem is not None:
            try:
                await asyncio.wait_for(sem.acquire(), timeout=0.01)
            except asyncio.TimeoutError:
                # 说明队列已满（信号量全被占用）
                raise ModelBusy(f"模型 {model_id} 排队已满 "
                                f"(max_queue_length={self.cfg.max_queue_length})，请稍后重试。")
        logger.info("排队: %s (锁定)", model_id)
        try:
            async with lock:
                # 懒加载：若未加载，现加载
                if eng.pipeline is None:
                    logger.info("模型 %s 未加载，懒加载中…", model_id)
                    try:
                        await asyncio.to_thread(eng.load)
                    except Exception as e:
                        raise ModelUnavailable(str(e)) from e
                if eng.status.state == "unloading":
                    await asyncio.sleep(0.05)  # 等待卸载完成
                    # 兜底重新加载
                    if eng.pipeline is None:
                        await asyncio.to_thread(eng.load)
                logger.info("推理开始: %s", model_id)
                t0 = time.perf_counter()
                audio, fmt = await asyncio.to_thread(worker, eng)
                dt = (time.perf_counter() - t0) * 1000
                logger.info("推理完成: %s  %.0fms", model_id, dt)
                return audio, fmt
        finally:
            if sem is not None:
                sem.release()

    def _do_synthesize(self, eng: Qwen3TtsEngine, req: SynthesisRequest) -> tuple[bytes, str]:
        """线程内执行完整合成，返回字节。"""
        # 解码参考音频
        ref_path = None
        if req.ref_audio_path:
            if not eng.pipeline or eng.cfg.capability != "voice_clone":
                raise ModelUnavailable("当前模型不支持语音克隆。")
            # 统一解码为 24k wav 临时文件（soundfile 可直接读 mp3，但为统一用 ffmpeg）
            wav_bytes = decode_to_wav_bytes(
                req.ref_audio_path, ffmpeg_path=self.cfg.ffmpeg_path,
                max_seconds=self.cfg.max_ref_audio_seconds)
            import tempfile
            fd, ref_path = tempfile.mkstemp(suffix=".wav")
            with os.fdopen(fd, "wb") as f:
                f.write(wav_bytes)

        try:
            wav, sr = eng.synthesize(
                text=req.text,
                language=req.language,
                voice_type=eng.cfg.capability,
                instruct=req.instruct if eng.cfg.capability == "voice_design" else None,
                ref_audio_path=ref_path,
                ref_text=req.ref_text,
                do_sample=req.do_sample,
            )
        finally:
            if ref_path:
                os.unlink(ref_path)

        wav_bytes = eng.to_wav_bytes(wav)
        if req.response_format == "mp3":
            mp3 = encode_mp3(wav_bytes, ffmpeg_path=self.cfg.ffmpeg_path,
                             bitrate=self.cfg.mp3_bitrate)
            return mp3, "audio/mpeg"
        return wav_bytes, "audio/wav"

    # ── 状态/查询 ──────────────────────────────────────────────────────
    def models_status(self) -> list[dict]:
        out = []
        now = time.time()
        for e in self.engines.values():
            j = e.partial_json()
            if j["status"] == LOADED:
                j["idle_seconds"] = int(now - (e.status.last_used_at or now))
            else:
                j["idle_seconds"] = None
            out.append(j)
        return out

    def resolve_voice_params(self, voice_id: Optional[str]) -> dict:
        """把 voice 注册名解析为合成参数。返回 {instruct|ref_audio_path|ref_text|language}。

        voice_id 为 None/"" → 使用默认语音；仍无 → 抛 VoiceNotFound。
        支持形如 "clone:xxx" 无（由 API 层处理临时参数）。
        """
        vid = voice_id or self.cfg.default_voice
        if not vid:
            raise ValueError("缺少 voice 参数且未配置默认语音。")
        v = self.voice_store.get(vid)
        params = {"language": v["language"]}
        if v["type"] == "design":
            params["instruct"] = v["instruct"]
        elif v["type"] == "clone":
            params["ref_audio_path"] = self.voice_store.resolve_audio_abs(v)
            params["ref_text"] = v["ref_text"]
        params["voice_record"] = v
        return params

    async def idle_unload_loop(self, interval: float = 5.0) -> None:
        """后台任务：按 unload_after_idle_seconds 卸载闲置模型。"""
        while True:
            await asyncio.sleep(interval)
            for m in self.cfg.models:
                if m.unload_after_idle_seconds <= 0:
                    continue
                eng = self.engines[m.id]
                if eng.pipeline is not None and eng.status.last_used_at:
                    idle = time.time() - eng.status.last_used_at
                    if idle > m.unload_after_idle_seconds:
                        logger.info("模型 %s 空闲 %.0fs，自动卸载", m.id, idle)
                        await self.unload_model(m.id)