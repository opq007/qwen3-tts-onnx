"""异步任务路由 — /api/tasks（提交、查询、下载、取消/清理）。

任务 worker：后台 asyncio.Task 串行执行（受 orchestrator 锁约束）。
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

from fastapi import (APIRouter, Depends, HTTPException, Request)
from fastapi.responses import Response

from ..services.orchestrator import (ModelBusy, ModelNotFound, ModelUnavailable,
                                     SynthesisRequest)
from ..store.voice_store import VoiceNotFound
from ..store.task_store import TaskNotFound, TaskStore
from .deps import get_orchestrator, get_task_store

logger = logging.getLogger("q3tts.tasks")
router = APIRouter(prefix="/api/tasks", tags=["tasks"])

_workers: dict[str, asyncio.Task] = {}


@router.post("")
async def create_task(request: Request,
                      orchestrator=Depends(get_orchestrator),
                      store: TaskStore = Depends(get_task_store)):
    cfg = request.app.state.config
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, detail={"code": "invalid_json", "message": "JSON 解析失败"})

    model = body.get("model", "")
    text = body.get("input", "")
    voice = body.get("voice")
    language = body.get("language")
    instruct = body.get("instruct")
    resp_fmt = body.get("response_format", "wav")
    from ..api.openai import ALLOWED_FORMATS, ALLOWED_MODELS
    if model not in ALLOWED_MODELS:
        raise HTTPException(400, detail={"code": "invalid_model",
                                         "message": f"model '{model}' 不存在"})
    if not text:
        raise HTTPException(400, detail={"code": "missing_parameter", "message": "input 不能为空"})
    if resp_fmt not in ALLOWED_FORMATS:
        raise HTTPException(400, detail={"code": "invalid_response_format",
                                         "message": "response_format 必须是 wav 或 mp3"})

    # voice 解析（异步任务不支持临时 multipart 参数，仅注册 voice）
    params = {}
    if voice:
        try:
            params = orchestrator.resolve_voice_params(voice)
        except VoiceNotFound:
            raise HTTPException(404, detail={"code": "voice_not_found",
                                             "message": f"voice '{voice}' 不存在"})
        except ValueError as e:
            raise HTTPException(400, detail={"code": "missing_parameter", "message": str(e)})
        eff_lang = params.get("language")
    else:
        eff_lang = None
    if language:
        eff_lang = language
    if instruct is not None:
        params["instruct"] = instruct
    if not eff_lang:
        raise HTTPException(400, detail={"code": "missing_parameter",
                                         "message": "缺少 language（或语音未绑定语言）"})

    task = store.create(
        model_id=model, input_text=text, language=eff_lang, voice_id=voice,
        instruct=params.get("instruct"),
        ref_audio=params.get("ref_audio_path"),
        ref_text=params.get("ref_text"),
        response_format=resp_fmt)
    task_id = task["task_id"]
    _workers[task_id] = asyncio.create_task(_run_task(task_id, request, orchestrator, store))
    return {"task_id": task_id, "status": "pending", "model": model,
            "created_at": task["created_at"]}


async def _run_task(task_id: str, request, orchestrator, store: TaskStore):
    t = store.get(task_id)
    model_id = t["model_id"]
    params = {}
    if t["voice_id"]:
        try:
            params = orchestrator.resolve_voice_params(t["voice_id"])
        except (VoiceNotFound, ValueError):
            pass
    req = SynthesisRequest(
        text=t["input_text"], language=t["language"] or params.get("language") or "auto",
        model_id=model_id,
        instruct=t["instruct"] or params.get("instruct"),
        ref_audio_path=params.get("ref_audio_path"),
        ref_text=params.get("ref_text"),
        response_format=t["response_format"])
    store.set_running(task_id, "tokenize")

    async def progress(step, frame, max_frames):
        store.set_progress(task_id, step, frame, max_frames)

    try:
        audio, ctype = await orchestrator.synthesize_task(req, progress_cb=progress)
        ext = "mp3" if t["response_format"] == "mp3" else "wav"
        rel = store.save_result(task_id, ext, audio)
        # 计算时长
        duration = None
        try:
            import numpy as np
            import soundfile as sf
            import io
            # 简单估算：wav 可从字节计算，mp3 不做
            if ext == "wav":
                data, sr = sf.read(io.BytesIO(audio), dtype="float32")
                duration = len(data) / sr
        except Exception:
            pass
        store.complete(task_id, rel, 24000, duration or 0.0)
        logger.info("任务 %s 完成: %s", task_id, rel)
    except asyncio.CancelledError:
        store.fail(task_id, "cancelled", "任务被取消")
        raise
    except ModelBusy as e:
        # 队列满 —— 让任务以可重试的失败状态结束
        store.fail(task_id, "model_busy", str(e))
        logger.warning("任务 %s 因队列满失败: %s", task_id, e)
    except Exception as e:
        store.fail(task_id, "inference_failed", str(e))
        logger.exception("任务 %s 失败", task_id)


@router.get("")
def list_tasks(status: Optional[str] = None, limit: int = 50, offset: int = 0,
               store: TaskStore = Depends(get_task_store)):
    return {"object": "list", "data": store.list(status=status, limit=min(limit, 200), offset=offset)}


@router.get("/{task_id}")
def get_task(task_id: str, store: TaskStore = Depends(get_task_store)):
    try:
        return store.get(task_id)
    except TaskNotFound:
        raise HTTPException(404, detail={"code": "task_not_found",
                                         "message": f"任务 {task_id} 不存在"})


@router.get("/{task_id}/audio")
def get_task_audio(task_id: str, store: TaskStore = Depends(get_task_store)):
    try:
        t = store.get(task_id)
    except TaskNotFound:
        raise HTTPException(404, detail={"code": "task_not_found",
                                         "message": f"任务 {task_id} 不存在"})
    if t["status"] != "completed" or not t["result_audio"]:
        raise HTTPException(409, detail={"code": "task_not_finished",
                                         "message": f"任务 {task_id} 未完成"})
    path = store.result_audio_abs(t)
    try:
        data = open(path, "rb").read()
    except OSError:
        raise HTTPException(500, detail={"code": "internal_error",
                                         "message": "结果文件缺失"})
    ctype = "audio/mpeg" if (t["response_format"] or "wav") == "mp3" else "audio/wav"
    return Response(content=data, media_type=ctype)


@router.delete("/{task_id}", status_code=204)
def delete_task(task_id: str, store: TaskStore = Depends(get_task_store)):
    try:
        t = store.get(task_id)
    except TaskNotFound:
        raise HTTPException(404, detail={"code": "task_not_found",
                                         "message": f"任务 {task_id} 不存在"})
    if t["status"] == "running":
        # 尝试取消
        wk = _workers.get(task_id)
        if wk and not wk.done():
            wk.cancel()
        store.cancel(task_id)
        return Response(status_code=204)
    store.delete(task_id)
    return Response(status_code=204)