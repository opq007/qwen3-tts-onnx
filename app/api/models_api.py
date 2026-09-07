"""服务管理路由 — /api/models（状态/加载/卸载）+ /api/config + /healthz。"""
from __future__ import annotations

import time

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from ..services.orchestrator import (ModelNotFound, ModelUnavailable,
                                     Orchestrator)
from .deps import get_orchestrator

router = APIRouter(prefix="/api", tags=["models"])
health_router = APIRouter(tags=["health"])


@router.get("/models/status")
def models_status(orchestrator: Orchestrator = Depends(get_orchestrator)):
    return {"models": orchestrator.models_status()}


@router.post("/models/{model_id}/load")
async def load_model(model_id: str,
                     orchestrator: Orchestrator = Depends(get_orchestrator)):
    try:
        state = await orchestrator.load_model(model_id, wait=True)
    except ModelNotFound:
        raise HTTPException(404, detail={"code": "model_not_found",
                                         "message": f"模型 {model_id} 不存在"})
    except Exception as e:
        raise HTTPException(500, detail={"code": "load_failed", "message": str(e)})
    return {"id": model_id, "status": state}


@router.post("/models/{model_id}/unload")
async def unload_model(model_id: str,
                       orchestrator: Orchestrator = Depends(get_orchestrator)):
    try:
        state = await orchestrator.unload_model(model_id)
    except ModelNotFound:
        raise HTTPException(404, detail={"code": "model_not_found",
                                         "message": f"模型 {model_id} 不存在"})
    return {"id": model_id, "status": state}


@router.get("/config")
def get_config_api(request: Request):
    cfg = request.app.state.config
    return {
        "host": cfg.host,
        "port": cfg.port,
        "auth_enabled": cfg.auth_enabled,
        "docs_public": cfg.docs_public,
        "data_dir": cfg.data_dir,
        "models_dir": cfg.models_dir,
        "default_voice": cfg.default_voice,
        "ffmpeg_path": cfg.ffmpeg_path,
        "mp3_bitrate": cfg.mp3_bitrate,
        "max_ref_audio_seconds": cfg.max_ref_audio_seconds,
        "max_upload_mb": cfg.max_upload_mb,
        "task_ttl_seconds": cfg.task_ttl_seconds,
        "max_queue_length": cfg.max_queue_length,
        "log_level": cfg.log_level,
        "models": [{"id": m.id, "name": m.name, "capability": m.capability,
                    "load_on_start": m.load_on_start,
                    "unload_after_idle_seconds": m.unload_after_idle_seconds,
                    "max_new_tokens": m.max_new_tokens,
                    "onnx_dir": m.onnx_dir, "tts_dir": m.tts_dir}
                   for m in cfg.models],
    }


@health_router.get("/healthz", include_in_schema=False)
def healthz(request: Request):
    orch: Orchestrator = request.app.state.orchestrator
    return {
        "status": "ok",
        "version": "0.1.0",
        "uptime_sec": int(time.time() - orch.startup_time),
        "models": [{"id": m["id"], "status": m["status"]} for m in orch.models_status()],
    }