"""OpenAI 兼容路由 — /v1/audio/speech + /v1/models。

voice 参数：语音库注册名（JSON）或 multipart 临时参数（不落库）。
response_format: wav | mp3；speed 预留入参（暂不实现变速）。
"""
from __future__ import annotations

import os
import uuid
from typing import Optional

from fastapi import (APIRouter, Depends, File, Form, HTTPException, Request,
                     UploadFile)
from fastapi.responses import Response

from ..config import LANGUAGES, MODEL_BASE, MODEL_DESIGN
from ..services.audio import guess_audio_format
from ..services.orchestrator import (ModelBusy, ModelNotFound, ModelUnavailable,
                                     Orchestrator, SynthesisRequest)
from ..store.voice_store import VoiceNotFound
from .deps import get_orchestrator

router = APIRouter(prefix="/v1", tags=["openai"])

ALLOWED_MODELS = {MODEL_BASE, MODEL_DESIGN}
ALLOWED_FORMATS = {"wav", "mp3"}


@router.post("/audio/speech")
async def audio_speech(request: Request,
                       orchestrator: Orchestrator = Depends(get_orchestrator)):
    """主合成接口。支持 JSON（OpenAI 兼容）与 multipart（临时参考音频，不落库）。

    JSON： {model, input, voice, response_format, speed}
    Multipart： text/model/language/voice/ref_audio/ref_text/instruct
    """
    cfg = request.app.state.config
    ctype = request.headers.get("content-type", "")

    if ctype.startswith("multipart/form-data"):
        return await _speech_multipart(request, orchestrator, cfg)
    return await _speech_json(request, orchestrator, cfg)


async def _speech_json(request, orchestrator, cfg):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, detail={"code": "invalid_json",
                                         "message": "请求 body 不是合法 JSON"})
    model = body.get("model", "")
    text = body.get("input", "")
    voice = body.get("voice")
    resp_fmt = body.get("response_format", "wav")
    speed = body.get("speed", 1.0)

    if model not in ALLOWED_MODELS:
        raise HTTPException(400, detail={"code": "invalid_model",
                                         "message": f"model '{model}' 不存在。"
                                                    f"可用: {sorted(ALLOWED_MODELS)}"})
    if not text:
        raise HTTPException(400, detail={"code": "missing_parameter",
                                         "message": "input 不能为空"})
    if resp_fmt not in ALLOWED_FORMATS:
        raise HTTPException(400, detail={"code": "invalid_response_format",
                                         "message": f"response_format 必须是 {sorted(ALLOWED_FORMATS)}"})
    # speed 预留（0.25~4.0 校验，但不实现变速）
    if speed is not None and not (0.25 <= float(speed) <= 4.0):
        raise HTTPException(400, detail={"code": "invalid_argument",
                                         "message": "speed 必须在 0.25~4.0 之间"})

    return await _run_speech(orchestrator, cfg, model=model, text=text, voice=voice,
                             resp_fmt=resp_fmt)


async def _speech_multipart(request, orchestrator, cfg):
    form = await request.form()
    model = form.get("model", "")
    text = form.get("text") or form.get("input", "")
    voice = form.get("voice")
    language = form.get("language")
    instruct = form.get("instruct")
    ref_text = form.get("ref_text")
    resp_fmt = form.get("response_format", "wav")
    ref_file: Optional[UploadFile] = form.get("ref_audio") or form.get("ref_audio_file")

    if model not in ALLOWED_MODELS:
        raise HTTPException(400, detail={"code": "invalid_model",
                                         "message": f"model '{model}' 不存在"})
    if not text:
        raise HTTPException(400, detail={"code": "missing_parameter",
                                         "message": "缺少 text"})

    # 临时克隆参数（不落库）
    ref_path = None
    if ref_file is not None:
        data = await ref_file.read()
        if len(data) > cfg.max_upload_mb * 1024 * 1024:
            raise HTTPException(413, detail={"code": "upload_too_large",
                                             "message": "上传过大"})
        ext = guess_audio_format(ref_file.filename)
        if ext not in ("wav", "mp3"):
            raise HTTPException(415, detail={"code": "unsupported_audio_format",
                                             "message": "仅支持 wav/mp3"})
        ref_path = os.path.join(cfg.tasks_dir, f"tmp-{uuid.uuid4().hex}.{ext}")
        with open(ref_path, "wb") as f:
            f.write(data)

    try:
        return await _run_speech(orchestrator, cfg, model=model, text=text, voice=voice,
                                 resp_fmt=resp_fmt, language=language,
                                 instruct=instruct, ref_text=ref_text,
                                 ref_audio_path=ref_path)
    finally:
        if ref_path and os.path.exists(ref_path):
            os.unlink(ref_path)


async def _run_speech(orchestrator, cfg, *, model, text, voice, resp_fmt,
                      language=None, instruct=None, ref_text=None, ref_audio_path=None):
    """构造 SynthesisRequest 并执行。临时参数优先于语音库注册参数。"""
    params: dict = {}
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

    # 临时参数覆盖
    if language:
        eff_lang = language
    if instruct is not None:
        params["instruct"] = instruct
    if ref_audio_path is not None:
        if ref_text is None:
            raise HTTPException(422, detail={"code": "missing_ref_text",
                                             "message": "临时克隆需要 ref_text"})
        params["ref_audio_path"] = ref_audio_path
        params["ref_text"] = ref_text
    if not eff_lang:
        raise HTTPException(400, detail={"code": "missing_parameter",
                                         "message": "缺少 language（或所用 voice 未绑定语言）"})
    if eff_lang not in LANGUAGES:
        raise HTTPException(400, detail={"code": "invalid_language",
                                         "message": f"语言必须是 {LANGUAGES}"})

    req = SynthesisRequest(text=text, language=eff_lang, model_id=model,
                           instruct=params.get("instruct"),
                           ref_audio_path=params.get("ref_audio_path"),
                           ref_text=params.get("ref_text"),
                           response_format=resp_fmt)
    try:
        audio, ctype = await orchestrator.synthesize_sync(req)
    except ModelBusy as e:
        raise HTTPException(429, detail={"code": "model_busy", "message": str(e)})
    except ModelNotFound as e:
        raise HTTPException(404, detail={"code": "model_not_found", "message": str(e)})
    except ModelUnavailable as e:
        raise HTTPException(503, detail={"code": "model_unavailable", "message": str(e)})
    except Exception as e:
        raise HTTPException(500, detail={"code": "inference_failed", "message": str(e)})
    return Response(content=audio, media_type=ctype)


@router.get("/models")
def list_models(orchestrator: Orchestrator = Depends(get_orchestrator)):
    from ..config import LANGUAGES
    data = []
    for j in orchestrator.models_status():
        data.append({
            "id": j["id"],
            "object": "model",
            "created": 1750000000,
            "owned_by": "local",
            "capability": j["capability"],
            "languages": LANGUAGES,
            "status": j["status"],
        })
    return {"object": "list", "data": data}