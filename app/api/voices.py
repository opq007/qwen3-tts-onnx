"""语音库管理路由 — /v1/voices。

创建克隆（multipart）+ 创建设计（JSON）+ 查改删 + 试听。
参考音频上传保存到 DATA_DIR/voices/（持久目录，非临时）。
"""
from __future__ import annotations

import io
import os
import uuid
from pathlib import Path
from typing import Optional

from fastapi import (APIRouter, Depends, File, Form, HTTPException, Request,
                     UploadFile)
from fastapi.responses import Response

from ..config import LANGUAGES, MODEL_BASE, MODEL_DESIGN
from ..services.audio import guess_audio_format
from ..store.schema import now_iso
from ..store.voice_store import (ALLOWED_EXTS, VoiceExists, VoiceNotFound,
                                 VoiceStore)
from .deps import get_voice_store, get_orchestrator

router = APIRouter(prefix="/v1/voices", tags=["voices"])


@router.get("")
def list_voices(type_: Optional[str] = None, language: Optional[str] = None,
                store: VoiceStore = Depends(get_voice_store)):
    if type_ and type_ not in ("clone", "design"):
        raise HTTPException(400, detail={"code": "invalid_argument",
                                         "message": "type 必须是 clone 或 design"})
    return {"object": "list", "data": store.list(type_=type_, language=language)}


@router.post("")
async def create_voice(
    request: Request,
    store: VoiceStore = Depends(get_voice_store),
):
    """创建设计语音（JSON）或克隆语音（multipart）。

    JSON:   {type: "design", voice_id, language, instruct, description}
    Multipart: type/voice_id/language/ref_audio/ref_text/description
    """
    cfg = request.app.state.config
    ctype = request.headers.get("content-type", "")

    if ctype.startswith("multipart/form-data"):
        form = await request.form()
        type_ = form.get("type") or "design"
        voice_id = form.get("voice_id", "")
        language = form.get("language", "")
        instruct = form.get("instruct")
        ref_text = form.get("ref_text")
        description = form.get("description", "") or ""
        ref_file: Optional[UploadFile] = form.get("ref_audio")
    else:
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(400, detail={"code": "invalid_json", "message": "body 不是 JSON"})
        type_ = body.get("type", "design")
        voice_id = body.get("voice_id", "")
        language = body.get("language", "")
        instruct = body.get("instruct")
        ref_text = body.get("ref_text")
        description = body.get("description", "") or ""
        ref_file = None

    if not voice_id:
        raise HTTPException(422, detail={"code": "missing_parameter",
                                         "message": "voice_id 必填"})
    if language not in LANGUAGES:
        raise HTTPException(400, detail={"code": "invalid_language",
                                         "message": f"语言必须是 {LANGUAGES}"})

    if type_ == "design":
        if not instruct:
            raise HTTPException(422, detail={"code": "missing_instruct",
                                             "message": "设计语音需要 instruct"})
        try:
            return store.create_design(voice_id, language, instruct, description)
        except VoiceExists:
            raise HTTPException(409, detail={"code": "voice_exists",
                                             "message": f"语音 {voice_id} 已存在"})
        except ValueError as e:
            raise HTTPException(422, detail={"code": "invalid_argument", "message": str(e)})

    elif type_ == "clone":
        if ref_file is None:
            raise HTTPException(422, detail={"code": "missing_ref_audio",
                                             "message": "克隆语音需要上传 ref_audio"})
        if not ref_text:
            raise HTTPException(422, detail={"code": "missing_ref_text",
                                             "message": "克隆语音需要 ref_text（参考音频转写文本）"})
        data = await ref_file.read()
        if len(data) > cfg.max_upload_mb * 1024 * 1024:
            raise HTTPException(413, detail={"code": "upload_too_large",
                                             "message": f"上传超过 {cfg.max_upload_mb}MB 限制"})
        ext = guess_audio_format(ref_file.filename)
        if ext not in ("wav", "mp3"):
            raise HTTPException(415, detail={"code": "unsupported_audio_format",
                                             "message": "仅支持 wav/mp3 参考音频"})
        tmp = os.path.join(cfg.voices_dir, f".upload-{uuid.uuid4().hex}.{ext}")
        Path(tmp).write_bytes(data)
        try:
            v = store.create_clone(voice_id, language, ref_text, tmp,
                                   audio_ext=f".{ext}", description=description)
        except VoiceExists:
            Path(tmp).unlink(missing_ok=True)
            raise HTTPException(409, detail={"code": "voice_exists",
                                             "message": f"语音 {voice_id} 已存在"})
        except ValueError as e:
            Path(tmp).unlink(missing_ok=True)
            raise HTTPException(422, detail={"code": "invalid_argument", "message": str(e)})
        return v

    raise HTTPException(400, detail={"code": "invalid_argument",
                                     "message": "type 必须是 clone 或 design"})


@router.get("/{voice_id}")
def get_voice(voice_id: str, store: VoiceStore = Depends(get_voice_store)):
    try:
        return store.get(voice_id)
    except VoiceNotFound:
        raise HTTPException(404, detail={"code": "voice_not_found",
                                         "message": f"语音 {voice_id} 不存在"})


@router.patch("/{voice_id}")
def update_voice(voice_id: str, body: dict,
                 store: VoiceStore = Depends(get_voice_store)):
    if body.get("type") or body.get("voice_id"):
        raise HTTPException(403, detail={"code": "immutable_field",
                                         "message": "type 与 voice_id 不可修改（先删再建）"})
    try:
        v = store.update(voice_id,
                         language=body.get("language"),
                         instruct=body.get("instruct"),
                         ref_text=body.get("ref_text"),
                         description=body.get("description"))
    except VoiceNotFound:
        raise HTTPException(404, detail={"code": "voice_not_found",
                                         "message": f"语音 {voice_id} 不存在"})
    return v


@router.delete("/{voice_id}", status_code=204)
def delete_voice(voice_id: str, store: VoiceStore = Depends(get_voice_store)):
    try:
        store.delete(voice_id)
    except VoiceNotFound:
        raise HTTPException(404, detail={"code": "voice_not_found",
                                         "message": f"语音 {voice_id} 不存在"})
    return Response(status_code=204)


@router.post("/{voice_id}/test")
async def test_voice(voice_id: str, body: dict,
                     store: VoiceStore = Depends(get_voice_store),
                     orchestrator=Depends(get_orchestrator)):
    try:
        v, text = store.test_text(voice_id, body.get("text"))
    except VoiceNotFound:
        raise HTTPException(404, detail={"code": "voice_not_found",
                                         "message": f"语音 {voice_id} 不存在"})

    from ..services.orchestrator import SynthesisRequest
    params = orchestrator.resolve_voice_params(voice_id)
    # 试听默认短文本
    text = body.get("text") or "这是一段试听语音，用来验证声音效果。"
    model_id = MODEL_BASE if v["type"] == "clone" else MODEL_DESIGN
    req = SynthesisRequest(text=text, language=v["language"], model_id=model_id,
                           instruct=params.get("instruct"),
                           ref_audio_path=params.get("ref_audio_path"),
                           ref_text=params.get("ref_text"),
                           response_format="wav")
    try:
        audio, ctype = await orchestrator.synthesize_sync(req)
    except Exception as e:
        raise HTTPException(500, detail={"code": "inference_failed", "message": str(e)})
    return Response(content=audio, media_type=ctype)