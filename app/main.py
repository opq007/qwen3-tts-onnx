"""FastAPI 应用装配 — 路由挂载 + 中间件 + 生命周期。

启动：
  uvicorn app.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .api import models_api, openai, tasks, voices
from .api.auth import AuthMiddleware
from .config import load_config
from .services.orchestrator import Orchestrator
from .store.schema import init_db
from .store.task_store import TaskStore
from .store.voice_store import VoiceStore

logger = logging.getLogger("q3tts")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s [%(name)s] %(message)s")

app = FastAPI(title="Qwen3-TTS ONNX", version="0.1.0",
              description="纯 CPU Qwen3-TTS 部署服务（语音克隆 + 声音设计）")


# 统一错误 envelope：{"error": {"code", "message"}}
from fastapi import HTTPException as FastAPIHTTPException, Request as Req
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse as _JR


@app.exception_handler(FastAPIHTTPException)
async def _http_exc_handler(request: Req, exc: FastAPIHTTPException):
    detail = exc.detail
    if isinstance(detail, dict) and "code" in detail:
        return _JR(status_code=exc.status_code,
                   content={"error": detail})
    return _JR(status_code=exc.status_code,
               content={"error": {"code": "http_error", "message": str(detail)}})


@app.exception_handler(RequestValidationError)
async def _validation_handler(request: Req, exc: RequestValidationError):
    return _JR(status_code=422,
               content={"error": {"code": "validation_error",
                                  "message": "参数校验失败",
                                  "details": exc.errors()[:5]}})


def _warn_missing_model_dirs(cfg) -> None:
    """启动前检查配置的模型目录是否存在，缺失时给出明确提示。"""
    for m in cfg.models:
        from pathlib import Path as _P
        onnx = _P(m.onnx_dir)
        if not onnx.exists() or not (onnx / "manifest.json").exists():
            logger.warning(
                "模型 %s 的 ONNX 目录缺失: %s —— 请运行 "
                "python scripts/download_models.py（或一键启动脚本自动下载）",
                m.id, m.onnx_dir)


def create_app() -> FastAPI:
    cfg = load_config()

    # 目录
    for d in (cfg.data_dir, cfg.voices_dir, cfg.tasks_dir):
        Path(d).mkdir(parents=True, exist_ok=True)

    # 存储
    conn = init_db(cfg.db_path)
    voice_store = VoiceStore(conn, cfg.voices_dir)
    task_store = TaskStore(conn, cfg.tasks_dir)
    orchestrator = Orchestrator(cfg, voice_store)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        """启动与关闭钩子（替代已弃用的 on_event）。"""
        from concurrent.futures import ThreadPoolExecutor
        import threading
        # 预加载（load_on_start=true）
        def _load_all():
            try:
                orchestrator.load_all_on_start()
            except Exception as e:
                logger.error("预加载失败（可稍后手动加载）: %s", e)

        orchestrator._executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 2)
        threading.Thread(target=_load_all, daemon=True).start()
        # 空闲卸载循环
        if any(m.unload_after_idle_seconds > 0 for m in cfg.models):
            asyncio.create_task(orchestrator.idle_unload_loop())
        # 清理过期任务
        try:
            n = task_store.cleanup_expired(cfg.task_ttl_seconds)
            if n:
                logger.info("清理 %d 个过期任务", n)
        except Exception as e:
            logger.warning("清理过期任务失败: %s", e)
        try:
            yield
        finally:
            if getattr(orchestrator, "_executor", None):
                orchestrator._executor.shutdown(wait=False)

    app.router.lifespan_context = lifespan
    app.state.config = cfg
    app.state.db = conn
    app.state.voice_store = voice_store
    app.state.task_store = task_store
    app.state.orchestrator = orchestrator

    # 启动前目录检查（模型未下载时给出明确指引）
    _warn_missing_model_dirs(cfg)

    # 鉴权中间件（最后注册 = 最外层，覆盖所有路由）
    # token 从 app.state.config.api_token 动态读取。
    # 注意：/（UI 登录页）与 /static 豁免鉴权 —— 前端加载后自行检测 401 显示登录页，
    # 后续所有 API 请求带 Authorization 头。
    app.add_middleware(AuthMiddleware, docs_public=cfg.docs_public)

    # 路由
    app.include_router(openai.router)
    app.include_router(voices.router)
    app.include_router(models_api.router)
    app.include_router(models_api.health_router)
    app.include_router(tasks.router)

    # 静态 UI（无鉴权路径 /static 与 / 由中间件处理：/ 不豁免 → 登录页经请求返回）
    web_dir = Path(__file__).resolve().parent.parent / "web"
    if web_dir.exists():
        app.mount("/static", StaticFiles(directory=web_dir), name="static")

        @app.get("/", include_in_schema=False)
        async def index():
            from fastapi.responses import FileResponse
            return FileResponse(web_dir / "index.html")
    else:
        @app.get("/", include_in_schema=False)
        async def index_missing():
            return {"msg": "web/ 目录缺失"}

    return app


create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))