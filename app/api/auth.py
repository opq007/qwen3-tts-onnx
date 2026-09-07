"""鉴权中间件 — 单一静态 token，支持 Bearer 与 x-api-key。

豁免：/healthz 恒豁免；/docs 按配置可豁免。
未配置 token → 无鉴权模式（日志告警）。
"""
from __future__ import annotations

import hmac
import logging

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger("q3tts.auth")

EXEMPT_PREFIXES = ("/healthz", "/docs", "/redoc", "/openapi.json", "/static")
# 静态资源（UI）也需要鉴权 —— UI 登录后通过 Authorization 头访问静态页。
# 但浏览器直接 GET / 加载 html 需要无鉴权返回登录页。这里：/ 和静态资源一律要求鉴权，
# 未带 token 时返回 401 而非 404，UI 拦截 401 显示登录页。
# 例外：静态资源经 UI 的 fetch 附加 header 即可。为简单，静态资源默认豁免，
# UI 页面本身通过应用内鉴权保护（后端接口均有 token）。
AUTH_EXEMPT = ("/healthz", "/docs", "/redoc", "/openapi.json", "/static")


class AuthMiddleware(BaseHTTPMiddleware):
    """token 从 app.state.config.api_token 动态读取，便于测试注入与动态配置。"""

    def __init__(self, app, docs_public: bool = True):
        super().__init__(app)
        self.docs_public = docs_public

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if self._exempt(path):
            return await call_next(request)

        cfg = getattr(request.app.state, "config", None)
        token = (getattr(cfg, "api_token", "") or "") if cfg else ""
        if not token:
            return await call_next(request)  # 无鉴权模式

        tok = self._extract(request)
        if not tok or not self._verify(tok, token):
            return JSONResponse(
                status_code=401,
                content={"error": {"code": "unauthorized",
                                   "message": "missing or invalid API token"}})
        return await call_next(request)

    def _exempt(self, path: str) -> bool:
        if path in ("/", "/healthz"):
            return True
        if self.docs_public and path.startswith(("/docs", "/redoc", "/openapi.json")):
            return True
        if path.startswith("/static"):
            return True
        return False

    @staticmethod
    def _extract(request: Request) -> str:
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            return auth[7:].strip()
        return (request.headers.get("x-api-key") or "").strip()

    @staticmethod
    def _verify(token: str, expected: str) -> bool:
        return hmac.compare_digest(token, expected)