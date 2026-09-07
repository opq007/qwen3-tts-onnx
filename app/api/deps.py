"""API 公共依赖 — 全局应用状态持有。

用 FastAPI app.state 挂载：
  state.config / state.db / state.voice_store / state.task_store / state.orchestrator
"""
from __future__ import annotations

from fastapi import Request

from ..config import Config


def get_config(request: Request) -> Config:
    return request.app.state.config


def get_orchestrator(request: Request):
    return request.app.state.orchestrator


def get_voice_store(request: Request):
    return request.app.state.voice_store


def get_task_store(request: Request):
    return request.app.state.task_store