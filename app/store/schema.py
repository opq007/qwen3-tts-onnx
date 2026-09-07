"""存储层 — SQLite schema 与连接管理。

SQLite WAL 模式，单进程单 worker（FastAPI 同步线程），天然串行写。
启动时按 user_version 执行增量迁移。
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

_SCHEMA_V1 = """
CREATE TABLE IF NOT EXISTS voices (
    voice_id      TEXT PRIMARY KEY,
    type          TEXT NOT NULL CHECK (type IN ('clone','design')),
    language      TEXT NOT NULL,
    instruct      TEXT,
    ref_audio     TEXT,
    ref_text      TEXT,
    description   TEXT DEFAULT '',
    duration_sec  REAL,
    audio_format  TEXT DEFAULT 'wav',
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tasks (
    task_id       TEXT PRIMARY KEY,
    model_id      TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'pending'
                  CHECK (status IN ('pending','running','completed','failed','cancelled')),
    input_text    TEXT NOT NULL,
    language      TEXT,
    voice_id      TEXT,
    instruct      TEXT,
    ref_audio     TEXT,
    ref_text      TEXT,
    response_format TEXT DEFAULT 'wav',
    result_audio  TEXT,
    sample_rate   INTEGER,
    duration_sec  REAL,
    progress_step TEXT,
    progress_frame INTEGER,
    max_frames    INTEGER,
    error_code    TEXT,
    error_message TEXT,
    created_at    TEXT NOT NULL,
    started_at    TEXT,
    finished_at   TEXT
);
CREATE INDEX IF NOT EXISTS idx_tasks_status_created ON tasks(status, created_at DESC);
"""


def init_db(db_path: str) -> sqlite3.Connection:
    """初始化数据库（建目录 + 建表），返回连接（row_factory=sq3.Row）。

    check_same_thread=False: FastAPI 在 threadpool 中执行同步路由，跨线程使用连接。
    单 worker 下由事件循环串行化访问，结合 WAL + busy_timeout 是安全的。
    """
    parent = os.path.dirname(os.path.abspath(db_path))
    Path(parent).mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA foreign_keys=ON")
    # 迁移
    ver = conn.execute("PRAGMA user_version").fetchone()[0]
    if ver < 1:
        conn.executescript(_SCHEMA_V1)
        conn.execute("PRAGMA user_version=1")
    conn.commit()
    return conn


def now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="seconds")