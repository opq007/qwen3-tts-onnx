"""任务存储 — tasks 表 CRUD + 异步任务结果管理。"""
from __future__ import annotations

import os
import sqlite3
import uuid
from pathlib import Path
from typing import Optional

from .schema import now_iso

TASK_STATUSES = ("pending", "running", "completed", "failed", "cancelled")


class TaskNotFound(Exception):
    pass


class TaskStore:
    def __init__(self, conn: sqlite3.Connection, tasks_dir: str):
        self.conn = conn
        self.tasks_dir = tasks_dir
        Path(tasks_dir).mkdir(parents=True, exist_ok=True)

    def create(self, model_id: str, input_text: str, language: Optional[str] = None,
               voice_id: Optional[str] = None, instruct: Optional[str] = None,
               ref_audio: Optional[str] = None, ref_text: Optional[str] = None,
               response_format: str = "wav") -> dict:
        task_id = str(uuid.uuid4())
        ts = now_iso()
        self.conn.execute(
            "INSERT INTO tasks (task_id,model_id,status,input_text,language,voice_id,"
            "instruct,ref_audio,ref_text,response_format,created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (task_id, model_id, "pending", input_text, language, voice_id,
             instruct, ref_audio, ref_text, response_format, ts))
        self.conn.commit()
        return self.get(task_id)

    def get(self, task_id: str) -> dict:
        row = self.conn.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
        if not row:
            raise TaskNotFound(task_id)
        return dict(row)

    def list(self, status: Optional[str] = None, limit: int = 50, offset: int = 0) -> list[dict]:
        q = "SELECT * FROM tasks"
        args: list = []
        if status:
            q += " WHERE status=?"
            args.append(status)
        q += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        args += [limit, offset]
        return [dict(r) for r in self.conn.execute(q, args).fetchall()]

    def set_running(self, task_id: str, progress_step: str = "queued") -> None:
        self.conn.execute(
            "UPDATE tasks SET status='running', started_at=?, progress_step=? WHERE task_id=?",
            (now_iso(), progress_step, task_id))
        self.conn.commit()

    def set_progress(self, task_id: str, step: str, frame: Optional[int] = None,
                     max_frames: Optional[int] = None) -> None:
        self.conn.execute(
            "UPDATE tasks SET progress_step=?, progress_frame=COALESCE(?,progress_frame),"
            " max_frames=COALESCE(?,max_frames) WHERE task_id=?",
            (step, frame, max_frames, task_id))
        self.conn.commit()

    def complete(self, task_id: str, result_audio: str, sample_rate: int,
                 duration_sec: float) -> None:
        self.conn.execute(
            "UPDATE tasks SET status='completed', result_audio=?, sample_rate=?,"
            " duration_sec=?, finished_at=? WHERE task_id=?",
            (result_audio, sample_rate, duration_sec, now_iso(), task_id))
        self.conn.commit()

    def fail(self, task_id: str, error_code: str, error_message: str) -> None:
        self.conn.execute(
            "UPDATE tasks SET status='failed', error_code=?, error_message=?, finished_at=? "
            "WHERE task_id=?",
            (error_code, error_message[:2000], now_iso(), task_id))
        self.conn.commit()

    def cancel(self, task_id: str) -> None:
        self.conn.execute(
            "UPDATE tasks SET status='cancelled', finished_at=? WHERE task_id=?",
            (now_iso(), task_id))
        self.conn.commit()

    def delete(self, task_id: str) -> None:
        # 级联删除结果文件
        t = self.get(task_id)
        if t.get("result_audio"):
            p = Path(t["result_audio"])
            if not p.is_absolute():
                p = Path(self.tasks_dir) / p
            try:
                if p.exists():
                    p.unlink()
            except OSError:
                pass
        self.conn.execute("DELETE FROM tasks WHERE task_id=?", (task_id,))
        self.conn.commit()

    def result_audio_abs(self, task: dict) -> str:
        p = Path(task["result_audio"])
        if not p.is_absolute():
            p = Path(self.tasks_dir) / p
        return str(p)

    def save_result(self, task_id: str, ext: str, data: bytes) -> str:
        fname = f"{task_id}.{ext.lstrip('.')}"
        dst = os.path.join(self.tasks_dir, fname)
        Path(dst).write_bytes(data)
        return os.path.relpath(dst, self.tasks_dir)

    def cleanup_expired(self, ttl_seconds: int) -> int:
        """删除 TTL 过期任务（仅 completed/failed/cancelled），返回清理数量。"""
        from datetime import datetime, timedelta, timezone
        cutoff = (datetime.now(timezone.utc) - timedelta(seconds=ttl_seconds))
        rows = self.conn.execute(
            "SELECT task_id FROM tasks WHERE status IN ('completed','failed','cancelled') "
            "AND finished_at < ?", (cutoff.isoformat(timespec="seconds"),)).fetchall()
        n = 0
        for r in rows:
            try:
                self.delete(r["task_id"])
                n += 1
            except (TaskNotFound, OSError):
                pass
        return n