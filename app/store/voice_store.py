"""语音库存储 — voices 表 CRUD + 参考音频文件管理。"""
from __future__ import annotations

import os
import re
import sqlite3
from pathlib import Path
from typing import Optional

from .schema import now_iso

VOICE_ID_RE = re.compile(r"^[A-Za-z0-9_\u4e00-\u9fa5-]{1,64}$")
ALLOWED_EXTS = {".wav", ".mp3"}


class VoiceNotFound(Exception):
    pass


class VoiceExists(Exception):
    pass


class VoiceStore:
    def __init__(self, conn: sqlite3.Connection, voices_dir: str):
        self.conn = conn
        self.voices_dir = voices_dir
        Path(voices_dir).mkdir(parents=True, exist_ok=True)

    # ── helpers ──────────────────────────────────────────────────────────
    @staticmethod
    def validate_voice_id(voice_id: str) -> str:
        if not VOICE_ID_RE.match(voice_id):
            raise ValueError(f"voice_id 非法：仅允许字母/数字/中文/_/-/，长度1-64，got {voice_id!r}")
        return voice_id

    def _audio_path(self, voice_id: str, ext: str) -> str:
        return os.path.join(self.voices_dir, f"{voice_id}.{ext.lstrip('.')}")

    # ── CRUD ─────────────────────────────────────────────────────────────
    def list(self, type_: Optional[str] = None, language: Optional[str] = None) -> list[dict]:
        q = "SELECT * FROM voices WHERE 1=1"
        args: list = []
        if type_:
            q += " AND type=?"
            args.append(type_)
        if language:
            q += " AND language=?"
            args.append(language)
        q += " ORDER BY created_at DESC"
        rows = self.conn.execute(q, args).fetchall()
        return [dict(r) for r in rows]

    def get(self, voice_id: str) -> dict:
        row = self.conn.execute("SELECT * FROM voices WHERE voice_id=?", (voice_id,)).fetchone()
        if not row:
            raise VoiceNotFound(voice_id)
        return dict(row)

    def create_clone(self, voice_id: str, language: str, ref_text: str,
                     audio_file: str, audio_ext: str, description: str = "",
                     duration_sec: Optional[float] = None) -> dict:
        """audio_file 为已保存的绝对路径（含扩展名）。audio_ext 为 .wav/.mp3。"""
        voice_id = self.validate_voice_id(voice_id)
        if not ref_text:
            raise ValueError("克隆语音需要 ref_text（参考音频转写文本）。")
        if audio_ext not in ALLOWED_EXTS:
            raise ValueError(f"参考音频格式不支持：{audio_ext}（仅支持 wav/mp3）")
        self._ensure_not_exists(voice_id)
        rel = self._save_audio_file(voice_id, audio_ext, audio_file)
        ts = now_iso()
        self.conn.execute(
            "INSERT INTO voices (voice_id,type,language,instruct,ref_audio,ref_text,"
            "description,duration_sec,audio_format,created_at,updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (voice_id, "clone", language, None, rel, ref_text, description,
             duration_sec, audio_ext.lstrip("."), ts, ts))
        self.conn.commit()
        return self.get(voice_id)

    def create_design(self, voice_id: str, language: str, instruct: str,
                      description: str = "") -> dict:
        voice_id = self.validate_voice_id(voice_id)
        if not instruct:
            raise ValueError("设计语音需要 instruct（声音描述指令）。")
        self._ensure_not_exists(voice_id)
        ts = now_iso()
        self.conn.execute(
            "INSERT INTO voices (voice_id,type,language,instruct,ref_audio,ref_text,"
            "description,duration_sec,audio_format,created_at,updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (voice_id, "design", language, instruct, None, None, description,
             None, "wav", ts, ts))
        self.conn.commit()
        return self.get(voice_id)

    def update(self, voice_id: str, language: Optional[str] = None,
               instruct: Optional[str] = None, ref_text: Optional[str] = None,
               description: Optional[str] = None) -> dict:
        v = self.get(voice_id)
        fields = []
        args: list = []
        if language is not None:
            fields.append("language=?"); args.append(language)
        if v["type"] == "design" and instruct is not None:
            fields.append("instruct=?"); args.append(instruct)
        if v["type"] == "clone" and ref_text is not None:
            fields.append("ref_text=?"); args.append(ref_text)
        if description is not None:
            fields.append("description=?"); args.append(description)
        if fields:
            fields.append("updated_at=?")
            args.append(now_iso())
            args.append(voice_id)
            self.conn.execute(f"UPDATE voices SET {', '.join(fields)} WHERE voice_id=?", args)
            self.conn.commit()
        return self.get(voice_id)

    def delete(self, voice_id: str) -> None:
        v = self.get(voice_id)
        # 级联删除音频文件
        if v.get("ref_audio"):
            p = Path(v["ref_audio"])
            if not p.is_absolute():
                p = Path(self.voices_dir) / p
            try:
                if p.exists():
                    p.unlink()
            except OSError:
                pass
        self.conn.execute("DELETE FROM voices WHERE voice_id=?", (voice_id,))
        self.conn.commit()

    def test_text(self, voice_id: str, text: Optional[str] = None) -> tuple[dict, str]:
        """返回 (voice, effective_text) 用于试听。"""
        v = self.get(voice_id)
        eff = text or "这是一段试听语音。这是一段试听语音。"
        return v, eff

    # ── 内部 ────────────────────────────────────────────────────────────
    def _ensure_not_exists(self, voice_id: str):
        if self.conn.execute("SELECT 1 FROM voices WHERE voice_id=?", (voice_id,)).fetchone():
            raise VoiceExists(voice_id)

    def _save_audio_file(self, voice_id: str, ext: str, src_path: str) -> str:
        """把上传的临时文件移动/复制到持久目录。返回相对路径。"""
        dst = self._audio_path(voice_id, ext)
        # 删除旧文件（若有）
        for e in ALLOWED_EXTS:
            old = self._audio_path(voice_id, e)
            if old != dst and Path(old).exists():
                try:
                    Path(old).unlink()
                except OSError:
                    pass
        # 若已在目标位置（multipart 已落到持久目录）则不重复移动
        if os.path.abspath(src_path) == os.path.abspath(dst):
            return os.path.relpath(dst, self.voices_dir)
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.move(src_path, dst)
        return os.path.relpath(dst, self.voices_dir)

    def resolve_audio_abs(self, v: dict) -> str:
        p = Path(v["ref_audio"])
        if not p.is_absolute():
            p = Path(self.voices_dir) / p
        return str(p)