"""音频工具 — ffmpeg 封装（mp3 解码/编码）+ 音频格式识别。"""
from __future__ import annotations

import io
import os
import subprocess
from typing import Optional

import numpy as np

SR = 24000


class AudioError(Exception):
    pass


def guess_audio_format(filename: str) -> str:
    """按扩展名/魔数推断音频格式，返回 'wav' | 'mp3' | None。"""
    name = (filename or "").lower()
    if name.endswith(".mp3"):
        return "mp3"
    if name.endswith(".wav") or name.endswith(".wave"):
        return "wav"
    # 魔数检测
    try:
        with open(filename, "rb") as f:
            head = f.read(12)
        if head.startswith(b"RIFF") and head[8:12] == b"WAVE":
            return "wav"
        if head.startswith(b"ID3") or (len(head) >= 2 and head[:2] in (b"\xff\xfb", b"\xff\xf3")):
            return "mp3"
    except OSError:
        pass
    return None


def _run_ffmpeg(args: list[str], ffmpeg_path: str = "ffmpeg", timeout: float = 60.0) -> bytes:
    try:
        proc = subprocess.run([ffmpeg_path, *args],
                              capture_output=True, timeout=timeout)
    except FileNotFoundError:
        raise AudioError("ffmpeg 未找到。请安装 ffmpeg 或配置 audio.ffmpeg_path。"
                         "Docker 镜像已内置。") from None
    except subprocess.TimeoutExpired:
        raise AudioError("ffmpeg 处理超时。") from None
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace")[-500:]
        raise AudioError(f"ffmpeg 失败: {err}")
    return proc.stdout


def decode_to_wav_bytes(src: str, ffmpeg_path: str = "ffmpeg",
                        max_seconds: Optional[int] = None) -> bytes:
    """把任意支持的音频(wav/mp3)解码为 24kHz 16bit 单声道 wav 字节。

    - wav: 用 soundfile + librosa 重采样（无需 ffmpeg）
    - mp3: 用 ffmpeg 解码
    """
    fmt = guess_audio_format(src)

    # wav 走 soundfile（无 ffmpeg 依赖）
    if fmt == "wav":
        import soundfile as sf
        data, sr = sf.read(src, dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        if max_seconds is not None and len(data) > max_seconds * SR:
            data = data[: max_seconds * SR]
        if sr != SR:
            import librosa
            data = librosa.resample(data, orig_sr=sr, target_sr=SR)
        buf = io.BytesIO()
        sf.write(buf, data.astype(np.float32), SR, format="WAV", subtype="PCM_16")
        return buf.getvalue()

    # mp3（或未知格式）：ffmpeg 解码
    args = ["-y", "-v", "error", "-i", src]
    if max_seconds is not None:
        args += ["-t", str(max_seconds)]
    args += ["-ar", str(SR), "-ac", "1", "-sample_fmt", "s16",
             "-f", "wav", "-"]
    return _run_ffmpeg(args, ffmpeg_path)


def encode_mp3(wav_bytes: bytes, ffmpeg_path: str = "ffmpeg",
               bitrate: str = "128k") -> bytes:
    """把 wav 字节编码为 mp3 字节。"""
    args = ["-y", "-v", "error", "-i", "pipe:0",
            "-b:a", bitrate, "-f", "mp3", "pipe:1"]
    try:
        proc = subprocess.run([ffmpeg_path, *args], input=wav_bytes,
                              capture_output=True, timeout=60.0)
    except FileNotFoundError:
        raise AudioError("ffmpeg 未找到。请安装 ffmpeg 或配置 audio.ffmpeg_path。") from None
    except subprocess.TimeoutExpired:
        raise AudioError("ffmpeg mp3 编码超时。") from None
    if proc.returncode != 0:
        raise AudioError("ffmpeg mp3 编码失败: "
                         f"{proc.stderr.decode('utf-8', errors='replace')[-300:]}")
    return proc.stdout


def save_temp_upload(data: bytes, dest: str) -> None:
    """把上传字节写入目标文件（服务在 multipart 中直接保存到持久目录）。"""
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    with open(dest, "wb") as f:
        f.write(data)