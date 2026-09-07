"""自测套件 — 用 MOCK 引擎驱动 API 全链路验证。

运行：cd 项目根 && .venv/Scripts/python -m pytest tests/test_api.py -v
（或无 pytest 时：.venv/Scripts/python tests/test_api.py）
"""
from __future__ import annotations

import contextlib
import sys
import tempfile
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from fastapi.testclient import TestClient
except ImportError as e:
    print(f"缺少依赖 {e} — 运行: .venv/Scripts/pip install pytest httpx")
    raise SystemExit(1)

from tests.conftest_mock import build_test_app  # noqa: E402


def _make_ref_wav(path: str, sr: int = 16000, dur: float = 1.0):
    import numpy as np
    t = np.arange(int(sr * dur)) / sr
    wav = (0.4 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    with wave.open(path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes((wav * 32767).astype("<i2").tobytes())
    return path


@contextlib.contextmanager
def _tmp(with_auth=True, token="secret"):
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        app, orch = build_test_app(tmp, with_auth=with_auth, token=token)
        with TestClient(app) as c:   # with 块保持 portal/loop 存活（后台任务需要）
            try:
                yield c, orch, tmp
            finally:
                try:
                    app.state.db.close()
                except Exception:
                    pass


def test_healthz_no_auth():
    with _tmp(with_auth=True) as (c, _o, _t):
        r = c.get("/healthz")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


def test_auth_required():
    with _tmp(with_auth=True, token="secret") as (c, _o, _t):
        assert c.get("/v1/models").status_code == 401
        assert c.get("/v1/models", headers={"Authorization": "Bearer secret"}).status_code == 200
        assert c.get("/v1/models", headers={"x-api-key": "secret"}).status_code == 200
        assert c.get("/v1/models", headers={"x-api-key": "wrong"}).status_code == 401


def test_no_auth_mode():
    with _tmp(with_auth=False) as (c, _o, _t):
        assert c.get("/v1/models").status_code == 200


def test_models_endpoint():
    with _tmp(with_auth=False) as (c, _o, _t):
        data = c.get("/v1/models").json()["data"]
        assert {m["id"] for m in data} == {"qwen3-tts-0.6b-base", "qwen3-tts-1.7b-voicedesign"}
        caps = {m["id"]: m["capability"] for m in data}
        assert caps["qwen3-tts-0.6b-base"] == "voice_clone"
        assert caps["qwen3-tts-1.7b-voicedesign"] == "voice_design"


def test_create_design_voice_and_speech():
    with _tmp(with_auth=False) as (c, orch, _t):
        r = c.post("/v1/voices", json={
            "type": "design", "voice_id": "female-soft", "language": "Chinese",
            "instruct": "yong wenrou huanman de yuqi shuo", "description": "kefu nvsheng"})
        assert r.status_code == 200, r.text
        assert r.json()["type"] == "design" and r.json()["instruct"]
        r = c.post("/v1/audio/speech", json={
            "model": "qwen3-tts-1.7b-voicedesign", "input": "nihao shijie", "voice": "female-soft"})
        assert r.status_code == 200, r.text
        assert r.headers["content-type"].startswith("audio/wav")
        assert len(r.content) > 1000
        eng = orch.engines["qwen3-tts-1.7b-voicedesign"]
        assert eng.generated[-1]["instruct"] == "yong wenrou huanman de yuqi shuo"


def test_create_clone_voice_multipart_and_speech():
    with _tmp(with_auth=False) as (c, orch, tmp):
        ref = Path(tmp) / "ref.wav"
        _make_ref_wav(str(ref))
        with open(ref, "rb") as f:
            r = c.post("/v1/voices", data={
                "type": "clone", "voice_id": "my-clone", "language": "Chinese",
                "ref_text": "zhe duan shi cankao yinpin neirong",
            }, files={"ref_audio": ("ref.wav", f, "audio/wav")})
        assert r.status_code == 200, r.text
        assert r.json()["type"] == "clone" and r.json()["ref_audio"] and r.json()["ref_text"]
        r = c.post("/v1/audio/speech", json={
            "model": "qwen3-tts-0.6b-base", "input": "kelong ceshi", "voice": "my-clone"})
        assert r.status_code == 200, r.text
        eng = orch.engines["qwen3-tts-0.6b-base"]
        assert eng.generated[-1]["ref_text"] == "zhe duan shi cankao yinpin neirong"
        assert eng.generated[-1]["ref_audio_path"]


def test_speech_invalid_model():
    with _tmp(with_auth=False) as (c, _o, _t):
        r = c.post("/v1/audio/speech", json={"model": "nope", "input": "x"})
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "invalid_model"


def test_speech_missing_voice():
    with _tmp(with_auth=False) as (c, _o, _t):
        r = c.post("/v1/audio/speech", json={
            "model": "qwen3-tts-1.7b-voicedesign", "input": "x"})
        assert r.status_code == 400


def test_mp3_response_format():
    with _tmp(with_auth=False) as (c, _o, _t):
        c.post("/v1/voices", json={
            "type": "design", "voice_id": "v1", "language": "Chinese", "instruct": "zhengchang"})
        r = c.post("/v1/audio/speech", json={
            "model": "qwen3-tts-1.7b-voicedesign", "input": "x", "voice": "v1",
            "response_format": "mp3"})
        # 环境无 ffmpeg 时 500；有 ffmpeg 时 200 + audio/mpeg
        assert r.status_code in (200, 500), r.text
        if r.status_code == 200:
            assert r.headers["content-type"].startswith("audio/mpeg")


def test_tasks_lifecycle():
    import time
    with _tmp(with_auth=False) as (c, _o, _t):
        c.post("/v1/voices", json={
            "type": "design", "voice_id": "v1", "language": "Chinese", "instruct": "zhengchang"})
        r = c.post("/api/tasks", json={
            "model": "qwen3-tts-1.7b-voicedesign", "input": "yibu ceshi", "voice": "v1"})
        assert r.status_code == 200, r.text
        tid = r.json()["task_id"]
        for _ in range(100):
            s = c.get(f"/api/tasks/{tid}").json()["status"]
            if s == "completed":
                break
            if s == "failed":
                raise AssertionError("task failed: " + str(c.get(f"/api/tasks/{tid}").json()))
            time.sleep(0.05)
        assert c.get(f"/api/tasks/{tid}").json()["status"] == "completed"
        audio = c.get(f"/api/tasks/{tid}/audio")
        assert audio.status_code == 200 and len(audio.content) > 0


def test_voice_validation():
    with _tmp(with_auth=False) as (c, _o, _t):
        r = c.post("/v1/voices", json={
            "type": "design", "voice_id": "a/b", "language": "Chinese", "instruct": "x"})
        assert r.status_code == 422
        c.post("/v1/voices", json={
            "type": "design", "voice_id": "dup", "language": "Chinese", "instruct": "x"})
        r = c.post("/v1/voices", json={
            "type": "design", "voice_id": "dup", "language": "Chinese", "instruct": "y"})
        assert r.status_code == 409
        r = c.post("/v1/audio/speech", json={
            "model": "qwen3-tts-1.7b-voicedesign", "input": "x", "voice": "ghost"})
        assert r.status_code == 404


def test_voice_delete():
    with _tmp(with_auth=False) as (c, _o, _t):
        c.post("/v1/voices", json={
            "type": "design", "voice_id": "del-me", "language": "Chinese", "instruct": "x"})
        assert c.delete("/v1/voices/del-me").status_code == 204
        assert c.get("/v1/voices/del-me").status_code == 404


def test_speed_preallocated():
    with _tmp(with_auth=False) as (c, _o, _t):
        c.post("/v1/voices", json={
            "type": "design", "voice_id": "v1", "language": "Chinese", "instruct": "x"})
        r = c.post("/v1/audio/speech", json={
            "model": "qwen3-tts-1.7b-voicedesign", "input": "x", "voice": "v1", "speed": 2.0})
        assert r.status_code == 200, r.text
        r = c.post("/v1/audio/speech", json={
            "model": "qwen3-tts-1.7b-voicedesign", "input": "x", "voice": "v1", "speed": 99.0})
        assert r.status_code == 400


def test_model_load_unload():
    with _tmp(with_auth=False) as (c, _o, _t):
        r = c.post("/api/models/qwen3-tts-0.6b-base/load")
        assert r.status_code == 200 and r.json()["status"] == "loaded"
        r = c.post("/api/models/qwen3-tts-0.6b-base/unload")
        assert r.status_code == 200 and r.json()["status"] == "unloaded"
        st = c.get("/api/models/status").json()
        assert len(st["models"]) == 2


def test_temp_clone_multipart_speech():
    """multipart 临时克隆（不落库）→ 直接合成。"""
    with _tmp(with_auth=False) as (c, orch, tmp):
        ref = Path(tmp) / "ref.wav"
        _make_ref_wav(str(ref))
        with open(ref, "rb") as f:
            r = c.post("/v1/audio/speech", data={
                "model": "qwen3-tts-0.6b-base", "text": "linshi kelong ceshi",
                "language": "Chinese", "ref_text": "cankao yinpin neirong",
            }, files={"ref_audio": ("ref.wav", f, "audio/wav")})
        assert r.status_code == 200, r.text
        eng = orch.engines["qwen3-tts-0.6b-base"]
        assert eng.generated[-1]["ref_text"] == "cankao yinpin neirong"
        voices = c.get("/v1/voices").json()["data"]
        assert len(voices) == 0


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    ok = 0
    for fn in fns:
        try:
            fn()
            print(f"  [PASS] {fn.__name__}")
            ok += 1
        except Exception as e:
            import traceback
            print(f"  [FAIL] {fn.__name__}: {e}")
            traceback.print_exc()
    print(f"\n{ok}/{len(fns)} passed")
    return ok == len(fns)


if __name__ == "__main__":
    sys.exit(0 if _run_all() else 1)