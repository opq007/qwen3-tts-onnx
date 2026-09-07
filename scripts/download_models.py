#!/usr/bin/env python3
"""模型下载脚本 — 从 HF 拉取 ONNX 子模型 + 原版 checkpoint 配置/分词文件。

用法:
  python scripts/download_models.py [--models-dir ./models] [--precision cpu_int4]
  python scripts/download_models.py --model 0.6b  # 仅下载一个模型

说明:
- ONNX 部分从 onnx-community 仓库拉取指定 precision 目录（默认 cpu_int4）
- 配置/分词部分从原版 Qwen 仓库拉取（config.json/generation_config.json/tokenizer 等）
- 需要 HF_TOKEN 环境变量拉取 gated 模型（可选）
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

try:
    from huggingface_hub import snapshot_download, hf_hub_download
except ImportError:
    print("需要 huggingface-hub: pip install huggingface-hub")
    raise SystemExit(1)

ONNX_REPOS = {
    "0.6b": "onnx-community/Qwen3-TTS-12Hz-0.6B-Base",
    "1.7b": "onnx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
}
ORIGINAL_REPOS = {
    "0.6b": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "1.7b": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
}
# 原版仓库只需这些文件（不含 safetensors 权重）
CONFIG_FILES = [
    "config.json", "generation_config.json", "tokenizer_config.json",
    "vocab.json", "merges.txt", "preprocessor_config.json", "tokenizer.json",
    "special_tokens_map.json",
]


def download_model(key: str, models_dir: str, precision: str, hf_token: str | None):
    base = Path(models_dir) / ("0.6B-Base" if key == "0.6b" else "1.7B-VoiceDesign")
    onnx_dir = base / precision
    original_dir = base / "original"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    original_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/2] 下载 ONNX 子模型 ({precision}) …")
    # 拉取 onnx 仓库的对应 precision 子目录
    snapshot_download(
        repo_id=ONNX_REPOS[key],
        allow_patterns=[f"{precision}/*", "README.md"],
        local_dir=str(base / "_onnx_tmp"),
        token=hf_token or None,
    )
    # 移动到目标目录
    tmp = base / "_onnx_tmp" / precision
    if tmp.exists():
        for f in tmp.iterdir():
            dst = onnx_dir / f.name
            if not dst.exists():
                import shutil
                if f.is_file():
                    shutil.copy2(f, dst)
                else:
                    shutil.copytree(f, dst)
    import shutil
    shutil.rmtree(base / "_onnx_tmp", ignore_errors=True)
    print(f"     ONNX 就绪: {onnx_dir}")

    print("[2/2] 下载原版 checkpoint 配置/分词文件 …")
    for name in CONFIG_FILES:
        try:
            p = hf_hub_download(repo_id=ORIGINAL_REPOS[key], filename=name,
                                local_dir=str(original_dir),
                                token=hf_token or None)
            print(f"     ✓ {name}")
        except Exception as e:
            print(f"     - 跳过 {name} ({e})")
    print(f"原版配置就绪: {original_dir}")
    print(f"模型 {key} 下载完成。")


def main():
    ap = argparse.ArgumentParser(description="下载 Qwen3-TTS ONNX 模型与配置")
    ap.add_argument("--models-dir", default="./models", help="模型根目录")
    ap.add_argument("--precision", default="cpu_int4", help="ONNX 精度目录")
    ap.add_argument("--model", choices=["0.6b", "1.7b", "all"], default="all")
    args = ap.parse_args()
    hf_token = os.environ.get("HF_TOKEN")

    keys = ["0.6b", "1.7b"] if args.model == "all" else [args.model]
    for k in keys:
        download_model(k, args.models_dir, args.precision, hf_token)
    print("全部完成。若使用默认配置，模型将出现在 ./models/ 下。")


if __name__ == "__main__":
    main()