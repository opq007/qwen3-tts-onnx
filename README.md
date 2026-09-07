# qwen3-tts-onnx

纯 CPU 环境下使用 ONNX Runtime 部署 Qwen3-TTS 的服务：
0.6B Base（语音克隆） + 1.7B VoiceDesign（声音设计）。

## 特性

- 🎙️ **语音克隆**（0.6B Base）：参考音频 + 参考文本 → 克隆音色合成
- 🎨 **声音设计**（1.7B VoiceDesign）：自然语言指令控制音色/情感/韵律
- 🗂️ **语音库**：预注册 voice（SQLite 持久化，参考音频存 `DATA_DIR/voices/`）
- 🎧 **双音频格式**：参考音频与合成结果均支持 **wav / mp3**（`ffmpeg`）
- 🔌 **OpenAI 兼容接口**：`POST /v1/audio/speech`、`GET /v1/models`
- 🔐 **鉴权**：静态 token（`Q3TTS_API_TOKEN`），Bearer / x-api-key
- 🌐 **可视化 UI**：单页四 Tab（合成/语音库/任务/模型管理）
- 🐳 **部署**：单一 Dockerfile → 常规 Docker + HF Spaces（`sdk: docker`）

## 一键启动

Windows 双击 `start.bat`，Linux/macOS 运行 `./start.sh`，脚本会自动：

1. 检查 Python ≥ 3.10，缺失时给出提示
2. 创建 `.venv` 虚拟环境并安装 `requirements.txt`（首次运行）
3. 自动生成 `config.yaml`（从 `config.example.yaml` 复制）
4. 检查 `models/` 下的 ONNX 模型，缺失时询问是否自动下载（`scripts/download_models.py`）
5. 检查 ffmpeg（仅 mp3 需要，wav 不依赖）
6. 启动服务并自动打开浏览器 `http://localhost:8000/`

```bash
# Linux / macOS（首次需加执行权限）
chmod +x start.sh && ./start.sh

# Windows：双击 start.bat，或命令行运行
.\start.bat
```

常用环境变量：

| 变量 | 说明 | 默认 |
|---|---|---|
| `Q3TTS_API_TOKEN` | 鉴权 token；设置后所有 API 需带 `Authorization: Bearer <token>` | 空 = 无鉴权 |
| `PORT` / `Q3TTS_PORT` | 服务端口 | `8000` |
| `DATA_DIR` | 数据目录（SQLite + voices/ + tasks/） | `./data` |
| `MODELS_DIR` | 模型根目录 | `./models` |
| `HF_TOKEN` | 下载 gated 模型用（可选） | — |

## 手动启动

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_models.py # 下载 ONNX 模型（首次）
cp config.example.yaml config.yaml
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 文档

- `docs/01-requirements.md` — 需求规格（grill-me 澄清定稿）
- `docs/02-architecture.md` — 架构设计（含鉴权、mp3 管线、部署）
- `docs/03-api-spec.md` — API 规范（OpenAI 兼容 + 扩展）
- `docs/04-data-model.md` — 数据模型与持久化
- `docs/05-ui-spec.md` — 静态 UI 规格
- `docs/06-constraints-risks.md` — 约束与风险（调研结论）