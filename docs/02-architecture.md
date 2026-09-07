# Qwen3-TTS ONNX 部署服务 — 架构设计

> 版本: 1.0
> 日期: 2026-09-07
> 关联: `docs/01-requirements.md`

## 1. 架构总览

```
┌─────────────────────────────────────────────────────────────────────┐
│                          FastAPI / Uvicorn                           │
│                                                                     │
│  ┌─────────────┐  ┌──────────────────┐  ┌─────────────────────────┐ │
│  │  OpenAI 兼容 │  │ 语音库管理 API    │  │ 服务管理/任务 API         │ │
│  │  /v1/*       │  │  /v1/voices/*    │  │  /api/models · /api/tasks│ │
│  └──────┬──────┘  └────────┬─────────┘  └────────────┬────────────┘ │
│         │                  │                         │              │
│         ▼                  ▼                         ▼              │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                     Service Layer (业务层)                     │  │
│  │  VoiceService · TaskService · ModelLifecycleService           │  │
│  └───────────────┬───────────────────────┬──────────────────────┘  │
│                  │                       │                          │
│                  ▼                       ▼                          │
│  ┌──────────────────────────┐  ┌──────────────────────────┐        │
│  │  VoiceStore (SQLite)     │  │  TaskStore (SQLite)      │        │
│  │  data/voices/ 上传目录    │  │  data/tasks/ 结果目录    │        │
│  └──────────────────────────┘  └──────────────────────────┘        │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                  Inference Orchestrator                       │  │
│  │  每模型: 串行推理队列 · 生命周期状态机 · 锁                     │  │
│  └───────┬──────────────────────────────┬──────────────────────┘  │
│          ▼                              ▼                          │
│  ┌──────────────────────┐  ┌──────────────────────┐                │
│  │  Qwen3TTSOnnxEngine  │  │  Qwen3TTSOnnxEngine  │                │
│  │  0.6B Base (克隆)     │  │  1.7B VoiceDesign    │                │
│  │  cpu_int4 · 9 组件    │  │  cpu_int4 · 8 组件   │                │
│  └──────────────────────┘  └──────────────────────┘                │
│                                                                     │
│  ┌────────────────────────────────────────────┐                     │
│  │        Static UI (原生单页, 无构建)          │  ← /static          │
│  └────────────────────────────────────────────┘                     │
└─────────────────────────────────────────────────────────────────────┘
```

**核心原则**：参考调研，运行时为**纯 `onnxruntime`**，直接复用 ONNX 社区仓库自带的 `inference.py`（manifest 驱动、numpy 自回归循环）作为推理引擎骨架。

## 2. 模块划分

### 2.1 推理引擎层 `qwen3_tts_onnx/`

| 模块 | 职责 |
|---|---|
| `engine.py` | `Qwen3TTSOnnxEngine`：按 manifest 加载子模型、构建 9/8 个 `ort.InferenceSession`、提供 `generate_voice_clone()` / `generate_voice_design()` 高层 API |
| `pipeline.py` | 复用/移植仓库 `inference.py`：prefill 组装、自回归循环、`code_predictor`×15、`residual_embed`、`tok_encoder`/`tok_decoder` 分块 |
| `manifest.py` | 解析 `manifest.json`，子模型→文件映射，EP 选择 |
| `tokenizer.py` | 文本 tokenizer 封装（`AutoTokenizer` + `trust_remote_code=True`，指向原版 Qwen checkpoint 目录） |
| `types.py` | 模型 ID、能力枚举、状态枚举、合成结果类型 |

> 两个模型共享同一份 `pipeline.py`（0.6B 与 1.7B 架构一致，仅 `hidden_size` 不同：1024 vs 2048）；差异仅在组件集（Base 多 `speaker_encoder`）与能力门控（`tts_model_type`）。

### 2.2 编排层

| 模块 | 职责 |
|---|---|
| `orchestrator.py` | `ModelRegistry`：持有两模型；**每模型一把 asyncio.Lock 串行队列**；同步/异步任务调度 |
| `lifecycle.py` | `ModelLifecycleService`：状态机（unloaded/loading/loaded/unloading）、`load_on_start`、`unload_after_idle_seconds`、内存统计 |

### 2.3 存储层

| 模块 | 职责 |
|---|---|
| `store/voice_store.py` | `VoiceStore`：SQLite CRUD、`data/voices/` 音频文件管理 |
| `store/task_store.py` | `TaskStore`：任务持久化（含结果路径）、状态/进度/错误 |
| `store/schema.py` | SQLite schema 与迁移（详见 `docs/04-data-model.md`） |
| `config.py` | 配置加载（YAML/环境变量，见 5） |

### 2.4 API 层 `api/`

| 模块 | 端点 |
|---|---|
| `openai.py` | `/v1/audio/speech`、`/v1/models` |
| `voices.py` | `/v1/voices` 系列 |
| `models_api.py` | `/api/models/*` |
| `tasks.py` | `/api/tasks` 系列 |
| `static.py` | 静态资源挂载与入口页 |
| `auth.py` | **鉴权中间件**：Bearer/x-api-key 校验 + 豁免清单 |

### 2.5 UI 层 `web/`（静态，无构建）

- `web/index.html` + `web/app.js` + `web/style.css`
- 原生单页多 Tab（详见 `docs/05-ui-spec.md`）

### 2.6 鉴权机制

- **策略**：单一静态 token（`Q3TTS_API_TOKEN` env / `server.api_token` config）
- **请求头**：同时接受 `Authorization: Bearer <token>` 与 `x-api-key: <token>`（`auth.py` 中间件统一解析）
- **豁免清单**：`GET /healthz`（恒豁免）；`/docs` 按 `server.docs_public` 配置豁免（默认 true）
- **UI 侧**：登录页输入 token → `localStorage` 存储 → 所有 fetch 附加 `Authorization: Bearer`；401 时返回登录页
- **未配置 token**：无鉴权模式启动（本地调试），日志 `WARNING`

### 2.7 音频格式管线（mp3）

```
输入侧: .mp3 ──ffmpeg──▶ 24kHz PCM ──▶ tok_encoder (克隆参考音频)
                                       (wav 直接送入, 无需转换)
输出侧: 24kHz PCM ──▶ WAV (soundfile)
              └────▶ MP3 (ffmpeg -b:a 128k)
```

- ffmpeg 调用封装于 `services/audio.py`（子进程，超时控制），Docker 镜像内置 ffmpeg 二进制

## 3. 推理流程（同步路径）

```
POST /v1/audio/speech
  → API 层解析 + voice 解析（语音库查名 / 临时参数）
  → VoiceService 组装生成参数（ref_audio/ref_text/instruct/language）
  → Orchestrator 选取目标模型 → 获取模型锁（串行）
  → 若模型 unloaded → lifecycle 触发加载（首次请求阻塞）
  → engine.generate_voice_clone() / generate_voice_design()
      ├─ tokenizer 文本编码
      ├─ prefill 组装（text_embed / codec_embed / speaker_encoder）
      ├─ 自回归循环（talker_cache → code_predictor×15 → residual_embed）
      └─ tok_decoder 分块解码 → 24kHz WAV bytes
  → 返回音频（application/octet-stream 或 audio/wav）
```

**异步路径**：POST 提交 → 写入 TaskStore（pending）→ 后台 worker 取锁执行同上流程 → 结果落 `data/tasks/` → 状态 completed；客户端轮询 `GET /api/tasks/{id}`。

## 4. 并发与模型生命周期设计

### 4.1 串行推理队列

- 每个模型维护 `asyncio.Lock`；推理请求 `async with lock` 排队
- CPU 自回归天然串行，多请求并发只会互相拖慢 + 内存膨胀 → 串行化是正确取舍
- 推理实际为 CPU 密集型：用线程池（`run_in_executor`）包裹，避免阻塞事件循环

### 4.2 同步/异步双形态

| 形态 | 端点 | 适用 |
|---|---|---|
| 同步 | `/v1/audio/speech` | OpenAI 兼容协议要求；短文本 |
| 异步 | `POST /api/tasks` + 轮询 | 长文本、调试、防止网关超时 |

### 4.3 生命周期状态机

```
         load             load            unload
unloaded ────► loading ────► loaded ──────────► unloading ──► unloaded
   ▲                                    │
   └──────────────── 空闲超时自动卸载 ────┘
```

- 默认 `load_on_start=true`（双模型常驻）
- `unload_after_idle_seconds`（默认关闭）：空闲超时自动卸载省内存
- 卸载接口：`POST /api/models/{name}/unload`；加载：`POST /api/models/{name}/load`

### 4.4 内存管理策略

- 加载时按 manifest 读取子模型；int4 权重自包含（无外部 `.data`）
- 提供 `GET /api/models/status` 返回加载状态 + 估算内存占用
- 建议 `max_new_tokens` 服务端上限（配置），防超长生成撑爆 KV cache

## 5. 配置设计（`config.yaml` + 环境变量覆盖）

```yaml
server:
  host: 0.0.0.0
  port: 8000

models:
  - id: qwen3-tts-0.6b-base
    name: 0.6B Base
    capability: voice_clone          # voice_clone | voice_design
    onnx_dir: ./models/0.6B-Base/cpu_int4
    tts_dir: ./models/0.6B-Base/original  # 原版 checkpoint(config+tokenizer)
    load_on_start: true
    unload_after_idle_seconds: 0     # 0=关闭
    max_new_tokens: 2048
    top_k: 50
    top_p: 1.0
    temperature: 0.9
  - id: qwen3-tts-1.7b-voicedesign
    name: 1.7B VoiceDesign
    capability: voice_design
    onnx_dir: ./models/1.7B-VoiceDesign/cpu_int4
    tts_dir: ./models/1.7B-VoiceDesign/original
    load_on_start: true
    ...

storage:
  db_path: ./data/app.db
  voices_dir: ./data/voices     # 参考音频持久目录
  tasks_dir: ./data/tasks       # 任务结果目录

voice:
  default_voice: ""             # 兜底语音(可空)
```

## 6. 目录结构（目标态）

```
qwen3-tts-onnx/
├── app/                    # 服务主包
│   ├── api/                # FastAPI 路由
│   ├── services/           # 业务层
│   ├── inference/          # ONNX 推理引擎(pipeline 移植)
│   ├── store/              # SQLite 存储
│   └── config.py
├── web/                    # 静态 UI(单页)
├── data/                   # 运行时数据
│   ├── voices/             # 参考音频上传目录(持久)
│   └── tasks/              # 任务结果
├── models/                 # 模型权重(下载脚本负责)
│   ├── 0.6B-Base/
│   └── 1.7B-VoiceDesign/
├── docs/                   # 本文档集
├── scripts/                # 下载/校验/启动脚本
├── Dockerfile              # 常规 + Spaces 共用(单一 Dockerfile)
├── README.md               # HF Spaces 引导(模型卡片说明)
└── requirements.txt
```

> 注：`docs/03-api-spec.md`、`docs/04-data-model.md`、`docs/05-ui-spec.md`、`docs/06-constraints-risks.md` 为本架构的配套规格。

## 7. 部署形态

### 7.1 单一 Dockerfile，双形态

同一镜像通过**环境变量**区分运行形态，无独立 Dockerfile：

```dockerfile
# 基础镜像: python:3.12-slim
# 1. 安装 ffmpeg + libsndfile (mp3 输入/输出 + 音频)
# 2. 安装 python 依赖 (onnxruntime/numpy/soundfile/fastapi/...)
# 3. 拷贝应用代码 + 静态 UI + 启动脚本
# 4. 默认 CMD: 读取 PORT/Q3TTS_API_TOKEN/DATA_DIR 启动 uvicorn
```

| 环境变量 | 常规 Docker | HF Spaces |
|---|---|---|
| `PORT` | `8000` | `7860` |
| `DATA_DIR` | `./data`（或挂载卷） | `/data` |
| `Q3TTS_API_TOKEN` | 可选（未设=无鉴权） | 必设（公网暴露） |
| `MODELS_DIR` | `./models` | `/data/models` |

### 7.2 模型获取策略

- **启动时下载**（镜像不含权重，保持小巧）：`scripts/download_models.py` 按配置从 HF 拉取：
  - ONNX 子模型（`cpu_int4/` 目录）
  - 原版 checkpoint 的 config/tokenizer 文件（`trust_remote_code` 所需）
- 下载缓存落 `MODELS_DIR`（Spaces 下为 `/data/models`，可持久）
- Spaces 可用 `HF_TOKEN` 拉取 gated 模型
- 首启较慢（下载 + 加载），CPU 加载本就慢，可接受；`load_on_start` 可配

### 7.3 HF Spaces 适配

- `README.md` 含 YAML 头：`sdk: docker`（HF 依仓库内 Dockerfile 自行构建镜像）
- 容器**只读文件系统**：全部可写数据经 `DATA_DIR=/data` 重定向（HF 持久卷）
- 监听 `0.0.0.0:7860`（`PORT=7860`）
- `/healthz` 无鉴权，供 Spaces 引导探测
- 默认 `load_on_start=true` 但受 Spaces 内存配额约束（详见 `06-constraints-risks.md`）

### 7.4 常规 Docker 部署

```bash
docker build -t qwen3-tts-onnx .
docker run -d -p 8000:8000 \
  -e Q3TTS_API_TOKEN=secret \
  -v ./data:/data \
  qwen3-tts-onnx
```

- 数据卷挂载 `DATA_DIR` 保证持久化与升级不丢
- `Q3TTS_API_TOKEN` 建议经 secrets 管理