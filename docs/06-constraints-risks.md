# Qwen3-TTS ONNX 部署服务 — 约束与风险

> 版本: 1.0
> 日期: 2026-09-07
> 依据: 对两个 HF 仓库 + 官方 `qwen-tts` 包的深度调研
> 关联: `docs/01-requirements.md` §4

## 1. 关键技术约束（调研结论，影响实现取舍）

### 1.1 运行时是纯 onnxruntime，**不是** onnxruntime-genai

- 两仓库自带 `inference.py`：9/8 个 ONNX 子模型 + numpy 自回归循环，`onnxruntime>=1.20` 直接驱动
- `onnxruntime-genai` 仅出现在构建工具链；`STATUS.md`/`test_modelbuilder_talker.py` 明确 **ModelBuilder 不可用于 talker**（MROPE 位置编码，`mrope_section=[24,20,20]`，genai 只支持标准 RoPE → 位置错误）
- → **实现必须复用/移植 `inference.py` 的编排逻辑**，不能走 genai `create_streaming_processor`

### 1.2 ONNX 仓库不含 tokenizer/config

- 必须另配原版 Qwen checkpoint 的：`config.json`、`generation_config.json`、`vocab.json`、`merges.txt`、`tokenizer_config.json`、`preprocessor_config.json`
- 文本 tokenizer 为自定义 `Qwen3TTSProcessor`（`trust_remote_code=True`）
- `config.json` 提供 token id 与 `tts_model_type`（能力门控依据）
- → 下载脚本需同时拉 ONNX 子模型 + 原版配置/分词文件

### 1.3 模型能力与组件差异

| | 0.6B Base | 1.7B VoiceDesign |
|---|---|---|
| `tts_model_type` | `base` | `voice_design` |
| 能力 | 语音克隆 | 声音设计（instruct） |
| 组件数 | 9（含 `speaker_encoder`） | 8（无 speaker_encoder） |
| 专属参数 | `ref_audio`+`ref_text`（ICL+x-vector） | `instruct` |
| `hidden_size` | 1024 | 2048 |

- 能力门控：对 Base 传 instruct、对 VoiceDesign 传 ref 都会报错 → 服务层必须按 capability 路由校验
- 预设人名音色（Vivian/Ryan）**不属于**这两个模型，属第三类 CustomVoice checkpoint —— 已明确 Out of Scope

### 1.4 CPU 仅 int4 量化可用（无 int8）

- 唯一 CPU 量化：int4（RTN）。codec 组件强制 fp32
- int4 权重自包含（无外部 `.data`）；fp16/fp32 有 1.77GB `.data` 兄弟文件（须一并下载）
- 磁盘：0.6B int4 ≈ 1.6GB；1.7B int4 ≈ 2.9GB
- **风险**：int4 存在精度损失，敏感场景需与 fp32 做对比验证（本服务 v1 采 int4 为默认，可配置切 fp32 校验）

### 1.5 CPU 推理速度慢（非实时）

- 1 token 自回归 + 每帧 16-17 次 `session.run()`（1 talker + 15 predictor + 1 residual）
- 约"每句数秒"；长文本分钟级
- KV cache 增长：28 层 × 8 KV 头 × 128 head_dim × 2 × 4B ≈ 229KB/帧；`max_new_tokens=2048` 时峰值可达 ~470MB
- → 必须设 `max_new_tokens` 服务端上限；长文本走异步任务

### 1.6 固定形状 codec

- `tok_encoder` 固定 1s（24000 samples）输入 → 参考音频须 `encode_chunked` 分块
- `tok_decoder` 固定 25 帧 → `decode_chunked` 分块（尾部重复-裁剪）
- 不可直接喂任意长度

### 1.7 无原生 `speaking_rate` 参数

- 官方 API 与 ONNX pipeline 均无数值语速参数
- 语速/情感/音色经 VoiceDesign 的 `instruct` 自然语言控制
- → `speed` 预留入参但**不实现**（见 `01-requirements.md` §3.5）

### 1.8 非流式

- ONNX 路径为非流式（生成完返回整段）。OpenAI 的 97ms 流式声称属 vLLM/DashScope 路径
- → `/v1/audio/speech` 整段返回；异步任务补足长文本场景

## 2. 内存与资源风险

| 场景 | 估算 | 缓解 |
|---|---|---|
| 双模型 int4 常驻 | 权重 4.5GB，运行峰值 6-10GB | 建议宿主机 ≥16GB；可配置单模型常驻 |
| 长文本 KV cache | 峰值 ~470MB/帧上限 | `max_new_tokens` 上限 |
| 多请求并发 | 内存翻倍 + CPU 争抢 | 串行推理队列（已设计） |
| 参考音频过大 | 上传耗时 + 存储膨胀 | 前端/后端限长（≤30s 建议，可配） |

## 3. 依赖与兼容风险

| 风险 | 说明 | 缓解 |
|---|---|---|
| Python 3.12 + librosa | 需 `numba>=0.60`/`llvmlite>=0.43` 引脚 | requirements 固定版本 |
| `trust_remote_code` | 自定义 processor 需远程代码执行 | 固化到本地缓存；版本锁定；仅信任 Qwen 官方 |
| onnxruntime 版本 | 需 ≥1.20 | requirements 约束 |
| tokenizer 依赖 transformers | 仅文本侧需要 | 轻量；模型权重不依赖 torch |
| 原版 checkpoint 下载 | 若只取配置/分词文件，无需 safetensors 权重 | 用 `huggingface_hub` 按文件拉取 |
| **ffmpeg 可用性** | mp3 解码（输入）与编码（输出）依赖系统 ffmpeg | Docker 镜像内置；非 Docker 部署检查 PATH；`audio.ffmpeg_path` 可配 |
| **mp3 → PCM 精度** | 参考音频经 mp3 有损压缩后解码，克隆音色有损 | 建议上传 wav 获得最佳克隆质量；mp3 作为便利格式支持 |

## 3.1 鉴权风险

| 风险 | 说明 | 缓解 |
|---|---|---|
| 静态 token 泄露 | token 硬编码/进版本库 | env / 配置注入；文档警告；日志脱敏 |
| 无鉴权模式误上线 | 未设 token 时服务裸露 | 启动日志 WARNING 大字告警；Dockerfile CMD 显式示例含 token |
| UI localStorage XSS | 静态页面注入 | 单页无外部输入渲染（不 `innerHTML` 用户内容）；严格转义 |
| brute-force | token 被暴力枚举 | v1 单机默认；可前置网关限速（文档建议） |

## 3.2 部署（Docker/Spaces）风险

| 风险 | 说明 | 缓解 |
|---|---|---|
| **Spaces 只读文件系统** | 容器除 `/data` `/tmp` 外只读；`data/` 写不进去 | `DATA_DIR=/data` 全量重定向（架构 §7.3） |
| **Spaces 端口 7860 强制** | 端口不符则引导失败 | `PORT=7860` 环境变量 |
| **Spaces 内存配额** | 双模型 int4 峰值 6-10GB 超 Spaces 免费配额（约 16GB CPU 限制） | Spaces 设 `load_on_start=false` 或仅单模型；`unload_after_idle_seconds` 省内存；长文本异步任务 |
| **镜像体积 vs 下载** | 内嵌权重 → 镜像 5GB+，Spaces 构建/传输受限 | **启动时下载**（架构 §7.2），镜像<1GB |
| 首次启动慢 | 下载 4.5GB + 加载耗时 | 文档告知；`/.` 健康探活仅反映进程存活，不反映模型就绪（可选 `GET /api/models/status` 查就绪） |
| 模型下载失败限流 | HF 429 | 下载脚本重试/断点；镜像可内置 `models/` 作为可选优化 |

## 4. 安全风险

| 风险 | 说明 | 缓解 |
|---|---|---|
| 路径穿越（上传/下载） | voice_id / task_id 拼路径 | 白名单校验 + `Path` 规范化 + 禁止绝对路径 |
| **无鉴权误部署** | 服务暴露后任意调用 | env token 必设（见 §3.1）；UI/API 统一拦截 |
| `trust_remote_code` 供应链 | 加载自定义 processor | 固定 commit/缓存 + 只信任官方源 |
| 资源耗尽 DoS | 无上限生成耗尽 CPU/内存 | `max_new_tokens`、队列上限、上传限长 |
| 任务文件膨胀 | 结果音频堆积 | TTL 清理 + 启动孤儿清理 |
| **mp3 上传滥用** | 恶意/超大 mp3 填满磁盘 | 上传大小上限 + `max_ref_audio_seconds` + 格式白名单（仅 wav/mp3） |

## 5. 功能范围风险（Out of Scope 带来的限制）

- 无普通 TTS / 无预设人名音色 / 无流式 / 无 mp3 —— 均为明确决策，若有需求需重新评估（尤其 CustomVoice 需新增第三类模型）
- `speed` 未实现：客户端期望变速会失效（返回原速）—— 文档与响应需明示

## 6. 推荐实现顺序（里程碑）

1. **M0 模型加载验证**：移植 `inference.py`，单模型加载 + selftest（`encode→decode` 往返）
2. **M1 推理引擎**：两个模型可生成音频（CLI 验证）
3. **M2 服务层**：FastAPI + 串行队列 + 生命周期 + 语音库存储
4. **M3 OpenAI 接口**：`/v1/audio/speech` + `/v1/models`
5. **M4 异步任务**：任务存储/进度/轮询
6. **M5 静态 UI**：四 Tab
7. **M6 收尾**：下载脚本、配置、日志、文档复核、性能基线

> 每一步均有明确验收（见各模块规格）。

## 7. 备选/演进方向（非本次范围，记录备用）

- **性能**：如 CPU 速度不达标，可评估 ggml/llama.cpp 社区移植 `andimarafioti/faster-qwen3-tts`（C++，非 onnxruntime）
- **格式**：mp3/opus 编码（需 `lameenc`/`pydub` 或 libsndfile-mp3）
- **自定义音色**：引入 `Qwen3-TTS-12Hz-*-CustomVoice` 提供 preset 人名音色
- **鉴权/多租户**：外部化部署时补充