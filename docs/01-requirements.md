# Qwen3-TTS ONNX 部署服务 — 需求规格

> 版本: 1.0
> 状态: 已澄清定稿
> 日期: 2026-09-07
> 澄清方式: grill-me 逐项需求访谈（已全部确认）

## 1. 背景与目标

在**纯 CPU 环境**下，使用 ONNX Runtime 部署 Qwen3-TTS 系列模型，提供 TTS 语音合成服务。

选用 Hugging Face ONNX Community 转换的两个模型：

| 模型 | HF 仓库 | 能力 | 说明 |
|---|---|---|---|
| **0.6B Base** | `onnx-community/Qwen3-TTS-12Hz-0.6B-Base` | **语音克隆** | 输入参考音频 + 参考文本，克隆该音色合成 |
| **1.7B VoiceDesign** | `onnx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | **声音设计** | 自然语言指令控制音色/情感/韵律 |

目标交付物：

1. **推理服务**：纯 CPU 上跑通两个模型的完整推理链路
2. **可视化静态界面**：方便功能调试与管理（单页，无前端构建）
3. **OpenAI 兼容接口**：提供 `voice` 相关的标准接口，便于对外提供服务
4. **需求持久化**：本组文档（`docs/`）作为后续实现依据
5. **鉴权**：对外接口与静态管理页面均需鉴权（环境变量/配置设置 token）
6. **部署形态**：支持常规 Docker 部署 + HF Spaces 环境下的 Docker 部署

## 2. 范围界定

### 2.1 功能范围（In Scope）

| # | 能力 | 承载模型 |
|---|---|---|
| 1 | 语音克隆（Voice Clone） | 0.6B Base |
| 2 | 声音设计（Voice Design，自然语言 instrruct 控） | 1.7B VoiceDesign |
| 3 | 语音库管理（预注册 + SQLite 持久化） | 两个模型共用 |
| 4 | OpenAI 兼容 voice 接口 | 两个模型共用 |
| 5 | 异步任务机制（长文本/调试场景） | 两个模型共用 |
| 6 | 模型生命周期管理（常驻/懒加载/卸载） | 两个模型各自可配 |
| 7 | 静态可视化 UI（合创/语音库/任务/服务 4 个 Tab） | — |
| 8 | **参考音频支持 wav 与 mp3 双格式**（上传/克隆） | `ffmpeg` 解码 |
| 9 | **合成输出支持 wav 与 mp3 双格式**（`response_format`） | `ffmpeg` 编码 |
| 10 | **统一鉴权**（对外接口 + 静态 UI） | 环境变量/配置 token |
| 11 | **常规 Docker 部署 + HF Spaces Docker 部署** | 单一 Dockerfile |

### 2.2 不在范围内（Out of Scope — 明确不加）

- ❌ 普通 TTS 合成（不克隆、不设计、直接用默认音色的平凡合成）——用户明确排除
- ❌ 预设人名音色（`speaker=Vivian` 等）——调研确认属第三类 **CustomVoice** checkpoint，本两个模型不支持
- ❌ 流式输出——调研确认 ONNX 导出路径为非流式（生成完返回整段）
- ❌ CUDA/GPU 推理——本服务目标纯 CPU
- ❌ `speed` 的变速实现——**预留入参但暂不实现**（见 3.6）
- ❌ 分角色/多用户鉴权——单 token 简单鉴权（见 3.7）
- ❌ ASR 自动转录参考文本——ref 文本必须手动提供
- ❌ 用户注册/多租户——单机单用户服务

### 2.3 关键边界说明

- 参考音频**必须**存于项目内**持久上传目录**（非临时目录），保证重启后语音库可继续使用
- ONNX 仓库**不含** config 与 tokenizer，需配合原版 Qwen checkpoint 的配置/分词文件
- 模型运行时为纯 `onnxruntime`（非 onnxruntime-genai、非 optimum）

## 3. 功能需求

### 3.1 语音合成

- 支持两种合成入口：
  - **语音克隆**（Base）：`文本 + 语言 + 参考音频 + 参考文本` → 24kHz WAV
  - **声音设计**（VoiceDesign）：`文本 + 语言 + instruct 指令` → 24kHz WAV
- 语音可通过**语音库注册名**（`voice` 参数）或**临时就地参数**（不落库）两种方式指定
- 输出格式：**WAV（24kHz, 16-bit PCM）或 MP3**（`response_format` 指定，经 `ffmpeg` 编码）

### 3.2 语音库管理

- 预注册语音：`clone` 或 `design` 两种类型
- SQLite 持久化，服务重启后语音条目不丢失
- 语音条目元数据：`voice_id`、`type`、`language`、`instruct`、`description`、时间戳
- 克隆语音的参考音频文件存于项目内 `data/voices/` 上传目录
- **参考音频支持 wav 与 mp3 双格式**（mp3 经 `ffmpeg` 解码为 24kHz PCM 后进入推理）
- 支持增、删、改、查、试听
- 提供**默认语音兜底**（config 可配 `default_voice`，API 调用省略 `voice` 时使用）

### 3.3 可视化界面（单页多 Tab）

| Tab | 功能 |
|---|---|
| 语音合成 | 选模型/选 voice 或临时参数 → 生成 → 播放/下载 |
| 语音库 | 展示、注册（克隆向导/设计向导）、编辑、删除、试听 |
| 任务列表 | 异步任务状态/进度查看、结果下载、错误展示 |
| 模型/服务 | 加载状态、内存、加载/卸载、配置与日志尾页 |

- 参考音频上传通过前端 FormData 上传到项目持久上传目录
- 支持"生成不入库"的临时参数（UI 与 API 均支持）

### 3.4 接口规格（总览）

| 分组 | 端点 | 用途 |
|---|---|---|
| OpenAI 兼容 | `POST /v1/audio/speech` | 主合成接口，返回音频 |
| OpenAI 兼容 | `GET /v1/models` | 列出可用模型 |
| 语音库管理 | `GET/POST /v1/voices` | 列表/创建 |
| 语音库管理 | `GET/PATCH/DELETE /v1/voices/{voice_id}` | 查/改/删 |
| 语音库管理 | `POST /v1/voices/{voice_id}/test` | 试听 |
| 服务管理 | `GET /api/models/status` | 模型状态/内存 |
| 服务管理 | `POST /api/models/{name}/load` · `/unload` | 加载/卸载 |
| 异步任务 | `POST /api/tasks` | 提交合成任务 |
| 异步任务 | `GET /api/tasks/{id}` | 查询任务状态 |
| 异步任务 | `GET /api/tasks/{id}/audio` | 下载结果 |
| 基础设施 | `GET /healthz` | 无鉴权健康探活（容器编排/Spaces 引导） |

> 完整字段与示例见 `docs/03-api-spec.md`。

### 3.5 `speed` 参数约定（重要）

- OpenAI 兼容接口接收 `speed` 入参（0.25 ~ 4.0），**预留但暂不实现变速**，当前原速返回
- 文档注明：模型原生不支持数值语速，语速/情感/音色经 VoiceDesign 的 `instruct` 自然语言控制
- 后续若实现：采用后处理变速（方案待定：WSOLA 保音调变速 vs 简单重采样）

### 3.6 并发与资源（非功能主轴）

- **并发模型**：单 worker + 每模型串行推理队列（CPU 自回归串行使然）
- **接口形态**：同步（OpenAI 兼容协议要求）+ 异步任务双形态
- **模型生命周期**：默认常驻；支持按配置或接口懒加载/卸载以省内存（见 4.4）

### 3.7 鉴权

- **单一静态 token**：经环境变量（如 `Q3TTS_API_TOKEN`）或配置文件设置
- 覆盖范围：**所有对外 API**（`/v1/*`、`/api/*`）+ **静态管理页面**（UI 登录页输入 token，存 localStorage）
- 兼容 OpenAI 生态客户端：同时接受 `Authorization: Bearer <token>` 与 `x-api-key: <token>` 两种请求头
- **豁免清单**：`/healthz`（容器探活/Spaces 引导需无鉴权探活，仅暴露状态）；`/docs` 默认豁免、可配置关闭豁免
- token 未设置时的行为：若未配置 token 则服务以**无鉴权**模式启动（便于本地快速调试），日志中告警

### 3.8 部署形态

- **单一 Dockerfile** 同时服务常规 Docker 部署与 HF Spaces（`sdk: docker`）
- 运行形态差异仅通过**环境变量**表达：

| 环境变量 | 常规部署默认 | HF Spaces |
|---|---|---|
| `PORT` | `8000` | `7860`（Spaces 硬要求） |
| `DATA_DIR` | `./data` | `/data`（Spaces 持久可写卷） |
| `MODELS_DIR` | `./models` | `/data/models`（可写目录下的模型缓存） |
| `Q3TTS_API_TOKEN` | 未设=无鉴权 | 必设（Spaces 暴露于公网） |

- **模型文件启动时下载**：镜像保持小巧通用；启动脚本按配置从 HF 拉取 ONNX 子模型 + 配置/分词文件（Spaces 可用 `HF_TOKEN` 拉 gated 模型）
- 镜像内置 `ffmpeg`（mp3 输入/输出解码编码）+ 运行时依赖
- Spaces 只读文件系统适配：所有可写数据（SQLite/音频/任务/模型缓存）经 `DATA_DIR` 重定向到持久卷

## 4. 非功能需求

### 4.1 性能预期

- 参考调研：1 token 自回归 + 16-17 次 `session.run()`/帧，速度约"每句数秒"，**非实时**
- 服务端必须对 `max_new_tokens` 设上限，防止超长文本无限生成
- 长文本走异步任务接口，避免网关超时

### 4.2 内存

- 双模型 int4 常驻：权重约 4.5 GB（0.6B≈1.6GB + 1.7B≈2.9GB），运行峰值约 6-10GB
- 支持按模型卸载以降低驻内存

### 4.3 环境依赖

- Python ≥ 3.10（开发建议 3.12）
- `onnxruntime>=1.20`（CPUExecutionProvider）、`numpy`、`soundfile`、`transformers`（仅文本 tokenizer，`trust_remote_code=True`）、`librosa`（含 `numba>=0.60`/`llvmlite>=0.43` 引脚）
- FastAPI / Uvicorn 服务层
- **`ffmpeg`**（参考音频 mp3 解码 + 合成结果 mp3 编码；Docker 镜像内置）

### 4.4 模型生命周期管理

| 状态 | 说明 | 触发 |
|---|---|---|
| `unloaded` | 未加载 | 启动配置 `models.load_on_start=false` / 卸载接口 |
| `loading` | 加载中 | 首次请求命中 / 加载接口 |
| `loaded` | 可推理 | 加载完成 |
| `unloading` | 卸载中 | 卸载接口 / 配置触发 |

- 支持 `load_on_start`（默认 true）
- 支持 `unload_after_idle_seconds` 惰性卸载（可配，默认关闭）
- 内存占用统计展示于服务管理 Tab

## 5. 已确认决策表（grill-me 结论汇总）

| # | 决策项 | 结论 |
|---|---|---|
| 1 | 功能范围 | A：双模型全功能（克隆 + 设计），**无普通 TTS** |
| 2 | voice 语义 | A：预注册语音库；SQLite 持久化 |
| 3 | 技术栈 | A：FastAPI + Uvicorn + 原生单页 UI（无构建） |
| 4 | 并发 | A：单 worker 串行队列；同步/异步双形态；默认常驻 + 可懒加载/卸载 |
| 5 | voice 数据模型 | ref 文本手动提供；type/language/instruct/description/时间戳；design 绑定语言；默认语音兜底 |
| 6 | speed / mp3 | speed 预留不实现；**输出 wav + mp3 双格式**（ffmpeg 编码） |
| 7 | 上传目录 | 项目内持久上传目录（非临时）；支持"临时生成不入库"交互 |
| 8 | 音频格式 | **输入（参考音频）支持 wav + mp3**（ffmpeg 解码）；输出同（见 #6） |
| 9 | 鉴权 | **单一静态 token**（环境变量/配置）；Bearer + x-api-key；UI localStorage；/healthz 豁免、/docs 可配豁免 |
| 10 | 部署 | **单一 Dockerfile**：常规 Docker + HF Spaces（`sdk: docker`）；`PORT`/`DATA_DIR` 环境变量区分；启动时下载模型 |
| 11 | mp3 实现 | 引入系统 `ffmpeg`（进 Docker 镜像）承担 mp3 解码与编码 |

## 6. 保留决策（后续实现/验证再做）

- `max_new_tokens` 具体上限值（服务配置）
- 变速后处理方案（当前不做）
- 鉴权重策略扩展（当前单 token）
- 多 worker 扩展（当前不做）