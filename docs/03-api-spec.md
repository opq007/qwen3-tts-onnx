# Qwen3-TTS ONNX 部署服务 — API 规范

> 版本: 1.0
> 日期: 2026-09-07
> 关联: `docs/01-requirements.md` §3.4 · `docs/02-architecture.md`

## 0. 约定

- Base URL: `http://<host>:<port>`
- 默认 JSON；音频返回原生二进制（`audio/wav` 或 `audio/mpeg`）
- 错误统一为 JSON：`{"error": {"code": "...", "message": "..."}}`（OpenAI 风格）
- **鉴权**：除豁免端点外，所有请求须带 token。支持两种头（二选一）：
  - `Authorization: Bearer <token>`
  - `x-api-key: <token>`
  - token 由 `Q3TTS_API_TOKEN` 环境变量或配置设置；未配置时服务以无鉴权模式启动（日志告警）
- **豁免端点**：`GET /healthz`（无鉴权探活）；`/docs` 默认豁免、可配置关闭
- 模型 ID：
  - `qwen3-tts-0.6b-base` — 0.6B Base（语音克隆）
  - `qwen3-tts-1.7b-voicedesign` — 1.7B VoiceDesign（声音设计）
- 支持语言（10 种）：Chinese、English、Japanese、Korean、German、French、Russian、Portuguese、Spanish、Italian

---

## 0.1 鉴权错误

未带/带错 token：**401**

```json
{"error": {"code": "unauthorized", "message": "missing or invalid API token"}}
```

## 0.2 健康探活

### `GET /healthz`（无鉴权）

```json
{"status": "ok", "version": "0.1.0", "uptime_sec": 120}
```

供容器编排探针与 HF Spaces 引导探测使用；仅暴露服务存活状态，不含敏感信息。

---

## 1. OpenAI 兼容组

### 1.1 `POST /v1/audio/speech` — 语音合成（同步）

OpenAI 兼容主接口。生成完成返回整段音频（非流式）。

**请求 body（application/json）**：

```jsonc
{
  "model": "qwen3-tts-1.7b-voicedesign",   // 必填, 模型 ID
  "input": "您好，欢迎使用语音合成服务。",   // 必填, 待合成文本
  "voice": "my-design-voice",              // 可选, 语音库注册名; 省略时用 default_voice
  "response_format": "wav",                // 可选, wav | mp3 (默认 wav)
  "speed": 1.0                             // 可选, 0.25~4.0; 预留入参, 暂不实现变速
}
```

**response_format**：

| 值 | Content-Type | 说明 |
|---|---|---|
| `wav` | `audio/wav` | 24kHz 16-bit PCM |
| `mp3` | `audio/mpeg` | 经 `ffmpeg` 编码（默认 128kbps，可配） |

**voice 的解析规则（重要）**：

| `voice` 值 | 解析结果 |
|---|---|
| 语音库注册名 | 命中 → 使用该语音的克隆/设计参数 |
| 省略 / 空串 | 使用 `config.voice.default_voice`；无默认则 400 |
| 不存在的名字 | 404 `voice_not_found` |

**临时参数（multipart/form-data，不落库）**：调用方可通过 multipart 携带一次性参考音频/设计指令而不入库（用于临时合成，与 UI"临时生成"对应）。

| multipart 字段 | 说明 |
|---|---|
| `text` | 合成文本（与 JSON 的 input 二选一） |
| `model` | 模型 ID |
| `language` | 语言，如 `Chinese` |
| `ref_audio` | (仅 Base) 参考音频文件（**.wav 或 .mp3**），克隆用 |
| `ref_text` | (仅 Base) 参考音频转写文本 |
| `instruct` | (仅 VoiceDesign) 声音设计指令 |
| `voice` |/ 可选 `voice_id` 兼容 |

**成功响应（200）**：`Content-Type: audio/wav` 或 `audio/mpeg`，返回 24kHz WAV / MP3 二进制。

**错误示例（400）**：

```json
{"error": {"code": "invalid_model", "message": "model 'foo' not found. available: qwen3-tts-0.6b-base, qwen3-tts-1.7b-voicedesign"}}
```

### 1.2 `GET /v1/models` — 模型列表

**成功响应（200）**：

```json
{
  "object": "list",
  "data": [
    {
      "id": "qwen3-tts-0.6b-base",
      "object": "model",
      "created": 1750000000,
      "owned_by": "local",
      "capability": "voice_clone",
      "languages": ["Chinese", "English", "Japanese", "Korean", "German", "French", "Russian", "Portuguese", "Spanish", "Italian"],
      "status": "loaded"
    },
    {
      "id": "qwen3-tts-1.7b-voicedesign",
      "object": "model",
      "created": 1750000000,
      "owned_by": "local",
      "capability": "voice_design",
      "languages": ["Chinese", "English", "Japanese", "Korean", "German", "French", "Russian", "Portuguese", "Spanish", "Italian"],
      "status": "loaded"
    }
  ]
}
```

---

## 2. 语音库管理组

### 2.1 `GET /v1/voices` — 语音列表

Query：`?type=clone|design`（可选）、`?language=Chinese`（可选）。

**成功响应**：

```json
{
  "object": "list",
  "data": [
    {
      "voice_id": "my-clone-1",
      "type": "clone",
      "language": "Chinese",
      "description": "自定义克隆音",
      "ref_audio": "/data/voices/my-clone-1.wav",
      "ref_text": "这段是参考音频的文字……",
      "duration_sec": 3.2,
      "created_at": "2026-09-07T10:00:00Z",
      "updated_at": "2026-09-07T10:00:00Z"
    }
  ]
}
```

> design 类型条目以 `instruct` 替代 `ref_audio/ref_text`。

### 2.2 `POST /v1/voices` — 创建语音条目

- **克隆类型**（multipart/form-data）：

| 字段 | 说明 |
|---|---|
| `voice_id` | 必填, 注册名（OpenAI `voice` 参数用） |
| `type=clone` | 类型 |
| `language` | 必填, 绑定语言 |
| `ref_audio` | 必填, 参考音频文件（**.wav 或 .mp3**，≤30s 建议） |
| `ref_text` | 必填, 参考音频转写文本 |
| `description` | 可选 |

- **设计类型**（application/json）：

```json
{
  "voice_id": "my-design-1",
  "type": "design",
  "language": "Chinese",
  "instruct": "用温柔缓慢的语气说中文",
  "description": "客服女声"
}
```

**成功响应（201）**：返回完整 voice 对象。
**错误**：`voice_id` 重复 → 409 `voice_exists`；缺 `ref_text`/`ref_audio` → 422 `missing_ref_text` 等。

### 2.3 `GET /v1/voices/{voice_id}` — 查询单条

**成功（200）** 返回 voice 对象；**404** `voice_not_found`。

### 2.4 `PATCH /v1/voices/{voice_id}` — 修改

JSON body：`{ "language": ..., "instruct": ..., "description": ..., "ref_text": ... }`（可空，允许局部更新）。**403** 不允许修改 `type` 与 `voice_id`（先删再建）。

### 2.5 `DELETE /v1/voices/{voice_id}` — 删除

删除条目并**级联删除**其 `data/voices/` 音频文件。**404** 未找到。**204** 成功。

### 2.6 `POST /v1/voices/{voice_id}/test` — 试听

Body：`{"text": "试听文本", "language": "Chinese"}`（text 可选，默认短句）。
调用该语音合成，返回音频（等价 `/v1/audio/speech` 但短句场景）。

---

## 3. 异步任务组

### 3.1 `POST /api/tasks` — 提交合成任务

请求 body 同 `/v1/audio/speech` 的 JSON 版（`model/input/voice/language/instruct/...`，不允许多部分）。

**成功响应（202）**：

```json
{
  "task_id": "6f1c2d3e-...",
  "status": "pending",
  "model": "qwen3-tts-1.7b-voicedesign",
  "created_at": "2026-09-07T10:05:00Z"
}
```

### 3.2 `GET /api/tasks/{task_id}` — 查询状态

**状态机**：`pending → running → completed / failed / cancelled`。

```json
{
  "task_id": "6f1c2d3e-...",
  "status": "running",
  "progress": {"step": "ar_loop", "frame": 32, "max_frames": 80},
  "model": "qwen3-tts-1.7b-voicedesign",
  "duration_ms": 4520,
  "created_at": "2026-09-07T10:05:00Z",
  "started_at": "2026-09-07T10:05:01Z",
  "finished_at": null,
  "error": null
}
```

**completed 时附加**：`result_audio`（`/api/tasks/{task_id}/audio`）、`sample_rate`、`duration_sec`。

### 3.3 `GET /api/tasks/{task_id}/audio` — 下载结果音频

**200** `audio/wav`；未完成 → 409 `task_not_finished`；不存在 → 404。

### 3.4 `DELETE /api/tasks/{task_id}` — 取消/清理

running 中 → 尝试取消（cancelled）；completed → 删除结果文件 + 记录。**204**。

### 3.5 `GET /api/tasks` — 任务列表

Query：`?status=pending|running|completed|failed&limit=50&offset=0`。按创建时间倒序。

---

## 4. 服务管理组

### 4.1 `GET /api/models/status` — 模型状态与内存

```json
{
  "models": [
    {
      "id": "qwen3-tts-0.6b-base",
      "status": "loaded",
      "capability": "voice_clone",
      "onnx_dir": "./models/0.6B-Base/cpu_int4",
      "estimated_ram_mb": 2600,
      "load_time_ms": 15300,
      "infer_count": 42,
      "last_used_at": "2026-09-07T10:04:00Z",
      "idle_seconds": 60
    }
  ],
  "server": {"version": "0.1.0", "uptime_sec": 3600}
}
```

### 4.2 `POST /api/models/{model_id}/load` — 手动加载

**200** `{"id": ..., "status": "loading"}`（异步加载，轮询 status 完成）或同步完成 `loaded`。

### 4.3 `POST /api/models/{model_id}/unload` — 手动卸载

**200** `{"id": ..., "status": "unloading"}`。`unload_after_idle_seconds>0` 时由生命周期服务自动触发同路径。
**409** 若该模型正在推理（新请求进入队列，卸载推迟）。

### 4.4 `GET /api/config` — 服务配置查看（只读）

返回 sanitized 后的配置（不含绝对路径泄露，隐藏 secrets）。

---

## 5. 错误码统一表

| HTTP | code | 场景 |
|---|---|---|
| 401 | `unauthorized` | 鉴权失败（未带/带错 token） |
| 400 | `invalid_model` / `invalid_language` / `invalid_response_format` | 无效枚举 |
| 400 | `missing_parameter` | 缺必填字段（text/model 等） |
| 404 | `voice_not_found` / `task_not_found` / `model_not_found` | 不存在 |
| 409 | `voice_exists` / `task_not_finished` / `model_busy` | 状态冲突 |
| 415 | `unsupported_audio_format` | 上传音频非 wav/mp3 |
| 422 | `missing_ref_text` / `missing_ref_audio` / `invalid_argument` | 参数校验失败 |
| 500 | `inference_failed` / `internal_error` | 服务/推理不可恢复错误 |
| 503 | `model_unavailable` | 模型加载中/卸载中/加载失败 |
| 429 | `too_many_requests` | 队列满（可配最大队列长度） |

## 6. OpenAI 兼容性说明（与官方差异）

| 项目 | 官方 OpenAI | 本服务 |
|---|---|---|
| `voice` | 固定内置 voice 名 | 语音库注册名（`/v1/voices` 管理） |
| `model` | `tts-1` 等 | `qwen3-tts-*` |
| `response_format` | wav/mp3/opus/flac | **wav \| mp3**（opus/flac 预留） |
| `speed` | 实现变速 | **预留, 不实现**（原速） |
| 流式 | 分块返回 | **非流式**整段返回 |
| 临时 audio 上传 | 不支持 | multipart 扩展（不落库，wav/mp3） |
| 鉴权 | `Authorization: Bearer` | 同 + `x-api-key`；单一静态 token |

以上差异在 `docs/01-requirements.md` 的"范围界定"与"speed 参数约定"中有据可依。