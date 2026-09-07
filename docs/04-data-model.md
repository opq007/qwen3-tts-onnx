# Qwen3-TTS ONNX 部署服务 — 数据模型与持久化

> 版本: 1.0
> 日期: 2026-09-07
> 关联: `docs/02-architecture.md` §2.3 · `docs/03-api-spec.md`

## 1. 存储布局

> 根目录由 **`DATA_DIR`** 环境变量决定（默认 `./data`），Spaces 环境设 `/data`（持久可写卷）。所有可写数据均在 `DATA_DIR` 下。

```
<DATA_DIR>/                # 默认 ./data；Spaces 为 /data
├── app.db                 # SQLite 主库（voice + task）
├── voices/                # 参考音频持久上传目录（用户要求：非临时目录）
│   └── <voice_id>.wav/.mp3 # 克隆参考音频（保存原始上传格式）
└── tasks/                 # 任务结果音频
    └── <task_id>.wav/.mp3  # 按 response_format 存储
```

- `DATA_DIR` 整体进入 `.gitignore`（运行时数据，不入库）
- `voices/` 与 `tasks/` 目录在服务启动时自动创建

## 2. SQLite Schema

> SQLite，WAL 模式。迁移由启动时 `schema.py` 执行（`PRAGMA user_version` + 增量迁移）。

### 2.1 `voices` 表

```sql
CREATE TABLE IF NOT EXISTS voices (
    voice_id      TEXT PRIMARY KEY,            -- OpenAI voice 参数注册名
    type          TEXT NOT NULL CHECK (type IN ('clone','design')),
    language      TEXT NOT NULL,               -- 绑定语言(10种之一: Chinese/English/...)
    instruct      TEXT,                        -- design 类型: 声音设计指令
    ref_audio     TEXT,                        -- clone 类型: 音频文件相对路径(DATA_DIR/voices/<id>.wav|.mp3)
    ref_text      TEXT,                        -- clone 类型: 参考音频转写文本
    description   TEXT DEFAULT '',             -- 备注
    duration_sec  REAL,                        -- 参考音频时长(可选, 便于展示)
    audio_format  TEXT DEFAULT 'wav',          -- 原始上传格式: wav | mp3
    created_at    TEXT NOT NULL,               -- ISO8601 UTC
    updated_at    TEXT NOT NULL
);
```

**约束规则**（服务层强制）：
- `type='clone'` → `ref_audio` 与 `ref_text` 必填，`instruct` 置 NULL
- `type='design'` → `instruct` 必填，`ref_audio/ref_text` 置 NULL
- `voice_id` 允许中文/字母/数字/`-_`；长度 ≤ 64
- `language` 限定 10 种枚举

### 2.2 `tasks` 表

```sql
CREATE TABLE tasks (
    task_id       TEXT PRIMARY KEY,            -- UUID
    model_id      TEXT NOT NULL,               -- qwen3-tts-0.6b-base | qwen3-tts-1.7b-voicedesign
    status        TEXT NOT NULL CHECK (status IN
                  ('pending','running','completed','failed','cancelled')),
    input_text    TEXT NOT NULL,
    language      TEXT,
    voice_id      TEXT,                        -- 使用的语音(可空=默认/temp)
    instruct      TEXT,                        -- 设计指令快照(异步任务用)
    ref_audio     TEXT,                        -- 克隆参考音频(任务作用域, 临时文件路径)
    ref_text      TEXT,
    result_audio  TEXT,                        -- 结果文件相对路径 data/tasks/<task_id>.wav
    sample_rate   INTEGER,                     -- 24000
    duration_sec  REAL,                        -- 结果时长(秒)
    progress_step TEXT,                        -- 进度: tokenize/prefill/ar_loop/decode
    progress_frame INTEGER,                    -- AR 循环当前帧
    max_frames    INTEGER,
    error_code    TEXT,
    error_message TEXT,
    created_at    TEXT NOT NULL,
    started_at    TEXT,
    finished_at   TEXT
);
CREATE INDEX idx_tasks_status_created ON tasks(status, created_at DESC);
```

**说明**：
- `ref_audio` 在任务作用域内：若来自 multipart 上传，落 `data/tasks/tmp_<task_id>.wav`，任务完成/失败后清理（不污染语音库）
- 队列满（可配 `max_queue_len`）时提交返回 429

### 2.3 无 `models` 表

模型清单来自 `config.yaml`（静态注册），不落库。运行状态（loaded/loading/idle 等）为**服务内内存态**，不持久化（重启后按配置重新加载）。

### 2.4 无 `config` 表

服务配置来自 `config.yaml` + 环境变量，只读展示（`GET /api/config`），不落库。

## 3. 上传文件管理

### 3.1 语音参考音频（持久）

- 路径：`<DATA_DIR>/voices/<voice_id>.<ext>`（`ext` = `wav` 或 `mp3`，保留原始上传格式）
- 校验：仅收 **wav / mp3**；mp3 经 `ffmpeg` 解码统一为 24kHz PCM 再进推理（`tok_encoder` 需 24kHz 输入）
- 保存策略：DTO 校验通过后写入；同名覆盖前先删除旧文件
- 删除语音条目（`DELETE /v1/voices/{id}`）时**级联删除**对应音频文件
- 清洗：`voice_id` 规范化（防路径穿越：仅允许 `[A-Za-z0-9_\u4e00-\u9fa5-]`，长度 ≤64）

### 3.2 任务结果音频（临时）

- 路径：`<DATA_DIR>/tasks/<task_id>.<ext>`（`ext` = `wav` 或 `mp3`，按 `response_format`）
- 保留策略：任务删除（`DELETE /api/tasks/{id}`）或 TTL 过期（可配 `task_ttl_seconds`，默认 7 天）时清理；服务启动时可选清理孤儿文件

## 4. 配置数据（`config.yaml`）

结构见 `docs/02-architecture.md` §5。关键字段约束：

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `server.host/port` | str/int | `0.0.0.0` / `8000` | Uvicorn 绑定 |
| `server.max_queue_length` | int | `16` | 同步接口排队上限(超限 429) |
| `server.api_token` | str | 空 | 鉴权 token；**env `Q3TTS_API_TOKEN` 优先**；空=无鉴权模式(告警) |
| `server.docs_public` | bool | `true` | `/docs` Swagger 是否豁免鉴权 |
| `data_dir` | str | `./data` | 可写数据根；Spaces 设 `/data`；**env `DATA_DIR` 优先** |
| `models[].load_on_start` | bool | `true` | 启动预加载 |
| `models[].capability` | enum | — | `voice_clone` \| `voice_design` |
| `models[].unload_after_idle_seconds` | int | `0` | `0`=不自动卸载 |
| `models[].max_new_tokens` | int | `2048` | 生成帧上限 |
| `models[].onnx_dir` / `tts_dir` | str | 见架构 | 模型目录（可含 `{DATA_DIR}` 占位） |
| `storage.voices_dir` / `tasks_dir` | str | `{data_dir}/voices` / `{data_dir}/tasks` | 相对 `DATA_DIR` |
| `voice.default_voice` | str | `""` | 兜底语音(空=需显式 voice) |
| `audio.ffmpeg_path` | str | `ffmpeg` | mp3 解码/编码；Docker 内置 |
| `audio.mp3_bitrate` | str | `128k` | mp3 输出码率 |
| `audio.max_ref_audio_seconds` | int | `30` | 参考音频长度上限 |

## 5. 一致性保证

- SQLite 单写者；服务为单进程单 worker，天然串行写
- 音频文件与 DB 记录的删除走同一事务/同一错误路径（先删文件，失败回滚标记，下次清理兜底）
- WAL checkpoint 由 SQLite 自动管理