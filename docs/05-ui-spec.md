# Qwen3-TTS ONNX 部署服务 — UI 规格说明

> 版本: 1.0
> 日期: 2026-09-07
> 关联: `docs/01-requirements.md` §3.3 · `docs/02-architecture.md`

## 1. 总体

- 静态单页，无前端构建：`web/index.html` + `web/style.css` + `web/app.js`
- FastAPI 挂载 `StaticFiles` 于 `/static`，根路由 `/` 返回 `index.html`
- 原生 JS（无框架）；fetch API 调后端；`<audio>` 元素播放
- 中文界面

## 2. 页面布局：顶部 Tab 导航 + 内容区

```
┌────────────────────────────────────────────────────────────┐
│  Qwen3-TTS ONNX · 🎙️语音合成 | 🗂️语音库 | 🤖任务 | ⚙️模型/服务 │
├────────────────────────────────────────────────────────────┤
│                    当前 Tab 内容区                           │
└────────────────────────────────────────────────────────────┘
```

## 3. Tab 1 🎙️ 语音合成

**模型选择**（radio 卡片）：
- `qwen3-tts-0.6b-base` · 语音克隆
- `qwen3-tts-1.7b-voicedesign` · 声音设计

**语音来源**（radio）：
- **用已注册语音**：下拉框（`GET /v1/voices` 按类型过滤）+ 试听按钮
- **临时参数（不入库）**：
  - Base：上传参考音频（file input）+ 参考文本（textarea）
  - VoiceDesign：指令文本（textarea，如"用温柔缓慢的语气说中文"）

**表单**：
- 合成文本（textarea，必填）
- 语言下拉（10 种）
- 高级（折叠）：`speed`（预留，灰掉/注明"暂不实现"）、`max_new_tokens`（可选）

**操作**：
- 【▶️ 同步合成】：按钮 + 加载动画；完成后预览（`<audio controls>`）+ 下载按钮（Blob）
- 【⏱️ 异步提交】：提交为任务 → 提示 task_id → 跳转 Tab 3 查看进度
- 结果区展示：时长、生成耗时、模型、是否使用了缓存等调试信息

## 4. Tab 2 🗂️ 语音库

**工具栏**：`+ 注册克隆语音`、`+ 注册设计语音`、刷新

**列表表格**（`GET /v1/voices`）：voice_id / 类型徽章 / 语言 / 指令或引用（截断展示）/ 更新时间 / 操作（试听 · 编辑 · 删除）

**注册向导（克隆）**（multipart `POST /v1/voices`）：
1. voice_id（文本）
2. 语言（下拉）
3. 参考音频（file，显示已选文件名/时长）
4. 参考文本（textarea，必填——强调"请准确转写，克隆质量依赖于此"）
5. description（可选）
6. 提交（前端校验缺失字段）

**注册向导（设计）**（JSON `POST /v1/voices`）：
1. voice_id
2. 语言
3. instruct 指令（textarea，必填）
4. description（可选）

**编辑**：语言/instruct/ref_text/description 可改（PATCH `PATCH /v1/voices/{id}`）；type 与 voice_id 不可改
**删除**：确认对话框（提示将同步删除参考音频文件）
**试听**：`POST /v1/voices/{id}/test`，返回音频 → 内嵌播放

**错误展示**：整齐的红色内联错误（如"voice_id 已存在"、"参考文本必填"）

## 5. Tab 3 🤖 任务列表

**工具栏**：刷新、过滤下拉（全部/pending/running/completed/failed/cancelled）

**表格**：task_id（截断）/ 模型 / 状态徽章 / 文本预览（截断 30 字）/ 进度条（AR frame/max）/ 创建时间 / 操作

**状态徽章**：
| 状态 | 徽章色 |
|---|---|
| pending | 灰 |
| running | 蓝（含进度条） |
| completed | 绿（可下载） |
| failed | 红（可查看错误） |
| cancelled | 灰 |

**操作**：下载（`GET /api/tasks/{id}/audio`）、取消（DELETE）、删除（DELETE）
**轮询**：任务列表与详情 3s 轮询（仅当有 active 任务时）

## 6. Tab 4 ⚙️ 模型/服务

**模型卡片 ×2**（`GET /api/models/status`）：
| 字段 | 展示 |
|---|---|
| 模型 ID / 能力 | 标题 |
| 状态（unloaded/loading/loaded/unloading） | 徽章 |
| 估算内存 | RAM MB |
| 加载耗时 / 推理次数 / 上次使用 | 明细 |
| 空闲秒数 | 明细 |
| 按钮 | [加载] [卸载]（按状态启用/禁用） |

**服务信息**：版本、uptime
**配置查看**（`GET /api/config`）：只读展示（json 折叠树）
**日志尾页**：轮询 `GET /api/logs?limit=50`（可配开启；滚动文本区 + 自动更新开关）

## 7. 交互约定与错误处理

- 全局 fetch 封装：统一 JSON 化、处理 `error` envelope、401/404/409/429/500 映射为可读信息
- 长时间同步请求：按钮 loading + 计时显示；超长文本提示"建议使用异步任务"
- 下载实现：`window.URL.createObjectURL(blob)` + `<a download>`
- 上传进度：`XMLHttpRequest.upload.onprogress`（可配）
- 所有 Tab 数据刷新 `f5` 快捷键与工具栏刷新按钮

## 8. 界面技术要求

- 单文件 CSS/JS，内联或独立文件均可（无构建）
- 响应式（桌面优先）
- 无外部 CDN 依赖（离线友好）
- 字体系统字体（中文界面，系统 UI 字体栈）