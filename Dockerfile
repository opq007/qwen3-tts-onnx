# Qwen3-TTS ONNX — 常规 Docker + HF Spaces 共用镜像
# 运行形态经环境变量区分：PORT / DATA_DIR / Q3TTS_API_TOKEN / HF_TOKEN
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    DATA_DIR=/data

# 1) 系统依赖：ffmpeg（mp3 输入/输出）+ 基础工具
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 2) Python 依赖
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3) 应用代码
COPY app/ ./app/
COPY web/ ./web/
COPY scripts/ ./scripts/
COPY config.example.yaml ./

# 4) 可写数据目录（Spaces 挂 /data；常规部署用卷）
RUN mkdir -p /data/voices /data/tasks /data/models

EXPOSE 7860

# 启动：uvicorn 绑定 PORT
CMD ["sh", "-c", "cd /app && uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]