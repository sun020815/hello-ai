# 用 python:3.14-slim 而不是 uv 官方镜像：ghcr.io 在国内经常拉不动，
# Docker Hub 可以走镜像源。uv 单独用 pip 装，走清华源。
FROM python:3.14-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple uv

WORKDIR /app

# 依赖单独一层：改代码不会让 uv sync 重跑
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# 代码层。.dockerignore 已经把 .env、.venv、缓存和评测产物排除掉了
COPY . .

# 非 root 跑。缓存文件写不进去也不影响功能（main.py 里已容错）
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/meta', timeout=4).status==200 else 1)"

# 单 worker：熔断器是进程内存态，多 worker 会各算各的（README 已知问题九）
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
