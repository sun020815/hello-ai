#!/usr/bin/env bash
# macOS / Linux 一键启动
set -e
cd "$(dirname "$0")"

if [ ! -f .env ]; then
  cp .env.example .env
  echo "已生成 .env，请填入 LLM_API_KEY 后重新运行"
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "缺少 uv，先装：curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi

echo "→ http://127.0.0.1:8000"
exec uv run uvicorn main:app --host 127.0.0.1 --port 8000
