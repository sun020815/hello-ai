"""临时对照：证明「同步路由的 40 并发墙」是 anyio 线程池容量，而不是 uvicorn 或代码。

跑法：.venv/Scripts/python.exe tp_limit.py            -> 默认 40
      THREAD_LIMIT=120 .venv/Scripts/python.exe tp_limit.py
跟 tp_app（同样的 /slow 路由、不改 limiter）对比 80 并发下的 RPS 与 p50。

留这个文件是为了让「40 这个数从哪来」可复现 —— 一旦哪天依赖升级导致默认值变化，
这个对照能立刻发现。
"""
import os
import time
from contextlib import asynccontextmanager

import anyio.to_thread
import uvicorn
from fastapi import FastAPI

LIMIT = int(os.getenv("THREAD_LIMIT", "0"))


@asynccontextmanager
async def lifespan(app):
    if LIMIT:
        anyio.to_thread.current_default_thread_limiter().total_tokens = LIMIT
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/slow")
def slow():
    """同步阻塞 0.5 秒。同步 def 路由会被 FastAPI 丢进 anyio 线程池，
    池子里有几个线程，就有几个能真正并行 —— 这个路由就是用来量那个数的。"""
    time.sleep(0.5)
    return {"ok": 1, "limit": LIMIT or 40}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8036")),
                log_level="warning")
