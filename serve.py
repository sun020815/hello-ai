"""公网部署用的启动入口。

不用 `uvicorn main:app --port $PORT`：发布沙箱不是通过 shell 执行启动命令的，
`$PORT` 不会被展开，uvicorn 收到的是空字符串，直接报
`Option '--port' requires an argument`（已踩过）。

改成由 Python 自己读环境变量，既不依赖 shell，也不依赖具体平台约定：
    本地  python serve.py            -> 3000
    沙箱  PORT=xxxx python serve.py  -> 用平台给的端口
"""
import os

import uvicorn

if __name__ == "__main__":
    port = int(os.getenv("PORT", "3000"))
    # 必须绑 0.0.0.0：平台的反代是从容器外部进来的，绑 127.0.0.1 连不上
    uvicorn.run("main:app", host="0.0.0.0", port=port, workers=1)
