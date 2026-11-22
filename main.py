import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

app = FastAPI()

LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.siliconflow.cn/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen3-8B")

@app.get("/")
def read_root():
    return {"message": "hello"}
@app.get("/greet/{name}")
def greet(name: str, city: str = "杭州"):
    return {"message": f"你好，{name}", "city": city}
class AskRequest(BaseModel):
    question: str
    top_k: int = 3


@app.post("/ask")
def ask(req: AskRequest):
    return {"received": req.question, "top_k": req.top_k}
KB = {
    "肌酸": "每天 3 到 5 克，什么时候吃都行，关键是每天吃。不需要冲击期。",
    "蛋白": "每公斤体重 1.6 到 2.2 克，分 3 到 4 餐吃，比一次吃完效果好。",
    "减脂": "先做 300 到 500 大卡的热量缺口，力量训练别停，否则掉的是肌肉。",
    "深蹲": "膝盖方向和脚尖一致，下蹲到大腿平行地面，起身时先顶髋。",
}


@app.get("/kb")
def kb(q: str):
    for key, answer in KB.items():
        if key in q:
            return {"matched": key, "answer": answer}
    return {"matched": None, "answer": "知识库里没有相关内容"}


@app.post("/chat")
def chat(req: AskRequest):
    if not LLM_API_KEY or LLM_API_KEY.startswith("sk-替换"):
        raise HTTPException(status_code=500, detail=".env 里的 LLM_API_KEY 还没填")

    client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "你是一名健身教练。回答要简短具体，给数字，不说套话。"},
                {"role": "user", "content": req.question},
            ],
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"调用模型失败：{e}")

    return {
        "answer": resp.choices[0].message.content,
        "model": resp.model,
        "tokens": resp.usage.total_tokens,
    }





