from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

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




