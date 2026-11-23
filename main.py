import json
import os
import urllib.request
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

app = FastAPI()

LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.siliconflow.cn/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen3-8B")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

# 相似度低于这个值就当「资料里没有」，不调模型
# 0.55 不是拍出来的：46 条评测集上扫了 0.45~0.72，0.45~0.56 是平台区间
# （整体准确率都在 91~93%），取区间中段比取边界更稳。见 tune_threshold.py
VEC_THRESHOLD = 0.55
# 最多取几条资料塞进 prompt
VEC_TOP_K = 2

# 两段式检索：向量先宽松召回 RECALL_K 条（不求准），reranker 再精排
RECALL_K = 4
RERANK_TOP_K = 2
# rerank 分数低于这个值就当「资料里没有」
# 注意它和余弦相似度不是一个量纲：0.55 那一套完全不能套用。
# 46 条评测集上以 0.002 步长扫了 0~0.10，0.006~0.018 是平台区间（整体准确率都是 87%），
# 取中段 0.01。即便这样，它仍不如纯向量（93%）。见 tune_rerank.py 与 README
RERANK_THRESHOLD = 0.01

# 知识库向量的本地缓存，避免每次重启都重新算（算一次要花钱）
CACHE_PATH = Path(__file__).parent / ".kb_vectors.json"


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
    "蛋白质": "每公斤体重 1.6 到 2.2 克，分 3 到 4 餐吃，比一次吃完效果好。",
    "减脂": "先做 300 到 500 大卡的热量缺口，力量训练别停，否则掉的是肌肉。",
    "深蹲": "膝盖方向和脚尖一致，下蹲到大腿平行地面，起身时先顶髋。",
}


# ---------- 老办法：比字面 ----------

def search_kb_keyword(q: str):
    """关键词匹配：问题里出现了哪个话题名，就算命中"""
    hits = []
    for key, text in KB.items():
        if key in q:
            hits.append({"topic": key, "text": text, "score": None})
    return hits


# ---------- 新办法：比意思 ----------

def cosine(a, b):
    """余弦相似度：两个向量方向有多接近，返回 -1 到 1，越大越像。
    手写版，只用 Python 内置函数，没装 numpy。"""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    return dot / (norm_a * norm_b)


def embed(texts):
    """把一串文字变成一串向量（每条 1024 个数字）"""
    client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    try:
        resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"调用 embedding 模型失败：{e}")
    return [d.embedding for d in resp.data]


# 知识库向量只在内存里存一份，第一次用到时才算
_kb_vectors = None


def get_kb_vectors():
    """算出（或从缓存读出）知识库每条资料的向量。
    注意嵌入的文本是「话题名：正文」，带上话题名能明显提高命中率。"""
    global _kb_vectors
    if _kb_vectors is not None:
        return _kb_vectors

    texts = [f"{name}：{text}" for name, text in KB.items()]

    # 缓存有效就复用：模型和文本都没变才算有效
    if CACHE_PATH.exists():
        try:
            cached = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
            if cached.get("model") == EMBED_MODEL and cached.get("texts") == texts:
                _kb_vectors = cached["vectors"]
                return _kb_vectors
        except Exception:
            pass  # 缓存文件坏了就重新算，不影响功能

    vectors = embed(texts)
    CACHE_PATH.write_text(
        json.dumps({"model": EMBED_MODEL, "texts": texts, "vectors": vectors}, ensure_ascii=False),
        encoding="utf-8",
    )
    _kb_vectors = vectors
    return _kb_vectors


def search_kb_vector(q: str, top_k: int = VEC_TOP_K, threshold: float = VEC_THRESHOLD):
    """向量检索：把问题也变成向量，跟每条资料算相似度，取最像的几条。
    返回 (命中的资料, 完整排名)"""
    q_vec = embed([q])[0]
    ranking = []
    for (name, text), kb_vec in zip(KB.items(), get_kb_vectors()):
        ranking.append({"topic": name, "text": text, "score": round(cosine(q_vec, kb_vec), 4)})
    ranking.sort(key=lambda x: x["score"], reverse=True)

    hits = [r for r in ranking[:top_k] if r["score"] >= threshold]
    return hits, ranking


# ---------- 第三段：reranker 精排 ----------

def rerank(query: str, docs):
    """把「问题 + 若干段资料」交给 reranker，它会逐条判断：这段资料到底有没有回答这个问题。

    跟 embedding 的区别：embedding 比的是「两段文字意思像不像」，
    reranker 比的是「这段资料是不是这个问题的答案」。后者才对得上「该不该答」。
    返回 [{"index": 原文下标, "score": 0~1}, ...]，按分数降序。"""
    if not LLM_API_KEY or LLM_API_KEY.startswith("sk-替换"):
        raise HTTPException(status_code=500, detail=".env 里的 LLM_API_KEY 还没填")

    payload = json.dumps({
        "model": RERANK_MODEL,
        "query": query,
        "documents": docs,
        "top_n": len(docs),
    }).encode()

    req = urllib.request.Request(
        LLM_BASE_URL.rstrip("/") + "/rerank",
        data=payload,
        headers={
            "Authorization": f"Bearer {LLM_API_KEY}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=40) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"调用 reranker 失败：{e}")

    return [
        {"index": r["index"], "score": round(r["relevance_score"], 4)}
        for r in data["results"]
    ]


def search_kb_rerank(q: str, top_k: int = RERANK_TOP_K, threshold: float = RERANK_THRESHOLD):
    """两段式检索：向量宽松召回 -> reranker 精排 -> 按 rerank 分数决定答不答。
    返回 (命中的资料, 完整排名)，排名里同时留着两套分数方便对比。"""
    q_vec = embed([q])[0]
    recalled = []
    for (name, text), kb_vec in zip(KB.items(), get_kb_vectors()):
        recalled.append({
            "topic": name,
            "text": text,
            "vec_score": round(cosine(q_vec, kb_vec), 4),
        })
    recalled.sort(key=lambda x: x["vec_score"], reverse=True)
    recalled = recalled[:RECALL_K]      # 召回阶段故意放宽，准确率交给下一段

    scores = rerank(q, [f"{r['topic']}：{r['text']}" for r in recalled])
    ranking = []
    for s in scores:
        item = dict(recalled[s["index"]])
        item["score"] = s["score"]       # 统一用 score 字段，方便前端只认一个
        ranking.append(item)
    ranking.sort(key=lambda x: x["score"], reverse=True)

    hits = [r for r in ranking[:top_k] if r["score"] >= threshold]
    return hits, ranking


@app.get("/kb")
def kb(q: str):
    hits = search_kb_keyword(q)
    if not hits:
        return {"matched": None, "answer": "知识库里没有相关内容"}
    return {"matched": hits[0]["topic"], "answer": hits[0]["text"]}


def call_llm(messages):
    """把「调用模型」这件事收成一个函数，两处共用"""
    if not LLM_API_KEY or LLM_API_KEY.startswith("sk-替换"):
        raise HTTPException(status_code=500, detail=".env 里的 LLM_API_KEY 还没填")

    client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    try:
        return client.chat.completions.create(model=LLM_MODEL, messages=messages)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"调用模型失败：{e}")


@app.post("/chat")
def chat(req: AskRequest):
    resp = call_llm([
        {"role": "system", "content": "你是一名健身教练。回答要简短具体，给数字，不说套话。"},
        {"role": "user", "content": req.question},
    ])

    return {
        "answer": resp.choices[0].message.content,
        "model": resp.model,
        "tokens": resp.usage.total_tokens,
    }


@app.post("/rag")
def rag(req: AskRequest, mode: str = "vector"):
    """mode=vector 纯向量检索（默认）
    mode=rerank 向量召回 + reranker 精排（两段式）
    mode=keyword 老的关键词匹配，方便对比"""
    if mode == "keyword":
        hits = search_kb_keyword(req.question)
        ranking = None
    elif mode == "rerank":
        hits, ranking = search_kb_rerank(req.question)
    else:
        hits, ranking = search_kb_vector(req.question)

    if not hits:
        return {
            "answer": "知识库里没有相关资料，我无法基于资料回答。",
            "sources": [],
            "ranking": ranking,
            "mode": mode,
            "tokens": 0,
        }

    context = "\n".join(f"- {h['topic']}：{h['text']}" for h in hits)

    resp = call_llm([
        {
            "role": "system",
            "content": (
                "你是一名健身教练。只能依据【资料】里的内容回答。"
                "资料没提到的信息，直接说「资料里没有这部分内容」，禁止补充资料以外的知识。"
                "回答简短具体，给数字。"
            ),
        },
        {
            "role": "user",
            "content": f"【资料】\n{context}\n\n【问题】\n{req.question}",
        },
    ])

    return {
        "answer": resp.choices[0].message.content,
        "sources": [h["topic"] for h in hits],
        "ranking": ranking,
        "mode": mode,
        "tokens": resp.usage.total_tokens,
    }
