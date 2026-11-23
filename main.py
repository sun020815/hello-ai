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
# 0.60 是扩库到 40 条之后重扫出来的：0.58~0.61 是平台区间（整体 93~94%），
# 取 0.60 偏保守一侧 —— 健身建议答错比不答更糟，宁可漏答也别给错方案。
# 4 条库时的 0.55 在 40 条库上会放行太多无关问题（拒答率只有 75%）。见 tune_threshold.py
VEC_THRESHOLD = 0.60
# 最多取几条资料塞进 prompt
VEC_TOP_K = 2

# 两段式检索：向量先宽松召回 RECALL_K 条（不求准），reranker 再精排。
# 库里只有 4 条时这个值等于「全召回」，reranker 没什么可挑的；
# 扩到 40 条之后必须放宽到十几条，否则真正该用的那条压根进不了候选，精排再准也救不回来。
RECALL_K = 12
RERANK_TOP_K = 2
# rerank 分数低于这个值就当「资料里没有」
# 注意它和余弦相似度不是一个量纲：0.60 那一套完全不能套用。
# 40 条库上以 0.005 步长扫了 0~0.20，0.055~0.085 是平台（整体 91%），取 0.06。
# 但它仍然不如纯向量（93%），原因也换了：不是候选太少，而是「语义近邻 ≠ 能回答」——
# 「膝盖半月板损伤怎么康复」的 rerank 最高分 0.4210，比该命中的最低分 0.0986 高得多，
# 因为深蹲那条资料确实在讲膝盖，可它没写康复。这条线切不开。见 tune_rerank.py 与 README
RERANK_THRESHOLD = 0.06

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


# 知识库从 kb.json 读，不再写死在代码里。
# 每条是一个 chunk：id 唯一、topic 是所属话题（评测按话题判命中）、text 是正文、source 是出处。
# 为什么拆出去：40 条正文塞进 .py 会让代码没法看，而且改内容不该动代码。
KB_PATH = Path(__file__).parent / "kb.json"
KB = json.loads(KB_PATH.read_text(encoding="utf-8"))

# 一个话题可能有多条 chunk（比如「肌酸」拆成剂量、时机、安全性、停用四条）
TOPICS = sorted({d["topic"] for d in KB})


# ---------- 老办法：比字面 ----------

def search_kb_keyword(q: str):
    """关键词匹配：问题里出现了哪个话题名，就算命中。
    注意这招的毛病：话题名必须一字不差地出现在问题里，「怎么变瘦」就永远命中不了「减脂」。"""
    hits = []
    for d in KB:
        if d["topic"] in q:
            hits.append({"id": d["id"], "topic": d["topic"], "text": d["text"],
                         "source": d["source"], "score": None})
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

    texts = [f"{d['topic']}：{d['text']}" for d in KB]

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


def pick_hits(ranking, top_k, threshold):
    """从排好序的候选里挑出要喂给模型的资料。两条规则：
    1. 分数过阈值 —— 不过线就当「资料里没有」，不硬凑；
    2. 同一话题只留一条（分最高的那条）—— 库里一个话题有多条 chunk，
       如果 top2 全是「减脂」，等于上下文被一个话题占满，用户问的可能是「减脂期蛋白吃多少」。
    多样性比同质的高分更值钱，这一条在扩库之后才成立。"""
    picked, seen = [], set()
    for r in ranking:
        if r["score"] < threshold:
            continue
        if r["topic"] in seen:
            continue
        seen.add(r["topic"])
        picked.append(r)
        if len(picked) >= top_k:
            break
    return picked


def search_kb_vector(q: str, top_k: int = VEC_TOP_K, threshold: float = VEC_THRESHOLD):
    """向量检索：把问题也变成向量，跟每条资料算相似度，取最像的几条。
    返回 (命中的资料, 完整排名)"""
    q_vec = embed([q])[0]
    ranking = []
    for d, kb_vec in zip(KB, get_kb_vectors()):
        ranking.append({
            "id": d["id"], "topic": d["topic"], "text": d["text"], "source": d["source"],
            "score": round(cosine(q_vec, kb_vec), 4),
        })
    ranking.sort(key=lambda x: x["score"], reverse=True)

    hits = pick_hits(ranking, top_k, threshold)
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
    for d, kb_vec in zip(KB, get_kb_vectors()):
        recalled.append({
            "id": d["id"],
            "topic": d["topic"],
            "text": d["text"],
            "source": d["source"],
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

    hits = pick_hits(ranking, top_k, threshold)
    return hits, ranking


# ---------- 第四段：混合检索（向量 + BM25）----------
# 现状：能跑，比纯向量高 1 条题（92.2% vs 92.0%），差异不显著，所以不作为默认。
# 为什么效果这么弱，见 README「混合检索：做了、量了、不采用」一节 —— 核心原因是
# 剩下的错误里只有 1 条是字面匹配问题，其余 6 条是语料分布、话题边界、语言鸿沟，
# 换排序算法都修不了。库扩到几千条时这套代码值得重新评估。

# 融合时向量分占多大比重。0.93 是在 87 条评测集上扫出来的平台区（0.90~0.96 都是 92.2%），
# 但 0.98 又会跌回 89.6% —— 这种锯齿本身就说明提升来自单条题的临界翻转，不是系统性改善。
HYBRID_W_VEC = 0.93

_bm25 = None


def get_bm25():
    """BM25 索引只在第一次用到时建，建一次就是纯内存计算，不花钱。"""
    global _bm25
    if _bm25 is None:
        from bm25 import BM25, tokenize_jieba
        # 嵌入文本和「向量」那一路保持一致，都是「话题名：正文」
        _bm25 = BM25([f"{d['topic']}：{d['text']}" for d in KB], tokenizer=tokenize_jieba)
    return _bm25


def hybrid_rank(q: str, q_vec):
    """纯计算：给定问题向量，返回融合后的排名。抽出来是为了让评测脚本
    能复用缓存的问题向量，不必重复调 embedding。"""
    bm_scores = get_bm25().scores(q)
    top_bm = max(bm_scores) or 1.0

    ranking = []
    for d, kb_vec, bm in zip(KB, get_kb_vectors(), bm_scores):
        vec_score = cosine(q_vec, kb_vec)
        fused = HYBRID_W_VEC * vec_score + (1 - HYBRID_W_VEC) * (bm / top_bm)
        ranking.append({
            "id": d["id"], "topic": d["topic"], "text": d["text"], "source": d["source"],
            # score 放的是「向量分」而不是融合分：门控（答不答）仍然只看向量分，
            # BM25 只参与排序。让 BM25 参与门控的那版实验把拒答率从 100% 打到 50%。
            "score": round(vec_score, 4),
            "fused": round(fused, 4),
            "bm25": bm,
        })
    ranking.sort(key=lambda x: x["fused"], reverse=True)
    return ranking


def search_kb_hybrid(q: str, top_k: int = VEC_TOP_K, threshold: float = VEC_THRESHOLD):
    q_vec = embed([q])[0]
    ranking = hybrid_rank(q, q_vec)
    hits = pick_hits(ranking, top_k, threshold)
    return hits, ranking


# ---------- 第五段：query 改写 ----------
# 动机是「文档语言 ≠ 查询语言」：资料里写的是陈述句（「沉肩，想的不是往下压」），
# 用户问的是问题句（「找不到发力感怎么办」），字面一个词都不重合。
# 做法：先让模型把口语问题改写成 3 条「资料里可能出现的表述」，再取每条 chunk 与
# 「原问题 + 3 条改写」的最大相似度。
#
# 实测结论：hit@1 从 90.9% 升到 92.2%、hit@2 从 94.8% 升到 97.4%，
# 但拒答率从 100% 掉到 90%，整体准确率 92.0% —— 与纯向量一模一样。
# 也就是说改写没有引入新信息，只是把工作点沿同一条曲线挪了位置。
# 原因是「取 K 条最大值」是极值统计，K 个样本的最大值期望必然高于 1 个样本，
# 该命中的和该拒答的一起被抬上去。详见 README 与 rewrite_experiment.py
REWRITE_THRESHOLD = 0.64
REWRITE_N = 3

REWRITE_PROMPT = (
    "你是检索助手。把用户的口语问题改写成 3 条健身资料里可能出现的表述。\n"
    "规则：\n"
    "1. 只换说法，不改变问题意图。不要回答这个问题。\n"
    "2. 用健身领域术语，比如 发力感、肌肉募集、离心控制、代偿、容量、渐进超负荷。\n"
    "3. 每条一句话，20 字以内。\n"
    "4. 三条之间角度要不同，不要只换同义词。\n"
    "只输出三行改写，不要编号，不要解释。"
)

REWRITE_CACHE = Path(__file__).parent / ".rewrites.json"
REWRITE_PROMPT_VER = "v1"


def clean_rewrite_line(line: str) -> str:
    """模型偶尔还是带编号或引号，清掉。"""
    s = line.strip().strip("\"'“”")
    for sep in (".", "、", ")", "）"):
        head, dot, tail = s.partition(sep)
        if dot and head.isdigit():
            s = tail.strip()
    return s


def rewrite_query(q: str):
    """口语问题 -> 3 条资料风格的表述。失败就退回原问题，不让检索整体挂掉。"""
    try:
        resp = call_llm([
            {"role": "system", "content": REWRITE_PROMPT},
            {"role": "user", "content": q},
        ])
    except Exception:
        return []
    lines = [clean_rewrite_line(l) for l in resp.choices[0].message.content.splitlines()]
    return [l for l in lines if l][:REWRITE_N]


def rewrite_many(questions):
    """批量改写 + 落盘缓存。只给评测和离线实验用，线上每次请求都是新问题，命中不了缓存。"""
    cache = {}
    if REWRITE_CACHE.exists():
        try:
            c = json.loads(REWRITE_CACHE.read_text(encoding="utf-8"))
            if c.get("model") == LLM_MODEL and c.get("prompt") == REWRITE_PROMPT_VER:
                cache = c.get("items", {})
        except Exception:
            pass

    todo = [q for q in questions if q not in cache]
    if todo:
        print(f"  需新改写 {len(todo)} 条（其余走缓存）")
        for i, q in enumerate(todo, 1):
            cache[q] = rewrite_query(q)
            print(f"    {i}/{len(todo)}", end="\r", flush=True)
        print()
        REWRITE_CACHE.write_text(
            json.dumps({"model": LLM_MODEL, "prompt": REWRITE_PROMPT_VER, "items": cache},
                       ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
    return cache


def search_kb_rewrite(q: str, top_k: int = VEC_TOP_K, threshold: float = REWRITE_THRESHOLD):
    """改写后检索：原问题 + 3 条改写一起嵌入，每条 chunk 取最大相似度。
    一次 embed 调用发 4 条文本，比逐条调用省 3 次往返。"""
    variants = [q] + rewrite_query(q)
    vecs = embed(variants)
    ranking = []
    for d, kb_vec in zip(KB, get_kb_vectors()):
        sims = [cosine(v, kb_vec) for v in vecs]
        ranking.append({
            "id": d["id"], "topic": d["topic"], "text": d["text"], "source": d["source"],
            "score": round(max(sims), 4),
            # 原问题自己的分留着，方便对比「是改写抬起来的还是本来就够」
            "orig_score": round(sims[0], 4),
        })
    ranking.sort(key=lambda x: x["score"], reverse=True)

    hits = pick_hits(ranking, top_k, threshold)
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
    mode=hybrid 向量 + BM25 混合检索（排序融合，门控仍看向量分）
    mode=rerank 向量召回 + reranker 精排（两段式）
    mode=rewrite 先把问题改写成 3 条资料风格表述，再取最大相似度
    mode=keyword 老的关键词匹配，方便对比"""
    if mode == "keyword":
        hits = search_kb_keyword(req.question)
        ranking = None
    elif mode == "rerank":
        hits, ranking = search_kb_rerank(req.question)
    elif mode == "hybrid":
        hits, ranking = search_kb_hybrid(req.question)
    elif mode == "rewrite":
        hits, ranking = search_kb_rewrite(req.question)
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
        "sources": [
            {"id": h["id"], "topic": h["topic"], "source": h["source"], "score": h["score"]}
            for h in hits
        ],
        "ranking": ranking,
        "mode": mode,
        "tokens": resp.usage.total_tokens,
    }
