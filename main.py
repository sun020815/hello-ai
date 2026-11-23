import json
import os
import random
import time
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from openai import OpenAI, APIConnectionError, APITimeoutError, RateLimitError
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


# ---------- 第零段：所有对外调用的超时与重试 ----------
# 这段不是为了「好看」，是因为 502 已经真真切切毁过三次实验数据，
# 其中一次是 87 条评测跑到七十几条时崩掉。
#
# 先说清楚 SDK 已经做了什么，免得重复造轮子：
#   openai 默认 max_retries=2，遇到 429 / 5xx / 连接错误本来就会重试。
# 那为什么还要自己写？两个洞：
#   1. 默认 read timeout 是 600 秒。上游连接被半关（代理切换时很常见）的时候，
#      请求会挂十分钟才报错 —— 这比立刻报错更糟，因为你分不清"在慢"还是"已经死了"。
#   2. 重试是静默的。跑评测时只能感觉到"这轮特别慢"，看不出是哪一次调用在重试、
#      重试了几次、最后到底成没成。出了问题无从归因。
# 所以：把 SDK 的重试关掉（max_retries=0），自己写循环。
#   次数、退避、日志全归我们管，而且 chat / embedding / rerank 三处共用一套策略 ——
#   rerank 走的是裸 urllib，SDK 那套重试根本管不到它，这段之前是零保护。

# 单次请求总超时（秒）。60 秒足够 8B 模型吐几百字，又不会让一个死连接拖住整个服务。
LLM_TIMEOUT = 60.0
# 嵌入只是把文字压成向量，正常几十毫秒，30 秒还没回来基本就是断了
EMBED_TIMEOUT = 30.0
# reranker 要把 12 段资料逐条过一遍，比嵌入慢，给 40 秒
RERANK_TIMEOUT = 40.0

MAX_ATTEMPTS = 3        # 一共试 3 次（1 次正常 + 2 次重试）
BACKOFF_BASE = 1.0      # 第 1 次重试等 1 秒，第 2 次等 2 秒
BACKOFF_CAP = 8.0       # 封顶：不管重试几次都不超过 8 秒
RETRY_JITTER = 0.5      # 抖动上限，防止所有请求同一时刻一起重试（惊群）

# 这几种状态码属于「再试一次可能就好」。
# 408 超时 / 409 冲突 / 425 太早 / 429 限流 / 5xx 上游挂了。
# 反过来 400 参数错、401 key 错、403 没权限、404 模型名错 —— 重试一百次结果一样，
# 只会把「立刻失败」变成「三十秒后失败」，中间还照样计费。所以必须区分。
RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}

_clients = {}


def get_client(timeout: float) -> OpenAI:
    """按超时时间复用 client。
    之前每次调用都 new 一个 OpenAI(...)，等于每次重新开一次连接池，
    keep-alive 完全没用上 —— 跑 87 条评测就是 87 次 TCP 握手外加 87 次 TLS 协商。"""
    if timeout not in _clients:
        _clients[timeout] = OpenAI(
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL,
            timeout=timeout,
            max_retries=0,      # 关掉 SDK 自带重试，改由 call_with_retry 统一管
        )
    return _clients[timeout]


def _should_retry(exc: Exception) -> bool:
    """这个错值不值得再试一次。"""
    # 先看状态码。注意 urllib.error.HTTPError 是 URLError 的子类，
    # 顺序反了的话 401 会被当成「网络问题」重试两次，白等三秒。
    status = getattr(exc, "status_code", None)
    if status is None and isinstance(exc, urllib.error.HTTPError):
        status = exc.code
    if status is not None:
        return status in RETRYABLE_STATUS

    # 没有状态码的：连接层错误。断了、超时、DNS 抽风，这些都值得重试。
    if isinstance(exc, (APITimeoutError, APIConnectionError, RateLimitError)):
        return True
    if isinstance(exc, (urllib.error.URLError, ConnectionError, TimeoutError)):
        return True
    return False


def _backoff(attempt: int) -> float:
    """指数退避：1s -> 2s -> 4s ...，加一点随机抖动。
    为什么要指数：上游 502 通常是瞬时过载，隔 0.1 秒重试等于继续加压；
    为什么要抖动：同一个服务里多个请求同时失败时，如果都等整 1 秒，
    会在同一刻一起回来，把上游再打挂一次。"""
    delay = min(BACKOFF_BASE * (2 ** (attempt - 1)), BACKOFF_CAP)
    return delay + random.uniform(0, RETRY_JITTER)


def call_with_retry(fn, label: str = "调用"):
    """把一次可能失败的外部调用包起来：失败 -> 判断值不值得重试 -> 退避 -> 再试。
    重试耗尽或遇到不该重试的错，就把最后一次的异常原样抛出，交给调用方翻译成 HTTP 状态码。"""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return fn()
        except HTTPException:
            # 我们自己抛的（key 没填之类），没有重试的意义
            raise
        except Exception as e:
            if attempt == MAX_ATTEMPTS or not _should_retry(e):
                raise
            wait = _backoff(attempt)
            # 打日志而不是静默重试：出问题时能从终端看出来是哪一步在挣扎
            print(
                f"  [{label}] 第 {attempt} 次失败（{type(e).__name__}: {e}），"
                f"{wait:.1f}s 后重试",
                flush=True,
            )
            time.sleep(wait)


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
    if not LLM_API_KEY or LLM_API_KEY.startswith("sk-替换"):
        raise HTTPException(status_code=500, detail=".env 里的 LLM_API_KEY 还没填")

    try:
        resp = call_with_retry(
            lambda: get_client(EMBED_TIMEOUT).embeddings.create(model=EMBED_MODEL, input=texts),
            label="embedding",
        )
    except Exception as e:
        raise HTTPException(
            status_code=502, detail=f"调用 embedding 模型失败（已试 {MAX_ATTEMPTS} 次）：{e}"
        )
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

    def _post():
        with urllib.request.urlopen(req, timeout=RERANK_TIMEOUT) as resp:
            body = resp.read().decode()
        try:
            return json.loads(body)
        except json.JSONDecodeError as e:
            # 状态码 200 但正文不是 JSON，通常是中间代理吐了个 HTML 错误页。
            # 这不是「代码写错了」，是响应被截断，属于连接层问题，值得重试。
            raise urllib.error.URLError(f"rerank 响应不是合法 JSON：{e}") from e

    try:
        data = call_with_retry(_post, label="rerank")
    except Exception as e:
        raise HTTPException(
            status_code=502, detail=f"调用 reranker 失败（已试 {MAX_ATTEMPTS} 次）：{e}"
        )

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

    try:
        return call_with_retry(
            lambda: get_client(LLM_TIMEOUT).chat.completions.create(
                model=LLM_MODEL, messages=messages
            ),
            label="chat",
        )
    except Exception as e:
        raise HTTPException(
            status_code=502, detail=f"调用模型失败（已试 {MAX_ATTEMPTS} 次）：{e}"
        )


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
            "degraded": False,
        }

    context = "\n".join(f"- {h['topic']}：{h['text']}" for h in hits)

    sources = [
        {"id": h["id"], "topic": h["topic"], "source": h["source"], "score": h["score"]}
        for h in hits
    ]

    try:
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
    except HTTPException as e:
        if e.status_code != 502:
            raise       # key 没填这类配置错误，不该伪装成「能答」，直接暴露出来
        # 走到这里说明：检索成功、生成失败。这时候甩一个 502 等于让用户白等一场，
        # 而资料是真实命中的 —— 把资料原样交出去，比一句报错有用。
        # 但必须打上 degraded 标记，否则调用方会以为这是模型润色过的答案。
        return {
            "answer": "模型服务暂时不稳定，重试后仍失败。以下是命中的原始资料：\n"
                      + "\n".join(f"- {h['topic']}：{h['text']}" for h in hits),
            "sources": sources,
            "ranking": ranking,
            "mode": mode,
            "tokens": 0,
            "degraded": True,
        }

    return {
        "answer": resp.choices[0].message.content,
        "sources": sources,
        "ranking": ranking,
        "mode": mode,
        "tokens": resp.usage.total_tokens,
        "degraded": False,
    }
