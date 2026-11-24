import hashlib
import json
import os
import queue
import random
import re
import threading
import time
import uuid
import urllib.error
import urllib.request
from collections import OrderedDict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

import anyio.to_thread

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI, APIConnectionError, APITimeoutError, RateLimitError
from pydantic import BaseModel, Field, StringConstraints

import plan_agent

# numpy 是可选的：装了就用矩阵乘算余弦（比逐条 Python 循环快 600~800 倍，
# 见 bench_scale.py），没装就退回纯 Python，功能不受影响，只是慢。
try:
    import numpy as np
except ImportError:      # pragma: no cover - 只在不装 numpy 的环境走到
    np = None

load_dotenv()

# ---------- 同步路由的并发上限 ----------
# /rag、/chat、/plan 都是同步 def，FastAPI 会把它们丢进 anyio 线程池执行，
# 池子默认只有 40 个线程。这不是「建议值」，是硬墙：第 41 个请求开始排队，
# 而排队时间会计进用户看到的延迟里。
#
# 压测实测（tp_limit.py，一个只 sleep 0.5 秒的同步路由）：
#   并发 40 -> 73.7 RPS，p50 505ms          （40 个线程刚好够）
#   并发 80 -> 76.0 RPS，p50 1003ms         （RPS 一点没涨，延迟翻倍 = 全在排队）
#   池子调到 120，同样的 80 并发 -> 141 RPS，p50 511ms
# 所以这个数是可以挪的，一个环境变量的事。
#
# 但默认不挪。原因有两层：
#   1. 线程是稀缺资源，每个线程要栈；32 位解释器下把池子开到几百，内存先炸。
#   2. 这个服务真正的瓶颈是上游模型（一次 5~10 秒），线程池放大只会让更多请求
#      同时压到上游，触发对方的并发限制 —— 墙从「自己排队」变成「对面拒绝」，
#      用户看到的东西没变好。
# 要调就显式给 THREAD_LIMIT，并且先想清楚上面两条。
THREAD_LIMIT = int(os.getenv("THREAD_LIMIT", "0"))


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    if THREAD_LIMIT:
        anyio.to_thread.current_default_thread_limiter().total_tokens = THREAD_LIMIT
    yield
    # 正常退出（SIGTERM / Ctrl-C）时把队列里还没落盘的问法写完。
    # 日志是异步写的，不 flush 就等于「每次重启丢掉最后几百毫秒的样本」。
    try:
        flush_query_log()
    except Exception:      # noqa: BLE001 —— 退出路径上不许再抛
        pass


app = FastAPI(lifespan=_lifespan)

LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.siliconflow.cn/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen3-8B")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")

# 主生成要不要开思维链（走 extra_body.enable_thinking）。
#   None  = 不传这个参数，完全交给供应商默认（线上现状，行为不变）
#   False = 显式关掉
# 默认 None 是为了「加开关不改行为」：改了默认值，所有历史结论都得重跑一遍。
_GEN_THINKING_ENV = os.getenv("GEN_THINKING", "").strip().lower()
GEN_THINKING = None if _GEN_THINKING_ENV == "" else (_GEN_THINKING_ENV not in ("0", "false", "off", "no"))



# ---------- 成本记账 ----------
# 每次外部调用记一笔 token，/stats 对外暴露，cost_report.py 拿它算单次成本。
# 只记 token、不记钱：单价会变，不同账号还有折扣，把人民币硬编码进代码是个陷阱
# （换个模型整张表就错了）。要知道多少钱，拿这里的数乘当时的单价。
COST = {
    "embed_tokens": 0, "rerank_tokens": 0,
    "prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0,
    "embed_calls": 0, "rerank_calls": 0, "chat_calls": 0,
}
_cost_lock = threading.Lock()


def add_cost(**kw):
    """累加 token / 调用次数。加了锁：uvicorn 多线程下 += 不是原子操作。"""
    with _cost_lock:
        for k, v in kw.items():
            COST[k] = COST.get(k, 0) + (v or 0)


def bill_chat(resp):
    """chat 类响应：生成 + 思维链分开记。
    reasoning_tokens 单列是因为它常常比正文还多 —— 看不清这块，成本会算少一半。"""
    u = getattr(resp, "usage", None)
    if not u:
        return
    detail = getattr(u, "completion_tokens_details", None)
    add_cost(
        prompt_tokens=getattr(u, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(u, "completion_tokens", 0) or 0,
        reasoning_tokens=getattr(detail, "reasoning_tokens", 0) or 0,
        chat_calls=1,
    )

# 精排默认用 Qwen3-Reranker-4B：bge 对口语疑问句系统性压分（同一道题 0.04 对 0.80），
# 换成 4B 后 20 条最难的问法从 8/20 命中升到 14/20，AUC 0.648 → 0.875。
RERANK_MODEL = os.getenv("RERANK_MODEL", "Qwen/Qwen3-Reranker-4B")

# 相似度低于这个值就当「资料里没有」，不调模型
# 0.60 是 v1 语料（126 条）上扫出来的：0.58~0.61 是平台区间（整体 93~94%），
# 取偏保守一侧 —— 健身建议答错比不答更糟，宁可漏答也别给错方案。
#
# v2 换成 105 条同一体系（谭成义）的语料后，这个阈值「没有变，但不能沿用原来的理由」。
# 在 98 条新评测集上重扫（tune_threshold.py，纯离线）发现：把阈值抬到 0.65 确实能把
# 拒答率从 70% 拉到 100%，但代价是同时砍掉 4 条真答案 —— 因为正确和错误的分数是交错的：
#     0.6144 含胸双杠（对）  0.6257 肌酸（错）  0.6281 三分化循环（对）
#     0.6327 保加利亚（对）  0.6330 减脂热量（错）  0.6405 生酮（错）  0.6491 半程（对）
# 0.6327 的正解被夹在 0.6257 和 0.6330 两个错解中间，没有任何一条线能切开。
# 所以这不是阈值问题，是「饮食配合」那两条 chunk 本身就在讲「每天吃多少」，
# 与「肌酸吃多少」「减脂吃多少卡」共享同一片语义空间 —— 内容层面的事，调参解决不了。
# 结论：阈值不动。那 3 条误命中留给生成层的 prompt 兜（system 里写了「资料没提到就说没有」），
# 但不能说这是已验证的 —— 端到端验证需要真实调用，见 README「v2 语料上的三路对比」。
VEC_THRESHOLD = 0.60
# 最多取几条资料塞进 prompt
VEC_TOP_K = 2

# 两段式检索：向量先宽松召回 RECALL_K 条（不求准），reranker 再精排。
# 库里只有 4 条时这个值等于「全召回」，reranker 没什么可挑的；
# 扩到 40 条之后必须放宽到十几条，否则真正该用的那条压根进不了候选，精排再准也救不回来。
RECALL_K = 12
# 精排过线之后，最多把几条资料塞进 prompt。
#
# 定 3 不是拍的，是 eval_compound.py 量出来的（35 条复合题 + 200 题老集）：
#   2 -> 3：复合题话题覆盖率 74.1% -> 86.4%（+12.3），完整覆盖 51.9% -> 74.1%
#   3 -> 4：覆盖率只再涨 1.3，完整覆盖不动
# 而 200 题老集上 2/3/4 的 hit@1、拒答率**一模一样**（92.8% / 80.6%），
# 平均过线条数从 0.86 只涨到 1.00 —— 因为门槛 0.95 很严，普通问题本来就只有
# 一条资料能过线，top_k 调大对它们没有任何影响。**加条数只对复合问题生效**，
# 对单意图问题既不帮忙也不添乱，所以这是一次零风险的改动。
#
# 3 而不是 4：4 的边际收益 1.3 个点，但塞进 prompt 的无关条目占比从 30.9% 涨到 32.4%，
# 上下文变长、模型被带偏的概率上升。3 是拐点。
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "3"))
# rerank 分数低于这个值就当「资料里没有」
# 注意它和余弦相似度不是一个量纲：0.60 那一套完全不能套用。
#
# 这个值的来历换了三次，本身就是「参数只在特定语料上成立」的证据：
#   4 条库：没意义，全召回
#   40 条库：扫 0~0.20，0.055~0.085 是平台，取 0.06
#   126 条拼凑语料：rerank 最高分 0.4210 比该命中的最低分 0.0986 还高，切不开，0.06 勉强用
#   105 条单一体系语料：分数整体上移，重扫 0~1.0 后 0.45~0.60 是平台（整体 94~95%），取 0.50
#
# 为什么取 0.50 而不是扫出来的「最优」0.55（34 条拒答题下）：
#
# 检索口径上 0.55 确实更好看 —— hit@1 93% / 拒答 88% / 整体 92%，
# 而 0.50 是 94% / 79% / 90%。看着该抬到 0.55。但端到端一测就露馅：
#
#   拒答题 34 条，阈值 0.50 → 检索拦 27 条，生成层再兜住 7 条，端到端 100%。
#   把阈值抬到 0.55，那 3 条（肌酸 0.5309 / 减脂蛋白 0.5443 / 史密斯 0.5458）
#   在检索层被拦下来，但它们在 0.50 下生成层本来就会认怂 —— 用户看到的东西
#   一模一样。也就是说 0.55 那 9 个点的拒答率提升，在端到端是零。
#
# 代价却是真的：「哥本哈根支撑」精排分 0.5404（实测有时过 0.55），
# 0.55 会把它误拒。它是真实问题、legs-05 专门讲了，没有第二道闸补救。
#
# 所以：**不要用错误层级的指标优化系统**。检索拒答率只是阈值开关的统计，
# 真正的守门人是生成层。检索阈值的主要价值是省 token（拦掉就不调模型）
# 和降低幻觉概率，不是「正确率」。
#
# 反方向也量过（e2e_hit.py，抽 20 条该命中的）：生成层误拒 1/20 = 5%。
# 幻觉 0/34、误拒 1/20，整体偏保守 —— 健身建议宁可不答，也别给错方案。
#
# 另外分数会抖：哥本哈根离线缓存 0.5404、实时调有时 >0.55；肌酸探针跑两次
# 差 0.001。边界题不可靠，这也是不把阈值贴着某个分数取的原因。
# 换 reranker 模型必须重扫：这是 bge-reranker-v2-m3 的输出分布，不是通用刻度。
# 实测过的分布差异：同一道题 bge 给 0.04、Qwen3-Reranker-4B 给 0.80，
# 阈值原样搬过去等于换了个判定标准，所以做成可配置，跟着模型一起换。
# 4B 在 200 题上的最优区间是 0.95~0.985（见 README 的模型对比表），默认取 0.95：
# 综合集 88% 与 0.985 的 92% 只差 4 点，但 0.985 会把口语集里 5 条本来答对的题拒掉。
RERANK_THRESHOLD = float(os.getenv("RERANK_THRESHOLD", "0.50"))

# ---------- doc2query：让精排也看到「这条资料能回答什么」 ----------
# 病因是诊断出来的，不是猜的（diag_lowscore.py 扫了 95 条命中题）：
#   低于 0.55 的只有 3 条，而且 100% 是「召回捞到了、精排给低分」，召回失败 0 条。
#   最典型的是「练了两个月不涨了怎么办」—— 向量排名第 1（就是 newbie-05），
#   rerank 只给 0.3222。原因很具体：那条 chunk 开头写「这套计划能练多久」，
#   精排看到主题句就判成「在讲计划时长」，而用户问的是进步停滞。
#
# 精排的输入一直只有「话题：正文」，等于要求它当场完成一次
# 「口语问法 -> 书面主题句」的翻译。doc2query 把这个翻译放到离线做：
# 给每条 chunk 生成几个用户问法，拼在正文后面一起送进去。
#
# 向量召回完全不动 —— 召回这条路诊断显示 0 失败，动它只引入风险。
# 代价是精排分数刻度会变，所以阈值必须重扫（见 tune_rerank.py）。
QUERIES_PATH = Path(__file__).parent / "kb_queries.json"
# 默认关闭。做成了一版、量过、没启用：
#   综合集 200 题 hit@1  92.6% -> 88.4%（分数整体下移，边缘正样本被推下去）
#   拒答率 74.3% -> 86.7%，但这是「拒答换命中」：放行的错答生成层本来就会兜住，
#   被拦的真答案没人救。口语集 24 题 +1 条，样本太少不构成结论。
# 真正有效的是改 chunk 的主题句（improve_leads.py）：只改 5 条，
# 四条题从 0.08~0.33 升到 0.54~0.96，而且只影响那几条，不动全局。
# 保留代码和 kb_queries.json，换更强的 reranker 或语料翻几倍后值得重测。
USE_DOC2QUERY = False
_kb_queries = None


def get_kb_queries():
    """{chunk_id: [问法, ...]}。文件不存在就返回空字典，等价于关闭。"""
    global _kb_queries
    if _kb_queries is None:
        try:
            _kb_queries = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))["queries"]
        except Exception:
            _kb_queries = {}
    return _kb_queries


def doc2query_signature():
    """精排缓存指纹。改了问法必须让缓存失效，否则会拿旧分数得出新结论。"""
    if not USE_DOC2QUERY:
        return "off"
    q = get_kb_queries()
    if not q:
        return "off"
    h = hashlib.md5(json.dumps(q, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:8]
    return f"on:{h}"


def rerank_text(item):
    """精排的候选文本。展示给用户的仍然是 item['text']，这里多出来的只喂模型。"""
    base = f"{item['topic']}：{item['text']}"
    if not USE_DOC2QUERY:
        return base
    qs = get_kb_queries().get(item["id"])
    if not qs:
        return base
    return f"{base}\n常见问法：{'；'.join(qs)}"

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


def _status_of(exc: Exception) -> int | None:
    """从异常里挖出 HTTP 状态码。两种来源：openai 的 status_code、urllib 的 code。
    抽成函数是因为重试和熔断都要判断，各写一份迟早不一致。"""
    status = getattr(exc, "status_code", None)
    if status is None and isinstance(exc, urllib.error.HTTPError):
        status = exc.code
    return status


def _should_retry(exc: Exception) -> bool:
    """这个错值不值得再试一次。"""
    # 先看状态码。注意 urllib.error.HTTPError 是 URLError 的子类，
    # 顺序反了的话 401 会被当成「网络问题」重试两次，白等三秒。
    status = _status_of(exc)
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


def call_with_retry(fn, label: str = "调用", stat: dict | None = None):
    """把一次可能失败的外部调用包起来：熔断检查 -> 重试 -> 记录成败。

    两层为什么要套在一起：重试解决「抖一下」，熔断解决「这一段就是挂」。
    上游整体挂掉的时候，3 次尝试等于把请求量乘 3 —— 那是加压，不是止损。
    熔断认了怂之后，请求一个包都不发就失败，上游才有喘息的机会。

    stat 传一个空字典进来，函数会把「实际试了几次」写进去。
    为什么要这个：错误信息里写死「已试 3 次」是骗人的 —— 401 这种错只试 1 次就放弃了，
    报告试了 3 次会让人以为网络真的抖了三次，排障方向直接跑偏。"""
    breaker = get_breaker(label)
    breaker.before_call()       # 熔断中直接抛 CircuitOpenError，不碰网络
    if stat is None:
        stat = {}

    try:
        result = _attempt_loop(fn, label, stat)
    except Exception as e:
        # 只有「上游真的有问题」才计入熔断。
        # 401 / 400 这类是上游明确拒绝了我们的请求 —— 它活着，只是我们错了。
        # 把这种错也算进熔断，会出现最坑的现场：key 填错之后所有请求都报「熔断中」，
        # 掩盖掉真正的原因。
        if _should_retry(e):
            breaker.on_failure(e)   # 把异常传进去，冷却时长按错误类型定
        else:
            breaker.on_client_error()
        raise

    breaker.on_success()
    return result


def _attempt_loop(fn, label: str, stat: dict):
    """光管重试，不管熔断。拆出来是为了让上面的 call_with_retry 只负责编排。"""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        stat["attempts"] = attempt
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


def external_error(exc: Exception, what: str, stat: dict) -> HTTPException:
    """把外部异常翻译成给调用方看的状态码。

    这一步不是形式主义。余额不足（402）如果原样包成 502，排障的人会去查网络、
    去怀疑上游挂了、去加熔断，而真正该做的是充值 —— 方向完全错。
    所以 4xx 一律翻成「我们自己这边的配置问题」，只有真正的上游故障才给 502。"""
    tried = stat.get("attempts", 1)
    status = getattr(exc, "status_code", None)
    if status == 402:
        return HTTPException(500, detail=f"{what}失败：上游账户余额不足（402），重试无用，需要充值")
    if status in (401, 403):
        return HTTPException(500, detail=f"{what}失败：密钥无效或无权限（{status}），重试无用")
    if status == 404:
        return HTTPException(500, detail=f"{what}失败：模型名不存在（404），重试无用")
    if status == 400:
        return HTTPException(500, detail=f"{what}失败：请求被上游拒绝（400）：{exc}")
    return HTTPException(502, detail=f"{what}失败（试了 {tried} 次）：{exc}")


# ---------- 第零段之二：熔断 ----------
# 重试和熔断解决的是两个不同的问题，不能互相替代：
#   重试：这一次可能只是抖了一下，再试也许就好。
#   熔断：这一段时间它就是不行的，别试了，试了也是白试，还把它的负载乘 3。
#
# 三个状态：
#   closed     正常放行，边跑边记录成功/失败
#   open       一律拒绝，直到冷却结束 —— 这段时间一个网络包都不发
#   half_open  冷却结束，只放一个请求去试探。成功就关回去，失败就再开
#
# 判定用两条规则，缺一条都会漏：
#   A. 滑动窗口错误率：最近 30 秒内失败占比超过 50%（且样本 ≥ 10 次）
#      —— 抓「间歇性故障」。30% 的请求失败但从不连续失败，连续计数永远凑不够数，
#         等于永远不熔断，而上游其实已经在带病工作。
#   B. 连续失败 5 次
#      —— 抓「低频调用 + 彻底挂掉」。embedding 这种可能几十秒才调一次，
#         窗口里凑不满 10 个样本，规则 A 永远不触发，只能靠这条兜住。
#
# 为什么规则 A 必须有 min_samples：只有 1 次调用失败时错误率就是 100%，
# 服务一启动第一次请求抖动就直接熔断 30 秒 —— 这是最典型的「小样本误判」。
# 样本不够时宁可不判，交给规则 B。

BREAKER_WINDOW = 30.0         # 滑动窗口长度（秒）
BREAKER_MIN_SAMPLES = 10      # 窗口内样本数少于这个就不做错误率判定
BREAKER_ERROR_RATE = 0.50     # 窗口内失败率超过这个就熔断
BREAKER_CONSECUTIVE = 5       # 连续失败兜底阈值
BREAKER_COOLDOWN = 30.0       # 默认冷却（5xx / 未知故障）


def cooldown_for(exc: Exception | None) -> float:
    """按错误类型给冷却时长。一刀切 30 秒是错的，两个方向都错：

    - 429 限流：上游活着，只是想让你慢一点。等 30 秒是白等，
      而且这 30 秒里本来可以正常处理的请求全被拒了。
    - 连接层彻底断了：等 5 秒就放探针等于自己打自己，需要更长的喘息时间。

    这些数字是经验值，没有压测依据。要定准得知道上游各类故障的实际恢复时间分布，
    现在没这个数据 —— 和阈值一样，认这一点比编一个精确数字好。"""
    status = _status_of(exc)

    if status == 429:
        return 5.0          # 限流：上游健康，短冷却
    if status == 408:
        return 10.0         # 请求超时：可能是瞬时拥塞
    if status == 503:
        return 20.0         # 服务不可用：过载，通常恢复比重启快
    if status is not None and 500 <= status < 600:
        return 30.0         # 502/504：网关或上游真挂了

    if isinstance(exc, (APITimeoutError, TimeoutError)):
        return 10.0
    if isinstance(exc, (APIConnectionError, urllib.error.URLError, ConnectionError)):
        return 15.0         # 连接层：可能只是网络抖，也可能彻底不通

    return BREAKER_COOLDOWN


class CircuitOpenError(Exception):
    """熔断中，请求根本没发出去。
    调用方应该翻译成 503 而不是 502 —— 两者的含义不同：
    502 是「我试过了，失败了」；503 是「我知道它会失败，所以没试」。
    对调用方来说前者是坏消息，后者是「请稍后再来」。"""


class CircuitBreaker:
    """一个上游一个熔断器。用 threading.Lock 是因为 FastAPI 的同步端点跑在线程池里，
    两个请求可能同时改状态 —— 竞态下最典型的 bug 是「探针放出去两个」，
    那等于上游在恢复期被双倍流量再打一次。"""

    def __init__(self, name, consecutive=BREAKER_CONSECUTIVE,
                 cooldown=BREAKER_COOLDOWN, window=BREAKER_WINDOW,
                 min_samples=BREAKER_MIN_SAMPLES, error_rate=BREAKER_ERROR_RATE,
                 clock=time.monotonic):
        self.name = name
        self.consecutive = consecutive
        self.cooldown = cooldown
        self.window = window
        self.min_samples = min_samples
        self.error_rate = error_rate
        self._clock = clock          # 测试里换成假时钟，不用真等 30 秒
        self._lock = threading.Lock()
        self.state = "closed"
        self.failures = 0            # 连续失败次数
        self.opened_at = None
        self.probing = False         # half_open 时，探针是不是已经飞出去了
        self.trips = 0               # 一共熔断过几次，给监控看
        self.last_reason = ""        # 最近一次是因为哪条规则熔断的
        # 滑动窗口：deque 里放 (时间戳, 是否成功)。用 deque 而不是 list 是因为
        # 每次都要从左边弹掉过期样本，list.pop(0) 是 O(n)，长期跑会越拖越慢。
        self.events = deque()

    def before_call(self):
        """放行或拒绝。拒绝时抛 CircuitOpenError，不发任何请求。"""
        with self._lock:
            if self.state == "closed":
                return

            if self.state == "open":
                waited = self._clock() - self.opened_at
                if waited < self.cooldown:
                    raise CircuitOpenError(
                        f"{self.name} 上游熔断中，约 {self.cooldown - waited:.0f} 秒后再试"
                    )
                self.state = "half_open"
                self.probing = True
                print(f"  [breaker:{self.name}] 冷却结束，放一个探针", flush=True)
                return

            # half_open：探针在飞的时候别的请求一律挡住。
            # 否则「只放一个」形同虚设，恢复期的上游会被正常流量再打挂一次。
            if self.probing:
                raise CircuitOpenError(f"{self.name} 正在探活，暂不接新请求")
            self.probing = True

    # ---------- 窗口维护 ----------

    def _prune(self, now):
        """弹掉窗口外的旧样本。不弹的话「最近 30 秒失败率」会变成「历史上失败率」，
        跑一整天之后永远是坏的。"""
        cutoff = now - self.window
        while self.events and self.events[0][0] < cutoff:
            self.events.popleft()

    def _record(self, ok, now):
        self.events.append((now, ok))
        self._prune(now)

    def _stats(self, now):
        self._prune(now)
        total = len(self.events)
        fails = sum(1 for _, ok in self.events if not ok)
        return total, fails

    # ---------- 结果回调 ----------

    def on_success(self):
        with self._lock:
            now = self._clock()
            if self.state != "closed":
                print(f"  [breaker:{self.name}] 探针成功，恢复正常", flush=True)
                # 恢复时必须清空窗口。不清的话，刚恢复的第一次请求会立刻
                # 被窗口里那些历史失败再判一次死刑 —— 刚救回来又打回去。
                self.events.clear()
                self.state = "closed"
                self.failures = 0
                self.opened_at = None
                self.probing = False
                return
            self.failures = 0
            self._record(True, now)

    def on_client_error(self):
        """上游回了 4xx：它活着，只是我们请求写错了。不计故障，
        但必须放掉探针，否则 half_open 会永远卡在「探针在飞」，后面全被拒。

        记为「成功样本」而不是「不记录」：4xx 证明上游在工作，
        让它按健康样本稀释错误率才是对的。否则一串 400 会把错误率推高，
        最后因为「我们自己的请求写错了」把上游熔断掉。"""
        with self._lock:
            self.probing = False
            if self.state == "half_open":
                self.state = "closed"
                self.failures = 0
                self.opened_at = None
                self.events.clear()
            self._record(True, self._clock())

    def on_failure(self, exc=None):
        """exc 用来决定冷却多长。不传就按默认 30 秒。"""
        with self._lock:
            self.probing = False
            now = self._clock()
            self._record(False, now)

            if self.state == "half_open":
                # 探针也失败了：说明还没好，立刻重新熔断，冷却从此刻重新计时
                self.failures += 1
                self._trip(exc, f"探针失败（{type(exc).__name__ if exc else '未知'}）")
                return

            self.failures += 1
            total, fails = self._stats(now)
            rate = fails / total if total else 0.0

            if total >= self.min_samples and rate >= self.error_rate:
                self._trip(exc, f"窗口 {total} 次里失败 {fails} 次（{rate:.0%}）")
            elif self.failures >= self.consecutive:
                self._trip(exc, f"连续失败 {self.failures} 次")

    def _trip(self, exc, reason):
        self.state = "open"
        self.opened_at = self._clock()
        self.trips += 1
        self.last_reason = reason
        # 冷却按错误类型定。注意这里是「本次」的冷却，会写进 self，
        # 因为 before_call 判断剩余时间要用同一个值。
        self.cooldown = cooldown_for(exc) if exc is not None else BREAKER_COOLDOWN
        print(f"  [breaker:{self.name}] {reason}，熔断 {self.cooldown:.0f} 秒", flush=True)

    def snapshot(self):
        with self._lock:
            total, fails = self._stats(self._clock())
            return {
                "state": self.state,
                "consecutive_failures": self.failures,
                "window_samples": total,
                "window_failures": fails,
                "window_error_rate": round(fails / total, 2) if total else 0.0,
                "last_trip_reason": self.last_reason,
                "cooldown_seconds": self.cooldown,
                "trips": self.trips,
                "cooldown_left": (
                    round(max(0.0, self.cooldown - (self._clock() - self.opened_at)), 1)
                    if self.state == "open" else 0.0
                ),
            }


_breakers = {}


def get_breaker(label: str) -> CircuitBreaker:
    """一个调用方一个熔断器，不共用一个全局的。
    理由：embedding 挂了不代表 chat 挂了（不同模型、不同配额），
    用一个全局熔断器会让一个接口的故障把另外两个也拖下水。"""
    if label not in _breakers:
        _breakers[label] = CircuitBreaker(label)
    return _breakers[label]


# 三个已知上游先登记好，这样 /breaker 从服务启动第一秒就有内容可看。
# 不预登记的话，得等某个接口被调用过一次它才冒出来 ——
# 排障时打开一看是空的，会误以为熔断器根本没生效。
for _label in ("chat", "embedding", "rerank"):
    get_breaker(_label)


STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.is_dir():
    # 演示页用原生 JS 单文件，不引外部 CDN —— 容器里没网也能打开。
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def read_root():
    """根路径直接给演示页。没有 static 目录就退回最简 JSON，
    不让一个缺失的前端把整个服务变成 500。

    Cache-Control: no-cache 并不是「不缓存」，是「用之前必须回源校验」。
    配合 FileResponse 自带的 ETag，没改版时是 304（一个来回，几十字节），
    改版后立刻拿到新的。不加的话浏览器走启发式缓存，改完前端用户可能
    还开着几天前的页面 —— 已经踩过一次：新加的入口在自己电脑上不显示。
    """
    page = STATIC_DIR / "index.html"
    if page.is_file():
        return FileResponse(page, headers={"Cache-Control": "no-cache"})
    return {"message": "hello", "demo": "static/index.html 不存在"}


@app.get("/meta")
def meta():
    """演示页启动时读这个，把「这套系统当前是什么配置」显示出来。

    为什么要有：排障时第一句总是「你跑的到底是哪个模型、哪个阈值」。
    这些值都来自环境变量，只写在文档里迟早和实际跑的不一致。
    """
    return {
        "kb_chunks": len(KB),
        "kb_topics": len({d["topic"] for d in KB}),
        "llm_model": LLM_MODEL,
        "embed_model": EMBED_MODEL,
        "rerank_model": RERANK_MODEL,
        "rerank_threshold": RERANK_THRESHOLD,
        "recall_k": RECALL_K,
        "rerank_top_k": RERANK_TOP_K,
        "fallback_enabled": FALLBACK_ENABLED,
        "uncertain_band": UNCERTAIN_BAND,
        "auth_required": bool(ACCESS_CODE),
        # 多轮：前端据此决定要不要显示「新对话」按钮和会话号
        "multiturn": CONDENSE_MODE != "off",
        "condense_mode": CONDENSE_MODE,
        "session_ttl_s": SESSION_TTL,
        "session_max_turns": SESSION_TURNS,
        # 问法日志：开着才记录真实提问。/queries 能看，refused=1 是选题清单
        "query_log": QUERY_LOG_ON,
        # 复合拆解：默认 off。开着也只对「两个以上分句」的句子生效
        "decomp_mode": DECOMP_MODE,
        "decomp_max_subs": DECOMP_MAX_SUBS,
        "multi_top_k": MULTI_TOP_K,
        # 排计划的动作库规模。页面上显示「33 个动作」，跟实际数据对得上，
        # 不靠文案里写死一个数 —— 加动作之后忘了改文案是最常见的过期信息。
        "exercises": len(PLAN_LIB.items) if PLAN_LIB else 0,
        "plan_max_days": plan_agent.MAX_DAYS,
        "plan_max_steps": plan_agent.MAX_STEPS,
    }


@app.get("/breaker")
def breaker_state():
    """熔断器的当前状态。线上排障第一件事就是看这个：
    上游明明通了但接口一直报 503，多半是熔断没恢复；
    反过来「为什么它不熔断」也能从这里看出失败次数涨没涨。
    不给这个口子，熔断就是一个黑盒 —— 出问题只能猜。"""
    return {name: b.snapshot() for name, b in _breakers.items()}


class AskRequest(BaseModel):
    """问答请求。

    question 必须限长。不设限是实测过的坑：3000 个重复的「深蹲」照样走完
    向量化 + 精排 + 生成，4.5 秒、钱照花，而且精排给了 0.9931 的高分——
    重复词把相似度堆上去了。500 字是真实口语问法（最长 40 字上下）的十倍，
    再长就不是提问了。

    session_id 也要限：它是会话表的 key，不设限就能拿 10 万个字符当 key 塞进去。
    """
    question: str = Field(min_length=1, max_length=500)
    # 多轮会话号。不传 = 单轮，行为跟以前完全一致（这是兼容性的底线：
    # 老的调用方、老的评测脚本一个字都不用改）。
    session_id: str | None = Field(default=None, max_length=64)
    # 曾经有个 top_k 字段，但 /rag 从来没读过它 —— 调用方传 999 也毫无作用。
    # 参数不生效比没有参数更坏：它会让人以为自己在控制返回条数。已删。


# 知识库从 kb.json 读，不再写死在代码里。
# 每条是一个 chunk：id 唯一、topic 是所属话题（评测按话题判命中）、text 是正文、source 是出处。
# 为什么拆出去：40 条正文塞进 .py 会让代码没法看，而且改内容不该动代码。
# 排计划用的动作库。跟 kb.json 分开：一份是散文（问答用），一份是结构化实体（排计划用），
# 两边靠 exercises.json 里的 ref 字段对齐，check_static.py 每次都会验这个对齐。
try:
    PLAN_LIB = plan_agent.Library.load()
except Exception as _e:      # noqa: BLE001 —— 动作库坏了不该让整个问答服务起不来
    PLAN_LIB = None
    print(f"[plan] 动作库加载失败，/plan 会返回 503：{_e}")

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

    stat = {}
    try:
        resp = call_with_retry(
            lambda: get_client(EMBED_TIMEOUT).embeddings.create(model=EMBED_MODEL, input=texts),
            label="embedding",
            stat=stat,
        )
    except CircuitOpenError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise external_error(e, "调用 embedding 模型", stat)
    u = getattr(resp, "usage", None)
    add_cost(embed_tokens=getattr(u, "prompt_tokens", 0) or 0, embed_calls=1)
    return [d.embedding for d in resp.data]


# 知识库向量只在内存里存一份，第一次用到时才算
_kb_vectors = None


_kb_matrix = None


def kb_matrix():
    """把全库向量整理成「一行一条、已 L2 归一化」的矩阵，只建一次。

    为什么单独建矩阵：余弦 = 归一化后的点积，所以一次矩阵乘就能拿到全库分数。
    原来的写法是 `for d, v in zip(KB, vectors): cosine(q, v)`，1024 维下每条要
    跑一遍 Python 循环 —— bench_scale.py 实测 119 条 57ms、1 万条 4.0 秒、
    10 万条 43.7 秒；换成矩阵乘是 0.1ms / 5.7ms / 66.8ms，快 600~800 倍。
    这个差值在 119 条时被 LLM 的 3 秒盖住了，看起来「没问题」，但它是唯一
    一个会随数据量线性爆炸的环节。"""
    global _kb_matrix
    if _kb_matrix is not None:
        return _kb_matrix
    if np is None:
        _kb_matrix = get_kb_vectors()
        return _kb_matrix
    mat = np.asarray(get_kb_vectors(), dtype=np.float32)
    if mat.ndim != 2 or mat.shape[0] == 0:
        _kb_matrix = mat
        return mat
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0        # 零向量不该让整库变 NaN
    _kb_matrix = mat / norms
    return _kb_matrix


def all_vec_scores(q_vec):
    """一条查询向量 → 全库余弦分，顺序与 KB 一致。
    numpy 没装时自动退回逐条 cosine（跟改造前一样慢，但不会挂）。"""
    if np is None:
        return [cosine(q_vec, v) for v in get_kb_vectors()]
    q = np.asarray(q_vec, dtype=np.float32)
    n = float(np.linalg.norm(q))
    if n:
        q = q / n
    return kb_matrix() @ q


def all_vec_scores_multi(q_vecs):
    """多个查询向量 → 分数矩阵，mat[i][j] = 第 i 条资料对第 j 个查询的余弦。
    改写检索要「4 个变体逐条比、取最大」，一次矩阵乘比 4 次单查更省。"""
    if np is None:
        return [[cosine(v, kb) for v in q_vecs] for kb in get_kb_vectors()]
    q = np.asarray(q_vecs, dtype=np.float32)
    norms = np.linalg.norm(q, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return kb_matrix() @ (q / norms).T


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
    try:
        CACHE_PATH.write_text(
            json.dumps({"model": EMBED_MODEL, "texts": texts, "vectors": vectors}, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError:
        pass        # 只读文件系统（容器里很常见）不该让检索挂掉，下次重启重算一遍而已
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


# 离门线多近算「不确定」。0.006 不是拍的：diag_stability.py 在 4B 上量过，
# 224 题里只有 2 条落在 0.006 以内，其中 1 条（髌骨软化 0.9463~0.9512）
# 连跑 5 次翻了 2 次面 —— 也就是说这个带子里的判定本身不可复现。
# 带宽取窄了抓不到翻面题，取宽了会把确定的答案也标成不确定（0.05 以内有 40 多条，
# 标了等于没标）。0.006 是「翻面实测区间」的实测宽度，不是调参调出来的。
UNCERTAIN_BAND = float(os.getenv("UNCERTAIN_BAND", "0.006"))


def near_gate(ranking, threshold, band: float | None = None) -> bool:
    """top1 的分跟门线贴得够近就返回 True —— 这个判定随时可能翻面。

    注意它**不代表答案错**：贴线过关的那条多半是对的，贴线没过的那条多半也没错，
    只是「这次刚好站在线这边」。所以用途是给调用方一个信号（前端标黄、
    离线统计、或者将来做人工复核队列），而不是自己再改判定 ——
    改判定的位置只有一个，就是 pick_hits 里那句 score < threshold。
    """
    if not ranking:
        return False                # 压根没召回到东西，不是「边界」是「没有」
    b = UNCERTAIN_BAND if band is None else band
    return abs(ranking[0]["score"] - threshold) < b


def public_sources(items):
    """对外暴露的资料字段。**带上 text**：只给话题名和分数的话，
    用户没法核对答案是不是真从资料里来的 —— 引用要能点开看原文才有意义。

    统一走这个函数而不是在三个 return 里各写一遍：之前 sources 在两处被
    分别裁剪，加字段时漏掉一处，前端就静默显示空白（2026-11-23 踩过）。
    """
    return [
        {
            "id": c["id"],
            "topic": c["topic"],
            "source": c.get("source", ""),
            "score": c.get("score"),
            "text": c.get("text", ""),
        }
        for c in items
    ]


def search_kb_vector(q: str, top_k: int = VEC_TOP_K, threshold: float = VEC_THRESHOLD):
    """向量检索：把问题也变成向量，跟每条资料算相似度，取最像的几条。
    返回 (命中的资料, 完整排名)"""
    q_vec = embed([q])[0]
    scores = all_vec_scores(q_vec)
    ranking = [
        {"id": d["id"], "topic": d["topic"], "text": d["text"], "source": d["source"],
         "score": round(float(s), 4)}
        for d, s in zip(KB, scores)
    ]
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

    stat = {}
    try:
        data = call_with_retry(_post, label="rerank", stat=stat)
    except CircuitOpenError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise external_error(e, "调用 reranker", stat)

    # reranker 按输入 token 计费，账单在 meta.billed_units 里（不是 usage 字段，
    # 跟 chat 不一样，所以单独读）。
    units = (data.get("meta") or {}).get("billed_units") or {}
    add_cost(rerank_tokens=units.get("input_tokens") or 0, rerank_calls=1)

    return [
        {"index": r["index"], "score": round(r["relevance_score"], 4)}
        for r in data["results"]
    ]


def rank_one(q: str, q_vec=None):
    """一条查询走完「全库向量召回 12 条 -> 精排 -> 排序」，返回完整 ranking。

    单独抽出来是因为口语改写要连跑 3 条。如果让两条路各写一遍召回和精排，
    改了一处忘了另一处，分数口径就会悄悄分叉 —— 这个项目已经因为
    「两份逻辑看起来一样」吃过一次亏（eval.py 覆盖写）。
    """
    if q_vec is None:
        q_vec = embed([q])[0]
    scores = all_vec_scores(q_vec)
    recalled = [
        {"id": d["id"], "topic": d["topic"], "text": d["text"], "source": d["source"],
         "vec_score": round(float(s), 4)}
        for d, s in zip(KB, scores)
    ]
    recalled.sort(key=lambda x: x["vec_score"], reverse=True)
    recalled = recalled[:RECALL_K]      # 召回阶段故意放宽，准确率交给下一段

    scores = rerank(q, [rerank_text(r) for r in recalled])
    ranking = []
    for s in scores:
        item = dict(recalled[s["index"]])
        item["score"] = s["score"]       # 统一用 score 字段，方便前端只认一个
        ranking.append(item)
    ranking.sort(key=lambda x: x["score"], reverse=True)
    return ranking


def search_kb_rerank(q: str, top_k: int = RERANK_TOP_K, threshold: float = RERANK_THRESHOLD):
    """两段式检索：向量宽松召回 -> reranker 精排 -> 按 rerank 分数决定答不答。
    返回 (命中的资料, 完整排名)，排名里同时留着两套分数方便对比。"""
    ranking = rank_one(q)
    return pick_hits(ranking, top_k, threshold), ranking


# ---------- 第三点五段：口语改写后再精排 ----------
# 证据链在三个脚本里，别凭印象：
#   diag_selfquery.py   420 个「这条资料该回答的问题」反向考语料，57 条 chunk 接不住自己
#                       第一版把 18% 判成「未召回」，其实是同话题兄弟抢走向量排名 —— 已改成双口径
#   ab_phrasing.py      语料一个字不动，只把问法换成书面：均分 0.155 -> 0.549，过门槛 0/20 -> 13/20
#   ab_rewrite_rerank.py 改写 + 精排（20 条最难）：过门槛 0/20 -> 11/20，top1 落对话题 8 -> 14
#
# 抽检 6 条时最能说明问题：问「为什么每个人合适的重量都不一样」，正文最后一句就是答案，
# reranker 给 0.031。所以不是「没有内容」，是它不认口语疑问句。
#
# 那为什么之前的 rewrite 模式只涨 1%？两个原因，都不是「改写没用」：
#   1. 它走纯向量，精排根本没接上，判别力丢了
#   2. 综合集 200 题里书面题已到 94.7%，天花板效应，没空间可涨
#
# 但改写不是万能药，更不能全量放开：
#   20 条最难问法里，有 10 条的改写把 top1 带到了**错的话题**
#   （「每次吸气要吸到多满」被改成「下拉时的呼吸」，4 次里 2 票跑高位下拉）。
#   把这些改写铺到所有请求上，等于给本来答对的 94.7% 引入漂移风险。
#
# 所以只在**原问法已经拒答**时才启动改写，风险只落在「本来什么都拿不到」的区间：
#   原问法过门槛 -> 直接答，一次额外调用都不发，行为完全不变
#   原问法拒答   -> 改写 3 条、各精排一次，谁分高用谁
#
# 实测收益（20 条最难口语问法，「安全命中」= 过门槛且话题对）：0/20 -> 8/20。
#
# 但这个「+8」是在一批**全部该答**的题上量的。加上负例之后结论翻了：
# 135 条口语集（110 该答 / 25 该拒）上，删掉负例就看不见问题 ——
# 被拒答的题里，有相当一部分是**本来就不该答**的（康复、装备、补剂、出界问题），
# 它们同样会进改写分支，改写完照样变成一句「很像健身问题」的话，然后被放行。
# diag_oral_gate.py 扫了 0.30~0.985 十个门槛，没有一个能救：
#   门槛 0.50（原默认）：命中 94.5%（+2），拒答 32%（-68 点），整体 83.0%
#   门槛 0.95（与基座同门线）：命中 94.5%（+2），拒答 60%，整体 88.1%
#   门槛 0.985：命中 93.6%（+1），拒答 68%，整体 88.9%
#   基座（不用改写）：命中 92.7%，拒答 100%，整体 94.1%
# 也就是「用 2 条命中换掉 17 条正确拒答」。对一个「宁可说资料不足」的产品，
# 这个交换是负的。所以 oral 不再作为可选演示模式的推荐路径，门槛也对齐基座 ——
# 留 0.50 等于留一个「一开就把拒答率打到三分之一」的开关。
ORAL_N = 3
ORAL_THRESHOLD = RERANK_THRESHOLD


def search_kb_rerank_oral(q: str, top_k: int = RERANK_TOP_K, threshold: float = ORAL_THRESHOLD):
    """先走正常精排；只有它拒答了，才用口语改写补一刀。

    改写复用 rewrite_query 那套 prompt —— 实测它比专门写的「书面化」prompt 更好
    （安全命中 8/20 vs 7/20），因为专门那版容易把问题里的具体对象写丢。

    返回的第三项 variants 只用来告诉调用方「这次改写成了什么」，没改写就是空列表。

    原问法这一刀**必须**用 RERANK_THRESHOLD，不能跟着 threshold 参数走（它历史上等于
    0.50，那是 bge 时代的刻度；现在已对齐到 0.95，但传参这条路本来就该堵上）。这里踩过一次：默认参数直接透传进
    search_kb_rerank，线上原问法的门线悄悄变成 0.50，而 eval.py 里基座用的是 0.95
    —— 线上和评测测的不是同一件事，且 0.50 正好是 4B 判「像但不是答案」那片区域，
    等于把一批半相关 chunk 全放进来。threshold 现在只作用于改写出来的候选。
    """
    hits, ranking = search_kb_rerank(q, top_k, RERANK_THRESHOLD)
    if hits:
        return hits, ranking, []        # 原问法就答了，不额外花钱、不改行为

    variants = rewrite_query(q)[:ORAL_N]
    if not variants:
        return hits, ranking, []        # 改写失败：维持原判，不因为锦上添花挂掉检索

    vecs = embed(variants)              # 3 条一起嵌入，省 2 次往返
    best_rank = None
    for v, vv in zip(variants, vecs):
        r = rank_one(v, vv)
        if best_rank is None or r[0]["score"] > best_rank[0]["score"]:
            best_rank = r
    if best_rank[0]["score"] <= ranking[0]["score"]:
        return hits, ranking, variants  # 改写没比原问法更强，维持拒答
    return pick_hits(best_rank, top_k, threshold), best_rank, variants


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
    for d, vec_score, bm in zip(KB, all_vec_scores(q_vec), bm_scores):
        vec_score = float(vec_score)
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
    mat = all_vec_scores_multi(vecs)
    ranking = []
    for i, d in enumerate(KB):
        sims = [float(x) for x in mat[i]]
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
def kb(q: str, request: Request):
    """关键词检索，只返回命中的第一条。演示页已不用它，留着是给评测脚本做对照。

    要口令：它等于把知识库挂成一个开放搜索框。内容本身是公开发过的文章，
    但一个对外服务该关的门要一起关，否则审计时会问「为什么只有这个没锁」。
    """
    check_code(request.headers.get("x-access-code"))
    hits = search_kb_keyword(q)
    if not hits:
        return {"matched": None, "answer": "知识库里没有相关内容"}
    return {"matched": hits[0]["topic"], "answer": hits[0]["text"]}


def call_llm(messages, temperature: float = 0.0, tools=None):
    """把「调用模型」这件事收成一个函数，多处共用。

    temperature 默认 0：同一道题必须答一样的，否则评测不可复现。
    这不是理论洁癖 —— 兜底判断里「每次吸气要吸到多满」第一次答、第二次拒，
    同一份资料同一个 prompt，结论却翻面。不可复现的系统，任何 A/B 结论都站不住。

    思维链由 GEN_THINKING 控制（None = 不传，用供应商默认；False = 显式关掉）。
    为什么要这个开关：实测输出里思维链占 75%，而输出单价是输入的 4 倍，
    等于一半的钱花在「想」上。关不关是质量与成本的取舍，得先量再定 ——
    见 ab_thinking.py。

    tools 只在 /plan 用（多步工具调用）。传了就不加 tool_choice 以外的任何东西：
    「自动决定要不要调工具」这件事必须留给模型，写死 required 就等于替它做了决定。
    """
    if not LLM_API_KEY or LLM_API_KEY.startswith("sk-替换"):
        raise HTTPException(status_code=500, detail=".env 里的 LLM_API_KEY 还没填")
    STATS["llm_calls"] += 1         # 只在下游真被调用时加，缓存命中不会走到这里

    extra = {} if GEN_THINKING is None else {"extra_body": {"enable_thinking": bool(GEN_THINKING)}}
    if tools:
        extra["tools"] = tools
    stat = {}
    try:
        return call_with_retry(
            lambda: get_client(LLM_TIMEOUT).chat.completions.create(
                model=LLM_MODEL, messages=messages, temperature=temperature, **extra
            ),
            label="chat",
            stat=stat,
        )
    except CircuitOpenError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise external_error(e, "调用模型", stat)


# ---------- 生成层兜底：检索拒答之后的第二道闸 ----------
# 病因不是模型不行，是代码把话堵死了：
#   if not hits: return "知识库里没有相关资料"
# 这句话在「语料里真的没有」时是对的，在「语料里有、但精排分数不够」时就是错的。
# 实测后者真实存在：「每次吸气要吸到多满」语料里有答案，4B 已经给到 0.802，
# 仍过不了 0.95 —— 用户拿到的是「没有」，而库里明明写着。
#
# 为什么不让检索层把阈值降下来：降了就成片漏放该拒的题（拒答率从 81% 掉到 70% 以下）。
# 一根线不能同时表达「能答就答」和「不懂别装」，所以拆成两层：
#   第一层精排分数说了算（严格，宁拒不答）
#   第二层只在第一层拒了时启动，把 top-N 弱相关资料交给生成层判断能不能答。
#
# 为什么不单独调一次「能不能答」再调一次「答」：慢一倍、贵一倍，而且两次调用会
# 给出互相矛盾的结论。用哨兵字符串，一次调用同时完成判断和作答。
# 为什么让它在 200 题上只赚 1 个点还默认开：赚的这 1 点不是重点。
# 重点是它把「检索层拒答」从终审降级成初筛 —— 后者本来就没资格终审，
# 它是按分数开关的，永远分不清「语料没有」和「语料有但问法怪」。
# 实测证据：20 条最难的极低分问法里，阈值拒掉 5 条，兜底救回 3 条且全部答对。
# 综合集只赚 1 点，是因为那 95 条命中题本来就被 4B 救得差不多了（假拒答只剩 2 条）。
# 真实用户问得比评测集口语，所以按评测集估的收益是下限，不是上限。
# 代价说清楚：每条被拒的题多一次生成调用（拒答率 19%），多 1~2 秒。
FALLBACK_ENABLED = True
FALLBACK_TOP_N = 2              # 给生成层几段资料：2 段够判断，多了反而能拼出答案
FALLBACK_FLOOR = 0.0            # top1 低于这个分就认定完全无关，连模型都不调（0 = 不设限）
FALLBACK_SENTINEL = "资料不足"
# prompt 的措辞是试出来的，不是写的。放宽成「有多少说多少」那一版更差：
# 肩袖损伤、半月板康复这类明确该拒的题都被硬答了（6 条测试里 5 条放行），
# 严格版是 3 答 3 拒、全对。所以「宁可说资料不足」这句不能去掉。
FALLBACK_PROMPT = (
    "你是一名健身教练。下面给你几段资料和一个问题。规则：\n"
    "1. 如果资料足以回答问题，直接给出简洁具体的答案，给数字，不说套话。\n"
    f"2. 如果资料不足以回答（问的东西资料里根本没提，或者只沾点边但答不了），"
    f"只输出这四个字：{FALLBACK_SENTINEL}。\n"
    "3. 不要补充资料以外的知识，不要解释你为什么答不了。\n"
    "宁可说资料不足，也不要猜。"
)


def answer_with_fallback(q: str, ranking, top_n: int = FALLBACK_TOP_N):
    """把弱相关资料交给生成层，让它判断「这些够不够回答」。

    返回 None 表示模型也认为答不了（应当拒答）；
    返回 dict 表示模型给出了答案。
    """
    if not ranking:
        return None
    if ranking[0]["score"] < FALLBACK_FLOOR:
        return None             # 完全无关，没必要花一次调用去确认

    ctx = ranking[:top_n]
    context = "\n".join(f"- {c['topic']}：{c['text']}" for c in ctx)
    try:
        resp = call_llm([
            {"role": "system", "content": FALLBACK_PROMPT},
            {"role": "user", "content": f"【资料】\n{context}\n\n【问题】\n{q}"},
        ])
    except HTTPException:
        return None             # 兜底本身失败就退回拒答，不能因为它把主链路带崩

    bill_chat(resp)
    text = resp.choices[0].message.content.strip()
    # 只认开头。模型有时会写「资料不足，因为……」或「资料不足。建议……」，
    # 用 in 全文判断会被「资料里没有提到 XX，但你要注意 YY」这种半答半拒蒙骗。
    head = text[:len(FALLBACK_SENTINEL) + 2]
    if FALLBACK_SENTINEL in head:
        return None
    return {"answer": text, "sources": ctx, "tokens": resp.usage.total_tokens}


# ---------- 第六段：上线三件套（缓存 / 限流 / 口令）----------
# 挂公网之前必须有的三层，缺任何一层都会被一个链接搞崩：
#   缓存   demo 的问题高度集中（页面那几个示例点来点去），一次回答重复计费很蠢；
#           更重要的是它把「被刷」的边际成本压到 0 —— 命中缓存不打上游。
#   限流   缓存挡不住「每次都换个问法」，所以还要有每分钟/每天的硬上限。
#   口令   链接会被转发。只靠限流，陌生人也能把日额度吃光；口令把「谁能用」收窄。
#
# 三层的顺序有讲究：**先查口令，再查缓存，最后才限流**。
# 缓存命中不消耗配额 —— 否则页面上的示例点两次就被 429，体验最差的地方
# 恰好是最便宜的地方。

ACCESS_CODE = os.getenv("ACCESS_CODE", "").strip()
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "12"))
DAILY_QUOTA = int(os.getenv("DAILY_QUOTA", "300"))
CACHE_MAX = int(os.getenv("CACHE_MAX", "500"))

# 多 worker 的状态共享。默认 memory，走下面那套原有代码，行为一个字节都没变；
# 只有显式 STATE_BACKEND=sqlite 才切到共享存储。
# 为什么开多 worker 必须切：缓存、限流、日额度都是进程内变量，4 个 worker
# 就是 4 份缓存、每人每分钟 48 次、日额度 1200 —— 限流失效比变快严重得多。
STATE_BACKEND = os.getenv("STATE_BACKEND", "memory").strip().lower()
STORE = None
if STATE_BACKEND == "sqlite":
    import shared_state

    STORE = shared_state.SqliteStore(os.getenv("STATE_DB", ".state.db"), CACHE_MAX)

_cache: "OrderedDict[str, dict]" = OrderedDict()      # 键 -> 完整返回体
_cache_lock = threading.Lock()
_rate: dict[str, deque] = {}                          # ip -> 请求时间戳
_rate_lock = threading.Lock()
_day = {"date": "", "used": 0}
STATS = {"requests": 0, "cache_hits": 0, "rate_limited": 0, "llm_calls": 0,
         "condense_calls": 0, "condense_failed": 0,
         "decomp_calls": 0, "decomp_failed": 0, "decomp_paths": 0,
         "plans": 0, "plan_tool_calls": 0, "plan_review": 0,
         # 问法日志改成异步写之后，这三个数是它健康与否的唯一证据：
         # 写成功多少、因队列满丢了多少、落盘失败了多少。
         "log_written": 0, "log_dropped": 0, "log_write_failed": 0,
         "started": time.time()}


def client_ip(request: Request) -> str:
    """取真实来源 IP。

    上线后请求会先到一个反代，`request.client.host` 永远是反代自己，按它限流
    等于全站共用一个桶 —— 一个人刷完，所有人都用不了。所以优先读 X-Forwarded-For
    的第一跳。这个头客户端可以伪造，但配合「每天总配额」已经够用：
    伪造只能绕过「每人每分钟」，绕不过「全站每天」。

    只取第一段（最靠近客户端那个），因为后面几段是各级代理自己追加的。
    """
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def check_code(code: str | None) -> None:
    """口令校验。没配 ACCESS_CODE 就放行 —— 本地开发和离线测试都靠这条。"""
    if not ACCESS_CODE:
        return
    if (code or "").strip() != ACCESS_CODE:
        raise HTTPException(status_code=401, detail="访问口令不正确")


def cache_key(mode: str, fallback: bool, question: str) -> str:
    """键必须带上 mode 和 fallback：同一句话在 rerank 和 vector 下答案不同，
    带不带兜底也不同。漏掉任何一个，用户就会拿到「另一个模式」的答案。"""
    return f"{mode}|{int(bool(fallback))}|{question.strip()}"


def cache_get(key: str) -> dict | None:
    if STORE is not None:
        body = STORE.cache_get(key)
        if body is not None:
            STATS["cache_hits"] += 1
        return body
    with _cache_lock:
        body = _cache.get(key)
        if body is None:
            return None
        _cache.move_to_end(key)       # LRU：刚被用过的挪到队尾
        STATS["cache_hits"] += 1
        return dict(body)             # 浅拷一层，避免调用方改到缓存里的对象


def cache_put(key: str, body: dict) -> None:
    if STORE is not None:
        STORE.cache_put(key, dict(body))
        return
    with _cache_lock:
        _cache[key] = dict(body)
        _cache.move_to_end(key)
        while len(_cache) > CACHE_MAX:
            _cache.popitem(last=False)


def rate_check(ip: str) -> float:
    """滑动窗口，返回 0 表示放行，返回正数表示要等几秒。

    用滑动窗口而不是固定窗口：固定窗口在整分钟边界上会放过 2 倍流量
    （59 秒打满 + 01 秒再打满），对「防刷」这种场景刚好是最坏的形状。
    """
    if STORE is not None:
        wait = STORE.rate_wait(ip, RATE_LIMIT_PER_MIN)
        if wait > 0:
            STATS["rate_limited"] += 1
        return wait
    now = time.monotonic()
    with _rate_lock:
        q = _rate.setdefault(ip, deque())
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= RATE_LIMIT_PER_MIN:
            STATS["rate_limited"] += 1
            return 60 - (now - q[0])
        q.append(now)

        # 顺手回收长期不活跃的 ip，否则这个字典只增不减 —— 被扫一遍就吃掉几 MB
        if len(_rate) > 2000:
            stale = [k for k, v in _rate.items() if not v or now - v[-1] > 300]
            for k in stale:
                _rate.pop(k, None)
        return 0.0


def daily_check() -> bool:
    """全站每天的总配额。这是最后一道闸：前面两层都能被绕过，这个绕不过。

    sqlite 后端下用一条 UPSERT + RETURNING，不会出现「两个 worker 同时
    读到 299 都放行」的竞态；memory 后端下只有一个进程，锁就够。
    """
    if STORE is not None:
        return STORE.daily_take(DAILY_QUOTA)
    today = time.strftime("%Y-%m-%d")
    with _rate_lock:
        if _day["date"] != today:
            _day["date"], _day["used"] = today, 0
        if _day["used"] >= DAILY_QUOTA:
            return False
        _day["used"] += 1
        return True


def mark_session(body: dict, ctx: dict | None) -> dict:
    """把「这一轮属于哪个会话、被理解成了什么、时间花在哪」写进返回体，并记进会话。

    缓存命中也得走这里：缓存里存的是**上一次请求的会话号**，不盖一遍，
    客户端就会从自己的会话跳到别人的会话上 —— 这种 bug 只在两个人同时
    问同一句话时出现，很难测到，但一旦出现就是「聊着聊着换了个人」。

    会话记录里**拒答和降级也留**：「那上斜呢」跟着的可能是一句「资料不足」，
    但这句话里的「上斜」仍然是下一轮补全要用的上下文。
    """
    if not ctx:
        return body
    body["session_id"] = ctx["sid"] or None
    body["resolved_question"] = ctx["resolved"]
    body["condensed"] = bool(ctx["condensed"])
    body["history_turns"] = ctx["hist"]
    # 拆出来的子问题：这一轮是几路检索的、拆成了什么，直接给调用方看。
    # 缓存命中时 ctx 里还没有 subs（拆解发生在算缓存键之后），
    # 就保留缓存体里那份 —— 用 setdefault 而不是覆盖，避免把「2 路」写成「0 路」。
    if ctx.get("subs"):
        body["sub_questions"] = list(ctx["subs"])
    else:
        body.setdefault("sub_questions", [])
    if ctx.get("tim"):
        # 分阶段耗时：慢的时候能自己说明慢在哪，不用回头猜。
        # 多轮引入了一段「补全」，它跟检索、生成不是一回事，混在一起看不出来。
        body["timings"] = {k: int(v) for k, v in ctx["tim"].items()}
    if ctx["sid"]:
        srcs = body.get("sources") or []
        session_append(ctx["sid"], ctx["raw"], ctx["resolved"],
                       srcs[0]["topic"] if srcs else None)
    return body


# ---------- 问法日志：让真人来出题 ----------
# 评测集的 86 条口语题是**对着知识库正文反推**出来的，那部分 hit@1 有 98% ——
# 用词必然重叠，这个数是自己骗自己。唯一能修的办法是让真人来问，
# 而不是继续自己写题。所以这里只做一件事：每来一个真实问题就落一行。
#
# 三个刻意的选择：
#   1. **不记 IP、不记 User-Agent**。这份日志会被拿来给人看，能少收就少收；
#      去重靠 session_id，不需要知道是谁在问。
#   2. **写盘失败不影响答题**。整个函数包在 try 里，出错只打一行 stderr ——
#      日志是旁路，不能让它把主链路拖死。
#   3. **内存里再留一份环形缓冲**。线上容器重启会丢文件，重启之后 /queries
#      至少还能给出本次运行的数据，不会「打开一看是空的」。
#
# 缓存命中也记：同一个问题被反复问，本身就是「这条内容最该做透」的信号，
# 只记冷启动等于只采到偏冷的样本。

QUERY_LOG_ON = os.getenv("QUERY_LOG_ON", "1").strip() not in ("0", "false", "off")
QUERY_LOG = Path(os.getenv("QUERY_LOG", ".qa_log.jsonl"))
QUERY_LOG_MAX_MB = float(os.getenv("QUERY_LOG_MAX_MB", "8"))
QUERY_LOG_RING = deque(maxlen=500)

# 问法日志原来是「每个请求在全局锁里 stat 一次文件、open 一次、write、close」。
# 压测（load_test.py）量出来这件事的代价：热缓存路径 342 RPS，关掉日志后 693 RPS
# —— 一个做统计的旁路功能，把整个服务的吞吐砍掉一半，而它连答案都不影响。
# 根因不是 write 慢，是「每个请求都 open/close 一次文件」+「全站在同一把锁上排队」。
#
# 改成：请求线程只做两件内存操作（进 ring、进队列），文件由一个后台线程攒批写。
# 一次 open 写 100 条，锁从「每请求一次」变成「每 100 条一次」，请求线程上不再有 IO。
#
# 代价要说清楚：进程被 kill 时，队列里还没落盘的那几十条会丢。这些是「用户问了什么」
# 的统计样本，丢几十条不影响任何判断；而为了不丢它让每个用户多等 2ms，是笔亏本买卖。
_logq: "queue.Queue[dict]" = queue.Queue(maxsize=int(os.getenv("QUERY_LOG_QUEUE", "5000")))
_logq_thread: threading.Thread | None = None
_logq_guard = threading.Lock()
LOG_BATCH = int(os.getenv("QUERY_LOG_BATCH", "100"))      # 一批最多写多少条
LOG_FLUSH_SEC = float(os.getenv("QUERY_LOG_FLUSH_SEC", "0.2"))  # 或者攒够 200ms 就先写


def _flush_log(buf: list[dict]) -> None:
    """把一批记录落到文件。整个函数只在后台线程里跑，所以不需要锁。"""
    if not buf:
        return
    try:
        size = QUERY_LOG.stat().st_size if QUERY_LOG.exists() else 0
        if size > QUERY_LOG_MAX_MB * 1024 * 1024:
            QUERY_LOG.replace(QUERY_LOG.with_name(QUERY_LOG.name + ".1"))
        # 一次 open 写一批：这才是省下来的东西。逐条写的话 100 条要 100 次 open/close。
        with QUERY_LOG.open("a", encoding="utf-8") as f:
            f.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in buf))
        STATS["log_written"] += len(buf)
    except OSError as e:
        STATS["log_write_failed"] += len(buf)
        print(f"[qa_log] 落盘失败 {len(buf)} 条：{type(e).__name__}: {e}")


def _log_writer() -> None:
    """后台写手。攒批 + 超时就写，避免「流量小的时候日志一直卡在内存里」。"""
    buf: list[dict] = []
    while True:
        try:
            buf.append(_logq.get(timeout=1.0))
        except queue.Empty:
            continue
        deadline = time.monotonic() + LOG_FLUSH_SEC
        while len(buf) < LOG_BATCH:
            left = deadline - time.monotonic()
            if left <= 0:
                break
            try:
                buf.append(_logq.get(timeout=left))
            except queue.Empty:
                break
        _flush_log(buf)
        buf = []


def flush_query_log(timeout: float = 3.0) -> bool:
    """等队列里的记录全部落盘。两处要用：进程正常退出前，和测试断言文件内容前。

    判断标准不能是「队列空了」——队列空的那一刻，写手可能正拿着一批在写文件。
    要「队列空」且「连续安静一个 flush 周期」，才算写完。
    """
    if _logq_thread is None or not _logq_thread.is_alive():
        return True
    end = time.monotonic() + timeout
    quiet_since: float | None = None
    while time.monotonic() < end:
        if _logq.empty():
            quiet_since = quiet_since or time.monotonic()
            if time.monotonic() - quiet_since > LOG_FLUSH_SEC + 0.05:
                return True
        else:
            quiet_since = None
        time.sleep(0.02)
    return False


def _ensure_log_writer() -> None:
    """懒启动写手线程。不在 import 时起，是为了 --reload 或测试里 import 模块
    不会凭空多出一个线程；双重检查是为了并发首请求不会起两个。"""
    global _logq_thread
    if _logq_thread is not None and _logq_thread.is_alive():
        return
    with _logq_guard:
        if _logq_thread is None or not _logq_thread.is_alive():
            t = threading.Thread(target=_log_writer, name="query-log-writer", daemon=True)
            t.start()
            _logq_thread = t


def log_query(body: dict, ctx: dict | None, mode: str = "", src: str = "",
              elapsed_ms: float = 0.0, cached: bool = False) -> None:
    """记一条真实问法。字段刻意少：能回答「问了什么、答没答出」就够了。"""
    if not QUERY_LOG_ON:
        return
    try:
        srcs = body.get("sources") or []
        fb = bool(body.get("fallback_used"))
        # 三个字段要分开，别混成一个 bool：
        #   ret 检索层自己命中（过阈值）—— 知识库里确实有这个答案
        #   fb  走生成层兜底救回来的 —— 检索没命中，是模型硬撑着答的
        #   ok  最终有没有给出答案（用户视角）
        # 「该补哪条内容」看的是 ret，不是 ok：兜底救回来的题，知识库里其实
        # 没有能过线的资料，恰恰最该补。混在一起就会漏掉这一整批。
        ret = bool(srcs) and not fb
        ok = bool(srcs)
        raw = ((ctx or {}).get("raw") or "").strip()
        resolved = ((ctx or {}).get("resolved") or "").strip()
        if not raw:
            return
        rec = {
            "t": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "q": raw,
            "mode": mode,
            "src": src or "api",
            "ok": ok,
            "ret": ret,
            "fb": fb,
            "topics": [s.get("topic") for s in srcs][:3],
            "top": round(float(srcs[0].get("score") or 0), 4) if srcs else None,
            "ms": int(elapsed_ms),
            "cached": bool(cached),
            "sid": (ctx or {}).get("sid") or "",
        }
        # 补全过的才记 rq：两个字段都在时，「补歪了」是可以看出来的
        if resolved and resolved != raw:
            rec["rq"] = resolved
        # 请求线程上只允许出现内存操作：
        #   deque.append 是 C 实现的原子操作，不需要锁，也不会有 IO 等待；
        #   put_nowait 队列满时立刻抛满，不阻塞 —— 宁可丢日志，不可让请求卡住。
        # _qlock 就此退场：它原来保护的是「ring + 文件」这一整段，现在那段只剩内存操作了。
        QUERY_LOG_RING.append(rec)
        _ensure_log_writer()
        try:
            _logq.put_nowait(rec)
        except queue.Full:
            STATS["log_dropped"] += 1
    except Exception as e:      # noqa: BLE001 —— 旁路功能，任何异常都不许往上冒
        print(f"[qa_log] 写日志失败（不影响答题）：{type(e).__name__}: {e}")


def tail_queries(n: int = 100) -> list[dict]:
    """取最近 n 条。优先读文件（跨重启），文件不在就退回内存缓冲。"""
    lines: deque[str] = deque(maxlen=n)
    try:
        if QUERY_LOG.exists():
            with QUERY_LOG.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        lines.append(line)
    except OSError:
        lines.clear()
    out = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    # 文件由后台线程攒批写，最多滞后 LOG_FLUSH_SEC（默认 200ms）。刚问完立刻看
    # /queries 会看不到那一条，所以用内存 ring 把文件里还没有的补上。
    # 去重键是 (时间, 问题)：rec 里没有唯一 id，而这两字段组合在实际使用中够用。
    # copy() 是原子的，直接迭代 deque 会撞上「迭代期间被 append」而报错。
    # 但 copy() 返回的还是 deque，deque 不支持切片 —— 所以再套一层 list。
    recent = list(QUERY_LOG_RING.copy())
    if not out:
        return recent[-n:]
    seen = {(r.get("t"), r.get("q")) for r in out[-max(n, 500):]}
    extra = []
    for r in recent:
        k = (r.get("t"), r.get("q"))
        if k not in seen:
            seen.add(k)
            extra.append(r)
    return (out + extra)[-n:]


def finish(body: dict, key: str | None, ctx: dict | None = None) -> dict:
    """统一的收尾：决定要不要写缓存、要不要记进会话。

    **降级结果不写缓存**：degraded 表示「上游这次抖了」，把它存下来等于
    把一次故障固化成永久的答案，后面所有人拿到的都是那段报错。
    """
    body.setdefault("cached", False)
    out = mark_session(body, ctx)
    # 顺序：**先补齐字段，再写缓存**。反过来的话，缓存里存的是缺了
    # sub_questions / timings 的版本，第二次命中时这些字段就是空的 ——
    # 现象是「第一次问能看到拆了几路，再问一次就没了」，比不返回更迷惑人。
    if key and not out.get("degraded"):
        cache_put(key, out)
    if ctx:
        t0 = ctx.get("t0")
        log_query(out, ctx, ctx.get("mode", ""), ctx.get("src", ""),
                  (time.perf_counter() - t0) * 1000 if t0 else 0.0,
                  cached=bool(out.get("cached")))
    return out


# ---------- 第七段：多轮会话 ----------
# 单轮系统碰到的第一个真问题不是「答得不好」，是「根本不理解」：
#   用户问「那上斜呢」—— 这五个字单看没有任何对象，向量化之后是一团噪声，
#   精排只能靠「上斜」两个字去撞，撞到哪个话题全凭运气。
# 所以多轮要解决的不是「记住聊过什么」，而是**把省略的那句话补回来**，
# 补成一句能独立检索的问题。这一步叫 query condensation（问题补全）。
#
# 三个刻意的设计：
#   1. 补全只用于**检索**，不改答案。答案仍然只能依据检索到的资料生成 ——
#      把历史对话喂进生成层看起来更「懂上下文」，但那样模型就有了一条
#      资料以外的信息来源，会开始编。
#   2. 补全是**有条件**的。每轮都补一次等于每轮多花一次调用（约 0.5 秒），
#      而真实对话里「完整问题」远多于「省略追问」。所以先看这句话像不像追问。
#   3. 会话状态是**进程内存**的。缓存和限流已经外置到 SQLite 了，会话没跟——
#      会话天然绑定一段连续对话，多 worker 下靠 session affinity 就能解，
#      外置它反而要每轮多一次读写。不假装它是持久化的。

SESSION_MAX = int(os.getenv("SESSION_MAX", "200"))       # 最多同时保留多少个会话
SESSION_TURNS = int(os.getenv("SESSION_TURNS", "6"))     # 每个会话保留最近几轮
SESSION_TTL = int(os.getenv("SESSION_TTL", "1800"))      # 空闲多久回收（秒）
CONDENSE_MODE = os.getenv("CONDENSE_MODE", "auto").strip().lower()   # auto | always | off
CONDENSE_CACHE_MAX = int(os.getenv("CONDENSE_CACHE_MAX", "1000"))
CONDENSE_FAST = os.getenv("CONDENSE_FAST", "1") not in ("0", "false", "False")
CONDENSE_MAX_TOKENS = int(os.getenv("CONDENSE_MAX_TOKENS", "64"))
# 补全最多等这么久。实测这个模型补全的中位延迟是 1.0 秒，但 8 次里有 2 次
# 超过 6 秒（供应商侧抖动）。等一句补全等 6 秒以上，用户体感就是「卡住了」，
# 所以上限压到 4 秒，超了就换一条零成本的退路（见 resolve_question 里的拼接）。
CONDENSE_TIMEOUT = float(os.getenv("CONDENSE_TIMEOUT", "4"))

_sessions: "OrderedDict[str, dict]" = OrderedDict()
_sess_lock = threading.Lock()
_cond: "OrderedDict[str, str]" = OrderedDict()            # 补全结果缓存（同一段对话会重复点）
_cond_lock = threading.Lock()


def new_session_id() -> str:
    """给客户端发一个会话号。用随机串而不是自增：自增号能被枚举，
    虽然这里没什么机密，但「能猜到的 id」迟早会在别的接口上变成漏洞。"""
    return uuid.uuid4().hex[:12]


def session_get(sid: str) -> list:
    """取某个会话的轮次列表（拷贝），顺手做过期回收。"""
    if not sid:
        return []
    now = time.time()
    with _sess_lock:
        s = _sessions.get(sid)
        if not s:
            return []
        if now - s["ts"] > SESSION_TTL:
            _sessions.pop(sid, None)
            return []
        _sessions.move_to_end(sid)
        return list(s["turns"])


def session_append(sid: str, raw: str, resolved: str, topic: str | None) -> None:
    """记一轮。存 resolved 而不是只存 raw：下一轮补全时，
    看到「用户原话 + 它被理解成了什么」比只看到原话准得多。"""
    if not sid:
        return
    now = time.time()
    with _sess_lock:
        s = _sessions.get(sid)
        if not s or now - s["ts"] > SESSION_TTL:
            s = {"turns": [], "ts": now}
            _sessions[sid] = s
        s["turns"].append({"q": raw, "resolved": resolved, "topic": topic, "ts": now})
        if len(s["turns"]) > SESSION_TURNS:
            del s["turns"][: len(s["turns"]) - SESSION_TURNS]
        s["ts"] = now
        _sessions.move_to_end(sid)
        while len(_sessions) > SESSION_MAX:
            _sessions.popitem(last=False)      # LRU：最久没动静的先走


def session_clear(sid: str) -> int:
    with _sess_lock:
        return 1 if _sessions.pop(sid, None) is not None else 0


def session_stats() -> dict:
    now = time.time()
    with _sess_lock:
        alive = sum(1 for s in _sessions.values() if now - s["ts"] <= SESSION_TTL)
        turns = sum(len(s["turns"]) for s in _sessions.values())
    return {"sessions": alive, "session_turns": turns, "session_max": SESSION_MAX,
            "condense_mode": CONDENSE_MODE, "condense_calls": STATS["condense_calls"]}


# 像「追问」的信号：短句，或者带省略/指代/接续的标志词。
# 这里故意宽松：把完整问题误判成追问的代价是**多一次调用**（补全后会原样返回），
# 把追问误判成完整问题的代价是**答错**。两种错误的价格不在一个量级上。
FOLLOWUP_HINTS = ("那", "呢", "还", "再", "换", "改", "也是", "同样", "一样",
                  "这个", "那个", "它", "一样", "继续", "然后", "以及", "顺便")


def needs_condense(question: str, history: list) -> bool:
    """要不要为这句话花一次补全调用。"""
    if CONDENSE_MODE == "off" or not history:
        return False
    if CONDENSE_MODE == "always":
        return True
    q = question.strip()
    if len(q) <= 6:
        return True                             # 「那上斜呢」「做几组」「那脚呢」必然是省略句
    # 这条线是量出来的，不是拍的：
    #   定 14 时，「推日都练哪些部位平时怎么安排」这种完整问题也被当成追问，
    #          白花一次调用，还会因为补全改写了措辞而丢掉缓存；
    #   定 8 时，仍然误伤「高位下拉握距多宽」（8 字、完整），而那次补全恰好超时，
    #          靠拼接退路才没答错；
    #   定 6 时，评测集里 21 条追问一条不漏（长的都带「那」），对照组一条不碰。
    # 更长的省略句靠下面的标志词兜住（「那如果我换成哑铃的话要做几组」有「那」）。
    return any(h in q for h in FOLLOWUP_HINTS)


CONDENSE_PROMPT = """用户在连续提问，最新一句话可能省略了主语或对象。把它补成一句能独立看懂的完整问题。

最重要的规则：**最新一句话里的那个对象，必须仍然是补全后句子的中心词**。
上文只用来补被省略的谓语、修饰和限定，不要把上一轮的整句句式套过来。

例一：上文在聊「自由泳怎么换气」，最新一句是「那打腿呢」。
正确：自由泳怎么打腿
错误：自由泳换气那打腿呢（把上文整句套过来，中心没落在打腿上）

例二：上文在聊「骑行装备怎么配」，最新一句是「那锁鞋呢」。
正确：骑行装备里锁鞋怎么选
错误：锁鞋怎么选（「锁鞋」单看太笼统，丢了「骑行」就没法定位）

其余规则：
1. 用上文里出现过的说法，不要换词、不要扩展问题范围。
2. 不回答问题，不解释，不加「以下是」「改写为」这类前缀。
3. 不要添加上文和最新一句话里都没出现的新对象、新数字。
4. 如果最新一句话本身已经完整、不需要上文，就原样输出。
5. 只输出这一句话，不要引号、不要编号、不要句末标点。

【上文】
{history}

【最新一句话】
{question}"""


_cond_fast_ok = [True]      # 用列表当可变标志，避免 global 满天飞

# 哪些错误说明「这个供应商不认 enable_thinking / max_tokens」。
# 必须精确到这个范围：第一版是「任何异常都把快路关掉」，结果第一次
# APITimeoutError（20 秒没回）就把快路永久关了，后面每条补全都在
# 慢路上等 6 秒然后超时 —— 一次网络抖动变成永久降级。
_PARAM_ERR_HINTS = ("enable_thinking", "max_tokens", "unexpected keyword",
                    "unknown parameter", "unsupported", "invalid")


def _param_error(e: Exception) -> bool:
    code = getattr(e, "status_code", None)
    if code == 400:
        return True
    msg = str(e).lower()
    return any(w in msg for w in _PARAM_ERR_HINTS)


def condense_llm(messages, max_tokens: int | None = None):
    """补全专用的模型调用：**关推理、限长、短超时、不重试**。

    四个「不」各有理由，都是量出来的：

    1. 关推理。同一个 prompt 实测：开推理 2.7~4.6 秒 / 输出 190~219 tokens；
       关掉 0.7~1.9 秒 / 输出 48~49 tokens。补全只要一句话，
       那 170 个 token 全花在「想」上，一点用都没有。
    2. 限长。同上，防止某个供应商偷偷把推理打开后输出一大段。
    3. 短超时。补全不在主链路上：等超了就用原话检索，最多少补一句，
       而让用户为了一句补全对着转圈等 10 秒是完全另一回事。
       实测这个模型的中位延迟 0.6~1.0 秒，但尾部能到 10 秒以上。
    4. 不重试。重试是给「抖一下」准备的，补全抖一下的代价（再等 3 秒）
       比它带来的收益（一句话补全得更准）大得多。主链路才配重试和熔断。

    后果说清楚：补全调用不再计入熔断器。上游整体挂掉时，每个追问会先白等
    最多 3 秒再退回原话 —— 这是为「不让辅助调用拖垮体验」付的固定成本。
    """
    if not LLM_API_KEY or LLM_API_KEY.startswith("sk-替换"):
        raise HTTPException(status_code=500, detail=".env 里的 LLM_API_KEY 还没填")
    STATS["llm_calls"] += 1
    fast = CONDENSE_FAST and _cond_fast_ok[0]
    n_tok = CONDENSE_MAX_TOKENS if max_tokens is None else max_tokens
    kw = {"max_tokens": n_tok, "extra_body": {"enable_thinking": False}} if fast else {}
    try:
        return get_client(CONDENSE_TIMEOUT).chat.completions.create(
            model=LLM_MODEL, messages=messages, temperature=0, **kw)
    except Exception as e:
        if fast and _param_error(e):
            # 只有「参数不被接受」才降级；超时/断网属于另一回事，下次还走快路
            _cond_fast_ok[0] = False
            print(f"[condense] 供应商不接受快速参数，之后走普通调用：{type(e).__name__} {str(e)[:80]}")
            return get_client(CONDENSE_TIMEOUT).chat.completions.create(
                model=LLM_MODEL, messages=messages, temperature=0)
        raise


def condense(question: str, history: list) -> str | None:
    """把省略的追问补成完整问题。**失败返回 None**。

    为什么是 None 而不是「返回原话」：上层需要区分两件事 ——
    「补全成功，模型认为这句话本来就完整」和「补全根本没说上话」。
    前者该原样去检索，后者该走退路（拿上一轮拼一个检索串）。
    两种情况都返回原话的话，上层只能猜。
    """
    if not history:
        return None
    lines = []
    for t in history[-3:]:
        line = f"- 用户：{t['q']}"
        if t.get("resolved") and t["resolved"] != t["q"]:
            line += f"（理解为：{t['resolved']}）"
        if t.get("topic"):
            line += f" → 命中话题：{t['topic']}"
        lines.append(line)
    hist_text = "\n".join(lines)

    ck = f"{hist_text}\u0001{question.strip()}"
    with _cond_lock:
        hit = _cond.get(ck)
        if hit is not None:
            _cond.move_to_end(ck)
            return hit

    try:
        STATS["condense_calls"] += 1
        resp = condense_llm([
            {"role": "system", "content": "你只做句子补全，不回答问题。"},
            {"role": "user", "content": CONDENSE_PROMPT.format(history=hist_text, question=question)},
        ])
        out = clean_rewrite_line(resp.choices[0].message.content or "")
    except HTTPException:
        STATS["condense_failed"] += 1        # 上游抖了 / key 没填
        return None
    except Exception:
        # 超时也走这里。**必须计数**：静默退回单轮时，用户看到的现象是
        # 「多轮好像没生效」，而日志里什么都没有 —— 这种沉默最难查。
        STATS["condense_failed"] += 1
        return None

    # 补全结果的三道保险：空的、太长（模型开始解释了）、没变化，都退回原话
    out = out.strip().strip("「」\"'。 ")
    if not out or len(out) > 60:
        STATS["condense_failed"] += 1
        return None
    with _cond_lock:
        _cond[ck] = out
        _cond.move_to_end(ck)
        while len(_cond) > CONDENSE_CACHE_MAX:
            _cond.popitem(last=False)
    return out


def resolve_question(question: str, sid: str, hist: list | None = None):
    """返回 (用于检索的问题, 上一轮轮数, 是否改写过)。

    补全成功就用补全结果；补全失败（超时/上游抖）就退一步：
    **把上一轮的问题和这一句拼起来**当检索串。

    为什么退路不是「用原话」：「那做几组」四个字单独检索是没有对象的，
    拼上「推日上斜练哪些部位」至少把对象带回来了。代价是查询变长、
    可能干扰精排，但那比直接拒答好 —— 实测这一句用原话会答「资料不足」。

    hist 可以由调用方先取好传进来：调用方需要先用 needs_condense 判断
    「这次会不会真打上游」，那一步也要读会话，读两次没必要。
    """
    if hist is None:
        hist = session_get(sid)
    if not needs_condense(question, hist):
        return question, len(hist), False
    out = condense(question, hist)
    if out is None:
        prev = (hist[-1].get("resolved") or hist[-1]["q"]) if hist else ""
        merged = f"{prev} {question.strip()}".strip()
        return merged, len(hist), merged != question.strip()
    return out, len(hist), out.strip() != question.strip()


@app.get("/queries")
def queries(request: Request, limit: int = 100, refused: int = 0, fmt: str = "json"):
    """看真实用户问了什么。**要口令** —— 这里存的是别人的原话。

    refused=1 只看「检索层没命中」的那批（含被生成层兜底救回来的）——
    那不是错误日志，是内容选题清单：粉丝在问、但知识库里没有能过线的资料，
    每一条都对应一篇文章的缺口。用 ret 而不是 ok 来筛，因为兜底救回来的
    同样属于「没有内容」，只是模型替它圆了一句。

    fmt=jsonl 直接给原始行，方便导出后扔进 analyze_queries.py。
    """
    check_code(request.headers.get("x-access-code"))
    limit = max(1, min(int(limit or 100), 2000))
    items = tail_queries(limit if not refused else min(limit * 5, 2000))
    if refused:
        # 老记录没有 ret 字段，退回看 ok，不然历史数据全被筛掉
        items = [r for r in items if not r.get("ret", r.get("ok"))][-limit:]
    else:
        items = items[-limit:]
    if fmt == "jsonl":
        from fastapi.responses import PlainTextResponse
        body = "\n".join(json.dumps(r, ensure_ascii=False) for r in items)
        return PlainTextResponse(body + "\n", media_type="application/x-ndjson")
    srcs = {}
    for r in items:
        srcs[r.get("src") or "api"] = srcs.get(r.get("src") or "api", 0) + 1
    return {
        "n": len(items),
        "refused": sum(1 for r in items if not r.get("ret", r.get("ok"))),
        "fallback": sum(1 for r in items if r.get("fb")),
        "unique": len({r.get("q") for r in items}),
        "by_src": srcs,
        "items": items,
    }


def _rss_mb() -> float | None:
    """当前进程的常驻内存。soak 测试靠它回答「跑一小时会不会一直涨」。

    不为一个数拉 psutil：两种平台的取值都只有几行。取不到就返回 None ——
    监控字段不许把 /stats 弄成 500。
    """
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            class _PMC(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t),
                            ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t),
                            ("PeakPagefileUsage", ctypes.c_size_t)]
            pmc = _PMC()
            pmc.cb = ctypes.sizeof(_PMC)
            ok = ctypes.windll.psapi.GetProcessMemoryInfo(
                ctypes.windll.kernel32.GetCurrentProcess(),
                ctypes.byref(pmc), pmc.cb)
            return round(pmc.WorkingSetSize / 1024 / 1024, 1) if ok else None
        with open("/proc/self/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return round(int(line.split()[1]) / 1024, 1)
    except Exception:      # noqa: BLE001
        return None
    return None


@app.get("/stats")
def stats(request: Request):
    """上线后看这个：缓存命中率、限流拦了多少、实际打了多少次上游、补全失败几次。

    不加这个接口，「加了缓存有没有用」只能靠感觉；面试时也少一个能讲的数字。

    **要口令。** 这里虽然没有用户原话，但有全站今天花了多少钱、额度还剩多少、
    有多少人正在被限流 —— 这些是运营数据，不该跟着公开链接一起发出去。
    原来漏了这道校验，公网上不带任何 header 就能把 cost 和 today_used 读走。
    """
    check_code(request.headers.get("x-access-code"))
    hit_rate = STATS["cache_hits"] / STATS["requests"] if STATS["requests"] else 0.0
    return {
        **STATS,
        "uptime_s": round(time.time() - STATS["started"], 1),
        # sqlite 后端下这几个数要从共享库取：本进程的 _cache 是空的，
        # 报 0 会让人以为缓存没生效。
        "state_backend": STATE_BACKEND,
        "cache_size": STORE.cache_size() if STORE else len(_cache),
        "cache_max": CACHE_MAX,
        "cache_hit_rate": round(hit_rate, 3),
        "rate_limit_per_min": RATE_LIMIT_PER_MIN,
        "daily_quota": DAILY_QUOTA,
        "today_used": STORE.daily_used() if STORE else _day["used"],
        "auth_required": bool(ACCESS_CODE),
        # token 消耗。只给数，不给钱 —— 单价按你自己的控制台乘（cost_report.py 有式子）。
        "cost": dict(COST),
        # soak 用：内存是不是稳定。数字本身不重要，一小时内有没有单调上涨才重要。
        "rss_mb": _rss_mb(),
        **session_stats(),
    }


@app.get("/session/{sid}")
def session_info(sid: str, request: Request):
    """看某个会话记了什么。多轮最容易出的 bug 是「记错了」而不是「没记」，
    所以把内部状态直接暴露出来 —— 调试时不用猜。

    要口令：turns 里是用户的原话。sid 是 12 位随机串，猜不到，
    但「猜不到」不等于「不该锁」—— 链接会被转发，日志会被贴进聊天记录。
    """
    check_code(request.headers.get("x-access-code"))
    turns = session_get(sid)
    return {"session_id": sid, "turns": turns, "n": len(turns),
            "ttl_s": SESSION_TTL, "max_turns": SESSION_TURNS}


@app.delete("/session/{sid}")
def session_drop(sid: str, request: Request):
    """清空一个会话。前端「新对话」按钮调它，请求里已经带了口令头。

    要口令的理由比读更硬：这是删除操作。不校验的话，任何人拿到 sid
    就能把别人的上下文清掉，而 sid 曾经是写在返回体里、可能被转发出去的。
    """
    check_code(request.headers.get("x-access-code"))
    return {"deleted": session_clear(sid), "session_id": sid}


# ---------- 第八段：复合问题拆解（做完了，但默认关着）----------
# 起因是一类「一句话问两件事」的输入：
#   「卧推的时候手该放多宽，高位下拉又该握多宽」
#   「增肌一天要吃多少蛋白，一周练几次比较合适」
# 单条 query 只有一个向量、一次精排，「手该放多宽」和「握多宽」会互相稀释，
# 排到第一的只有一个，另一件事就被丢了。做法是把它拆成 2~3 条子查询、
# 各自检索，再把结果合并进同一段上下文。
#
# **但这个功能默认是关的，因为量完之后发现它不划算。** 证据在 eval_compound.py
# 和 .eval_compound.json，35 条复合题（另附 7 条单意图 + 8 条负例）上五组对照：
#
#   策略                 话题覆盖   完整覆盖   无关条目   拒答
#   A 单路 2 条           74.1%     51.9%     33.3%    100%
#   B 单路 3 条           86.4%     74.1%     30.9%    100%
#   D 多路 4 条（轮流）    90.1%     81.5%     39.8%    100%
#   E 多路 2 条（轮流）    75.9%     55.6%     37.0%    100%
#
# 读法：
#   1. 从 2 条加到 3 条，白拿 +12.3 个点，不需要任何额外调用；
#   2. 在同样的 2 条预算下（A vs E），拆解只值 +1.9 —— 这是「拆解本身」的价值，
#      不含条数带来的好处；
#   3. 拆解真正能多覆盖的 2.5 个点，代价是一路 LLM 调用（中位 0.8 秒，
#      4 秒超时下实测 10/35 条失败）、每路一次 rerank、无关条目占比再涨 7 个点。
#
# 为什么在小库上不划算，这个原因比结论重要：
# 库里只有 105 条 chunk，top4 就等于看了全库的 3.8%。随便什么问法都能撞上
# 好几个话题，单路本来就「够宽」，拆解只是在已经够宽的召回面上再摊一次。
# 库涨到几千条时 top4 只覆盖 0.1%，一条 query 只能咬住一个子问题 ——
# 那时候拆解才是必需品。所以代码留着、开关留着，`mode=multi` 可以直接对比。
#
# 拒答没有被拆解稀释，这一点是设计出来的：每路**各自过绝对门槛**再融合，
# 不是融合完再过门槛。反过来做的话，两路各 0.7 的弱相关项加分会盖过
# 某一路 0.96 的真命中，而 0.7 本来是应该被拒掉的东西。实测 8 条负例
# （含「生理期能不能练腿，要不要减量」这种拆开后极易误答的）全部仍然拒答。

DECOMP_MODE = os.getenv("DECOMP_MODE", "off").strip().lower()      # off | auto | always
DECOMP_MAX_SUBS = int(os.getenv("DECOMP_MAX_SUBS", "3"))
DECOMP_TIMEOUT = float(os.getenv("DECOMP_TIMEOUT", "4"))
# 拆解要输出 2~3 行，condense 那 64 token 卡在边界上，给到 96
DECOMP_MAX_TOKENS = int(os.getenv("DECOMP_MAX_TOKENS", "96"))
MULTI_TOP_K = int(os.getenv("MULTI_TOP_K", "4"))   # 多路融合后给模型几段

DECOMP_PROMPT = """用户问了一句话。判断这句话里是不是**同时问了两个以上互不相干的问题**。

如果是，把它拆成 2~3 个能各自独立去检索的子问题；如果不是，原样输出这一句。

判断标准（很重要，别搞错）：
- 两个问题各自有**不同的对象**（比如「卧推握多宽」和「下拉握多宽」），才算复合。
- 同一个对象的两个侧面（比如「卧推下放几秒」和「卧推要不要停」）**不算**复合，不要拆。
- 只是问得啰嗦、加了语气词、举了例子，都不算复合。

硬约束：
1. 子问题里**不许出现原句没有的对象、部位、数字**。原句没说「哑铃」，就不许在子问题里加「哑铃」。
2. 每条子问题都要能被单独看懂，不许用「它」「这个」指代。
3. 不改写原句的意思，不扩范围，不回答问题。
4. 每行一条，行首写「- 」。最多 3 行。
5. 如果不需要拆，只输出一行「- 原句」（把原句一个字不改抄一遍）。

例一：原句「自由泳怎么换气，骑行的时候又该怎么呼吸」
- 自由泳怎么换气
- 骑行的时候怎么呼吸

例二：原句「自由泳换气要练多久才能顺」
- 自由泳换气要练多久才能顺

例三：原句「帮我看看这段代码哪里错了」
- 帮我看看这段代码哪里错了

【原句】
{question}"""


def parse_subs(text: str) -> list[str]:
    """解析「- xxx」行。解析不出就返回空列表，由上层退回单路。"""
    out = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        for pre in ("- ", "-", "·", "•", "1.", "2.", "3."):
            if line.startswith(pre):
                line = line[len(pre):].strip()
                break
        line = line.strip("「」\"'。 ")
        if line:
            out.append(line)
    return out


DECOMP_HINTS = ("还", "又", "另外", "分别", "各", "顺便", "以及", "同时")
# 疑问词：一句话里出现两次，基本就是两个问题（「吃多少蛋白」+「练几次」）。
# 这一条是补丁，来源很具体：「增肌一天要吃多少蛋白，一周练几次比较合适」
# 只有一个逗号、也没有「还」「又」，被前两条规则漏掉了。
QWORDS = ("多少", "几次", "几组", "几个", "多久", "多宽", "多高", "多重",
          "怎么", "要不要", "能不能", "哪些", "什么", "为什么", "是不是", "哪里")


def need_decompose(question: str) -> bool:
    """要不要为这句话花一次拆解调用。

    这里只是**省调用**的预筛，不是判定器 —— 拆不拆最终由模型说了算。
    所以线放得很松：宁可多调一次（代价 0.8 秒），也别把真复合题漏掉。
    三种信号，任何一条命中就交给模型判断：
      1. 至少两个分句（逗号/问号数 ≥ 2）；
      2. 一个分句 + 并列连接词（「还」「又」「分别」这类）；
      3. 一个分句 + 两个以上疑问词（「吃多少」+「练几次」）。
    「练腿日隔多久练一次比较好」这种 12 个字、零分句的句子直接挡掉。

    注意它**不管 DECOMP_MODE**：这个是「这句话像不像复合」的形态判断，
    跟「这个功能开不开」是两件事，混在一起以后就没法单独调了。
    """
    q = question.strip()
    if len(q) < 12:
        return False                    # 12 个字装不下两个问题
    clauses = q.count("，") + q.count(",") + q.count("；") + q.count(";") \
        + q.count("？") + q.count("?")
    if clauses >= 2:
        return True
    if clauses >= 1 and any(h in q for h in DECOMP_HINTS):
        return True
    return sum(q.count(w) for w in QWORDS) >= 2


def decompose(question: str) -> list[str] | None:
    """拆出子问题。**不拆、失败都返回 None**，上层一律退回单路。

    失败率要说清楚：4 秒超时下实测 35 条里 10 条超时（28.6%），放到 8 秒
    几乎全部成功 —— 说明不是模型慢，是长尾。所以这里跟 condense 一个策略：
    短超时、不重试、失败静默退回。拆解是**锦上添花**的一层，
    它不该有能力把一次本来能答的提问变慢或者变失败。
    """
    try:
        STATS["decomp_calls"] += 1
        resp = condense_llm([
            {"role": "system", "content": "你只做句子拆分，不回答问题。"},
            {"role": "user", "content": DECOMP_PROMPT.format(question=question)},
        ], max_tokens=DECOMP_MAX_TOKENS)
        subs = parse_subs(resp.choices[0].message.content or "")
    except Exception:
        STATS["decomp_failed"] += 1     # 超时、上游抖、key 没填 —— 一律退回单路
        return None

    if not subs or len(subs) > DECOMP_MAX_SUBS:
        STATS["decomp_failed"] += 1
        return None
    # 模型判断「不用拆」，会原样抄一遍。抄得不完全一样也不算拆 ——
    # 单条结果只可能有两种意思：要么不用拆，要么它自己就是那句原话。
    if len(subs) == 1:
        return None
    if any(s == question.strip() for s in subs):
        return None
    STATS["decomp_paths"] += len(subs)
    return subs


def rrf_fuse(rankings):
    """Reciprocal Rank Fusion：只用名次，不用原始分数来融合。

    为什么不直接比 rerank 分数：每一路的分数分布不一样 ——
    「卧推握多宽」的 top1 可能是 0.99，「腰疼能不能练」的 top1 可能只有 0.85，
    直接取最大值等于让「分数虚高的那一路」绑架整个排序。
    RRF 把每一路都压成名次，跨路可比。k=60 是常用常数，作用是压低头部名次的
    绝对差距 —— 不设 k 的话第 1 名和第 2 名的权重差一倍，太陡。

    实测在这个库上 RRF 和「每路轮流取一条」结果完全一样（见 eval_compound.py
    的 C/D 两组都是 90.1%），因为大部分路只过门槛 1~2 条。留着 RRF 是因为
    库变大后每路都会有多条，那时两者才会分叉。
    """
    agg = {}
    for ranking in rankings:
        for i, r in enumerate(ranking):
            cur = agg.setdefault(r["id"], {"item": dict(r), "rrf": 0.0,
                                           "best": 0.0, "hits": 0})
            cur["rrf"] += 1.0 / (60 + i + 1)
            cur["best"] = max(cur["best"], r["score"])
            cur["hits"] += 1
    return sorted(agg.values(), key=lambda x: (-x["rrf"], -x["best"]))


def search_kb_multi(q: str, subs: list[str], top_k: int = MULTI_TOP_K):
    """多路检索 + 融合。返回 (命中的资料, 用于判定的 ranking)。

    **每路各自过绝对门槛，再融合** —— 顺序不能反，理由在第八段开头。
    只要有一路过线就答，一路都不过线就拒。
    """
    paths = list(dict.fromkeys([q] + (subs or [])))     # 主查询永远算第一路，且排最前
    rankings, passed = [], []
    for p in paths:
        rk = rank_one(p)
        rankings.append(rk)
        kept = pick_hits(rk, 99, RERANK_THRESHOLD)      # 只卡门槛，先不截断
        if kept:
            passed.append(kept)

    if not passed:
        return [], (rankings[0] if rankings else [])

    picked, seen = [], set()
    for it in rrf_fuse(passed):
        topic = it["item"]["topic"]
        if topic in seen:               # 同话题只留一条，跟 pick_hits 一个规矩
            continue
        seen.add(topic)
        row = dict(it["item"])
        row["score"] = round(it["best"], 4)      # 对外报 rerank 分，不报 RRF 分
        row["paths"] = it["hits"]                # 这一条被几路同时命中
        picked.append(row)
        if len(picked) >= top_k:
            break

    # 判「答不答」用的是「最贴线的那一路」：多路的门是「或」，所以依据应该是
    # 各路 top1 里最高的那个，不是主查询那一路。near_gate 只看 ranking[0]，
    # 所以把分数最高的那条换到队首，后面的顺序不动。
    ranking = rankings[0] if rankings else []
    tops = [rk[0] for rk in rankings if rk]
    if tops:
        best = max(tops, key=lambda r: r["score"])
        ranking = [best] + [r for r in ranking if r["id"] != best["id"]]
    return picked, ranking


@app.post("/chat")
def chat(req: AskRequest, request: Request):
    # 这个口子没有检索、直接打模型，是最容易被拿来白嫖的一个 ——
    # 所以口令、限流的判断跟 /rag 完全一致，不能因为「只是个演示接口」就漏掉。
    check_code(request.headers.get("x-access-code"))
    STATS["requests"] += 1
    wait = rate_check(client_ip(request))
    if wait > 0:
        raise HTTPException(status_code=429, detail=f"问得太快了，{int(wait) + 1} 秒后再试",
                            headers={"Retry-After": str(int(wait) + 1)})
    if not daily_check():
        raise HTTPException(status_code=429, detail="今天的演示额度用完了，明天再来")

    resp = call_llm([
        {"role": "system", "content": "你是一名健身教练。回答要简短具体，给数字，不说套话。"},
        {"role": "user", "content": req.question},
    ])

    bill_chat(resp)
    return {
        "answer": resp.choices[0].message.content,
        "model": resp.model,
        "tokens": resp.usage.total_tokens,
    }


# ---------- 寒暄和边界：不该进检索的那几句话 ----------

# 为什么单独拎出来：真实问法日志里「你叫什么名字」被拒答成了
# 「知识库里没有相关资料」。技术上没错——库里确实没有——但用户看到的是
# 一个不会说人话的东西。寒暄、问用法、道谢这三类，答案根本不在知识库里，
# 也不该让 reranker 去判断，直接给固定话术。
#
# 判据是**整句匹配**，不是包含：像「你好，推日练什么」这种半句寒暄半句正事，
# 必须继续往下走去检索，否则真正的提问会被一句「在」吞掉。
_SMALLTALK = [
    (r"你(是)?(谁|啥|什么|叫什么(名字)?|是真人吗|是ai吗|是不是ai|是机器人吗)|你叫什么|介绍一下你(自己)?|你能干什么|你是干嘛的",
     lambda: f"我是壮壮说健身的知识库问答。库里 {len(KB)} 条正文，讲的是三分化训练、"
             f"动作细节、组次安排和饮食配合。问训练上的问题就行，比如「推日之后二头还练吗」。"
             f"答不了的我会直接说没有，不瞎编。"),
    (r"(这个|那个|它)?(怎么|如何)(用|使用|问|玩)|能问(什么|啥)|可以问(什么|啥)|(这个|那个)(是|能)(干|做)(什么|啥)|有什么(用|功能)|怎么提问",
     lambda: "直接把问题打出来，越具体越好。可以说「深蹲膝盖要不要过脚尖」，"
             "也可以说「推日先练胸还是先练肩」。库里没有的我会说没有，"
             "你把想问的发到公众号，问得多的我先补。"),
    (r"你好|您好|hi|hello|hey|在吗|在不在|有人吗",
     lambda: "在。训练上有什么想问的直接说，推日拉日腿日、动作细节、组次安排、吃多少都能问。"),
    (r"谢谢|多谢|感谢|谢了|好的|好|嗯|ok|收到|明白了|懂了|再见|拜拜|晚安",
     lambda: "行。还有别的想问就直接打出来。"),
]

_PUNCT_RE = re.compile(r"[\s，。！？、,.!?;:；：\"'“”‘’()（）\[\]【】…—~-]")


def smalltalk_reply(question: str):
    """整句命中寒暄就返回固定话术，否则 None（继续走检索）。"""
    q = _PUNCT_RE.sub("", (question or "")).lower()
    q = re.sub(r"[啊呀呢嘛吧哦呗]+$", "", q)   # 句尾语气词不改变意思，「你是谁啊」和「你是谁」是一句
    if not q or len(q) > 20:      # 太长的一定是真问题，别拦
        return None
    for pat, make in _SMALLTALK:
        if re.fullmatch(pat, q):
            return make()
    return None


# 拒答文案。原来那句「我无法基于资料回答」是技术口径，用户看到只会觉得被拒绝了。
# 改成把拒答变成选题收集：告诉他这条我这儿没有，以及去哪儿说。
REFUSAL_TEXT = ("这条我库里还没有。想问的可以发到公众号后台，问得多的我先补进去。")


@app.post("/rag")
def rag(req: AskRequest, request: Request,
        mode: Literal["rerank", "vector", "hybrid", "rewrite", "oral", "keyword", "multi"] = "rerank",
        fallback: bool | None = None, src: str = ""):
    """mode=rerank 向量召回 + reranker 精排（默认，200 题整体 88% / 拒答 81%）
    mode=vector 纯向量检索（整体 86% / 拒答 70%）
    mode=hybrid 向量 + BM25 混合检索（排序融合，门控仍看向量分）
    mode=rewrite 先把问题改写成 3 条资料风格表述，再取最大相似度
    mode=oral    先改写成 3 条书面表述，再各走一遍精排，取最高分那次
    mode=keyword 老的关键词匹配，方便对比
    mode=multi   复合问题拆解：拆成 2~3 条子查询各检索一遍再融合（默认关，理由见第八段）
    fallback=1   检索拒答后，把 top-2 弱相关资料交给生成层判断能不能答（生成层兜底）

    返回里的 uncertain 表示「这次的判定贴着门线」（|top1 - 门槛| < 0.006）：
    同一句话连跑 5 次有过翻面的实测区间（diag_stability.py）。它不影响答不答，
    只告诉调用方「这条结论不稳」，前端标黄、离线复核都靠它。

    为什么默认从 vector 换成 rerank：同一批 98 条题上，rerank 精排 + 0.50 门控
    比纯向量高 8 个点（94% vs 86%），拒答率也从 70% 提到 90%。代价是每题多一次
    上游调用（约 0.27 秒），换来的主要不是排序变好，而是**判别力**——
    reranker 的分数在单一体系语料上能把「该答的」和「该拒的」切开，余弦相似度切不开。

    上线三层（口令 -> 缓存 -> 限流）都在这个函数入口。缓存排在限流前面的理由
    见上面那段注释；唯一的例外是补全：它会真打一次上游，所以限额要提前到它之前。
    缓存命中直接返回，`cached: true` 且 `tokens: 0` —— 缓存命中没收钱，就别报钱。
    """
    check_code(request.headers.get("x-access-code"))
    STATS["requests"] += 1
    # src 是埋点来源，会原样写进问法日志。不截断的话，一个 1MB 的 src
    # 能让每次请求都往日志里写 1MB —— 这个参数的长相就决定了它不该超过几十字节。
    src = (src or "")[:32]

    # 寒暄、问用法、道谢：这三类的答案根本不在知识库里，走检索只会得到一个
    # 「这条我库里还没有」。所以在这里就分出去，不进缓存、不进问法日志。
    # 不进日志是刻意的：问法日志是**内容缺口清单**，「你叫什么名字」混进去会
    # 让缺口看起来比实际严重，也会污染「高频问题」排序。
    greet = smalltalk_reply(req.question)
    if greet:
        return {"answer": greet, "sources": [], "ranking": [], "mode": mode,
                "variants": [], "tokens": 0, "degraded": False,
                "fallback_used": False, "uncertain": False,
                "smalltalk": True, "cached": False}

    # 多轮第一步：把省略的追问补成一句能独立检索的话。
    # **必须在算缓存键之前** —— 「那上斜呢」跟在推日后面和跟在拉日后面
    # 是两个完全不同的问题，拿原话当键，第二个会话会直接吃到第一个的答案。
    sid = (req.session_id or "").strip()
    t0 = time.perf_counter()

    # 补全会真的打一次上游模型（中位 0.79 秒）。所以限流和日额度必须**在它之前**
    # 过闸，不能等走到检索阶段再查 —— 这条路径原来能被绕过，实测过：
    # 「短问句 + 每次换一个 X-Forwarded-For + 随机 session_id」可以让服务无限次
    # 替调用方调模型，而每分钟限流和每天 300 的额度都在补全之后才判，
    # 等于额度只拦得住生成、拦不住补全。补全花了钱，就该在花钱前拦。
    hist_now = session_get(sid)
    prepaid = False
    if needs_condense(req.question, hist_now):
        pre_wait = rate_check(client_ip(request))
        if pre_wait > 0:
            raise HTTPException(
                status_code=429,
                detail=f"问得太快了，{int(pre_wait) + 1} 秒后再试（每分钟 {RATE_LIMIT_PER_MIN} 次）",
                headers={"Retry-After": str(int(pre_wait) + 1)},
            )
        if not daily_check():
            raise HTTPException(status_code=429, detail="今天的演示额度用完了，明天再来")
        prepaid = True        # 已经扣过这份额度，下面不要再扣第二次

    query, hist_n, condensed = resolve_question(req.question, sid, hist=hist_now)
    # 这一轮属于谁、被理解成了什么、各段花了多久 —— 后面所有 return 都带着它，
    # 所以只在这里建一次。传六个位置参数那种写法已经不够用了。
    ctx = {"sid": sid, "raw": req.question, "resolved": query,
           "hist": hist_n, "condensed": condensed, "subs": [],
           "mode": mode, "src": src, "t0": t0,
           "tim": {"condense_ms": (time.perf_counter() - t0) * 1000, "decomp_ms": 0}}

    # 兜底开关在本函数内部被用过两次（决定缓存键、决定走不走兜底），
    # 所以先算一次：同一个值算两遍，迟早有一处忘了跟着改。
    use_fb = FALLBACK_ENABLED if fallback is None else fallback
    key = cache_key(mode, use_fb, query)

    cached = cache_get(key)
    if cached is not None:
        cached["cached"] = True
        cached["tokens"] = 0
        out = mark_session(cached, ctx)
        # 缓存命中也要记日志：同一个问题被反复问，是「这条内容最该做透」的信号
        log_query(out, ctx, mode, src, (time.perf_counter() - t0) * 1000, cached=True)
        return out

    # 补全那条路径上面已经扣过额度了，这里不再扣第二次 —— 一次请求一份额度，
    # 否则一句「那上斜呢」会被算成两次，用户莫名其妙被限。
    if not prepaid:
        wait = rate_check(client_ip(request))
        if wait > 0:
            raise HTTPException(
                status_code=429,
                detail=f"问得太快了，{int(wait) + 1} 秒后再试（每分钟 {RATE_LIMIT_PER_MIN} 次）",
                headers={"Retry-After": str(int(wait) + 1)},
            )
        if not daily_check():
            raise HTTPException(status_code=429, detail="今天的演示额度用完了，明天再来")

    t1 = time.perf_counter()
    variants = []           # 只有 oral 模式有；放外面是为了三个 return 的字段一致
    subs = []               # 只有 multi 模式有：拆出来的子问题
    # 全链路都用 query（补全后的问题）：检索、兜底、生成提问三处必须一致。
    # 只拿补全结果去检索、却拿原话去生成，模型会对着「那上斜呢」四个字打转。
    if mode == "multi":
        # 拆解要单独计时：它是这个模式里唯一一段「可能不发生」的耗时，
        # 混进 retrieve_ms 就看不出来慢在拆还是在检索。
        t_dec = time.perf_counter()
        # mode=multi 是显式请求，所以这里不受 DECOMP_MODE=off 影响 ——
        # 「全局默认关」管的是默认模式要不要自动拆，不是「这个功能坏没坏」。
        # 想彻底关掉就把分支删了或者让前端不传 mode=multi。
        if DECOMP_MODE == "always" or need_decompose(query):
            subs = decompose(query) or []
        ctx["tim"]["decomp_ms"] = (time.perf_counter() - t_dec) * 1000
        ctx["subs"] = subs
        hits, ranking = search_kb_multi(query, subs)
    elif mode == "keyword":
        hits = search_kb_keyword(query)
        ranking = None
    elif mode == "rerank":
        hits, ranking = search_kb_rerank(query)
    elif mode == "oral":
        hits, ranking, variants = search_kb_rerank_oral(query)
    elif mode == "hybrid":
        hits, ranking = search_kb_hybrid(query)
    elif mode == "rewrite":
        hits, ranking = search_kb_rewrite(query)
    else:
        hits, ranking = search_kb_vector(query)

    ctx["tim"]["retrieve_ms"] = (time.perf_counter() - t1) * 1000
    t2 = time.perf_counter()

    # 这次判定用的是哪条门线。每种模式各有一条，不能统一按 RERANK_THRESHOLD 算：
    # 拿错门线会得出一堆假的「不确定」（vector 的 0.60 跟 4B 的 0.95 不是一个刻度）。
    if mode == "oral":
        # variants 非空 = 改写这条路真的参与了，此时最终排序来自改写候选，
        # 它比的是 ORAL_THRESHOLD；基座那一次仍然比的是 RERANK_THRESHOLD。
        gate = ORAL_THRESHOLD if variants else RERANK_THRESHOLD
    elif mode in ("vector", "hybrid"):
        gate = VEC_THRESHOLD
    elif mode == "rewrite":
        gate = REWRITE_THRESHOLD
    else:
        gate = RERANK_THRESHOLD
    uncertain = near_gate(ranking, gate)

    if not hits:
        # 第二道闸：检索层严格拒了，但召回里确实有弱相关资料。
        # 让生成层自己判断「这些够不够回答」——它能拒，也能把本来被埋掉的答案捞出来。
        # fallback 三态在函数入口就算好了：不传走全局默认，显式 ?fallback=0 必须能关掉
        # （写成 `fallback or FALLBACK_ENABLED` 就关不掉了，False or True 还是 True）。
        fb = None
        if use_fb and ranking:
            fb = answer_with_fallback(query, ranking)

        if fb:
            return finish({
                "answer": fb["answer"],
                "sources": public_sources(fb["sources"]),
                "ranking": ranking,
                "mode": mode,
                "variants": variants,
                "tokens": fb["tokens"],
                "degraded": False,
                "fallback_used": True,      # 调用方据此知道：这条不是精排过线的，是兜底兜回来的
                "uncertain": uncertain,     # 兜底能兜回来，本身就说明它贴着门线
            }, key, ctx)

        return finish({
            "answer": REFUSAL_TEXT,
            "sources": [],
            "ranking": ranking,
            "mode": mode,
            "variants": variants,
            "tokens": 0,
            "degraded": False,
            "fallback_used": False,
            "uncertain": uncertain,
        }, key, ctx)

    context = "\n".join(f"- {h['topic']}：{h['text']}" for h in hits)

    sources = public_sources(hits)

    try:
        resp = call_llm([
            {
                "role": "system",
                "content": (
                    "你是一名健身教练，跟学员说话，不用书面语。"
                    "回答只用下面给你的内容，不许补充别的知识；内容里完全没有的，直接说「这个我还答不上来」。"
                    "只有部分相关时，把相关的那部分直接讲清楚，不要先用「答不上来」开头再补 —— 那等于绕一圈才回答。"
                    "回答简短具体，给数字。答案里不要出现「资料」「根据资料」这类词 —— 学员不关心你的资料来源。"
                ),
            },
            {
                "role": "user",
                "content": f"【资料】\n{context}\n\n【问题】\n{query}",
            },
        ])
        ctx["tim"]["llm_ms"] = (time.perf_counter() - t2) * 1000
    except HTTPException as e:
        if e.status_code not in (502, 503):
            raise       # key 没填这类配置错误，不该伪装成「能答」，直接暴露出来
        # 走到这里说明：检索成功、生成失败。
        # 502 = 试过了确实失败，503 = 上游熔断中压根没试。
        # 两种情况下资料都是真实命中的 —— 把资料原样交出去，比一句报错有用。
        # 但必须打上 degraded 标记，否则调用方会以为这是模型润色过的答案。
        return finish({
            "answer": "模型服务暂时不稳定（重试或熔断），以下是命中的原始资料：\n"
                      + "\n".join(f"- {h['topic']}：{h['text']}" for h in hits),
            "sources": sources,
            "ranking": ranking,
            "mode": mode,
            "variants": variants,
            "tokens": 0,
            "degraded": True,
            "fallback_used": False,
            "uncertain": uncertain,
        }, key, ctx)

    bill_chat(resp)
    return finish({
        "answer": resp.choices[0].message.content,
        "sources": sources,
        "ranking": ranking,
        "mode": mode,
        "variants": variants,
        "tokens": resp.usage.total_tokens,
        "degraded": False,
        "fallback_used": False,
        "uncertain": uncertain,
    }, key, ctx)


# ---------- 第九段：多步工具调用（排周计划）----------
# /rag 解决的是「一个问题、一段回答」，它的形状决定了它答不了「给我排一周计划」：
# 检索出来的是一段段散文，模型拿着散文拼计划，拼出来的东西既没法校验，
# 也没法解释「为什么周三排这个动作」。
#
# 这里换一种做法：给模型三个工具（查动作库 / 查单个动作 / 交草稿自检），
# 让它先查、再排、再自查。三条硬约束写在代码里，不写在提示词里：
#
#   1. 动作名必须在 exercises.json 里存在（34 个动作，每个都指回一条已发布正文）。
#      模型编出来的名字会被剔除并记进 dropped，不会静默混进计划。
#   2. 组次/间歇/要点全部由服务端从库里填 —— 模型只交动作名，不交数字。
#      所以「模型瞎写 4 组 6 次侧平举」这件事在结构上就不可能发生。
#   3. 伤病冲突、天数不符、同部位连练，代码层独立校验一遍。模型可以自己调
#      check_plan 自查，但**它查不查、说没问题，都不影响服务端再查一次**。
#
# 为什么不用流式：排计划是一次性交付物（要打印、要给学员），不是聊天，流式在这里
# 只增加「拼到一半没法校验」的麻烦。

class PlanRequest(BaseModel):
    goal: str = Field("增肌", max_length=40)
    days: int = Field(3, ge=2, le=5)
    equipment: str = Field("健身房", max_length=20)
    session_min: int = Field(60, ge=20, le=180)
    level: str = Field("新手", max_length=10)
    # 元素也要限长。`list[str]` 只限了列表长度，单个元素随便多长，
    # 塞一万字的「膝」进去照样会进模型上下文 —— 它最终是拼进 prompt 的。
    injuries: list[Annotated[str, StringConstraints(max_length=20)]] = Field(
        default_factory=list, max_length=4)
    note: str = Field("", max_length=200)


@app.post("/plan")
def plan(req: PlanRequest, request: Request, src: str = ""):
    """排一周训练计划。多步：查动作库 → 组计划 → 自检 → 出 JSON。

    返回里的 trace 是这条链的账本：每一步调了什么工具、参数是什么、
    查到几条、花了多少毫秒。面试和排查都看它 —— 「AI 排的计划」如果不可追溯，
    那它就只是一段看起来合理的文字。

    needs_review=true 表示有硬问题（动作被剔除 / 伤病冲突 / 天数不符 / 没排出来），
    这时候计划**不进缓存**：一次没排好的结果被缓存下来，就等于把它固化成了标准答案。
    """
    check_code(request.headers.get("x-access-code"))
    STATS["requests"] += 1
    src = (src or "")[:32]      # 同 /rag：埋点参数不该有任意长度
    prof = req.model_dump()
    t0 = time.perf_counter()

    # 缓存键用整个档案：目标、天数、器械、水平、伤病任何一个不同，计划就该不同。
    key = cache_key("plan", False, json.dumps(prof, sort_keys=True, ensure_ascii=False))
    hit = cache_get(key)
    if hit is not None:
        hit["cached"] = True
        hit["tokens"] = 0               # 缓存命中没打上游，就别报钱
        return hit

    wait = rate_check(client_ip(request))
    if wait > 0:
        raise HTTPException(status_code=429, detail=f"排得太快了，{int(wait) + 1} 秒后再试",
                            headers={"Retry-After": str(int(wait) + 1)})
    if not daily_check():
        raise HTTPException(status_code=429, detail="今天的演示额度用完了，明天再来")

    if PLAN_LIB is None:
        raise HTTPException(status_code=503, detail="动作库没加载成功，排不了计划")

    def _llm(messages, tools=None):
        """把 call_llm 包一层：每次调用都记账，链路上任何一步的 token 都不漏。

        不在这里做重试和熔断 —— call_llm 里已经有了，这里再包一层会变成
        「重试的重试」，一次抖动变成九次上游调用。
        """
        r = call_llm(messages, tools=tools)
        bill_chat(r)
        return r

    try:
        out = plan_agent.plan_week(prof, _llm, lib=PLAN_LIB)
    except HTTPException as e:
        if e.status_code not in (502, 503):
            raise            # 没填 key 这类配置错误必须暴露，不能伪装成「排不出来」
        raise HTTPException(status_code=503, detail="排计划的模型服务暂时不稳定，等一下再试")

    STATS["plans"] += 1
    STATS["plan_tool_calls"] += len(out["trace"])
    if out["needs_review"]:
        STATS["plan_review"] += 1

    out["profile"] = prof
    out["cached"] = False
    out["timings"] = {"total_ms": int((time.perf_counter() - t0) * 1000)}
    if not out["needs_review"]:
        cache_put(key, out)

    # 不写进问法日志：那份日志是「内容缺口清单」，一条训练计划不是缺口。
    # 混进去会让缺口统计失真，也会把「接住率」拉高 —— 排计划成功不等于内容覆盖变好。
    return out
