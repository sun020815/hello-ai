import json
import os
import random
import threading
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
RERANK_TOP_K = 2
# rerank 分数低于这个值就当「资料里没有」
# 注意它和余弦相似度不是一个量纲：0.60 那一套完全不能套用。
#
# 这个值的来历换了三次，本身就是「参数只在特定语料上成立」的证据：
#   4 条库：没意义，全召回
#   40 条库：扫 0~0.20，0.055~0.085 是平台，取 0.06
#   126 条拼凑语料：rerank 最高分 0.4210 比该命中的最低分 0.0986 还高，切不开，0.06 勉强用
#   105 条单一体系语料：分数整体上移，重扫 0~1.0 后 0.45~0.60 是平台（整体 94~95%），取 0.50
#
# 为什么取 0.50 而不是全量最优的 0.535：两者整体只差 1 条（94% vs 95%），
# 差别只在「把哪一条错掉」：
#   0.50  放行「肌酸一天吃多少」(0.5309 → 实测 0.527)，检索口径拒答率 90%
#   0.535 误拒「哥本哈根支撑」(0.5404)，检索口径拒答率 100%
# 选 0.50 是因为前者不算真错：肌酸那条虽然被放行，但生成层拿到资料后发现
# 里面没有肌酸，会回答「资料里没有肌酸一天吃多少的内容」—— 实测验证过。
# 也就是说**检索口径 90%、端到端 100%**，拒答本来就有两道闸。
# 而 0.535 误拒的那条是真实问题、真实有答案，谁也不会替它补救。
# 另外 0.535 距离 0.5404 只有 0.0054，贴着悬崖；五折交叉验证里有一折的
# 独立最优漂到了 0.640，说明最优点的位置会随样本移动。0.50 留有余量。
# 交叉验证留出集平均 94%，不是过拟合出来的假象。
#
# 换 reranker 模型必须重扫：这是 bge-reranker-v2-m3 的输出分布，不是通用刻度。
RERANK_THRESHOLD = 0.50

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
            breaker.on_failure()
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
#   closed     正常放行，数「连续」失败几次
#   open       一律拒绝，直到冷却结束 —— 这段时间一个网络包都不发
#   half_open  冷却结束，只放一个请求去试探。成功就关回去，失败就再开
#
# 为什么必须「连续」而不是「累计」：累计计数永远回不到零，跑一天下来总会攒够 5 次
# 偶发失败，然后无辜地熔断。一次成功就清零，「连续」才反映「现在是不是正在挂」。

BREAKER_THRESHOLD = 5       # 连续失败 5 次才熔断。太小会被一次抖动误伤，太大就没有保护意义
BREAKER_COOLDOWN = 30.0     # 熔断 30 秒后放一个探针


class CircuitOpenError(Exception):
    """熔断中，请求根本没发出去。
    调用方应该翻译成 503 而不是 502 —— 两者的含义不同：
    502 是「我试过了，失败了」；503 是「我知道它会失败，所以没试」。
    对调用方来说前者是坏消息，后者是「请稍后再来」。"""


class CircuitBreaker:
    """一个上游一个熔断器。用 threading.Lock 是因为 FastAPI 的同步端点跑在线程池里，
    两个请求可能同时改状态 —— 竞态下最典型的 bug 是「探针放出去两个」，
    那等于上游在恢复期被双倍流量再打一次。"""

    def __init__(self, name, threshold=BREAKER_THRESHOLD,
                 cooldown=BREAKER_COOLDOWN, clock=time.monotonic):
        self.name = name
        self.threshold = threshold
        self.cooldown = cooldown
        self._clock = clock          # 测试里换成假时钟，不用真等 30 秒
        self._lock = threading.Lock()
        self.state = "closed"
        self.failures = 0            # 连续失败次数
        self.opened_at = None
        self.probing = False         # half_open 时，探针是不是已经飞出去了
        self.trips = 0               # 一共熔断过几次，给监控看

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

    def on_success(self):
        with self._lock:
            if self.state != "closed":
                print(f"  [breaker:{self.name}] 探针成功，恢复正常", flush=True)
            self.state = "closed"
            self.failures = 0
            self.opened_at = None
            self.probing = False

    def on_client_error(self):
        """上游回了 4xx：它活着，只是我们请求写错了。不计故障，
        但必须放掉探针，否则 half_open 会永远卡在「探针在飞」，后面全被拒。"""
        with self._lock:
            self.probing = False
            if self.state == "half_open":
                self.state = "closed"
                self.failures = 0
                self.opened_at = None

    def on_failure(self):
        with self._lock:
            self.probing = False
            if self.state == "half_open":
                # 探针也失败了：说明还没好，立刻重新熔断，冷却从此刻重新计时
                self._trip()
                return
            self.failures += 1
            if self.failures >= self.threshold:
                self._trip()

    def _trip(self):
        self.state = "open"
        self.opened_at = self._clock()
        self.trips += 1
        print(
            f"  [breaker:{self.name}] 连续失败 {self.failures} 次，"
            f"熔断 {self.cooldown:.0f} 秒",
            flush=True,
        )

    def snapshot(self):
        with self._lock:
            return {
                "state": self.state,
                "consecutive_failures": self.failures,
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


@app.get("/")
def read_root():
    return {"message": "hello"}


@app.get("/breaker")
def breaker_state():
    """熔断器的当前状态。线上排障第一件事就是看这个：
    上游明明通了但接口一直报 503，多半是熔断没恢复；
    反过来「为什么它不熔断」也能从这里看出失败次数涨没涨。
    不给这个口子，熔断就是一个黑盒 —— 出问题只能猜。"""
    return {name: b.snapshot() for name, b in _breakers.items()}


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

    stat = {}
    try:
        data = call_with_retry(_post, label="rerank", stat=stat)
    except CircuitOpenError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise external_error(e, "调用 reranker", stat)

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

    stat = {}
    try:
        return call_with_retry(
            lambda: get_client(LLM_TIMEOUT).chat.completions.create(
                model=LLM_MODEL, messages=messages
            ),
            label="chat",
            stat=stat,
        )
    except CircuitOpenError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise external_error(e, "调用模型", stat)


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
def rag(req: AskRequest, mode: str = "rerank"):
    """mode=rerank 向量召回 + reranker 精排（默认，98 题整体 94% / 拒答 90%）
    mode=vector 纯向量检索（整体 86% / 拒答 70%）
    mode=hybrid 向量 + BM25 混合检索（排序融合，门控仍看向量分）
    mode=rewrite 先把问题改写成 3 条资料风格表述，再取最大相似度
    mode=keyword 老的关键词匹配，方便对比

    为什么默认从 vector 换成 rerank：同一批 98 条题上，rerank 精排 + 0.50 门控
    比纯向量高 8 个点（94% vs 86%），拒答率也从 70% 提到 90%。代价是每题多一次
    上游调用（约 0.27 秒），换来的主要不是排序变好，而是**判别力**——
    reranker 的分数在单一体系语料上能把「该答的」和「该拒的」切开，余弦相似度切不开。
    """
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
        if e.status_code not in (502, 503):
            raise       # key 没填这类配置错误，不该伪装成「能答」，直接暴露出来
        # 走到这里说明：检索成功、生成失败。
        # 502 = 试过了确实失败，503 = 上游熔断中压根没试。
        # 两种情况下资料都是真实命中的 —— 把资料原样交出去，比一句报错有用。
        # 但必须打上 degraded 标记，否则调用方会以为这是模型润色过的答案。
        return {
            "answer": "模型服务暂时不稳定（重试或熔断），以下是命中的原始资料：\n"
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
