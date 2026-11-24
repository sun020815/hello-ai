"""上线三件套的离线测试：缓存 / 限流 / 口令。不联网、不花钱。

为什么单独一个文件：`test_web.py` 管的是「页面和接口的字段对不对得上」，
这里管的是「挂到公网之后会不会被一个链接搞崩」。两件事的失效方式完全不同，
混在一起，改前端的人会被限流测试绊住。

跑法：uv run python test_guard.py

测法说明：把 `app.get_client` 换成假客户端，`/rag` 走完整真实代码路径
（口令 -> 缓存 -> 限流 -> 拒答/回答 -> 写缓存 -> 计数），但不出网。
为什么不直接替换 `app.call_llm`：那等于把计数、重试、熔断整段跳过，
`/stats` 里的 llm_calls 就永远测不到 —— 而「缓存到底省没省下上游调用」
正是这个文件最该证明的事。
"""
import sys
import time

from fastapi.testclient import TestClient

import main as app

fails = []


def check(name, cond, extra=""):
    print(("  ok   " if cond else "  FAIL ") + name + (f"   {extra}" if extra != "" else ""))
    if not cond:
        fails.append(name)


class FakeUsage:
    total_tokens = 42


class FakeMessage:
    content = "假答案：卧推时脚踩实、臀按凳、肩胛后缩。"


class FakeChoice:
    message = FakeMessage()


class FakeResp:
    choices = [FakeChoice()]
    model = "fake-model"
    usage = FakeUsage()


class FakeCompletions:
    """只实现 call_llm 用到的那一条链：client.chat.completions.create(...)"""

    def create(self, model=None, messages=None, temperature=None):
        return FakeResp()


class FakeClient:
    class chat:
        completions = FakeCompletions()


HIT = {"id": "push-01", "topic": "推日", "text": "推日练胸、肩中束和三头。",
       "source": "谭成义 三分化·推日", "score": 0.991}


def fake_search(q, top_k=2, threshold=None):
    return [HIT], [HIT]


def reset():
    app._cache.clear()
    app._rate.clear()
    app._day["date"] = ""
    app._day["used"] = 0
    for k, v in list(app.STATS.items()):
        if isinstance(v, int):
            app.STATS[k] = 0
    app.STATS["started"] = time.time()


def calls():
    return app.STATS["llm_calls"]


app.search_kb_rerank = fake_search
# 每种 mode 的检索函数都要换成假的：不换的话 vector/hybrid 会真去调 embedding，
# 测试就变成了「联网且花钱」—— 而且它一失败，表现成 500，看起来像缓存键写错了。
app.search_kb_vector = fake_search
app.search_kb_hybrid = fake_search
app.search_kb_rewrite = fake_search
app.search_kb_keyword = fake_search
app.get_client = lambda timeout: FakeClient()

# 口令在本地 .env 里可能是配着的（公网部署要的），但测试必须跟本机配置无关 ——
# 否则「全部通过」只在没配口令的那台机器上成立。口令单独在第四节里显式设置。
app.ACCESS_CODE = ""
# 测试流量不许进真实问法日志
app.QUERY_LOG_ON = False
client = TestClient(app.app)


print("【一、缓存：同一句话第二次不再打上游】")
reset()
r1 = client.post("/rag", json={"question": "卧推的时候身体老是晃"})
r2 = client.post("/rag", json={"question": "卧推的时候身体老是晃"})
d1, d2 = r1.json(), r2.json()
check("第一次 200", r1.status_code == 200, r1.status_code)
check("第一次 cached=False", d1.get("cached") is False, str(d1.get("cached")))
check("第一次真调了模型", calls() == 1, calls())
check("第二次 cached=True", d2.get("cached") is True, str(d2.get("cached")))
check("第二次没再调模型", calls() == 1, calls())
check("第二次 tokens=0（命中没收钱就别报钱）", d2.get("tokens") == 0, d2.get("tokens"))
check("两次答案一致", d1.get("answer") == d2.get("answer"))
check("cache_hits 记了 1 次", app.STATS["cache_hits"] == 1, app.STATS["cache_hits"])

print("\n【二、缓存键必须带上 mode 和兜底开关】")
reset()
client.post("/rag?mode=rerank", json={"question": "推日练什么"})
client.post("/rag?mode=vector", json={"question": "推日练什么"})
check("换 mode 不复用答案", calls() == 2, calls())
client.post("/rag?mode=rerank&fallback=0", json={"question": "推日练什么"})
check("换 fallback 不复用答案", calls() == 3, calls())
r_back = client.post("/rag?mode=rerank&fallback=1", json={"question": "推日练什么"})
# 默认值就等于 fallback=1，所以这一下必须命中第一次的缓存 ——
# 如果实现把「没传」和「传 1」当成两个键，用户切一下开关就白花一次钱。
check("default 与 fallback=1 是同一个键", r_back.json().get("cached") is True,
      str(r_back.json().get("cached")))
check("键的空间是 3 条（rerank / vector / rerank+fb0）", len(app._cache) == 3, len(app._cache))

print("\n【三、限流：换着问法刷也会被拦，但缓存命中不算配额】")
reset()
old_per_min = app.RATE_LIMIT_PER_MIN
app.RATE_LIMIT_PER_MIN = 2
r1 = client.post("/rag", json={"question": "问题 A"})
r2 = client.post("/rag", json={"question": "问题 B"})
r3 = client.post("/rag", json={"question": "问题 C"})
check("第 1 次放行", r1.status_code == 200, r1.status_code)
check("第 2 次放行", r2.status_code == 200, r2.status_code)
check("第 3 次 429", r3.status_code == 429, r3.status_code)
check("429 带 Retry-After", r3.headers.get("retry-after") is not None,
      r3.headers.get("retry-after", ""))
check("429 的说明里有人话", "秒后再试" in r3.json().get("detail", ""),
      r3.json().get("detail", ""))
check("被拦的那次没打上游", calls() == 2, calls())
# 缓存命中必须绕过限流，否则页面上的示例点两次就 429
r_again = client.post("/rag", json={"question": "问题 A"})
check("超额后命中缓存仍能返回", r_again.status_code == 200, r_again.status_code)
check("并且标记为缓存", r_again.json().get("cached") is True)
app.RATE_LIMIT_PER_MIN = old_per_min

print("\n【四、口令：没配就放行，配了就只认对的】")
reset()
app.ACCESS_CODE = ""
r = client.post("/rag", json={"question": "没配口令"})
check("未配置口令时放行", r.status_code == 200, r.status_code)

app.ACCESS_CODE = "kanwo123"
reset()
r_no = client.post("/rag", json={"question": "没带口令"})
check("不带口令 401", r_no.status_code == 401, r_no.status_code)
r_bad = client.post("/rag", json={"question": "带错口令"},
                    headers={"X-Access-Code": "wrong"})
check("错口令 401", r_bad.status_code == 401, r_bad.status_code)
check("口令错时一次上游都没打", calls() == 0, calls())
r_ok = client.post("/rag", json={"question": "带对口令"},
                   headers={"X-Access-Code": "kanwo123"})
check("对的口令 200", r_ok.status_code == 200, r_ok.status_code)
check("对的口令真的打了上游", calls() == 1, calls())

# /chat 是另一个能直接打模型的口子，同样要挡
r_chat = client.post("/chat", json={"question": "x"})
check("/chat 也校验口令", r_chat.status_code == 401, r_chat.status_code)
app.ACCESS_CODE = ""

print("\n【五、/stats 的数字要对得上发生的事】")
reset()
old_per_min = app.RATE_LIMIT_PER_MIN
app.RATE_LIMIT_PER_MIN = 1
client.post("/rag", json={"question": "统计用 A"})
client.post("/rag", json={"question": "统计用 A"})      # 缓存命中
client.post("/rag", json={"question": "统计用 B"})      # 429
s = client.get("/stats").json()
app.RATE_LIMIT_PER_MIN = old_per_min
check("requests=3", s.get("requests") == 3, s.get("requests"))
check("cache_hits=1", s.get("cache_hits") == 1, s.get("cache_hits"))
check("rate_limited=1", s.get("rate_limited") == 1, s.get("rate_limited"))
check("llm_calls=1（只有真正打上游的那次）", s.get("llm_calls") == 1, s.get("llm_calls"))
check("命中率约 0.333", abs(s.get("cache_hit_rate", 0) - 1 / 3) < 0.01, s.get("cache_hit_rate"))
check("auth_required 是布尔", isinstance(s.get("auth_required"), bool),
      type(s.get("auth_required")).__name__)

print("\n【六、降级结果不许进缓存】")
reset()
orig_get_client = app.get_client


class BrokenCompletions:
    def create(self, model=None, messages=None, temperature=None):
        raise RuntimeError("upstream boom")     # 无状态码 => 不该重试，直接 502


class BrokenClient:
    class chat:
        completions = BrokenCompletions()


app.get_client = lambda timeout: BrokenClient()
r_deg = client.post("/rag", json={"question": "会降级的题"})
check("降级返回 200 且 degraded=True",
      r_deg.status_code == 200 and r_deg.json().get("degraded") is True,
      r_deg.json().get("degraded"))
check("降级结果没写进缓存", len(app._cache) == 0, len(app._cache))
app.get_client = orig_get_client

# ---------- 七、补全也要花钱，额度必须在它之前判 ----------
# 修复前 resolve_question 跑在 rate_check / daily_check 之前：短问句会先调一次
# 补全模型，然后才轮到额度判断。结果是「额度只拦得住生成，拦不住补全」——
# 伪造 X-Forwarded-For 再换 session_id，就能无限次让服务替调用方调模型。
print("\n【七、额度用尽时，补全不该先花一次钱】")
app.session_append("guard-sid", "推日练哪些部位", "推日练哪些部位", "推日")
called = []
orig_condense = app.condense
orig_daily = app.daily_check
app.get_client = lambda timeout: FakeClient()
app.condense = lambda q, hist: (called.append(q), "补全后的问题")[1]
app.daily_check = lambda: False
try:
    # 换个 XFF，避开前面几组测试已经用掉的每分钟额度，确保拦它的是「日额度」
    r = client.post("/rag", json={"question": "那上斜呢", "session_id": "guard-sid"},
                    headers={"X-Forwarded-For": "10.77.77.77"})
    check("额度满了返回 429", r.status_code == 429, r.status_code)
    check("condense 一次都没被调用（钱没花）", len(called) == 0, called)
finally:
    app.condense = orig_condense
    app.daily_check = orig_daily
    app.get_client = orig_get_client
    app.session_clear("guard-sid")

print()
if fails:
    print(f"失败 {len(fails)} 项：")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("全部通过")
