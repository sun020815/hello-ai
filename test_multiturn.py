"""多轮会话的离线契约测试：不联网、不花钱、不要 key。

为什么单独一个文件：多轮最容易错的不是「补全得准不准」（那是模型的事），
而是**状态串了**——会话记错、缓存键用错、两个人问同一句话互相覆盖。
这些全是纯逻辑错误，用假客户端就能钉死，一次真实调用都不用花。

跑法：uv run python test_multiturn.py
"""
import main as app
from fastapi.testclient import TestClient

fails = []
def check(name, cond, extra=""):
    print(("  ok   " if cond else "  FAIL ") + name + ("" if cond else f"   -> {extra}"))
    if not cond:
        fails.append(name)

# 关掉一切会干扰的外部条件：口令、限流、日额度
app.ACCESS_CODE = ""
app.RATE_LIMIT_PER_MIN = 100000
app.DAILY_QUOTA = 100000

SEEN = []          # 检索层真正收到的 query，用来证明「用的是补全后的那句」
TOPIC_BY_WORD = [
    ("上斜", "上胸"), ("推日", "推日"), ("拉日", "拉日"), ("深蹲", "深蹲"),
    ("二头", "拉日"), ("卧推", "卧推技术"), ("硬拉", "硬拉"),
]


def fake_search(q, top_k=3, threshold=0.95):
    SEEN.append(q)
    topic = next((t for w, t in TOPIC_BY_WORD if w in q), None)
    if topic is None:
        return [], [{"id": "x", "topic": "无关", "text": "", "score": 0.2}]
    hit = {"id": f"{topic}-01", "topic": topic, "text": f"{topic}的资料", "score": 0.99}
    return [hit], [hit]


def last_question(user):
    """补全 prompt 的最后一段就是「最新一句话」。"""
    return user.rsplit("【最新一句话】", 1)[-1].strip()


def history_part(user):
    return user.split("【最新一句话】", 1)[0]


class Resp:
    def __init__(self, text):
        msg = type("M", (), {"content": text})()
        self.choices = [type("C", (), {"message": msg})()]
        self.usage = type("U", (), {"total_tokens": 11})()
        self.model = "fake"


class Completions:
    """假上游。靠第一条 system 消息区分「补全」和「作答」这两类调用。

    补全的返回是**按上文决定**的，这一点是故意的：测「缓存键必须用补全后的问题」
    那一条时，需要同一句「那上斜呢」在不同上文下补出不同结果。
    """
    def create(self, model=None, messages=None, temperature=None, **kw):
        # **kw 是必须的：补全走的是 max_tokens + extra_body 那条快路，
        # 假客户端不接这两个参数就会被判成「供应商不支持」，测的全是退化路径。
        # 只有补全那条路会带 extra_body；生成那条不带，不能一起断言
        if "extra_body" in kw:
            assert not kw["extra_body"].get("enable_thinking", True), \
                "补全必须关推理，否则白等几秒"
        system = messages[0]["content"]
        user = messages[-1]["content"]
        if "句子补全" in system:
            # 真实的补全模型对「本来完整的句子」会原样返回（prompt 第 4 条要求），
            # 假客户端必须照这个行为演，否则测出来的缓存行为是假的。
            latest = last_question(user)
            if latest == "那上斜呢":
                return Resp("上斜卧推怎么练")
            if latest.startswith("做几组"):
                return Resp(("推日" if "推日" in history_part(user) else "拉日") + "做几组")
            return Resp(latest)
        return Resp("这是假答案")


class Client:
    class chat:
        completions = Completions()


app.search_kb_rerank = fake_search
app.get_client = lambda timeout: Client()


def fresh():
    """每个用例前清空会话与缓存，避免互相污染。"""
    app._sessions.clear()
    app._cache.clear()
    app._cond.clear()
    SEEN.clear()
    app.STATS["condense_calls"] = 0
    app._cond_fast_ok[0] = True


c = TestClient(app.app)


def ask(q, sid=None, **kw):
    body = {"question": q}
    if sid:
        body["session_id"] = sid
    return c.post("/rag", json=body, params=kw).json()


print("\n【一、不传 session_id：行为必须与单轮完全一致】")
fresh()
d = ask("深蹲膝盖要不要过脚尖")
check("resolved_question 就是原话", d["resolved_question"] == "深蹲膝盖要不要过脚尖", d["resolved_question"])
check("condensed 为 False", d["condensed"] is False)
check("history_turns 为 0", d["history_turns"] == 0, d["history_turns"])
check("session_id 为 None", d["session_id"] is None, d["session_id"])
check("没有建任何会话", len(app._sessions) == 0, len(app._sessions))
check("没有花补全调用", app.STATS["condense_calls"] == 0, app.STATS["condense_calls"])

print("\n【二、第一轮：完整长句不该触发补全】")
fresh()
d = ask("推日都练哪些部位平时怎么安排", sid="s1")
check("第一轮不补全", d["condensed"] is False)
check("resolved 等于原话", d["resolved_question"] == "推日都练哪些部位平时怎么安排")
check("history_turns 是 0（本轮不把自己算进上文）", d["history_turns"] == 0, d["history_turns"])
check("会话记了 1 轮", len(app.session_get("s1")) == 1, len(app.session_get("s1")))
check("会话里存了命中话题", app.session_get("s1")[0]["topic"] == "推日", app.session_get("s1"))

print("\n【三、第二轮短追问：补全并真的用它去检索】")
fresh()
ask("推日都练哪些部位平时怎么安排", sid="s1")
SEEN.clear()
d = ask("那上斜呢", sid="s1")
check("标记为补全过", d["condensed"] is True)
check("resolved 是补全后的句子", d["resolved_question"] == "上斜卧推怎么练", d["resolved_question"])
check("检索层收到的是补全后的问题", SEEN == ["上斜卧推怎么练"], SEEN)
check("history_turns 是 1", d["history_turns"] == 1, d["history_turns"])
check("答的是上胸（跟着补全结果走）", d["sources"] and d["sources"][0]["topic"] == "上胸", d.get("sources"))

print("\n【四、最关键的一条：同一句追问，不同上文不能共用缓存】")
fresh()
ask("推日都练哪些部位平时怎么安排", sid="a")
da = ask("做几组", sid="a")
ask("拉日都练哪些部位平时怎么安排", sid="b")
db = ask("做几组", sid="b")
check("两个会话补全出不同的问题", da["resolved_question"] != db["resolved_question"],
      (da["resolved_question"], db["resolved_question"]))
check("第二个会话没有被第一个的缓存吃掉", db["cached"] is False, db["cached"])
check("两次答的是不同话题", da["sources"][0]["topic"] != db["sources"][0]["topic"],
      (da["sources"][0]["topic"], db["sources"][0]["topic"]))
check("缓存里是两条独立的键", len(app._cache) >= 4, len(app._cache))

print("\n【五、缓存命中：会话号要盖成当前这个，不能带出旧号】")
fresh()
d1 = ask("推日都练哪些部位平时怎么安排", sid="x")
d2 = ask("推日都练哪些部位平时怎么安排", sid="y")
check("第二次命中缓存", d2["cached"] is True, d2["cached"])
check("session_id 被盖成 y", d2["session_id"] == "y", d2["session_id"])
check("tokens 记 0", d2["tokens"] == 0, d2["tokens"])
check("y 也记了一轮", len(app.session_get("y")) == 1, len(app.session_get("y")))

print("\n【六、拒答也要进会话（下一轮的上下文来自这句被拒的话）】")
fresh()
d = ask("帮我写一首关于春天的诗", sid="n1", fallback="0")
check("确实是拒答", d["sources"] == [], d["sources"])
hist = app.session_get("n1")
check("拒答仍记了 1 轮", len(hist) == 1, hist)
check("拒答轮的话题为空", hist and hist[0]["topic"] is None, hist)

print("\n【七、CONDENSE_MODE=off：退回单轮】")
fresh()
app.CONDENSE_MODE = "off"
try:
    ask("推日都练哪些部位平时怎么安排", sid="s2")
    d = ask("那上斜呢", sid="s2")
    check("不补全", d["condensed"] is False)
    check("resolved 就是原话", d["resolved_question"] == "那上斜呢", d["resolved_question"])
    check("没花补全调用", app.STATS["condense_calls"] == 0, app.STATS["condense_calls"])
finally:
    app.CONDENSE_MODE = "auto"

print("\n【八、会话轮数上限：只留最近 N 轮】")
fresh()
old_n = app.SESSION_TURNS
app.SESSION_TURNS = 3
try:
    for i in range(6):
        ask(f"深蹲第{i}个问题要练多久", sid="s3")
    check("只留 3 轮", len(app.session_get("s3")) == 3, len(app.session_get("s3")))
    check("留下的是最近那几轮", app.session_get("s3")[-1]["q"].startswith("深蹲第5个"), app.session_get("s3"))
finally:
    app.SESSION_TURNS = old_n

print("\n【九、DELETE /session 清空；不存在的会话不报错】")
fresh()
ask("深蹲膝盖要不要过脚尖", sid="s4")
r = c.delete("/session/s4")
check("删除返回 1", r.json()["deleted"] == 1, r.json())
check("删完就空了", app.session_get("s4") == [])
check("再删一次返回 0 而不是报错", c.delete("/session/s4").json()["deleted"] == 0)
check("GET 不存在的会话返回空列表", c.get("/session/nope").json()["n"] == 0)

print("\n【十、会话 TTL：过期就当没有上文】")
fresh()
ask("推日都练哪些部位平时怎么安排", sid="s5")
app._sessions["s5"]["ts"] -= (app.SESSION_TTL + 10)
d = ask("那上斜呢", sid="s5")
check("过期后不算上文", d["history_turns"] == 0, d["history_turns"])
check("不补全", d["condensed"] is False)

print("\n【十一、补全失败：返回 None，并由上层用「上一轮 + 这一句」兜住】")
fresh()
ask("推日都练哪些部位平时怎么安排", sid="s6")


class Boom:
    def create(self, model=None, messages=None, temperature=None, **kw):
        raise RuntimeError("upstream boom")


saved = app.get_client
app.get_client = lambda timeout: type("C", (), {"chat": type("Ch", (), {"completions": Boom()})()})()
try:
    hist = app.session_get("s6")
    check("condense 失败返回 None（不是原话）", app.condense("那上斜呢", hist) is None)
    q, n, changed = app.resolve_question("那上斜呢", "s6")
    check("退路是「上一轮 + 这一句」", q.endswith("那上斜呢") and "推日" in q, q)
    check("退路也算改写（不是原话）", changed is True)
    check("失败被计数", app.STATS["condense_failed"] >= 1, app.STATS["condense_failed"])
finally:
    app.get_client = saved

print("\n【十二、检索层调用次数：命中缓存时不再检索】")
fresh()
ask("推日都练哪些部位平时怎么安排", sid="s7")
n1, c1 = len(SEEN), app.STATS["condense_calls"]
d = ask("推日都练哪些部位平时怎么安排", sid="s7")
check("第二次没有新增检索", len(SEEN) == n1, (n1, len(SEEN)))
check("第二次命中答案缓存", d["cached"] is True, d["cached"])
check("补全结果被缓存（没再花调用）", app.STATS["condense_calls"] == c1,
      (c1, app.STATS["condense_calls"]))

print()
if fails:
    print(f"失败 {len(fails)} 项：" + "；".join(fails))
    raise SystemExit(1)
print("全部通过")
