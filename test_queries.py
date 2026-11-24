"""问法日志的离线契约测试：不联网、不花钱。

为什么单独一个文件：日志是**旁路**，最危险的失败不是「没记上」，是
「记日志这一步把答题搞挂了」，以及「把测试流量记进真实日志」——
后者更隐蔽，等你两周后去看数据，里面一半是自己跑测试留下的假问法。

钉住六件事：
    1. 答题会记、拒答也记（拒答那批才是选题清单）
    2. **缓存命中也记**（重复提问是最强的高频信号）
    3. src 能透传（分得清公众号来的和自测的）
    4. 补全过的才带 rq
    5. /queries 要口令，refused=1 只给答不出的
    6. 写盘炸了不许影响答题

跑法：uv run python test_queries.py
"""
import json
import tempfile
from pathlib import Path

import main as app
from fastapi.testclient import TestClient

fails = []


def check(name, cond, extra=""):
    print(("  ok   " if cond else "  FAIL ") + name + ("" if cond else f"   -> {extra}"))
    if not cond:
        fails.append(name)


# 关掉会干扰的外部条件
app.ACCESS_CODE = ""
app.RATE_LIMIT_PER_MIN = 100000
app.DAILY_QUOTA = 100000
app.QUERY_LOG_ON = True
TMP = Path(tempfile.mkdtemp()) / "qa.jsonl"
app.QUERY_LOG = TMP


def fake_search(q, top_k=3, threshold=0.95):
    """问什么就答什么；带「诗」的一律答不出，用来造拒答样本。"""
    if "诗" in q or "月" in q:
        return [], [{"id": "x", "topic": "无关", "text": "", "score": 0.30}]
    topic = "推日" if "推日" in q else ("上胸" if "上斜" in q else "深蹲")
    hit = {"id": f"{topic}-01", "topic": topic, "text": f"{topic}的资料", "score": 0.99}
    return [hit], [hit]


class Resp:
    def __init__(self, text):
        msg = type("M", (), {"content": text})()
        self.choices = [type("C", (), {"message": msg})()]
        self.usage = type("U", (), {"total_tokens": 11})()
        self.model = "fake"


class Completions:
    def create(self, model=None, messages=None, temperature=None, **kw):
        system = messages[0]["content"]
        user = messages[-1]["content"]
        if "句子补全" in system:
            latest = user.rsplit("【最新一句话】", 1)[-1].strip()
            if latest == "那做几组":
                return Resp("推日做几组")
            return Resp(latest)
        return Resp("这是假答案")


class Client:
    class chat:
        completions = Completions()


app.search_kb_rerank = fake_search
app.get_client = lambda timeout: Client()

c = TestClient(app.app)


def fresh():
    if TMP.exists():
        TMP.unlink()
    app._cache.clear()
    app._sessions.clear()
    app._cond.clear()
    app.QUERY_LOG_RING.clear()


def lines():
    if not TMP.exists():
        return []
    raw = TMP.read_text(encoding="utf-8").strip()
    return [json.loads(x) for x in raw.splitlines() if x.strip()] if raw else []


def ask(q, sid=None, **kw):
    body = {"question": q}
    if sid:
        body["session_id"] = sid
    return c.post("/rag", json=body, params=kw).json()


print("【一、答题记一条，字段对得上】")
fresh()
d = ask("深蹲膝盖要不要过脚尖", src="mp")
recs = lines()
check("落了一行", len(recs) == 1, len(recs))
r0 = recs[0] if recs else {}
check("记录原话", r0.get("q") == "深蹲膝盖要不要过脚尖", r0.get("q"))
check("ok=true", r0.get("ok") is True, r0.get("ok"))
check("记了命中话题", r0.get("topics") == ["深蹲"], r0.get("topics"))
check("记了精排分", r0.get("top") == 0.99, r0.get("top"))
check("src 透传 mp", r0.get("src") == "mp", r0.get("src"))
check("没记 IP/UA", not any(k in r0 for k in ("ip", "ua", "user_agent")), list(r0))
check("有耗时", isinstance(r0.get("ms"), int), r0.get("ms"))

print("\n【二、拒答的三种状态要分开：检索没命中 / 兜底救回 / 用户侧有没有答案】")
fresh()
ask("帮我写一首关于月亮的诗")
recs = lines()
check("拒答也落了一行", len(recs) == 1, len(recs))
r0 = recs[0] if recs else {}
check("检索层没命中（ret=false）", r0.get("ret") is False, r0.get("ret"))
check("走了生成层兜底（fb=true）", r0.get("fb") is True, r0.get("fb"))
check("用户侧算答出来（ok=true，模型硬撑的）", r0.get("ok") is True, r0.get("ok"))

fresh()
ask("帮我写一首关于月亮的诗", fallback=0)
r0 = lines()[0] if lines() else {}
check("关掉兜底后真的拒答（ok=false）", r0.get("ok") is False, r0.get("ok"))
check("也标了 ret=false", r0.get("ret") is False, r0.get("ret"))
check("fb=false", r0.get("fb") is False, r0.get("fb"))
check("没有 topics", r0.get("topics") == [], r0.get("topics"))

print("\n【三、缓存命中也记（重复提问是最强信号）】")
fresh()
ask("推日都练哪些部位")
ask("推日都练哪些部位")
recs = lines()
check("两次都记了", len(recs) == 2, len(recs))
check("第二次标 cached", recs and recs[-1].get("cached") is True, recs[-1].get("cached") if recs else None)
check("第二次 ret 仍为 True", recs and recs[-1].get("ret") is True, None)

print("\n【四、补全过的才带 rq】")
fresh()
ask("推日都练哪些部位", sid="s-log-1")     # 第一轮，完整句
ask("那做几组", sid="s-log-1")            # 第二轮，省略追问
recs = lines()
check("两轮都记了", len(recs) == 2, len(recs))
check("完整句没有 rq", "rq" not in (recs[0] if recs else {}), recs[0] if recs else None)
check("追问记了 rq", recs and recs[-1].get("rq") == "推日做几组", recs[-1].get("rq") if recs else None)
check("追问的原话保留", recs and recs[-1].get("q") == "那做几组", recs[-1].get("q") if recs else None)

print("\n【五、/queries 要口令，支持过滤与导出】")
app.ACCESS_CODE = "secret-123"
try:
    r = c.get("/queries")
    check("无口令 401", r.status_code == 401, r.status_code)
    r = c.get("/queries", headers={"x-access-code": "wrong"})
    check("错口令 401", r.status_code == 401, r.status_code)
    r = c.get("/queries", headers={"x-access-code": "secret-123"})
    check("对口令 200", r.status_code == 200, r.status_code)
    body = r.json()
    check("返回条数正确", body.get("n") == 2, body.get("n"))
    check("有 by_src 汇总", isinstance(body.get("by_src"), dict), body.get("by_src"))
    check("items 是列表", isinstance(body.get("items"), list), type(body.get("items")))

    # 补一条真拒答（关兜底），用来测过滤 —— 注意要带口令，否则 401 写不进去
    c.post("/rag", json={"question": "帮我写一首关于月亮的诗"},
           params={"fallback": "0"}, headers={"x-access-code": "secret-123"})
    r = c.get("/queries", params={"refused": 1}, headers={"x-access-code": "secret-123"})
    got = [x["q"] for x in r.json()["items"]]
    check("refused=1 只给检索没命中的", got == ["帮我写一首关于月亮的诗"], got)

    r = c.get("/queries", params={"fmt": "jsonl"}, headers={"x-access-code": "secret-123"})
    check("jsonl 是 ndjson", "ndjson" in r.headers.get("content-type", ""),
          r.headers.get("content-type"))
    check("jsonl 行数对", len([x for x in r.text.strip().splitlines() if x]) == 3, r.text[:60])
finally:
    app.ACCESS_CODE = ""

print("\n【六、关掉就不写；写盘炸了不影响答题】")
fresh()
app.QUERY_LOG_ON = False
d = ask("深蹲膝盖要不要过脚尖")
check("关了之后不落盘", len(lines()) == 0, len(lines()))
check("但答题照常", d.get("answer") == "这是假答案", d.get("answer"))
app.QUERY_LOG_ON = True

bad = Path(tempfile.mkdtemp()) / "no_such_dir" / "qa.jsonl"
old = app.QUERY_LOG
app.QUERY_LOG = bad
d = ask("深蹲膝盖要不要过脚尖")
check("目录不存在时答题不崩", d.get("answer") == "这是假答案", d.get("answer"))
check("内存缓冲还在", len(app.QUERY_LOG_RING) >= 1, len(app.QUERY_LOG_RING))
app.QUERY_LOG = old

print("\n【七、/meta 声明了这个能力】")
m = c.get("/meta").json()
check("meta.query_log 为 True", m.get("query_log") is True, m.get("query_log"))

print()
if fails:
    print(f"{len(fails)} 项失败：{fails}")
    raise SystemExit(1)
print("全部通过")
