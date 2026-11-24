"""复合拆解的离线契约测试：不联网、不花钱、不要 key。

跟 test_multiturn 一个路子：模型拆得准不准是模型的事（那是 eval_compound.py
要量的），这里要钉死的是**纯逻辑**——解析、预筛、失败退回、门控不被稀释、
缓存键不串。这些错了，线上表现是「偶尔答非所问」，最难查。

跑法：uv run python test_decompose.py
"""
import main as app
from fastapi.testclient import TestClient

fails = []
def check(name, cond, extra=""):
    print(("  ok   " if cond else "  FAIL ") + name + ("" if cond else f"   -> {extra}"))
    if not cond:
        fails.append(name)

app.ACCESS_CODE = ""
app.RATE_LIMIT_PER_MIN = 100000
app.DAILY_QUOTA = 100000
# 测试流量不许进真实问法日志：两周后去看数据，一半是自己跑测试留下的假问法
app.QUERY_LOG_ON = False

SEEN_RANK = []          # 检索层每次收到的 query，用来证明「真的分了几路」


# ---------- 假检索层 ----------
# 故意让「长复合句」整体命中不了（模拟真实情况：一句话塞两个问题时，
# 精排分数被稀释），只有拆开之后的短句才能过线。这是拆解唯一的价值所在，
# 也正好是「拆解会不会把拒答打开」的风险点。
def fake_rank_one(q, q_vec=None):
    SEEN_RANK.append(q)
    long = len(q) > 14
    table = [
        ("上斜", "上胸", 0.98),
        ("卧推", "卧推技术", 0.98 if not long else 0.42),
        ("下拉", "高位下拉", 0.97 if not long else 0.40),
        ("蛋白", "饮食配合", 0.97),
        ("几次", "分化与频率", 0.96),
        ("深蹲", "深蹲", 0.99),
    ]
    out = []
    for i, (w, topic, sc) in enumerate(table):
        if w in q:
            out.append({"id": f"{topic}-01", "topic": topic, "text": f"{topic}的资料",
                        "source": "t", "score": sc})
    # 补几条低分的，模拟「召回永远有 12 条」这个事实
    for i in range(3):
        out.append({"id": f"noise-{i}", "topic": f"噪声{i}", "text": "", "source": "t",
                    "score": 0.2})
    out.sort(key=lambda r: r["score"], reverse=True)
    return out


class Resp:
    def __init__(self, text):
        msg = type("M", (), {"content": text})()
        self.choices = [type("C", (), {"message": msg})()]
        self.usage = type("U", (), {"total_tokens": 30})()
        self.model = "fake"


SPLIT = {
    "卧推的时候手该放多宽，高位下拉又该握多宽": ["卧推的时候手该放多宽", "高位下拉该握多宽"],
    "增肌一天要吃多少蛋白，一周练几次比较合适": ["增肌一天要吃多少蛋白", "增肌一周练几次比较合适"],
    "肌酸一天吃多少，训练前要不要喝咖啡": ["肌酸一天吃多少", "训练前要不要喝咖啡"],
}


def fake_condense_llm(messages, max_tokens=None):
    """假拆解模型。只认 SPLIT 表里的句子，其余一律「- 原句」。"""
    user = messages[-1]["content"]
    q = user.rsplit("【原句】", 1)[-1].strip()
    if q in SPLIT:
        return Resp("\n".join(f"- {s}" for s in SPLIT[q]))
    return Resp(f"- {q}")


saved_rank, saved_llm = app.rank_one, app.condense_llm
app.rank_one = fake_rank_one
app.condense_llm = fake_condense_llm

c = TestClient(app.app)


def fresh():
    app._sessions.clear()
    app._cache.clear()
    SEEN_RANK.clear()
    for k in ("decomp_calls", "decomp_failed", "decomp_paths", "llm_calls"):
        app.STATS[k] = 0


print("\n【一、parse_subs：四种输出格式都要认】")
check("标准「- 」", app.parse_subs("- 甲\n- 乙") == ["甲", "乙"], app.parse_subs("- 甲\n- 乙"))
check("裸破折号", app.parse_subs("-甲\n-乙") == ["甲", "乙"])
check("编号", app.parse_subs("1. 甲\n2. 乙") == ["甲", "乙"])
check("带句号引号", app.parse_subs("- 「甲。」") == ["甲"], app.parse_subs("- 「甲。」"))
check("空行忽略", app.parse_subs("\n- 甲\n\n") == ["甲"])
check("空文本给空列表", app.parse_subs("") == [])
check("None 不炸", app.parse_subs(None) == [])

print("\n【二、need_decompose：预筛只负责省调用，不负责判定】")
for q, exp in [
    ("卧推的时候手该放多宽，高位下拉又该握多宽", True),
    ("增肌一天要吃多少蛋白，一周练几次比较合适", True),
    ("深蹲膝盖要不要过脚尖", False),
    ("练腿日隔多久练一次比较好", False),
    ("一周练几次比较合适", False),
    ("卧推手肘该收着还是打开", False),
]:
    check(f"{'要拆' if exp else '不拆'}：{q}", app.need_decompose(q) is exp,
          app.need_decompose(q))
check("太短的句子直接挡掉（省钱）", app.need_decompose("那上斜呢") is False)

print("\n【三、decompose：三种「不该拆」都要返回 None】")
fresh()
check("模型说不用拆 → None", app.decompose("深蹲膝盖要不要过脚尖") is None)
check("拆出 1 条 → None", app.decompose("随便一句话") is None)
check("正常拆解 → 列表", app.decompose("卧推的时候手该放多宽，高位下拉又该握多宽")
      == ["卧推的时候手该放多宽", "高位下拉该握多宽"])

_orig = app.condense_llm
app.condense_llm = lambda messages, max_tokens=None: Resp("- 甲\n- 乙\n- 丙\n- 丁")
check("超过 3 路 → None", app.decompose("甲，乙，丙，丁") is None)
app.condense_llm = lambda messages, max_tokens=None: Resp("- 甲\n- 甲")
check("子问题里夹着原句 → None（说明它在复读，不是真拆）",
      app.decompose("甲") is None)
app.condense_llm = lambda messages, max_tokens=None: (_ for _ in ()).throw(RuntimeError("upstream boom"))
check("上游炸了 → None，不抛给调用方", app.decompose("卧推握多宽，下拉握多宽") is None)
check("失败被计数", app.STATS["decomp_failed"] >= 1, app.STATS["decomp_failed"])
app.condense_llm = _orig

print("\n【四、search_kb_multi：每路各自过门槛，融合后同话题只留一条】")
fresh()
q = "卧推的时候手该放多宽，高位下拉又该握多宽"
subs = ["卧推的时候手该放多宽", "高位下拉该握多宽"]
picked, ranking = app.search_kb_multi(q, subs)
topics = [p["topic"] for p in picked]
check("主查询过不了线、子查询能过 → 仍然答", len(picked) >= 2, topics)
check("两个话题都在", set(topics) >= {"卧推技术", "高位下拉"}, topics)
check("没有重复话题", len(topics) == len(set(topics)), topics)
check("每条都带着「被几路命中」", all("paths" in p for p in picked), picked)
check("跑了几路就检索几次", len(SEEN_RANK) == 3, SEEN_RANK)
check("判定的 top1 是各路里分最高的那条", ranking[0]["score"] >= 0.97,
      ranking[0]["score"])

print("\n【五、最关键：拆解不能把拒答打开】")
fresh()
# 两路都不含知识库里任何话题 → 每路 top1 都是 0.2，必须拒
bad = "肌酸一天吃多少，训练前要不要喝咖啡"
picked, ranking = app.search_kb_multi(bad, SPLIT[bad])
check("两路都不过线 → 一条都不给", picked == [], [p["topic"] for p in picked])
check("但 ranking 还留着（用于判 uncertain / 兜底）", ranking and ranking[0]["score"] < 0.95,
      ranking[0]["score"] if ranking else None)

# 一路过、一路不过：只该答过线的那一路，不能因为「有一路过」就把不过的那条也塞进来
mixed = "深蹲膝盖要不要过脚尖，肌酸一天吃多少"
picked, ranking = app.search_kb_multi(mixed, ["深蹲膝盖要不要过脚尖", "肌酸一天吃多少"])
check("只保留过线的那一路", [p["topic"] for p in picked] == ["深蹲"],
      [p["topic"] for p in picked])

print("\n【六、/rag?mode=multi 端到端：字段、缓存键、口子一致性】")
fresh()
d = c.post("/rag", json={"question": q}, params={"mode": "multi"}).json()
check("返回 sub_questions", d.get("sub_questions") == subs, d.get("sub_questions"))
check("mode 记为 multi", d["mode"] == "multi")
check("timings 里有 decomp_ms", "decomp_ms" in d["timings"], d.get("timings"))
check("答到了两个话题", len({s["topic"] for s in d["sources"]}) >= 2,
      [s["topic"] for s in d["sources"]])

fresh()
a = c.post("/rag", json={"question": q}, params={"mode": "rerank"}).json()
b = c.post("/rag", json={"question": q}, params={"mode": "multi"}).json()
check("multi 和 rerank 的缓存互不串", b["cached"] is False, b["cached"])
check("multi 的 sub_questions 不为空", b["sub_questions"] == subs)
check("rerank 的 sub_questions 是空", a["sub_questions"] == [], a["sub_questions"])
check("两种模式答案不同（分了几路确实不一样）", a["sources"] != b["sources"],
      ([s["topic"] for s in a["sources"]], [s["topic"] for s in b["sources"]]))

fresh()
d1 = c.post("/rag", json={"question": q}, params={"mode": "multi"}).json()
d2 = c.post("/rag", json={"question": q}, params={"mode": "multi"}).json()
check("第二次命中缓存", d2["cached"] is True, d2["cached"])
check("缓存命中也要带 sub_questions（不能写成空）", d2["sub_questions"] == subs,
      d2["sub_questions"])
check("缓存命中不再检索", len(SEEN_RANK) == 3, SEEN_RANK)

print("\n【七、负例端到端：拆了也还是拒】")
fresh()
d = c.post("/rag", json={"question": bad, "session_id": "m1"},
           params={"mode": "multi", "fallback": "0"}).json()
check("仍然拒答", d["sources"] == [], d["sources"])
check("虽然拆成了两路（说明真拆了，不是没跑）", len(d["sub_questions"]) == 2,
      d["sub_questions"])

print("\n【八、DECOMP_MODE=always：绕过预筛，但拆不拆仍由模型定】")
fresh()
old = app.DECOMP_MODE
app.DECOMP_MODE = "always"
try:
    n0 = app.STATS["decomp_calls"]
    c.post("/rag", json={"question": "卧推手放多宽"}, params={"mode": "multi"})
    check("always 下短句也送进拆解了（预筛被绕过）",
          app.STATS["decomp_calls"] > n0, (n0, app.STATS["decomp_calls"]))
    check("模型说不用拆 → 仍然单路",
          c.post("/rag", json={"question": "卧推手放多宽"},
                 params={"mode": "multi"}).json()["sub_questions"] == [])
finally:
    app.DECOMP_MODE = old

print("\n【九、mode=multi 不带 session_id：不该建会话】")
fresh()
c.post("/rag", json={"question": q}, params={"mode": "multi"})
check("没建会话", len(app._sessions) == 0, len(app._sessions))

app.rank_one, app.condense_llm = saved_rank, saved_llm

print()
if fails:
    print(f"失败 {len(fails)} 项：" + "；".join(fails))
    raise SystemExit(1)
print("全部通过")
