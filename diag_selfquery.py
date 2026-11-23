"""体检：每条 chunk 自己的问法，能不能被自己拿回来。

doc2query 原本是为了给精排补输入，结果没启用（拒答换命中，净亏）。
但那 420 个问法本身是一份现成的「反向考题」：
它们是「这条资料应该能回答的问题」，理应由这条资料自己拿回来。

但「自己拿回来」有两个层次，混在一起就会得出错误结论：
  话题级——用户这么问，这个**话题**里有没有一条能接住？  接不住 = 语料真的空洞
  chunk 级——接住的是不是**它自己**那条？              不是 = 兄弟冗余 / 主题句不聚焦

第一版脚本只算了 chunk 级，结果 18% 被判成「未召回」。看着像语料烂了，
实际大部分是「这个动作做几组几次」被同话题讲组次的那条抢走了 —— 用户照样有答案，
只是答它的不是出题的那条。拿这个数字去改语料，会朝错误方向改。

所以本脚本分三层报：
  1. 话题级命中：语料覆盖能力（能不能回答）
  2. chunk 级命中：主题句聚焦度（该不该由这条答）
  3. 五类归因：把「真没人答」从「兄弟答了」里摘出来

三类真病，三个修法：
  自己答但低分   主题句跑偏：reranker 读完不认为这段在答这个问题
  异话题抢答     话题边界模糊，或这个问法本来就属于别的主题
  没人答         语料空洞（本轮预期极少，因为问法就是从正文生成的）

局限：这些问法是模型对着正文写的，天生「顺着这条资料」。
分数高不能证明用户会这么问，只能证明「这条资料至少接得住一种正常问法」。
本脚本用来找**明显接不住的**，不是用来算检索准确率。

全流程走缓存（.selfquery.json），断了能续跑，不重复花钱。
"""

import json
import sys
from pathlib import Path

import main as app

HERE = Path(__file__).parent
CACHE = HERE / ".selfquery.json"
TH = app.RERANK_THRESHOLD
LOW = 0.55          # 「贴着门槛」也算隐患，0.50 是生死线，0.55 是观察线
BATCH = 16


# ---------- 缓存 ----------

def load_cache():
    if CACHE.exists():
        try:
            return json.loads(CACHE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"qvecs": {}, "reranks": {}}


def save_cache(c):
    CACHE.write_text(json.dumps(c, ensure_ascii=False), encoding="utf-8")


# ---------- 计算 ----------

def fill_vectors(cache, questions):
    """批量补向量。一条一个请求最容易被限流，16 条一批。"""
    qvecs = cache["qvecs"]
    todo = [q for q in questions if q not in qvecs]
    if not todo:
        return qvecs
    print(f"  向量：需新算 {len(todo)} 条", flush=True)
    for i in range(0, len(todo), BATCH):
        part = todo[i:i + BATCH]
        for q, v in zip(part, app.embed(part)):
            qvecs[q] = v
        save_cache(cache)
        print(f"    {min(i + BATCH, len(todo))}/{len(todo)}", flush=True)
    return qvecs


def fill_reranks(cache, questions):
    """对每个问法：全库向量取前 12 -> 精排 -> {chunk_id: score}。"""
    qvecs = cache["qvecs"]
    rr = cache["reranks"]
    todo = [q for q in questions if q not in rr]
    if not todo:
        return rr
    print(f"  精排：需新算 {len(todo)} 条", flush=True)

    kb_vecs = app.get_kb_vectors()
    assert cache.get("kb_size", len(app.KB)) == len(app.KB), "语料条数变了，旧缓存作废"

    for i, q in enumerate(todo, 1):
        recalled = sorted(
            ((app.cosine(qvecs[q], kv), d) for d, kv in zip(app.KB, kb_vecs)),
            key=lambda x: x[0], reverse=True,
        )[: app.RECALL_K]
        cand = [d for _, d in recalled]
        out = app.rerank(q, [app.rerank_text(d) for d in cand])
        rr[q] = {cand[s["index"]]["id"]: s["score"] for s in out}
        if i % 20 == 0 or i == len(todo):
            save_cache(cache)
            print(f"    {i}/{len(todo)}", flush=True)
    return rr


# ---------- 分析 ----------

def vec_rank_of(q, cid, qvecs, kb_vecs):
    """自己的 chunk 在全库向量里排第几（判断是表示层问题还是精排问题）。"""
    if q not in qvecs:
        return None
    sims = sorted(
        ((app.cosine(qvecs[q], kv), d["id"]) for d, kv in zip(app.KB, kb_vecs)),
        reverse=True,
    )
    for rank, (_, x) in enumerate(sims, 1):
        if x == cid:
            return rank
    return None


def classify(r):
    """五类归因。优先级：先看这条自己答没答，再看是谁答的。"""
    if r["top_id"] == r["cid"]:
        return "自己答" if r["self_score"] >= TH else "自己答但低分"
    if r["top_topic"] == r["topic"]:
        return "同话题兄弟答"
    if r["top_score"] < TH:
        return "没人答"
    return "异话题抢答"


ORDER = ["自己答", "自己答但低分", "同话题兄弟答", "异话题抢答", "没人答"]


def main():
    queries = app.get_kb_queries()
    if not queries:
        print("没有 kb_queries.json，先跑 build_doc2query.py")
        return

    # --limit N 只跑前 N 条，用来验证链路通不通，别一上来就烧 400 次调用
    if "--limit" in sys.argv:
        k = int(sys.argv[sys.argv.index("--limit") + 1])
        queries = dict(list(queries.items())[:k])
        print(f"（只测前 {k} 条）")

    questions = [q for qs in queries.values() for q in qs]
    topic_of = {d["id"]: d["topic"] for d in app.KB}
    print(f"语料 {len(app.KB)} 条，问法 {len(questions)} 个"
          f"（平均每条 {len(questions) / len(queries):.1f} 个）\n")

    cache = load_cache()
    print("=== 计算（缓存命中就不重复花钱）===")
    fill_vectors(cache, questions)
    fill_reranks(cache, questions)
    cache["kb_size"] = len(app.KB)
    save_cache(cache)

    qvecs, rr = cache["qvecs"], cache["reranks"]
    kb_vecs = app.get_kb_vectors()

    # ---- 逐问法定性 ----
    rows = []
    for cid, qs in queries.items():
        for q in qs:
            scores = rr.get(q) or {}
            ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
            self_rank = next((i for i, (x, _) in enumerate(ranked, 1) if x == cid), 99)
            top_id, top_score = (ranked[0] if ranked else (None, 0.0))
            rows.append({
                "cid": cid,
                "topic": topic_of[cid],
                "q": q,
                "self_score": scores.get(cid),       # None = 自己没进召回池
                "self_rank": self_rank,
                "top_id": top_id,
                "top_topic": topic_of.get(top_id),
                "top_score": top_score,
                "vec_rank": vec_rank_of(q, cid, qvecs, kb_vecs),
            })
    for r in rows:
        r["dx"] = classify(r)

    n = len(rows)
    cnt = {k: sum(1 for r in rows if r["dx"] == k) for k in ORDER}

    # 话题级：top1 是不是落在这个问法所属话题里
    topic_hit = sum(1 for r in rows if r["top_topic"] == r["topic"])
    self_hit = sum(1 for r in rows if r["top_id"] == r["cid"])
    passed = sum(1 for r in rows if (r["self_score"] or 0) >= TH)

    print("\n" + "=" * 76)
    print("【两个口径，分开看】")
    print("-" * 76)
    print(f"  话题级命中（这个问题，这个赛道有人接）  {topic_hit}/{n}  "
          f"({topic_hit / n * 100:.1f}%)")
    print(f"  chunk 级命中（接住的是出题那条自己）    {self_hit}/{n}  "
          f"({self_hit / n * 100:.1f}%)")
    print(f"  自己分数过门槛 {TH:.2f}                 {passed}/{n}  "
          f"({passed / n * 100:.1f}%)")

    print("\n【五类归因】")
    print("-" * 76)
    gloss = {
        "自己答": "正常，这条自己的问法自己接住，且过门槛",
        "自己答但低分": "**真病**：答的是它，但精排不认，主题句没对着问题说话",
        "同话题兄弟答": "不是缺陷：兄弟条答了，用户照样有答案，只是这条不聚焦",
        "异话题抢答": "**要查**：别的话题分更高，话题边界可能糊了",
        "没人答": "**真病**：整库都没接住，语料或表达有洞",
    }
    for k in ORDER:
        print(f"  {k:<12}{cnt[k]:>4} 条  ({cnt[k] / n * 100:>5.1f}%)   {gloss[k]}")

    scores = sorted(r["self_score"] for r in rows if r["self_score"] is not None)
    if scores:
        qq = lambda p: scores[min(len(scores) - 1, int(len(scores) * p))]
        print(f"\n  自评分数分位  p10={qq(0.10):.3f}  中位={qq(0.50):.3f}  p90={qq(0.90):.3f}")

    # 召回池外的，按「谁接的」再拆一次 —— 这是第一版误判的地方
    out_pool = [r for r in rows if r["self_score"] is None]
    if out_pool:
        same = sum(1 for r in out_pool if r["top_topic"] == r["topic"])
        print(f"\n  自己没进召回池的 {len(out_pool)} 条里："
              f"{same} 条是同话题兄弟接的（不算缺陷），{len(out_pool) - same} 条是别的话题接的")
    print("=" * 76)

    # ---- 按 chunk 汇总 ----
    by_chunk = {}
    for r in rows:
        by_chunk.setdefault(r["cid"], []).append(r)

    summary = []
    for cid, rs in by_chunk.items():
        ss = [r["self_score"] if r["self_score"] is not None else 0.0 for r in rs]
        bad = [r for r in rs if r["dx"] in ("自己答但低分", "没人答")]
        summary.append({
            "cid": cid,
            "topic": topic_of[cid],
            "n": len(rs),
            "min": min(ss),
            "avg": sum(ss) / len(ss),
            "hit1": sum(1 for r in rs if r["top_id"] == cid),
            "topic_hit": sum(1 for r in rs if r["top_topic"] == r["topic"]),
            "bad": len(bad),
            "rs": rs,
            "text": next(d["text"] for d in app.KB if d["id"] == cid),
        })
    # 有真病的排最前，其次按平均分
    summary.sort(key=lambda s: (-s["bad"], s["min"], s["avg"]))

    sick = [s for s in summary if s["bad"]]
    print(f"\n【真病清单】{len(sick)} 条 chunk 至少有一个问法自己接不住")
    print("-" * 76)
    for i, s in enumerate(sick, 1):
        marks = " ".join(
            (f"{r['self_score']:.2f}" if r["self_score"] is not None else "--")
            + ("!" if r["dx"] in ("自己答但低分", "没人答") else "")
            for r in s["rs"]
        )
        print(f"{i:>2}. [{s['cid']}] {s['topic']}  均{s['avg']:.2f}  自己答{s['hit1']}/{s['n']}"
              f"  话题内{s['topic_hit']}/{s['n']}")
        print(f"    {s['text'][:56]}...")
        print(f"    分数（! = 自己接不住）：{marks}")
        for r in s["rs"]:
            if r["dx"] not in ("自己答但低分", "没人答"):
                continue
            if r["dx"] == "没人答":
                why = f"top1 是「{r['top_id']}」但只 {r['top_score']:.3f}，整库都没接住"
            else:
                why = f"自己排 {r['self_rank']}，只给 {r['self_score']:.3f}"
            print(f"      x {r['q']}  -> {why}")

    # ---- 兄弟冗余：不影响用户，但说明 chunk 边界该收 ----
    redun = [s for s in summary if not s["bad"] and s["hit1"] < s["n"]]
    if redun:
        print(f"\n【不算病，但该知道】{len(redun)} 条 chunk 的问法都被同话题兄弟接走了")
        print("-" * 76)
        for s in redun[:10]:
            others = sorted({r["top_id"] for r in s["rs"] if r["top_id"] != s["cid"]})
            print(f"  [{s['cid']}] {s['topic']}  均{s['avg']:.2f}  接走它问法的是：{', '.join(others)}")

    # ---- 按话题 ----
    print("\n【按话题】话题级命中率 = 这个赛道能不能接住自己的问法")
    print("-" * 76)
    tp = {}
    for r in rows:
        tp.setdefault(r["topic"], []).append(r)
    # 注意：key 里不能引用 for 的循环变量 t —— 排序发生在绑定之前，会 NameError
    for t, rs in sorted(tp.items(),
                        key=lambda kv: sum(1 for r in kv[1] if r["top_topic"] == kv[0]) / len(kv[1])):
        th_ = sum(1 for r in rs if r["top_topic"] == t)
        sh_ = sum(1 for r in rs if r["top_id"] == r["cid"])
        sick_n = sum(1 for r in rs if r["dx"] in ("自己答但低分", "没人答"))
        bar = "#" * max(1, int(th_ / len(rs) * 14))
        print(f"  {t:<10}话题 {th_ / len(rs) * 100:>5.1f}% {bar:<15} "
              f"自己 {sh_ / len(rs) * 100:>5.1f}%  病 {sick_n}/{len(rs)}")

    # ---- 落盘 ----
    out = HERE / ".selfquery_report.json"
    out.write_text(json.dumps({
        "threshold": TH,
        "global": {"n": n, "topic_hit": topic_hit, "self_hit": self_hit,
                   "passed": passed, "by_class": cnt},
        "chunks": [{k: v for k, v in s.items() if k not in ("rs", "text")}
                   | {"class": sorted({r["dx"] for r in s["rs"] if r["dx"] != "自己答"})}
                   for s in summary],
        "queries": [{k: r[k] for k in ("cid", "topic", "q", "self_score", "self_rank",
                                       "top_id", "top_topic", "top_score", "vec_rank", "dx")}
                    for r in rows],
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n明细已存 {out.name}")


if __name__ == "__main__":
    main()
