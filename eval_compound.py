"""复合问题的检索评测：单路 vs 多路，三组消融。

要回答的问题只有一个：**「把一句话拆成两路去检索」到底有没有用？**

所以必须同时跑三组，少一组结论就是假的：
  A 单路 top2   现在线上的样子
  B 单路 top4   只把条数加到 4，不拆
  C 多路 top4   拆成 2~3 路，每路独立 rank_one，RRF 融合后取 4

  A -> B 是「多给几条」的效果，B -> C 才是「拆解」本身的效果。
  如果 C 和 B 差不多，那结论就是「加 top_k 就够了，不用拆」——这个结论
  比硬说拆解有用值钱，因为它少一层调用、少一次失败概率。

指标：
  cov       期望话题被覆盖的比例（复合题的核心指标，不是 top1 命中）
  full      期望话题全部覆盖的题占比
  junk      返回的 sources 里不属于任何期望话题的条数占比（防止「多给几条」变成乱塞）
  refuse    该拒答的题里成功拒掉的比例

用法：uv run python eval_compound.py
"""
import json
import time
from pathlib import Path

import main as app
from probe_decompose import decompose_v1

HERE = Path(__file__).parent
CASES = json.loads((HERE / "eval_cases_compound.json").read_text(encoding="utf-8"))
DEC_CACHE = HERE / ".eval_compound_decomp.json"
RANK_CACHE = HERE / ".eval_compound_ranks.json"

RRF_K = 60          # RRF 的标准常数，作用是压低头部名次的绝对差距
MULTI_TOP = 4       # 多路融合后给模型几段


# ---------- 缓存：每个 query 的完整 ranking 只算一次 ----------
def load_cache(path):
    if path.exists():
        try:
            c = json.loads(path.read_text(encoding="utf-8"))
            if c.get("model") == app.RERANK_MODEL:
                return c["data"]
        except Exception:
            pass
    return {}


def save_cache(path, data):
    path.write_text(json.dumps({"model": app.RERANK_MODEL, "data": data},
                               ensure_ascii=False), encoding="utf-8")


RANKS = load_cache(RANK_CACHE)
DECS = load_cache(DEC_CACHE)


def rank_cached(q: str):
    """一个 query 走完整的两段式检索，结果按 query 字符串缓存。"""
    if q in RANKS:
        return RANKS[q]
    ranking = app.rank_one(q)
    RANKS[q] = [{"id": r["id"], "topic": r["topic"], "text": r["text"],
                 "source": r.get("source", ""), "score": r["score"]} for r in ranking[: app.RECALL_K]]
    return RANKS[q]


def decompose_cached(q: str):
    if q in DECS:
        return DECS[q]
    subs = decompose_v1(q)
    DECS[q] = subs
    return subs


# ---------- 三种策略 ----------
def strategy_single(q, top_k):
    ranking = rank_cached(q)
    return app.pick_hits(ranking, top_k, app.RERANK_THRESHOLD), ranking


def rrf_fuse(rankings):
    """Reciprocal Rank Fusion：只用名次，不用分数。

    为什么不直接比 rerank 分数：每个子查询的分数分布不一样 ——
    「卧推握多宽」的 top1 可能是 0.99，「腰疼能不能练」的 top1 可能只有 0.85，
    直接取最大值等于让「分数虚高的那一路」绑架整个排序。
    RRF 把每一路都压成名次，跨路可比，这也是它在多路检索里成为默认做法的原因。
    """
    agg = {}
    for ranking in rankings:
        for i, r in enumerate(ranking):
            cur = agg.setdefault(r["id"], {"item": dict(r), "rrf": 0.0, "best": 0.0, "hits": 0})
            cur["rrf"] += 1.0 / (RRF_K + i + 1)
            cur["best"] = max(cur["best"], r["score"])
            cur["hits"] += 1
    out = sorted(agg.values(), key=lambda x: (-x["rrf"], -x["best"]))
    return out


def strategy_multi(q, subs, top_k=MULTI_TOP):
    """多路检索 + RRF 融合。

    **每路先过绝对门槛再融合**，不是融合完再过门槛 —— 这条顺序决定了拒答
    会不会被拆解稀释。先融合的话，两路各自 0.7 的弱相关项加起来可能超过
    某一路 0.96 的真命中，而 0.7 本来是应该被拒掉的东西。
    """
    paths = list(dict.fromkeys([q] + (subs or [])))      # 去重，主查询永远算第一路
    passed = []
    rankings = []
    for p in paths:
        rk = rank_cached(p)
        rankings.append(rk)
        kept = app.pick_hits(rk, 99, app.RERANK_THRESHOLD)   # 只过门槛，先不截断
        if kept:
            passed.append(kept)

    if not passed:
        return [], rankings

    fused = rrf_fuse(passed)
    picked, seen = [], set()
    for it in fused:
        topic = it["item"]["topic"]
        if topic in seen:           # 同话题只留一条，跟 pick_hits 一个规矩
            continue
        seen.add(topic)
        row = dict(it["item"])
        row["score"] = round(it["best"], 4)     # 对外仍然报 rerank 分，不是 RRF 分
        row["paths"] = it["hits"]
        picked.append(row)
        if len(picked) >= top_k:
            break
    return picked, rankings


def strategy_roundrobin(q, subs, top_k=MULTI_TOP):
    """每路轮流贡献一条，不做分数融合。

    为什么值得跟 RRF 比：RRF 会把「两路都出现在中段」的资料顶上来，
    而这类资料通常是泛化条目（「肩部训练」这种总论）。实测 junk 就是这么涨的。
    round-robin 更保守 —— 每路只交自己最自信的那一条，正好对上「一句话里
    有几个问题，就每个问题给一段最相关的资料」这个需求。
    """
    paths = list(dict.fromkeys([q] + (subs or [])))
    per = []
    for p in paths:
        kept = app.pick_hits(rank_cached(p), 99, app.RERANK_THRESHOLD)
        if kept:
            per.append(kept)
    picked, seen = [], set()
    level = 0
    while len(picked) < top_k and any(level < len(x) for x in per):
        for lst in per:
            if level >= len(lst):
                continue
            it = lst[level]
            if it["topic"] in seen:
                continue
            seen.add(it["topic"])
            picked.append(dict(it))
            if len(picked) >= top_k:
                break
        level += 1
    return picked


# ---------- 打分 ----------
def score_case(case, picked):
    expect = set(case.get("expect") or [])
    got = [p["topic"] for p in picked]
    if case["kind"] == "refuse":
        return {"ok": not picked, "cov": 1.0 if not picked else 0.0,
                "full": not picked, "junk": 0.0}
    if not expect:
        return {"ok": True, "cov": 1.0, "full": True, "junk": 0.0}
    covered = expect & set(got)
    junk = sum(1 for t in got if t not in expect)
    return {
        "ok": bool(covered),
        "cov": len(covered) / len(expect),
        "full": covered == expect,
        "junk": junk / len(got) if got else 0.0,
    }


def summarize(rows):
    hit = [r for r in rows if r["kind"] != "refuse"]
    ref = [r for r in rows if r["kind"] == "refuse"]
    n = len(hit) or 1
    return {
        "cov": sum(r["cov"] for r in hit) / n,
        "full": sum(1 for r in hit if r["full"]) / n,
        "junk": sum(r["junk"] for r in hit) / n,
        "refuse": (sum(1 for r in ref if r["ok"]) / len(ref)) if ref else 1.0,
        "n_hit": len(hit), "n_ref": len(ref),
    }


def pct(x):
    return f"{x*100:5.1f}%"


def main():
    print(f"评测集 {len(CASES)} 条，门槛 {app.RERANK_THRESHOLD}，多路取 {MULTI_TOP} 条\n")

    # 1) 先补拆解
    todo = [c["q"] for c in CASES if c["q"] not in DECS]
    if todo:
        print(f"拆解缓存缺 {len(todo)} 条，开始补（每条间隔 1.2s 防限速）…")
        for i, q in enumerate(todo, 1):
            decompose_cached(q)
            print(f"  {i}/{len(todo)}", end="\r", flush=True)
            time.sleep(1.2)
        print()
        save_cache(DEC_CACHE, DECS)

    # 2) 再补检索
    queries = set()
    for c in CASES:
        subs = DECS.get(c["q"]) or [c["q"]]
        queries.add(c["q"]); queries.update(subs)
    todo = [q for q in queries if q not in RANKS]
    if todo:
        print(f"检索缓存缺 {len(todo)} 个 query，开始补…")
        for i, q in enumerate(todo, 1):
            rank_cached(q)
            print(f"  {i}/{len(todo)}", end="\r", flush=True)
        print()
        save_cache(RANK_CACHE, RANKS)

    # 3) 六组策略
    out = {}
    for name in ("single2", "single3", "single4", "multi", "round", "round2"):
        rows = []
        for c in CASES:
            q = c["q"]
            if name == "single2":
                picked, _ = strategy_single(q, 2)
            elif name == "single3":
                picked, _ = strategy_single(q, 3)
            elif name == "single4":
                picked, _ = strategy_single(q, 4)
            elif name == "multi":
                picked, _ = strategy_multi(q, DECS.get(q))
            elif name == "round":
                picked = strategy_roundrobin(q, DECS.get(q))
            else:
                picked = strategy_roundrobin(q, DECS.get(q), top_k=2)
            rows.append({"q": q, "kind": c["kind"], **score_case(c, picked),
                         "got": [p["topic"] for p in picked]})
        out[name] = summarize(rows)
        out[name + "_rows"] = rows

    # 4) 逐题对比
    print("\n逐题（cov：期望话题覆盖比例）")
    print(f"  {'题':<34} {'A':>6} {'B3':>6} {'B4':>6} {'E2':>6} {'C':>6}   覆盖到的话题(C)")
    for i, c in enumerate(CASES):
        a = out["single2_rows"][i]
        b3 = out["single3_rows"][i]
        b = out["single4_rows"][i]
        d = out["multi_rows"][i]
        e = out["round2_rows"][i]
        short = c["q"][:32]
        print(f"  {short:<34} {pct(a['cov'])} {pct(b3['cov'])} {pct(b['cov'])} {pct(e['cov'])} {pct(d['cov'])}   {d['got']}")

    print(f"\n{'策略':<14} {'cov':>8} {'full':>8} {'junk':>8} {'refuse':>8}")
    for name, label in (("single2", "A 单路2"), ("single3", "B3 单路3"),
                        ("single4", "B4 单路4"), ("round2", "E 多路2"),
                        ("multi", "C 多路RRF4"), ("round", "D 多路轮流4")):
        s = out[name]
        print(f"{label:<14} {pct(s['cov'])} {pct(s['full'])} {pct(s['junk'])} {pct(s['refuse'])}")

    a, b, c, d = out["single2"], out["single4"], out["multi"], out["round"]
    e = out["round2"]
    print(f"\n条数效应 (A->B4)：cov {pct(b['cov']-a['cov'])}  full {pct(b['full']-a['full'])}")
    print(f"拆解效应 (B4->C)：cov {pct(c['cov']-b['cov'])}  full {pct(c['full']-b['full'])}  junk {pct(c['junk']-b['junk'])}")
    print(f"拆解效应 (B4->D)：cov {pct(d['cov']-b['cov'])}  full {pct(d['full']-b['full'])}  junk {pct(d['junk']-b['junk'])}")
    print(f"同预算对比 (A2->E2)：cov {pct(e['cov']-a['cov'])}  full {pct(e['full']-a['full'])}  junk {pct(e['junk']-a['junk'])}")

    Path(HERE / ".eval_compound.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                                  encoding="utf-8")


if __name__ == "__main__":
    main()
