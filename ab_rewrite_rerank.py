"""把「口语改写」接到「精排」前面 —— 这个组合之前没测过。

已有的两条路各缺一半：
  rewrite 模式：改写 + 纯向量。改写补了语域落差，但放弃了精排的判别力（94.7% -> 88.4%）
  rerank  模式：精排 + 原始口语问法。判别力在，但口语问法被系统性压分

ab_phrasing.py 已经量出压分有多狠：同一批 20 个问题，只把问法换成书面表述、语料一个字不动，
平均分 0.155 -> 0.549，过门槛 0/20 -> 13/20，20 条里 18 条变高、0 条变低。
所以修法很明确：在入口把口语问法转成书面，再走精排，语料不动。

本脚本验的就是这个组合，用最难的那批（rerank 原始分最低的 20 条口语问法）。

指标：
  top1 是不是自己那条 chunk   —— 排序对不对
  最高分过不过 0.50           —— 会不会被拒答
两条都看，因为「排第一」和「过门槛」是两件事，之前吃过只看其一的亏。
"""

import json
from pathlib import Path

import main as app

HERE = Path(__file__).parent
PAIRS = HERE / ".phrasing_ab.json"
OUT = HERE / ".rewrite_rerank.json"


def best_over_variants(q, variants):
    """原问题 + 改写，逐条走完整精排，取最高分那一次。
    不合并成一条长查询 —— 合起来会让 reranker 同时看到几种意图，分数反而钝。"""
    best = None
    detail = []
    for v in [q] + variants:
        _, ranking = app.search_kb_rerank(v)
        top = ranking[0]
        detail.append({"q": v, "top_id": top["id"], "top_topic": top["topic"],
                       "score": top["score"]})
        if best is None or top["score"] > best["score"]:
            best = detail[-1]
    return best, detail


def main():
    topic_of = {d["id"]: d["topic"] for d in app.KB}
    pairs = json.loads(PAIRS.read_text(encoding="utf-8"))
    # 复用上一轮的 20 条口语问法
    oral = []
    seen = set()
    for k in pairs:
        cid, q = k.split("|", 1)
        if (cid, q) in seen:
            continue
        seen.add((cid, q))
        oral.append((cid, q))

    print(f"口语问法 {len(oral)} 条，逐条改写（有缓存就复用）")
    cache = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    todo = [q for _, q in oral if q not in cache]
    rewrites = app.rewrite_many(todo) if todo else {}
    for k, v in rewrites.items():
        cache.setdefault(k, {"variants": v})

    print(f"\n{'chunk':<11}{'原始':>7}{'改后':>7}{'差':>8}  {'自中':<5} 最好的改写")
    print("-" * 92)
    rows = []
    for cid, q in oral:
        if "result" not in cache[q]:
            best, detail = best_over_variants(q, cache[q]["variants"])
            cache[q]["result"] = {"best": best, "detail": detail}
            OUT.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        res = cache[q]["result"]
        base = next(d for d in res["detail"] if d["q"] == q)
        best = res["best"]
        best_w = max((d for d in res["detail"] if d["q"] != q),
                     key=lambda d: d["score"], default=None)
        rows.append({
            "cid": cid, "q": q,
            "base": base["score"], "best": best["score"],
            "base_self": base["top_id"] == cid,
            "best_self": best["top_id"] == cid,
            "best_topic_ok": best["top_topic"] == topic_of[cid],
        })
        mark = "是" if best["top_id"] == cid else "-"
        print(f"{cid:<11}{base['score']:>7.3f}{best['score']:>7.3f}"
              f"{best['score'] - base['score']:>+8.3f}  {mark:<5} "
              f"{(best_w['q'][:26] if best_w else '')}")

    n = len(rows)
    tb = app.RERANK_THRESHOLD
    print("-" * 92)
    print(f"  平均分        {sum(r['base'] for r in rows) / n:.3f}  ->  "
          f"{sum(r['best'] for r in rows) / n:.3f}")
    print(f"  过门槛 {tb}      {sum(1 for r in rows if r['base'] >= tb)}/{n}  ->  "
          f"{sum(1 for r in rows if r['best'] >= tb)}/{n}")
    print(f"  top1 是自己   {sum(r['base_self'] for r in rows)}/{n}  ->  "
          f"{sum(r['best_self'] for r in rows)}/{n}")
    print(f"  top1 落对话题 {sum(1 for r in rows if r['best_topic_ok'])}/{n}")
    print("\n注意：这是「最好那条改写」的上界，线上要么多跑几次取最大（更贵），")
    print("      要么只跑一条（更省但会低一些）。别拿上界当线上效果。")


if __name__ == "__main__":
    main()
