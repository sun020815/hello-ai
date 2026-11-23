# -*- coding: utf-8 -*-
"""生成层兜底 A/B：检索拒答之后，再让生成层判断一次「这些资料够不够回答」。

要回答的问题只有一个：**被精排阈值拦下的题里，有多少其实答得出来，又会不会
顺手把「本来就该拒」的题也放进来。**

两组用同一份精排分数（`.eval_reranks.json`），唯一的差别是 hits 为空时：
  A 组（现状）：直接返回「知识库里没有相关资料」
  B 组（兜底）：把 top-2 弱相关资料交给生成层，由它决定答还是拒

「答对」的判定沿用检索口径（top1 的话题 == 期望话题），因为兜底给的就是这两段资料，
模型只能基于它们作答。这个口径偏乐观：答了不等于答到点子上，所以脚本会把
兜底答案原文打出来，必须人工抽样复核。

用法：
  uv run python e2e_fallback.py --limit 20     # 先看 20 条的行为
  uv run python e2e_fallback.py                # 全量 200 + 口语 24
  uv run python e2e_fallback.py --report       # 只读缓存出报告
"""

import json
import sys
from pathlib import Path

import main as app

HERE = Path(__file__).parent
CACHE = HERE / ".fallback.json"
TH = app.RERANK_THRESHOLD


def ranking_of(q):
    """从缓存的精排分数还原完整排名（12 条）。"""
    sc = SCORES[q]
    by_id = {d["id"]: d for d in app.KB}
    rows = [{"id": i, "topic": by_id[i]["topic"], "text": by_id[i]["text"],
             "source": by_id[i]["source"], "score": s} for i, s in sc.items()]
    rows.sort(key=lambda x: x["score"], reverse=True)
    return rows


def fallback(q, ranking, cache):
    """走真实兜底逻辑，结果缓存下来（判断一次要花一次生成调用）。"""
    if q in cache:
        return cache[q]
    got = app.answer_with_fallback(q, ranking)
    rec = {
        "answered": got is not None,
        "answer": (got["answer"][:160] if got else ""),
        "top_score": ranking[0]["score"] if ranking else None,
    }
    cache[q] = rec
    CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    return rec


def evaluate(cases, cache, do_fallback):
    rows = []
    for c in cases:
        r = ranking_of(c["q"])
        kept = app.pick_hits(r, app.RERANK_TOP_K, TH)
        if kept:
            rows.append({"q": c["q"], "kind": c["kind"], "expect": c["expect"],
                         "answered": True, "topic": kept[0]["topic"], "via": "阈值"})
            continue
        if not do_fallback:
            rows.append({"q": c["q"], "kind": c["kind"], "expect": c["expect"],
                         "answered": False, "topic": None, "via": "阈值"})
            continue
        fb = fallback(c["q"], r, cache)
        rows.append({"q": c["q"], "kind": c["kind"], "expect": c["expect"],
                     "answered": fb["answered"], "topic": r[0]["topic"],
                     "via": "兜底", "ans": fb["answer"]})
    return rows


def summarize(name, rows):
    hit = [r for r in rows if r["kind"] == "hit"]
    ref = [r for r in rows if r["kind"] == "refuse"]
    h_ok = sum(1 for r in hit if r["answered"] and r["topic"] == r["expect"])
    h_bad = sum(1 for r in hit if r["answered"] and r["topic"] != r["expect"])
    h_ref = sum(1 for r in hit if not r["answered"])
    r_ok = sum(1 for r in ref if not r["answered"])
    r_bad = sum(1 for r in ref if r["answered"])
    n = len(rows)
    print(f"\n【{name}】{n} 题")
    print(f"  该命中 {len(hit)}：答对 {h_ok} / 答错 {h_bad} / 误拒 {h_ref}")
    print(f"  该拒答 {len(ref)}：拒对 {r_ok} / 误答 {r_bad}")
    acc = (h_ok + r_ok) / n
    if hit and ref:
        print(f"  整体 {acc:.1%}   命中率 {h_ok / len(hit):.1%}   拒答率 {r_ok / len(ref):.1%}")
    else:
        print(f"  整体 {acc:.1%}（本组缺少一类样本，分项率不显示）")
    return {"acc": (h_ok + r_ok) / n, "h_ok": h_ok, "h_ref": h_ref, "r_ok": r_ok, "r_bad": r_bad}


if __name__ == "__main__":
    CASES = json.loads((HERE / "eval_cases.json").read_text(encoding="utf-8"))
    COL = json.loads((HERE / "eval_cases_colloquial.json").read_text(encoding="utf-8"))
    SCORES = json.loads((HERE / ".eval_reranks.json").read_text(encoding="utf-8"))["scores"]

    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])
        CASES, COL = CASES[:limit], COL[:limit]

    cache = {}
    if CACHE.exists():
        cache = json.loads(CACHE.read_text(encoding="utf-8"))

    print(f"精排模型 {app.RERANK_MODEL} 阈值 {TH}")
    # A 组不花钱，先跑
    a = evaluate(CASES, cache, do_fallback=False)
    b = evaluate(CASES, cache, do_fallback=True)

    sa = summarize("A 现状：检索拒了就拒", a)
    sb = summarize("B 兜底：拒了再让生成层判一次", b)

    print(f"\n净变化：整体 {sa['acc']:.1%} -> {sb['acc']:.1%}"
          f"（{(sb['acc'] - sa['acc']) * 100:+.1f} 点）")

    # 关键：兜底救回了哪些命中题，又放进去了哪些该拒的
    saved = [r for r, x in zip(a, b) if r["kind"] == "hit" and not r["answered"] and x["answered"]]
    leaked = [r for r, x in zip(a, b) if r["kind"] == "refuse" and not r["answered"] and x["answered"]]
    print(f"\n兜底救回（该答、被阈值拒、兜底答了）：{len(saved)} 条")
    for r, x in zip(a, b):
        if r["kind"] == "hit" and not r["answered"] and x["answered"]:
            print(f"  [{x['topic']}{'✓' if x['topic'] == x['expect'] else '✗'}] {r['q'][:22]:<24}{x.get('ans', '')[:60]}")
    print(f"\n兜底误放（该拒、兜底却答了）：{len(leaked)} 条")
    for r, x in zip(a, b):
        if r["kind"] == "refuse" and not r["answered"] and x["answered"]:
            print(f"  [{x['topic']}] {r['q'][:22]:<24}{x.get('ans', '')[:60]}")

    # 口语集单独看（全是该命中，主要看误拒有没有减少）
    if not limit:
        ca = evaluate(COL, cache, do_fallback=False)
        cb = evaluate(COL, cache, do_fallback=True)
        print("\n【口语集 24 条】")
        for name, rows in (("A", ca), ("B", cb)):
            ok = sum(1 for r in rows if r["answered"] and r["topic"] == r["expect"])
            bad = sum(1 for r in rows if r["answered"] and r["topic"] != r["expect"])
            ref = sum(1 for r in rows if not r["answered"])
            print(f"  {name}: 答对 {ok} / 答错 {bad} / 拒答 {ref}")
