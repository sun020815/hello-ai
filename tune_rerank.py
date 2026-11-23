# -*- coding: utf-8 -*-
"""扫 rerank 阈值：用评测缓存（.eval_reranks.json）离线算，不花一分钱。

rerank 分数和余弦相似度不是一个量纲，0.55 那套数字完全不能套用，
必须单独扫一遍才知道该取多少。用法：uv run python tune_rerank.py

先跑 eval.py 把缓存生成出来，否则这个脚本没东西可读。
"""
import json
from pathlib import Path

import main as app

HERE = Path(__file__).parent
CASES = json.loads((HERE / "eval_cases.json").read_text(encoding="utf-8"))
CACHE_PATH = HERE / ".eval_reranks.json"
SCORES = json.loads(CACHE_PATH.read_text(encoding="utf-8"))["scores"]   # {问题: {chunk_id: 分数}}
BY_ID = {d["id"]: d for d in app.KB}


def ranking_of(q):
    """把缓存里的 {id: 分数} 还原成排好序的列表"""
    rows = [{"id": i, "topic": BY_ID[i]["topic"], "score": s} for i, s in SCORES[q].items()]
    rows.sort(key=lambda x: x["score"], reverse=True)
    return rows


def score_at(th):
    """给定阈值，算 hit@1 / hit@2 / 拒答率 / 整体准确率"""
    h1 = h2 = 0
    hit_n = ref_ok = ref_n = 0
    for c in CASES:
        # 和线上 search_kb_rerank 同一套挑法
        kept = [r["topic"] for r in app.pick_hits(ranking_of(c["q"]), app.RERANK_TOP_K, th)]
        got = kept[0] if kept else None
        if c["kind"] == "refuse":
            ref_n += 1
            ref_ok += got is None
        else:
            hit_n += 1
            h1 += got == c["expect"]
            h2 += c["expect"] in kept
    n = len(CASES)
    return {
        "th": th,
        "hit1": h1 / hit_n,
        "hit2": h2 / hit_n,
        "refuse": ref_ok / ref_n,
        "acc": (h1 + ref_ok) / n,
    }


def main():
    print("【先看两类分数的分布：能不能用一条线切开】\n")
    hit_tops, ref_tops = [], []
    for c in CASES:
        top = ranking_of(c["q"])[0]
        (ref_tops if c["kind"] == "refuse" else hit_tops).append((top["score"], c["q"]))

    hit_tops.sort()
    ref_tops.sort(reverse=True)
    print(f"  应拒答的问题，最高分：{ref_tops[0][0]:.4f}  （{ref_tops[0][1]}）")
    print(f"  应拒答的问题，次高分：{ref_tops[1][0]:.4f}  （{ref_tops[1][1]}）")
    print(f"  该命中的问题，最低分：{hit_tops[0][0]:.4f}  （{hit_tops[0][1]}）")
    print(f"  该命中的问题，次低分：{hit_tops[1][0]:.4f}  （{hit_tops[1][1]}）")
    print("\n  → 如果「该拒答的最高分」比「该命中的最低分」还高，就说明两类分数有重叠，")
    print("     没有任何一条线能完全切开。下面是扫出来的实际代价。\n")

    print(f"{'阈值':>8}{'hit@1':>9}{'hit@2':>9}{'拒答率':>9}{'整体准确':>10}")
    print("-" * 46)
    best = None
    th = 0.0000
    while th <= 0.2001:
        m = score_at(th)
        print(f"{th:>8.4f}{m['hit1']*100:>8.0f}%{m['hit2']*100:>8.0f}%"
              f"{m['refuse']*100:>8.0f}%{m['acc']*100:>9.0f}%")
        if best is None or m["acc"] > best["acc"] or (
            abs(m["acc"] - best["acc"]) < 1e-9 and m["refuse"] > best["refuse"]
        ):
            best = m
        th = round(th + 0.005, 4)

    print("-" * 46)
    print(f"\n整体准确率最高的一档：阈值 {best['th']:.3f} → "
          f"hit@1 {best['hit1']*100:.0f}% / 拒答 {best['refuse']*100:.0f}% / 整体 {best['acc']*100:.0f}%")

    vec = json.loads((HERE / "eval_results.json").read_text(encoding="utf-8"))["vector"]["summary"]
    print(f"对照纯向量：                "
          f"hit@1 {vec['hit1']*100:.0f}% / 拒答 {vec['refuse']*100:.0f}% / 整体 {vec['acc']*100:.0f}%")


if __name__ == "__main__":
    main()
