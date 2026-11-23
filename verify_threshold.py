# -*- coding: utf-8 -*-
"""阈值二选一的端到端裁决：不看检索指标，看生成层实际吐出什么。

为什么必须做这一步：检索口径说「0.55 拒答率 88%，比 0.50 的 79% 好」，
但那只是「放行 / 拦截」的开关统计。真正决定用户会不会被误导的，是
放行之后生成层会不会照着错误资料编出一个像样的答案。

一共只有 7 条边界题，各调一次生成，成本几分钱。
"""
import json
from pathlib import Path

import main as app
from main import AskRequest

# 0.50 放行、0.55 拦截的题（来自 tune_rerank.py 的误判清单）
BOUNDARY = [
    ("肌酸一天吃多少", "拒答", 0.5309, "饮食配合"),
    ("减脂期蛋白质要吃更多吗", "拒答", 0.5443, "饮食配合"),
    ("史密斯机卧推能替代自由卧推吗", "拒答", 0.5458, "卧推技术"),
    ("哥本哈根支撑有什么用", "命中", 0.5404, "腿日"),
]

# 两个阈值都拦不住的（要靠改内容，不在这里测）
HARD = [
    ("肩峰撞击综合征怎么治", 0.8466, "肩部训练"),
    ("腰椎间盘突出能练硬拉吗", 0.8402, "硬拉"),
    ("卧推手腕疼怎么办", 0.7882, "热身与激活"),
    ("腹肌怎么练", 0.6430, "推日"),
]


def ask_with(q, th):
    """改全局阈值为 th，走完整 /rag 链路（检索 + 生成 + 降级），返回结果。"""
    old = app.RERANK_THRESHOLD
    app.RERANK_THRESHOLD = th
    try:
        r = app.rag(AskRequest(question=q))
    finally:
        app.RERANK_THRESHOLD = old
    return r


def show(q, th, verdict_hint):
    r = ask_with(q, th)
    got = r["sources"][0]["topic"] if r["sources"] else "拒答"
    ans = r["answer"].replace("\n", " ").strip()
    print(f"  [{th:.2f}] 命中={got:<10} 答: {ans[:96]}")
    return got, ans


def main():
    print("=" * 78)
    print("A. 边界四题：0.50 放行 / 0.55 拦截 —— 看 0.50 放行后生成层编不编")
    print("=" * 78)
    rows = []
    for q, truth, score, topic in BOUNDARY:
        print(f"\n{q}   （标注={truth}，精排分 {score:.4f}，最高分 chunk 属「{topic}」）")
        got50, ans50 = show(q, 0.50, truth)
        got55, ans55 = show(q, 0.55, truth)
        rows.append((q, truth, score, got50, ans50, got55, ans55))

    print()
    print("=" * 78)
    print("B. 高难四题：两个阈值都放行 —— 这就是「内容问题」的直接证据")
    print("=" * 78)
    for q, score, topic in HARD:
        print(f"\n{q}   （精排分 {score:.4f}，最高分 chunk 属「{topic}」）")
        show(q, 0.55, "拒答")

    print()
    print("=" * 78)
    print("C. 判定辅助：生成层有没有说出「资料里没有」")
    print("=" * 78)
    print(f"{'问题':<26}{'0.50 命中':<12}{'0.50 认怂':<10}{'0.55 命中':<12}{'0.55 认怂':<10}")
    print("-" * 72)
    for q, truth, score, got50, ans50, got55, ans55 in rows:
        def admits(a):
            return "是" if ("没有" in a or "无法" in a or "不涉及" in a) else "否"
        print(f"{q[:24]:<26}{got50:<12}{admits(ans50):<10}{got55:<12}{admits(ans55):<10}")


if __name__ == "__main__":
    main()
