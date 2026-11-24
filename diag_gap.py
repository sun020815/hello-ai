"""诊断：真实问法日志里没接住的题，卡在哪一层。

区分三种情况，处理方式完全不同：
  - 向量召回就没有相关 chunk   -> 语料层面没有，只能补内容
  - 召回到了但精排没过线       -> 排序问题（口语问法不认），不是没内容
  - 精排过线了却被别处拦住     -> 门槛 / 逻辑问题，改代码
"""
import sys

import main

QS = [
    "减脂热量缺口到底是三五百大卡还是八百一千大卡",
    "怎么把胳膊练粗",
    "臀腿，女生训练",
    "肩锁关节疼痛",
    "给我一个详细的全身健身计划",
    "每次吸气要吸到多满才算够",
]

for q in QS:
    print("=" * 64)
    print("Q:", q)
    ranking = main.rank_one(q)
    if not ranking:
        print("  召回 0 条")
        continue
    print("  向量 top3：")
    for r in sorted(ranking, key=lambda x: x["vec_score"], reverse=True)[:3]:
        print(f"    {r['vec_score']:.4f}  {r['id']:<12} {r['topic']}")
    print("  精排 top3：")
    for r in ranking[:3]:
        flag = "  <= 过线" if r["score"] >= main.RERANK_THRESHOLD else ""
        print(f"    {r['score']:.4f}  {r['id']:<12} {r['topic']}{flag}")
    top = ranking[0]["score"]
    print(f"  结论：{'过线' if top >= main.RERANK_THRESHOLD else '精排未过线'}"
          f"  阈值 {main.RERANK_THRESHOLD} / 差 {main.RERANK_THRESHOLD - top:.4f}")
    print(f"  最像那条正文：{ranking[0]['text'][:56]}…")

sys.exit(0)
