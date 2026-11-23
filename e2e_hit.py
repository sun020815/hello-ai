# -*- coding: utf-8 -*-
"""反方向的风险：生成层会不会把该答的题也拒掉。

端到端拒答率 100% 听着很好，但如果代价是「什么都答资料里没有」，那这个系统
等于废了。所以必须测另一个方向：该命中的题，生成层有没有正常作答。

抽样跑（固定 seed 可复现），不跑全部 90 条，因为结论看的是比例不是逐条。

用法：uv run python e2e_hit.py [--n 20]
"""
import json
import random
import sys
from pathlib import Path

import main as app
from main import AskRequest

ADMIT = ("资料里没有", "资料中没有", "没有这部分", "没有相关", "无法基于",
         "不涉及", "没有提到", "未提及", "没有讲", "不在资料")


def admits(ans):
    return any(k in ans for k in ADMIT)


def main():
    n = 20
    if "--n" in sys.argv:
        n = int(sys.argv[sys.argv.index("--n") + 1])

    cases = json.loads((Path(__file__).parent / "eval_cases.json").read_text(encoding="utf-8"))
    hits = [c for c in cases if c["kind"] == "hit"]

    # 只看检索就答对的题（hit@1 正确），这样测的是纯粹的「生成层会不会误拒」
    # 检索本来就答错的不算误拒，那是检索的锅
    import tune_rerank as T
    correct = []
    for c in hits:
        kept = [r["topic"] for r in app.pick_hits(T.ranking_of(c["q"]), app.RERANK_TOP_K,
                                                  app.RERANK_THRESHOLD)]
        if kept and kept[0] == c["expect"]:
            correct.append(c)

    random.seed(7)
    sample = random.sample(correct, min(n, len(correct)))

    print(f"检索 hit@1 正确共 {len(correct)} 条，抽 {len(sample)} 条走端到端\n")
    refused = []
    for i, c in enumerate(sample, 1):
        r = app.rag(AskRequest(question=c["q"]))
        ans = r["answer"].replace("\n", " ").strip()
        bad = admits(ans) and not r["sources"] is None
        # 更严格的判定：检索命中了，但答案里在说「资料里没有」
        mis = bool(r["sources"]) and admits(ans)
        if mis:
            refused.append((c["q"], ans))
        print(f"  {i:>2}/{len(sample)}  {'误拒' if mis else '正常':<4} "
              f"[{r['sources'][0]['topic'] if r['sources'] else '拒答'}] "
              f"{c['q'][:22]:<24}{ans[:52]}", flush=True)

    print()
    print("=" * 70)
    print(f"抽样 {len(sample)} 条该命中的题，生成层误拒 {len(refused)} 条"
          f"（{len(refused)/len(sample)*100:.0f}%）")
    print("=" * 70)
    if refused:
        print("\n误拒明细：")
        for q, a in refused:
            print(f"  {q}\n    {a[:160]}")


if __name__ == "__main__":
    main()
