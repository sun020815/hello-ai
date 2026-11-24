"""量化类问法诊断：答案在库里，但精排不认。

背景：「一周减几斤算正常」精排 0.90 被拒，而 lose-01 正文里明写着
「一周掉零点二五到零点五公斤」。这与 miss=0（gold 100% 在召回 12 条内）矛盾。

一条问法要过三关：
  1. 向量召回 12 条里有没有 gold      -> 没有：embedding 层面不认，补内容也没用
  2. gold 在精排里排第几、多少分      -> 分低：reranker 不认这个问法
  3. top1 是不是 gold                -> 是别的：知识点被抢答，属标注歧义

三组对照，差别只在「问法形式」，内容指向同一个 gold：
  A 量化（几斤 / 多少克 / 几次）      <- 怀疑对象
  B 同义非量化（怎么定 / 安排）
  C 阿拉伯数字（0.5 公斤 / 175 克）   <- 库里全是中文数字，怀疑不匹配
"""
import statistics as st

import main

# (问法, gold_id)
A_QUANT = [
    ("一周减几斤算正常", "lose-01"),
    ("减脂一周掉多少斤正常", "lose-01"),
    ("减脂期每天吃多少大卡", "lose-diet-01"),
    ("减脂期碳水一天吃多少克", "lose-diet-01"),
    ("减脂期蛋白质吃多少克", "lose-diet-01"),
    ("减脂期脂肪吃多少克", "lose-diet-01"),
    ("有氧一周做几次", "cardio-01"),
    ("有氧每次做多久", "cardio-01"),
    ("一天吃几个蛋黄", "diet-food-04"),
    ("坚果一天吃多少", "diet-food-04"),
    ("炒菜油一天用多少", "diet-food-04"),
    ("减脂期蛋白一天吃几克", "diet-food-01"),
]

B_PLAIN = [
    ("减脂期掉秤速度应该是多少", "lose-01"),
    ("减脂掉体重太快好不好", "lose-01"),
    ("减脂期热量该怎么定", "lose-diet-01"),
    ("减脂期碳水怎么安排", "lose-diet-01"),
    ("减脂期三大营养素怎么分配", "lose-diet-01"),
    ("减脂期脂肪怎么安排", "lose-diet-01"),
    ("减脂期有氧怎么安排", "cardio-01"),
    ("有氧的强度和时长怎么选", "cardio-01"),
    ("蛋黄怎么吃", "diet-food-04"),
    ("坚果怎么安排", "diet-food-04"),
    ("炒菜油怎么控制", "diet-food-04"),
    ("减脂期蛋白怎么定", "diet-food-01"),
]

C_ARABIC = [
    ("减脂期一周掉 0.5 公斤正常吗", "lose-01"),
    ("减脂期每天吃 1700 大卡可以吗", "lose-diet-01"),
    ("减脂期碳水吃 175 克可以吗", "lose-diet-01"),
    ("蛋白质一天吃 126 克够吗", "lose-diet-01"),
    ("有氧每周做 3 次够吗", "cardio-01"),
    ("有氧每次 30 分钟够吗", "cardio-01"),
    ("一天吃 2 个蛋黄可以吗", "diet-food-04"),
    ("炒菜油一天 2 小勺够吗", "diet-food-04"),
]


def probe(q, gold):
    ranking = main.rank_one(q)
    vec_rank = next((i for i, r in enumerate(sorted(
        ranking, key=lambda x: x["vec_score"], reverse=True)) if r["id"] == gold), None)
    # 向量排名要按全库算，rank_one 已经截断到 12 条，所以 vec_rank 只在 12 内有效
    g = next((r for r in ranking if r["id"] == gold), None)
    top = ranking[0] if ranking else None
    return {
        "q": q, "gold": gold,
        "in_recall": g is not None,
        "gold_score": g["score"] if g else 0.0,
        "rank": next((i for i, r in enumerate(ranking) if r["id"] == gold), -1),
        "top_id": top["id"] if top else "-",
        "top_score": top["score"] if top else 0.0,
        "top_topic": top["topic"] if top else "-",
        "pass": bool(top and top["score"] >= main.RERANK_THRESHOLD),
    }


def run(title, cases):
    print("=" * 74)
    print(title)
    print("=" * 74)
    rows = []
    for q, gold in cases:
        r = probe(q, gold)
        rows.append(r)
        tag = "过线" if r["pass"] else "拒  "
        missed = "" if r["in_recall"] else "  !!gold 未进召回"
        print(f"  {tag} top={r['top_score']:.4f} gold={r['gold_score']:.4f}"
              f"(第{r['rank'] + 1})  {q}{missed}")
        if r["top_id"] != gold:
            print(f"        抢答：{r['top_id']} / {r['top_topic']} {r['top_score']:.4f}")
    gs = [r["gold_score"] for r in rows]
    ts = [r["top_score"] for r in rows]
    n_pass = sum(1 for r in rows if r["pass"])
    print(f"  ---- gold 精排均分 {st.mean(gs):.4f} | top1 均分 {st.mean(ts):.4f}"
          f" | 过线 {n_pass}/{len(rows)} | gold 未进召回 {sum(1 for r in rows if not r['in_recall'])}")
    return rows


if __name__ == "__main__":
    print(f"阈值 {main.RERANK_THRESHOLD} / 召回 {main.RECALL_K} 条 / 库 {len(main.KB)} 条\n")
    ra = run("A 量化问法（几斤 / 多少克 / 几次）", A_QUANT)
    rb = run("B 同义非量化（怎么定 / 安排）", B_PLAIN)
    rc = run("C 阿拉伯数字问法", C_ARABIC)

    print("\n" + "=" * 74)
    print("A vs B 配对差（同一 gold，只换问法形式）")
    print("=" * 74)
    diffs = []
    for (qa, ga), (qb, gb) in zip(A_QUANT, B_PLAIN):
        sa = next(r["gold_score"] for r in ra if r["q"] == qa)
        sb = next(r["gold_score"] for r in rb if r["q"] == qb)
        diffs.append(sa - sb)
        print(f"  {sa - sb:+.4f}  {qa}  vs  {qb}")
    print(f"  平均 {st.mean(diffs):+.4f}  中位 {st.median(diffs):+.4f}"
          f"  量化更低的 {sum(1 for d in diffs if d < 0)}/{len(diffs)} 条")
