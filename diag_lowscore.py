"""诊断「同义换措辞」的低分题：病因在召回，还是在精排。

两段式检索有两道关，修法完全不同：

  正确答案没进 12 条召回池   -> 向量召回问题，改语料表示（doc2query / 别名）
  进了池子但精排给了低分     -> reranker 对口语问法不敏感，改精排输入

不区分就动手，等于给感冒的人做手术。所有计算走本地缓存，不花钱。
"""

import json
from pathlib import Path

import main as app

HERE = Path(__file__).parent
LOW = 0.55          # 只看低于这个分的命中题，0.50 是门槛，0.55 是「贴着门槛」


def main():
    cases = json.loads((HERE / "eval_cases.json").read_text(encoding="utf-8"))
    rr = json.loads((HERE / ".eval_reranks.json").read_text(encoding="utf-8"))["scores"]
    qcache = json.loads((HERE / ".eval_qvecs.json").read_text(encoding="utf-8"))
    qvec = dict(zip(qcache["questions"], qcache["vectors"]))
    kbvec = json.loads((HERE / ".kb_vectors.json").read_text(encoding="utf-8"))["vectors"]

    by_id = {d["id"]: d for d in app.KB}
    # 话题 -> 该话题下所有 chunk id
    topic_ids = {}
    for d in app.KB:
        topic_ids.setdefault(d["topic"], []).append(d["id"])

    rows = []
    for c in cases:
        if c["kind"] != "hit":
            continue
        q = c["q"]
        if q not in rr:
            continue                    # 这题还没跑过 rerank
        scores = rr[q]
        right = topic_ids.get(c["expect"], [])
        in_pool = [i for i in right if i in scores]
        if in_pool:
            best_id = max(in_pool, key=lambda i: scores[i])
            rr_score = scores[best_id]
        else:
            best_id, rr_score = None, 0.0

        # 正确 chunk 在向量侧的排名（全库算，不受召回截断影响）
        vec_rank = None
        if q in qvec:
            qv = qvec[q]
            sims = sorted(
                ((app.cosine(qv, kv), d["id"]) for d, kv in zip(app.KB, kbvec)),
                reverse=True,
            )
            for rank, (_, cid) in enumerate(sims, 1):
                if cid in right:
                    vec_rank = rank
                    break
        rows.append({"q": q, "expect": c["expect"], "rr": rr_score,
                     "in_pool": bool(in_pool), "best_id": best_id,
                     "vec_rank": vec_rank})

    rows.sort(key=lambda r: r["rr"])
    low = [r for r in rows if r["rr"] < LOW]

    print(f"命中题 {len(rows)} 条，其中 rerank 分低于 {LOW} 的 {len(low)} 条")
    print(f"（门槛 {app.RERANK_THRESHOLD}，低于它的会被拒答）\n")

    if not low:
        print("没有低分命中题。")
        return

    print(f"{'rerank':>7}{'向量排名':>9}  {'在召回池':<8} 问题")
    print("-" * 78)
    for r in low:
        pool = "是" if r["in_pool"] else "**否**"
        vr = f"{r['vec_rank']}" if r["vec_rank"] else "?"
        print(f"{r['rr']:>7.4f}{vr:>9}  {pool:<8} {r['q']}  -> {r['expect']}")

    # 归因
    no_pool = [r for r in low if not r["in_pool"]]
    in_pool_low = [r for r in low if r["in_pool"]]
    print()
    print(f"  召回没捞到正确答案：{len(no_pool)} 条  <- 改向量 / 加口语别名")
    print(f"  捞到了但精排分低：{len(in_pool_low)} 条  <- 改 rerank 输入")
    if no_pool:
        print()
        print("  召回失败的题（正确答案的向量排名越靠后越难救）：")
        for r in no_pool:
            print(f"    向量第 {r['vec_rank']} 名  {r['q']}")

    # 召回池外的题，即使 rerank 完美也救不了
    print()
    deep = [r for r in low if r["vec_rank"] and r["vec_rank"] > app.RECALL_K]
    print(f"  向量排名超出召回窗口（>{app.RECALL_K} 名）的：{len(deep)} 条")
    for r in deep:
        print(f"    第 {r['vec_rank']} 名  {r['q']}  -> {r['expect']}")


if __name__ == "__main__":
    main()
