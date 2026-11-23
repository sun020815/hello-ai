"""两种改写 prompt 对打，并且把「高分但跑偏」单独摘出来。

上一轮的 11/20 是按「top1 分数过门槛」算的，这个口径有个洞：
改写可能把问题漂移成**另一个问题**（「重量为什么因人而异」被改成「训练容量为何因人而异」），
然后高分命中另一条 chunk。分数好看，答案错得更彻底 —— 比拒答更坏。

所以这里同时报两个数：
  safe  = 过门槛 **且** top1 落在原问题所属话题（用户能拿到对路的答案）
  drift = 过门槛但 top1 跑到别的话题（危险，宁可拒答）

两个 prompt：
  old  app.REWRITE_PROMPT（v1，要求「改写成资料里可能出现的表述」）
  new  app.ORAL_PROMPT（v1，要求「书面表述 + 保留具体对象」）
同一批 20 条极低分口语问法，语料不动。

old 的分数直接读 .rewrite_rerank.json（已经不花钱算过了），只新算 new。
"""

import json
from pathlib import Path

import main as app

HERE = Path(__file__).parent
OLD = HERE / ".rewrite_rerank.json"
CACHE = HERE / ".oral_prompt_ab.json"
TH = app.RERANK_THRESHOLD


def scores_of(variants):
    """3 条改写各走一遍精排，返回 {variant: {chunk_id: score}}。走 .oral_prompt_ab.json 缓存。"""
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    kb_vecs = app.get_kb_vectors()
    todo = [v for v in variants if v not in cache]
    if todo:
        print(f"  需新算 {len(todo)} 条改写")
        for i, v in enumerate(todo, 1):
            vv = app.embed([v])[0]
            cand = sorted(((app.cosine(vv, kv), d) for d, kv in zip(app.KB, kb_vecs)),
                          key=lambda x: x[0], reverse=True)[: app.RECALL_K]
            out = app.rerank(v, [app.rerank_text(d) for _, d in cand])
            cache[v] = {cand[s["index"]][1]["id"]: s["score"] for s in out}
            print(f"    {i}/{len(todo)}", end="\r", flush=True)
        print()
        CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    return cache


def judge(var_names, score_map, cid, topic):
    """在一组候选改写里取最高分，返回 (分数, 是否落对自己话题, top1 id)。"""
    best_v = max(var_names, key=lambda v: max(score_map[v].values()))
    ranked = sorted(score_map[best_v].items(), key=lambda kv: kv[1], reverse=True)
    top_id, top_s = ranked[0]
    return top_s, app_topic(top_id) == topic, top_id, best_v


_topic_of = {d["id"]: d["topic"] for d in app.KB}


def app_topic(cid):
    return _topic_of[cid]


def main():
    pa = json.loads((HERE / ".phrasing_ab.json").read_text(encoding="utf-8"))
    old = json.loads(OLD.read_text(encoding="utf-8"))
    pairs = [(k.split("|", 1)[0], k.split("|", 1)[1]) for k in pa]
    pairs = list(dict.fromkeys(pairs))

    new_variants = {q: (app.oral_variants(q) or [q]) for _, q in pairs}
    flat = list(dict.fromkeys(v for vs in new_variants.values() for v in vs))
    new_scores = scores_of(flat)

    print(f"\n{'chunk':<11}{'old分':>7}{'new分':>7}{'  old安全':<9}{'new安全':<8} new 改写")
    print("-" * 96)
    agg = {"old": [0, 0], "new": [0, 0]}   # [过门槛, 安全]
    rows = []
    for cid, q in pairs:
        topic = _topic_of[cid]
        # old：直接用上一轮存下的最高分那次
        od = old[q]["result"]["best"]
        o_s, o_safe = od["score"], od["top_topic"] == topic
        # new
        n_s, n_safe, n_top, n_bestv = judge(new_variants[q], new_scores, cid, topic)

        agg["old"][0] += o_s >= TH
        agg["old"][1] += (o_s >= TH and o_safe)
        agg["new"][0] += n_s >= TH
        agg["new"][1] += (n_s >= TH and n_safe)
        rows.append((cid, q, o_s, n_s, o_safe, n_safe))
        flag = "" if (n_s >= TH) == (o_s >= TH) else ("  new赢" if n_s > o_s else "  **new输**")
        print(f"{cid:<11}{o_s:>7.3f}{n_s:>7.3f}  {'是' if o_safe else '否':<9}"
              f"{'是' if n_safe else '否':<8} {n_bestv[:30]}{flag}")

    n = len(pairs)
    print("-" * 96)
    for k, name in [("old", "旧 prompt"), ("new", "新 prompt")]:
        p, safe = agg[k]
        print(f"  {name}：过门槛 {p}/{n}　安全命中（过门槛且话题对）{safe}/{n}　"
              f"跑偏 {p - safe}/{n}")
    print("\n比较口径：看「安全命中」，不看「过门槛」。跑偏的高分是负资产。")
    print("注意 old 的分数取自上一轮，new 是这批现算的；语料、阈值、检索链路完全相同。")


if __name__ == "__main__":
    main()
