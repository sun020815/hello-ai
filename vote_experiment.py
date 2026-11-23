# -*- coding: utf-8 -*-
"""投票门控实验：改写能不能顺便判断「该不该答」。

思路：如果 4 条表述（原问题 + 3 条改写）各自排第一的话题高度一致，
说明这是同一个真实意图；如果散在 4 个话题上，说明改写是被术语硬拽过去的。
前者的典型是「找不到发力感」，后者的典型是「推荐一部健身电影」。

这跟前两轮的区别：不是又在调阈值，而是换了一个判据。
"""
import json
from pathlib import Path

import eval as E
import main as app

HERE = Path(__file__).parent
CACHE = HERE / ".rewrite_vote.json"

rewrites = json.loads((HERE / ".rewrites.json").read_text(encoding="utf-8"))["items"]
qs = [c["q"] for c in E.CASES]


def get_variant_topics():
    """每条表述单独排一次，记下 top1 话题和分数。结果缓存，避免重算 348 次嵌入。"""
    if CACHE.exists():
        try:
            c = json.loads(CACHE.read_text(encoding="utf-8"))
            if c.get("n") == len(qs):
                return c["rows"]
        except Exception:
            pass

    kb_vecs = app.get_kb_vectors()
    rows = {}
    flat = [(q, t) for q in qs for t in [q] + rewrites[q]]
    vecs = []
    for part in E.chunked([t for _, t in flat], 16):
        vecs.extend(app.embed(part))

    for (q, _), v in zip(flat, vecs):
        sims = [(app.cosine(v, kv), d["topic"]) for d, kv in zip(app.KB, kb_vecs)]
        sims.sort(reverse=True)
        rows.setdefault(q, []).append({"score": round(sims[0][0], 4), "topic": sims[0][1]})
    CACHE.write_text(json.dumps({"n": len(qs), "rows": rows}, ensure_ascii=False),
                     encoding="utf-8")
    return rows


rows = get_variant_topics()


def vote(tops):
    """返回 (票数最多的话题, 票数)"""
    cnt = {}
    for t in tops:
        cnt[t["topic"]] = cnt.get(t["topic"], 0) + 1
    topic, n = max(cnt.items(), key=lambda x: x[1])
    return topic, n


def evaluate(need_votes, gate):
    """need_votes：至少几条表述排第一是同一话题才算「可信」。
    gate：原问题自己的分也要过这个底线，防止 1 条擦边 + 3 条垃圾凑票。"""
    out = []
    for case in E.CASES:
        tops = rows[case["q"]]
        topic, n = vote(tops)
        best = max(t["score"] for t in tops)

        if case["kind"] == "refuse":
            ok = not (n >= need_votes and best >= gate)
            out.append({"ok": ok, "hit2": ok, "kind": "refuse", "q": case["q"],
                        "got": topic if not ok else None})
        else:
            # 排序仍按最高分那条表述的名次走，这里简化成「话题对不对」
            ok = n >= need_votes and best >= gate and topic == case["expect"]
            out.append({"ok": ok, "hit2": ok, "kind": "hit", "q": case["q"],
                        "got": topic, "exp": case["expect"]})
    hit = [r for r in out if r["kind"] == "hit"]
    ref = [r for r in out if r["kind"] == "refuse"]
    return {
        "hit1": sum(r["ok"] for r in hit) / len(hit),
        "hit2": sum(r["hit2"] for r in hit) / len(hit),
        "refuse": sum(r["ok"] for r in ref) / len(ref),
        "acc": sum(r["ok"] for r in out) / len(out),
    }, out


if __name__ == "__main__":
    print("===== 投票分布：该命中的 vs 该拒答的 =====")
    for kind in ("hit", "refuse"):
        cnt = {}
        for case in E.CASES:
            if case["kind"] != kind:
                continue
            _, n = vote(rows[case["q"]])
            cnt[n] = cnt.get(n, 0) + 1
        total = sum(cnt.values())
        dist = "  ".join(f"{k}票:{v}条({v/total*100:.0f}%)" for k, v in sorted(cnt.items()))
        print(f"  {kind:7s} {dist}")
    print()

    print("===== 门控效果 =====")
    for need in (2, 3, 4):
        for gate in (0.45, 0.50, 0.55):
            m, _ = evaluate(need, gate)
            print(f"  需{need}票一致 且最高分>={gate:.2f}   hit@1 {m['hit1']*100:5.1f}%  "
                  f"拒答 {m['refuse']*100:5.1f}%  整体 {m['acc']*100:5.1f}%")

    print("\n===== 关键样本的一致性 =====")
    for q in ["找不到发力感怎么办", "意念口诀怎么用", "练的时候肌肉没感觉",
              "推荐一部健身题材的电影", "膝盖半月板损伤怎么康复", "健身教练证怎么考",
              "膝盖疼吃什么药", "一天要吃多少蛋白质"]:
        tops = rows[q]
        topic, n = vote(tops)
        detail = " / ".join(f"{t['topic']}({t['score']})" for t in tops)
        print(f"  {q[:16]:18s} {n}票 → {topic:6s}  {detail}")
