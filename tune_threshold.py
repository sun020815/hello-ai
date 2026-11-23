# -*- coding: utf-8 -*-
"""阈值扫描：把评测集在不同阈值下的表现一次算出来，选拐点。

问题向量只算一次（并缓存），所以扫几十个阈值和扫 1 个的成本一样。
用法：uv run python tune_threshold.py

注意：扩库之后这个脚本必须重跑一遍再确定 VEC_THRESHOLD。
40 条 chunk 之间的分数分布比 4 条时更挤，原来 0.55 那个平台区间大概率会移动。
"""
import main as app
from eval import CASES, get_q_vectors


def rank_all():
    """每条问题对全库 chunk 的完整排名，只算一次（问题向量走 eval.py 的缓存）"""
    q_vecs = get_q_vectors()
    kb_vecs = app.get_kb_vectors()
    out = []
    for qv in q_vecs:
        r = sorted(
            ({"id": d["id"], "topic": d["topic"], "score": app.cosine(qv, kv)}
             for d, kv in zip(app.KB, kb_vecs)),
            key=lambda x: x["score"], reverse=True,
        )
        out.append(r)
    return out


def score_at(rankings, th, top_k=app.VEC_TOP_K):
    hit1 = hit2 = 0
    ref_ok = ref_n = 0
    for case, ranking in zip(CASES, rankings):
        # 和线上同一套挑法：先过阈值，再按话题去重，最后取 top_k
        kept = [r["topic"] for r in app.pick_hits(ranking, top_k, th)]
        top = kept[0] if kept else None
        if case["kind"] == "refuse":
            ref_n += 1
            ref_ok += 1 if top is None else 0
        else:
            hit1 += 1 if top == case["expect"] else 0
            hit2 += 1 if case["expect"] in kept else 0
    n_hit = sum(1 for c in CASES if c["kind"] == "hit")
    n = len(CASES)
    return {
        "hit1": hit1 / n_hit, "hit2": hit2 / n_hit,
        "refuse": ref_ok / ref_n if ref_n else 1.0,
        "acc": (hit1 + ref_ok) / n,
    }


def run():
    rankings = rank_all()
    rows = []
    for i in range(40, 78):          # 0.40 ~ 0.77，扩库后分布会往两边散，范围放宽
        th = i / 100
        rows.append((th, score_at(rankings, th)))

    print("阈值     hit@1   hit@2   拒答率   整体准确率")
    print("-" * 48)
    best = max(rows, key=lambda x: (x[1]["acc"], x[1]["refuse"]))
    for th, m in rows:
        mark = "  ← 当前" if abs(th - app.VEC_THRESHOLD) < 1e-9 else ""
        mark += "  ← 最优" if th == best[0] else ""
        print(f"{th:.2f}    {m['hit1']*100:5.0f}%  {m['hit2']*100:5.0f}%  "
              f"{m['refuse']*100:5.0f}%   {m['acc']*100:5.0f}%{mark}")

    print(f"\n最优阈值 {best[0]:.2f}：整体准确率 {best[1]['acc']*100:.0f}%，"
          f"hit@1 {best[1]['hit1']*100:.0f}%，拒答率 {best[1]['refuse']*100:.0f}%")

    flat = [th for th, m in rows if m["acc"] == best[1]["acc"]]
    if len(flat) > 1:
        print(f"注意：{flat[0]:.2f} ~ {flat[-1]:.2f} 之间准确率一样，取中间值比取边界更稳。")
    return rows


if __name__ == "__main__":
    run()
