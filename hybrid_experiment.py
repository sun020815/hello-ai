# -*- coding: utf-8 -*-
"""混合检索的离线实验：向量 + BM25 怎么融合最好。

不调 embedding（复用 eval.py 的问题向量缓存），只算 BM25 和排序，零成本，
可以随便扫参数。选定方案后再改 main.py。

跑法：uv run python hybrid_experiment.py
"""
import eval as E
import main as app
from bm25 import BM25

CASES = E.CASES
TOP_K = app.VEC_TOP_K
VEC_GATE = app.VEC_THRESHOLD          # 0.60
# BM25 救援通道还要卡一道向量分下限：纯噪音问题（例如「推荐一部健身电影」）
# 里「健身」这个词也能命中一堆资料，但向量分很低，0.45 这道闸能挡住它们。
VEC_FLOOR = 0.45


def evaluate(score_rows, gate_bm=None):
    """score_rows: 每题 [(chunk下标, 排序分)] 降序。

    gate_bm 为 None 时，门控完全交给向量分（跟现在的纯向量逻辑一致）；
    传数字时额外开一条「BM25 救援」通道，用于对比 —— 实测那条路会救错。
    """
    rows = []
    for qi, (case, ranked) in enumerate(zip(CASES, score_rows)):
        vec_of = dict(vec_raw[qi])
        bm_first = bm_orders[qi][0]
        bm_first_score = bm_scores[qi][bm_first]

        kept, seen = [], set()
        for i, _ in ranked:
            ok = vec_of.get(i, 0) >= VEC_GATE
            if not ok and gate_bm is not None:
                ok = (i == bm_first and bm_first_score >= gate_bm
                      and vec_of.get(i, 0) >= VEC_FLOOR)
            if not ok:
                continue
            topic = app.KB[i]["topic"]
            if topic in seen:
                continue
            seen.add(topic)
            kept.append(topic)
            if len(kept) >= TOP_K:
                break

        got = kept[0] if kept else None
        if case["kind"] == "refuse":
            rows.append({"ok": got is None, "hit2": got is None, "kind": "refuse"})
        else:
            rows.append({"ok": got == case["expect"],
                         "hit2": case["expect"] in kept, "kind": "hit"})
    hit = [r for r in rows if r["kind"] == "hit"]
    ref = [r for r in rows if r["kind"] == "refuse"]
    return {
        "hit1": sum(r["ok"] for r in hit) / len(hit),
        "hit2": sum(r["hit2"] for r in hit) / len(hit),
        "refuse": sum(r["ok"] for r in ref) / len(ref),
        "acc": sum(r["ok"] for r in rows) / len(rows),
    }


def show(name, m):
    print(f"  {name:32s} hit@1 {m['hit1']*100:5.1f}%  hit@2 {m['hit2']*100:5.1f}%  "
          f"拒答 {m['refuse']*100:5.1f}%  整体 {m['acc']*100:5.1f}%")


q_vecs = E.get_q_vectors()
print(f"知识库 {len(app.KB)} 条 | 评测 {len(CASES)} 条\n")

kb_vecs = app.get_kb_vectors()
vec_raw = []
for qv in q_vecs:
    pairs = [(i, round(app.cosine(qv, v), 4)) for i, v in enumerate(kb_vecs)]
    pairs.sort(key=lambda p: p[1], reverse=True)
    vec_raw.append(pairs)

# ---------- 1. BM25 单独用 ----------
for with_topic in (False, True):
    texts = [f"{d['topic']}：{d['text']}" if with_topic else d["text"] for d in app.KB]
    bm = BM25(texts)
    raw = [bm.scores(c["q"]) for c in CASES]
    orders = [sorted(range(len(r)), key=lambda i: -r[i]) for r in raw]
    label = "带话题名" if with_topic else "纯正文"
    print(f"===== BM25 单独用（{label}）=====")
    for t in (0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0):
        # BM25 单独用时，排序分就是 BM25 分；把门控设成阈值本身即可
        rows = []
        for qi, case in enumerate(CASES):
            kept, seen = [], set()
            for i in orders[qi]:
                if raw[qi][i] < t:
                    continue
                topic = app.KB[i]["topic"]
                if topic in seen:
                    continue
                seen.add(topic)
                kept.append(topic)
                if len(kept) >= TOP_K:
                    break
            got = kept[0] if kept else None
            if case["kind"] == "refuse":
                rows.append({"ok": got is None, "hit2": got is None, "kind": "refuse"})
            else:
                rows.append({"ok": got == case["expect"],
                             "hit2": case["expect"] in kept, "kind": "hit"})
        hit = [r for r in rows if r["kind"] == "hit"]
        ref = [r for r in rows if r["kind"] == "refuse"]
        show(f"BM25 阈值 {t:.0f}", {
            "hit1": sum(r["ok"] for r in hit) / len(hit),
            "hit2": sum(r["hit2"] for r in hit) / len(hit),
            "refuse": sum(r["ok"] for r in ref) / len(ref),
            "acc": sum(r["ok"] for r in rows) / len(rows),
        })
    print()

# 往下用「纯正文」那版
bm = BM25([d["text"] for d in app.KB])
bm_scores = [bm.scores(c["q"]) for c in CASES]
bm_orders = [sorted(range(len(r)), key=lambda i: -r[i]) for r in bm_scores]

print("===== 纯向量基线（门控 = 向量分 >= 0.60）=====")
show("基线", evaluate(vec_raw))
print()

# ---------- 2. RRF 排序 + 向量门控 ----------
print("===== RRF 排序（门控仍只看向量分）=====")
for cand in (6, 12, 30, len(app.KB)):
    fused = []
    for qi in range(len(CASES)):
        score = {}
        for rank, (i, _) in enumerate(vec_raw[qi][:cand], 1):
            score[i] = score.get(i, 0.0) + 1 / (60 + rank)
        for rank, i in enumerate(bm_orders[qi][:cand], 1):
            score[i] = score.get(i, 0.0) + 1 / (60 + rank)
        fused.append(sorted(score.items(), key=lambda x: -x[1]))
    show(f"RRF cand={cand}", evaluate(fused))
print()

# ---------- 3. 加权排序 + 向量门控 ----------
print("===== 加权排序（门控仍只看向量分）=====")
for w in (0.7, 0.8, 0.9, 0.95, 1.0):
    fused = []
    for qi in range(len(CASES)):
        top_bm = max(bm_scores[qi]) or 1.0
        merged = {}
        for i, s in vec_raw[qi]:
            merged[i] = w * s
        for i, s in enumerate(bm_scores[qi]):
            merged[i] = merged.get(i, 0.0) + (1 - w) * (s / top_bm)
        fused.append(sorted(merged.items(), key=lambda x: -x[1]))
    show(f"w_vec={w:.2f}", evaluate(fused))
print()

# ---------- 4. 对照：开 BM25 救援通道会怎样 ----------
print("===== 对照：RRF + BM25 救援通道（说明为什么不用）=====")
fused12 = []
for qi in range(len(CASES)):
    score = {}
    for rank, (i, _) in enumerate(vec_raw[qi][:12], 1):
        score[i] = score.get(i, 0.0) + 1 / (60 + rank)
    for rank, i in enumerate(bm_orders[qi][:12], 1):
        score[i] = score.get(i, 0.0) + 1 / (60 + rank)
    fused12.append(sorted(score.items(), key=lambda x: -x[1]))
for g in (6.0, 8.0, 10.0):
    show(f"救援阈值 BM25>={g:.0f}", evaluate(fused12, gate_bm=g))
