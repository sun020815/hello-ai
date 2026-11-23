# -*- coding: utf-8 -*-
"""query 改写实验：只测检索，不生成答案。

核心问题不是「改写能不能提高相似度」——一定能，取 K 条的最大值只会把分数抬高。
真正要回答的是：这份提高是给了该命中的题，还是也给了该拒答的题。

所以同时测两种门控：
  A. 排序用改写后的最高分，门控（答不答）仍看原问题的分  <- 排序/门控拆开
  B. 排序和门控都用改写后的最高分                      <- 直接替换，阈值必须重扫

改写和向量都落盘缓存，反复跑不额外花钱。用法：uv run python rewrite_experiment.py
"""
import json
from pathlib import Path

import eval as E
import main as app

HERE = Path(__file__).parent
CACHE = HERE / ".rewrites.json"

# 改写的三条要求：换说法不换意图、用资料里的术语、三条角度不同。
# 最后一句「不要回答」很重要，否则模型会直接编答案，嵌入的是答案不是问题。
REWRITE_PROMPT = (
    "你是检索助手。把用户的口语问题改写成 3 条健身资料里可能出现的表述。\n"
    "规则：\n"
    "1. 只换说法，不改变问题意图。不要回答这个问题。\n"
    "2. 用健身领域术语，比如 发力感、肌肉募集、离心控制、代偿、容量、渐进超负荷。\n"
    "3. 每条一句话，20 字以内。\n"
    "4. 三条之间角度要不同，不要只换同义词。\n"
    "只输出三行改写，不要编号，不要解释。"
)
PROMPT_VER = "v1"


def clean_line(line: str) -> str:
    """模型偶尔还是带编号或引号，清掉。"""
    s = line.strip().strip("\"'“”")
    for sep in (".", "、", ")", "）"):
        head, dot, tail = s.partition(sep)
        if dot and head.isdigit():
            s = tail.strip()
    return s


def get_rewrites():
    """改写缓存统一由 main.rewrite_many 管，这里只做一次调用，
    免得 experiment 和 eval 各写一份格式不同的 .rewrites.json。"""
    return app.rewrite_many([c["q"] for c in E.CASES])


def max_and_mean_scores(questions, rewrites):
    """每条问题：原问题 + 3 条改写，各自对全库算相似度，再逐 chunk 取最大/平均。
    返回 {q: (max列表, mean列表)}，顺序与 app.KB 一致。
    向量走 eval.get_flat_vectors 的缓存，反复跑不重复调 embedding。"""
    kb_vecs = app.get_kb_vectors()
    flat, groups = [], []
    for q in questions:
        group = [q] + rewrites[q]
        groups.append(group)
        flat.extend(group)

    vec_map = dict(zip(flat, E.get_flat_vectors(flat)))

    out = {}
    for q, group in zip(questions, groups):
        sims = [[app.cosine(vec_map[t], kv) for kv in kb_vecs] for t in group]
        n = len(group)
        out[q] = (
            [max(s[i] for s in sims) for i in range(len(kb_vecs))],
            [sum(s[i] for s in sims) / n for i in range(len(kb_vecs))],
        )
    return out


def build_ranking(scores, gate_scores=None):
    """排序按 scores，score 字段放 gate_scores（不给就同一份）。
    pick_hits 只认 score 字段，所以这样就能做到「排序和门控用不同分数」。"""
    gate = gate_scores if gate_scores is not None else scores
    ranking = []
    for d, s, g in zip(app.KB, scores, gate):
        ranking.append({"id": d["id"], "topic": d["topic"], "text": d["text"],
                        "source": d["source"], "score": round(g, 4)})
    order = sorted(range(len(ranking)), key=lambda i: -scores[i])
    return [ranking[i] for i in order]


def run(preds, threshold, name, quiet=False):
    rows = []
    for case, ranking in zip(E.CASES, preds):
        kept = app.pick_hits(ranking, app.VEC_TOP_K, threshold)
        got = kept[0]["topic"] if kept else None
        if case["kind"] == "refuse":
            rows.append({"ok": got is None, "hit2": got is None, "kind": "refuse",
                         "q": case["q"], "got": got})
        else:
            rows.append({"ok": got == case["expect"], "kind": "hit", "q": case["q"],
                         "got": got, "exp": case["expect"],
                         "hit2": case["expect"] in [k["topic"] for k in kept]})
    hit = [r for r in rows if r["kind"] == "hit"]
    ref = [r for r in rows if r["kind"] == "refuse"]
    m = {
        "hit1": sum(r["ok"] for r in hit) / len(hit),
        "hit2": sum(r["hit2"] for r in hit) / len(hit),
        "refuse": sum(r["ok"] for r in ref) / len(ref),
        "acc": sum(r["ok"] for r in rows) / len(rows),
    }
    if not quiet:
        print(f"  {name:34s} hit@1 {m['hit1']*100:5.1f}%  hit@2 {m['hit2']*100:5.1f}%  "
              f"拒答 {m['refuse']*100:5.1f}%  整体 {m['acc']*100:5.1f}%")
    return m, rows


if __name__ == "__main__":
    qs = [c["q"] for c in E.CASES]
    print(f"知识库 {len(app.KB)} 条 | 评测集 {len(qs)} 条 | 改写模型 {app.LLM_MODEL}")
    rewrites = get_rewrites()
    print("  改写样例：", rewrites[qs[0]])

    sc = max_and_mean_scores(qs, rewrites)
    q_vecs = E.get_q_vectors()
    kb_vecs = app.get_kb_vectors()

    orig = {}
    for q, v in zip(qs, q_vecs):
        orig[q] = [app.cosine(v, kv) for kv in kb_vecs]

    print("\n===== 基线 =====")
    base_preds = [build_ranking(orig[q]) for q in qs]
    base_m, base_rows = run(base_preds, app.VEC_THRESHOLD, f"纯向量 (阈值 {app.VEC_THRESHOLD})")

    print("\n===== A. 改写只改排序，门控仍看原问题 =====")
    preds_a = [build_ranking(sc[q][0], orig[q]) for q in qs]
    for t in (0.58, 0.60, 0.62):
        m, rows = run(preds_a, t, f"排序=改写max 门控=原始 (阈值 {t})")

    print("\n===== B. 改写同时改排序和门控（阈值重扫）=====")
    preds_b = [build_ranking(sc[q][0]) for q in qs]
    for t in (0.62, 0.64, 0.66, 0.68, 0.70, 0.72, 0.74):
        run(preds_b, t, f"全用改写max (阈值 {t})")

    print("\n===== C. 用 4 条的平均值而不是最大值 =====")
    preds_c = [build_ranking(sc[q][1]) for q in qs]
    for t in (0.52, 0.55, 0.58, 0.60, 0.62):
        run(preds_c, t, f"全用改写mean (阈值 {t})")

    # 逐条看 A 在最佳阈值下改了什么
    print("\n===== A 逐条变化（阈值 0.60）=====")
    _, rows_a = run(preds_a, 0.60, "A@0.60", quiet=True)
    for b, a in zip(base_rows, rows_a):
        if b["got"] != a["got"]:
            tag = "修好" if (a["ok"] and not b["ok"]) else ("弄坏" if (b["ok"] and not a["ok"]) else "换话题")
            print(f"  [{tag}] {b['q'][:20]:22s} {b['got'] or '拒答':6s} → {a['got'] or '拒答'}")

    # ---------- 最关键的一张表：同样拒答率下比命中率 ----------
    # 如果改写真有信息量，它应该在每个拒答率档位上都压过纯向量。
    # 如果两条曲线重合，说明改写只是把工作点挪了位置。
    print("\n===== 匹配对比：同样拒答率下，谁的命中率更高 =====")
    curves = {}
    for name, preds in (("纯向量", base_preds), ("改写max", preds_b)):
        pts = []
        for t in (x / 100 for x in range(38, 80)):
            m, _ = run(preds, t, name, quiet=True)
            pts.append((t, m["hit1"], m["refuse"]))
        curves[name] = pts

    print(f"  {'拒答率档位':14s} {'纯向量最佳 hit@1':>18s} {'改写最佳 hit@1':>16s}  差异")
    for target in (1.00, 0.90, 0.80, 0.70):
        line = f"  ≥{target*100:3.0f}%{'':9s}"
        vals = {}
        for name, pts in curves.items():
            ok = [(h, t) for t, h, r in pts if r >= target - 1e-9]
            vals[name] = max(ok) if ok else (None, None)
        b, rb = vals["纯向量"]
        w, rw_ = vals["改写max"]
        if b is None or w is None:
            print(line + "   数据不足")
            continue
        diff = (w - b) * 100
        print(f"{line} {b*100:15.1f}% (阈{rb:.2f}) {w*100:14.1f}% (阈{rw_:.2f})  "
              f"{diff:+.1f} 点")

    json.dump({"rewrites": rewrites}, open(".rewrite_sample.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
