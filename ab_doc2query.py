"""A/B：doc2query 到底有没有用，只改精排输入，其他全部相同。

两个坑必须先说清，否则这组对比会给出假结论：

1. 阈值不能共用。开了 doc2query 之后精排分数整体上移，拿同一个 0.50 去比
   是拿新刻度套旧尺子。所以**每个配置用自己的最优阈值**，再比。
2. 光在书面评测集上比不出东西。评测集的问法本身偏书面（「卧推手肘打开多少度」），
   doc2query 要解决的是口语鸿沟，问题不在这一侧。所以两个集都跑：
     eval_cases.json           200 题，书面偏多，用来确认「没把原来的弄坏」
     eval_cases_colloquial.json 24 题，纯口语，用来看「到底修好了多少」

排序指标（hit@1 / hit@2）不受阈值影响，先看它；拒答率和整体准确率各自扫最优。
"""

import json
from pathlib import Path

import main as app

HERE = Path(__file__).parent
CACHE = {False: HERE / ".ab_rerank_off.json", True: HERE / ".ab_rerank_on.json"}
TH_GRID = [round(i * 0.01, 2) for i in range(0, 100)]


def load_cases(name):
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def scores_for(cases, use_doc2query):
    """对每个问题算全库精排分：向量召回 12 条 -> 精排 -> {chunk_id: score}。"""
    app.USE_DOC2QUERY = use_doc2query
    path = CACHE[use_doc2query]
    cache = {}
    if path.exists():
        cache = json.loads(path.read_text(encoding="utf-8")).get("scores", {})

    todo = [c["q"] for c in cases if c["q"] not in cache]
    if todo:
        print(f"  [doc2query={use_doc2query}] 需新算 {len(todo)} 条")
        # 批量嵌入，别一条一个请求
        qvecs = {}
        for i in range(0, len(todo), 16):
            part = todo[i:i + 16]
            for q, v in zip(part, app.embed(part)):
                qvecs[q] = v
        kb_vecs = app.get_kb_vectors()
        for i, q in enumerate(todo, 1):
            recalled = sorted(
                ((app.cosine(qvecs[q], kv), d) for d, kv in zip(app.KB, kb_vecs)),
                key=lambda x: x[0], reverse=True,
            )[: app.RECALL_K]
            cand = [d for _, d in recalled]
            out = app.rerank(q, [app.rerank_text(d) for d in cand])
            cache[q] = {cand[s["index"]]["id"]: s["score"] for s in out}
            if i % 25 == 0 or i == len(todo):
                print(f"    {i}/{len(todo)}", flush=True)
        path.write_text(json.dumps({"scores": cache}, ensure_ascii=False), encoding="utf-8")
    return cache


TOPIC_OF = {d["id"]: d["topic"] for d in app.KB}


def topic_of(cid):
    return TOPIC_OF[cid]


def metrics(cases, scores, th):
    """自洽口径：命中题必须在阈值以上、且 top1 对，才算答对。
    不能像 eval.py 那样「hit@1 只看排序、拒答只看阈值」——那个口径下
    抬高阈值只改善拒答、不惩罚命中，整体准确率会单调上升，于是永远选最高阈值。
    """
    h1 = h2 = ref_ok = hit_n = ref_n = 0
    for c in cases:
        ranked = sorted(scores[c["q"]].items(), key=lambda kv: kv[1], reverse=True)
        kept = [cid for cid, sc in ranked if sc >= th][: app.RERANK_TOP_K]
        got = [topic_of(x) for x in kept]
        if c["kind"] == "refuse":
            ref_n += 1
            ref_ok += not kept
        else:
            hit_n += 1
            h1 += bool(got) and got[0] == c["expect"]
            h2 += c["expect"] in got
    n = len(cases)
    return {
        "hit1": h1 / hit_n if hit_n else 0,
        "hit2": h2 / hit_n if hit_n else 0,
        "refuse": ref_ok / ref_n if ref_n else 0,
        "acc": (h1 + ref_ok) / n,
        "n": n,
    }


def best_threshold(cases, scores):
    # 平局时优先拒答率高的一档（更保守），再不行取阈值高的那一档
    best = max(((metrics(cases, scores, t), t) for t in TH_GRID),
               key=lambda pair: (pair[0]["acc"], pair[0]["refuse"], pair[1]))
    return best[1], best[0]


def at_refuse_rate(cases, scores, target=0.90):
    """固定拒答率，看谁的命中更好。两个配置的分数刻度不同，直接比同一个阈值不公平。"""
    best = None
    for t in TH_GRID:
        m = metrics(cases, scores, t)
        d = abs(m["refuse"] - target)
        if best is None or d < best[0] - 1e-9:
            best = (d, t, m)
    return best[1], best[2]


def main():
    written = load_cases("eval_cases.json")
    colloquial = load_cases("eval_cases_colloquial.json")

    print(f"书面/综合集 {len(written)} 题，口语集 {len(colloquial)} 题\n")
    print("=== 计算精排分数 ===")
    off_w = scores_for(written, False)
    on_w = scores_for(written, True)
    off_c = scores_for(colloquial, False)
    on_c = scores_for(colloquial, True)

    print("\n" + "=" * 78)
    print("【综合集】200 题（95 命中 + 105 拒答），每个配置扫自己的最优阈值")
    print(f"{'':<12}{'hit@1':>9}{'hit@2':>9}{'拒答率':>9}{'整体':>9}{'阈值':>8}")
    print("-" * 78)
    summary = {}
    for label, sc in [("doc2query 关", off_w), ("doc2query 开", on_w)]:
        th, m = best_threshold(written, sc)
        summary[label] = (th, m)
        print(f"{label:<12}{m['hit1'] * 100:>8.1f}%{m['hit2'] * 100:>8.1f}%"
              f"{m['refuse'] * 100:>8.1f}%{m['acc'] * 100:>8.1f}%{th:>8.2f}")

    print("\n【综合集】把拒答率都压到 90%，比命中（同一起跑线）")
    for label, sc in [("doc2query 关", off_w), ("doc2query 开", on_w)]:
        th, m = at_refuse_rate(written, sc, 0.90)
        print(f"  {label:<14} 阈值 {th:.2f} -> hit@1 {m['hit1'] * 100:.1f}%  "
              f"hit@2 {m['hit2'] * 100:.1f}%  实际拒答 {m['refuse'] * 100:.0f}%")

    print("\n【口语集】24 题，全是该命中的，只看排序（阈值不参与）")
    print(f"{'':<12}{'hit@1':>9}{'hit@2':>9}")
    print("-" * 30)
    for label, sc in [("doc2query 关", off_c), ("doc2query 开", on_c)]:
        m = metrics(colloquial, sc, 0.0)
        print(f"{label:<12}{m['hit1'] * 100:>8.1f}%{m['hit2'] * 100:>8.1f}%")
    print("=" * 78)

    # ---- 口语集逐条：这是判断「修好了没有」的地方 ----
    print("\n口语集逐题（off -> on，分数为精排最高分）：")
    fixed = broke = 0
    for c in colloquial:
        q = c["q"]

        def verdict(scores):
            ranked = sorted(scores[q].items(), key=lambda kv: kv[1], reverse=True)
            return topic_of(ranked[0][0]), ranked[0][1]

        o, o_max = verdict(off_c)
        n, n_max = verdict(on_c)
        ok_o, ok_n = o == c["expect"], n == c["expect"]
        if ok_o and ok_n:
            continue
        if not ok_o and ok_n:
            flag, fixed = "  修好", fixed + 1
        elif ok_o and not ok_n:
            flag, broke = "  **弄坏**", broke + 1
        else:
            flag = ""
        print(f"  {q:<16} 期望 {c['expect']:<8} off={o}({o_max:.3f}) on={n}({n_max:.3f}){flag}")
    print(f"\n  修好 {fixed} 条，弄坏 {broke} 条")


if __name__ == "__main__":
    main()
