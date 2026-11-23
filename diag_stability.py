"""边界抖动量化：同一道题连续精排 N 次，看分数、排序、判定到底会不会变。

为什么单独测这个：/rag 的判定只有一行 `score >= RERANK_THRESHOLD`（0.95），
分数抖 0.001 就可能翻面。上一轮做生成层兜底时观察到「每次吸气要吸到多满」
第一次答、第二次拒，当时归因到 temperature=0.7，改成 0 之后抖动仍在 ——
真凶要么在这里（reranker 服务端），要么在召回（embedding 抖导致 top-12 换人）。

所以每次重复都把三件事一起记下来，才能分清是哪一环不稳：
  1. query 向量与第一次的余弦相似度      -> embedding 稳不稳
  2. 召回的 12 条 id 序列是否一致        -> 召回稳不稳
  3. top1 分数 / 排序 / 阈值判定是否变   -> 精排稳不稳

判定用线上同一个 pick_hits，不另写一套口径（这个项目已经因为两份逻辑分叉吃过亏）。

用法：
  uv run python diag_stability.py              # 默认贴边 20 条 + 对照 6 条，各 5 次
  uv run python diag_stability.py --n 3 --edge 10
"""

import json
import sys
import time
from pathlib import Path

import main as app

HERE = Path(__file__).parent
TH = app.RERANK_THRESHOLD
OUT = HERE / ".stability.json"

# 4B 的分数极度集中在两端：115 条在 0.99 以上、61 条在 0.20 以下。
# 真正「一抖就翻面」的只有贴着 0.95 的那一小撮，所以样本按 |Δ| 排序取，不随机抽。
EDGE_BAND = 0.04        # |score - 阈值| <= 0.04 视为贴边


def parse_args():
    n, edge, ctrl = 5, 20, 3
    a = sys.argv
    if "--n" in a:
        n = int(a[a.index("--n") + 1])
    if "--edge" in a:
        edge = int(a[a.index("--edge") + 1])
    if "--ctrl" in a:
        ctrl = int(a[a.index("--ctrl") + 1])
    return n, edge, ctrl


def load_candidates():
    """从已缓存的评测分数里挑样本，不再花钱重跑一遍 200 题。"""
    scores = json.loads((HERE / ".eval_reranks.json").read_text(encoding="utf-8"))["scores"]
    cases = json.loads((HERE / "eval_cases.json").read_text(encoding="utf-8"))
    col = json.loads((HERE / "eval_cases_colloquial.json").read_text(encoding="utf-8"))
    by_id = {d["id"]: d for d in app.KB}
    kind = {c["q"]: c.get("kind", "?") for c in cases}
    kind.update({c["q"]: "hit(口语)" for c in col})

    rows = []
    for q, sc in scores.items():
        idx, s = max(sc.items(), key=lambda kv: kv[1])
        rows.append({
            "q": q,
            "kind": kind.get(q, "?"),
            "base": s,
            "base_topic": by_id[idx]["topic"],
            "delta": s - TH,
        })
    return rows


def run_once(q):
    """走一遍线上链路，但把中间产物留下。刻意不调 app.rank_one：它把 embedding
    丢在里面，测不出「向量抖」和「精排抖」的区别。"""
    q_vec = app.embed([q])[0]
    kb_vecs = app.get_kb_vectors()

    recalled = []
    for d, kv in zip(app.KB, kb_vecs):
        recalled.append({
            "id": d["id"], "topic": d["topic"], "text": d["text"],
            "source": d["source"], "vec_score": round(app.cosine(q_vec, kv), 4),
        })
    recalled.sort(key=lambda x: x["vec_score"], reverse=True)
    recalled = recalled[: app.RECALL_K]

    scores = app.rerank(q, [app.rerank_text(r) for r in recalled])
    ranking = []
    for s in scores:
        item = dict(recalled[s["index"]])
        item["score"] = s["score"]
        ranking.append(item)
    ranking.sort(key=lambda x: x["score"], reverse=True)

    hits = app.pick_hits(ranking, app.RERANK_TOP_K, TH)
    return {
        "q_vec": q_vec,
        "recall_ids": [r["id"] for r in recalled],
        "top_id": ranking[0]["id"],
        "top_topic": ranking[0]["topic"],
        "top_score": ranking[0]["score"],
        "top2_ids": [r["id"] for r in ranking[:2]],
        "n_hits": len(hits),
        "answered": bool(hits),
    }


def main():
    n, edge, ctrl = parse_args()
    rows = load_candidates()

    band = sorted([r for r in rows if abs(r["delta"]) <= EDGE_BAND], key=lambda r: abs(r["delta"]))
    chosen = band[:edge]

    # 对照组：分数离阈值很远的题。如果它们也抖，说明「抖」是全局现象，
    # 不是边界特性，那整套阈值判定都要重新考虑。
    far_high = sorted([r for r in rows if r["base"] >= 0.999], key=lambda r: -r["base"])[:ctrl]
    far_low = sorted([r for r in rows if r["base"] <= 0.05], key=lambda r: r["base"])[:ctrl]
    for r in chosen:
        r["group"] = "贴边"
    for r in far_high:
        r["group"] = "远高"
    for r in far_low:
        r["group"] = "远低"
    sample = chosen + far_high + far_low

    print(f"精排模型 {app.RERANK_MODEL}   阈值 {TH}   每题重复 {n} 次")
    print(f"样本：贴边 {len(chosen)} 条（|Δ|≤{EDGE_BAND}）+ 远高 {len(far_high)} + 远低 {len(far_low)}"
          f" = {len(sample)} 条，共 {len(sample) * n} 次真实调用\n")

    results = []
    t0 = time.time()
    for i, s in enumerate(sample, 1):
        runs = []
        for _ in range(n):
            try:
                runs.append(run_once(s["q"]))
            except Exception as e:
                runs.append({"error": f"{type(e).__name__}: {e}"[:120]})
        ok = [r for r in runs if "error" not in r]
        if not ok:
            print(f"  [{i}/{len(sample)}] {s['q'][:22]}  全部失败：{runs[0]['error']}")
            continue

        scores = [r["top_score"] for r in ok]
        tops = [r["top_id"] for r in ok]
        verdicts = [r["answered"] for r in ok]
        recalls = [tuple(r["recall_ids"]) for r in ok]
        sims = [app.cosine(ok[0]["q_vec"], r["q_vec"]) for r in ok]

        rec = {
            "q": s["q"], "kind": s["kind"], "group": s["group"],
            "base": s["base"], "base_topic": s["base_topic"],
            "scores": scores,
            "min": min(scores), "max": max(scores), "spread": round(max(scores) - min(scores), 4),
            "top_changed": len(set(tops)) > 1,
            "flip": len(set(verdicts)) > 1,
            "recall_changed": len(set(recalls)) > 1,
            "min_embed_sim": round(min(sims), 6),
            "errors": len(runs) - len(ok),
        }
        results.append(rec)
        print(f"  [{i}/{len(sample)}] {s['q'][:20]:<22} "
              f"{rec['min']:.4f}~{rec['max']:.4f} 极差{rec['spread']:.4f} "
              f"{'翻面' if rec['flip'] else '    '} {'top1变' if rec['top_changed'] else '     '}"
              f" {'召回变' if rec['recall_changed'] else ''}", flush=True)

    if not results:
        print("没有成功样本")
        return

    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---------- 汇总 ----------
    print("\n" + "=" * 96)
    print(f"{'组':<6}{'题数':>4}{'翻面':>6}{'top1变':>8}{'召回变':>8}"
          f"{'最大极差':>10}{'中位极差':>10}{'embed最差相似':>14}")
    for g in ("贴边", "远高", "远低"):
        rs = [r for r in results if r["group"] == g]
        if not rs:
            continue
        sp = sorted(r["spread"] for r in rs)
        print(f"{g:<6}{len(rs):>4}{sum(r['flip'] for r in rs):>6}"
              f"{sum(r['top_changed'] for r in rs):>8}{sum(r['recall_changed'] for r in rs):>8}"
              f"{max(sp):>10.4f}{sp[len(sp) // 2]:>10.4f}"
              f"{min(r['min_embed_sim'] for r in rs):>14.6f}")

    print("\n【翻面的题（同一问题重复问，一次答一次拒）】")
    flips = [r for r in results if r["flip"]]
    if flips:
        for r in sorted(flips, key=lambda r: -r["spread"]):
            print(f"  {r['min']:.4f}~{r['max']:.4f}  极差 {r['spread']:.4f}  "
                  f"基线 {r['base']:.4f}  {r['q']}")
    else:
        print("  无")

    print("\n【top1 被换掉的题（抖动改变了「答什么」）】")
    changed = [r for r in results if r["top_changed"]]
    if changed:
        for r in changed:
            print(f"  {r['min']:.4f}~{r['max']:.4f}  {r['q']}")
    else:
        print("  无")

    print("\n【极差最大的 5 条】")
    for r in sorted(results, key=lambda r: -r["spread"])[:5]:
        print(f"  {r['spread']:.4f}  {r['q'][:34]}  ({r['min']:.4f}~{r['max']:.4f})")

    print(f"\n耗时 {time.time() - t0:.0f} 秒，明细写入 {OUT.name}")
    if any(r["errors"] for r in results):
        print(f"有 {sum(r['errors'] for r in results)} 次调用失败（已重试，仍失败才计入）")


if __name__ == "__main__":
    main()
