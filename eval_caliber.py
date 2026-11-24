"""评测口径改造：把「整体准确率」拆成召回 / 准入门 / 排序四层。

背景：原来只有一个 hit@1（第一条话题是否等于标注）。这个数把三种完全
不同性质的失败混成一个：向量没召回、召回了但分数不够被门槛拦、过线了
但排到第二。三者的修法完全不同（换 embedding / 调阈值 / 换精排），
混在一起看就会去优化错的地方。

四桶互斥，加起来正好等于正例总数：

  ok    gold 过阈值且排在第一位 —— 真答到了
  rank  gold 过阈值但不是第一   —— 资料在上下文里，只是不是第一条
  gate  gold 排第一但没过阈值   —— 找到了，门槛拦了（该答没答）
  miss  召回的 12 条里没有该话题 —— 向量层就没捞到

负例单独看：kept 非空 = 误放。

用法：
    .venv/Scripts/python.exe eval_caliber.py            # 两套集都跑
    .venv/Scripts/python.exe eval_caliber.py --set main # 只跑综合集
"""
import json
import sys

import main as app
import eval as E

SETS = {
    "main": "eval_cases.json",
    "oral": "eval_cases_colloquial.json",
}


def rank_for(q, scores, by_id):
    ranking = [{"id": i, "topic": by_id[i]["topic"], "score": v}
               for i, v in scores[q].items()]
    ranking.sort(key=lambda x: x["score"], reverse=True)
    return ranking


def analyze(cases, scores, by_id):
    rows = []
    for c in cases:
        ranking = rank_for(c["q"], scores, by_id)
        kept = app.pick_hits(ranking, app.RERANK_TOP_K, app.RERANK_THRESHOLD)
        got = [k["topic"] for k in kept]

        if c["kind"] == "refuse":
            rows.append({"q": c["q"], "kind": "refuse", "bucket": "ref",
                         "false_pass": bool(kept), "got": got,
                         "ranking": ranking,
                         "top": ranking[0]["score"] if ranking else 0.0})
            continue

        exp = c["expect"]
        gold = [r for r in ranking if r["topic"] == exp]
        if not gold:
            bucket, best = "miss", None
        else:
            best = gold[0]                      # 该话题下分最高的那一条
            if best["score"] < app.RERANK_THRESHOLD:
                bucket = "gate"
            elif got and got[0] == exp:
                bucket = "ok"
            else:
                bucket = "rank"
        rows.append({"q": c["q"], "kind": "hit", "bucket": bucket,
                     "expect": exp, "got": got, "ranking": ranking,
                     "best": best, "hit2": exp in got,
                     "note": c.get("note", "")})
    return rows


def bucket_report(rows, label, n_all):
    pos = [r for r in rows if r["kind"] == "hit"]
    neg = [r for r in rows if r["kind"] == "refuse"]
    cnt = {b: sum(1 for r in pos if r["bucket"] == b) for b in ("ok", "rank", "gate", "miss")}
    false_pass = sum(1 for r in neg if r["false_pass"])

    print(f"\n{'=' * 70}")
    print(f"集：{label}   n={n_all}（正例 {len(pos)} / 负例 {len(neg)}）"
          f"   top_k={app.RERANK_TOP_K}   阈值={app.RERANK_THRESHOLD}")
    print(f"{'=' * 70}")

    hit1 = cnt["ok"] / len(pos)
    hit_ctx = sum(1 for r in pos if r["hit2"]) / len(pos)
    recall = (len(pos) - cnt["miss"]) / len(pos)
    refuse = (len(neg) - false_pass) / len(neg) if neg else 1.0
    acc = (cnt["ok"] + len(neg) - false_pass) / n_all

    print(f"\n【旧口径（一个数）】")
    print(f"  hit@1 {hit1 * 100:5.1f}%   hit@ctx {hit_ctx * 100:5.1f}%   "
          f"拒答 {refuse * 100:5.1f}%   整体 {acc * 100:5.1f}%")
    print(f"  hit@ctx = 答案是否进入喂给模型的上下文（kept 最多 {app.RERANK_TOP_K} 条）。")
    print(f"  它和 hit@1 的差 = 排序抖动，不是答不出来："
          f"差 {(hit_ctx - hit1) * 100:.1f} 个点")

    print(f"\n【新口径（四个桶，加起来 = 正例 {len(pos)}）】")
    print(f"  {'桶':<6} {'条数':>5} {'占比':>7}   说明")
    desc = {
        "ok": "gold 过线且排第一 —— 真答到",
        "rank": "gold 过线但非第一 —— 资料在上下文里",
        "gate": "gold 排第一但没过线 —— 门槛拦了",
        "miss": "召回 12 条里没这个话题 —— 向量层没捞到",
    }
    for b in ("ok", "rank", "gate", "miss"):
        print(f"  {b:<6} {cnt[b]:>5} {cnt[b] / len(pos) * 100:6.1f}%   {desc[b]}")

    print(f"\n【可修复量】")
    print(f"  召回@12（gold 进了候选）   {recall * 100:5.1f}%"
          f"   ← 向量层的能力上限，miss 越小越不该动 embedding")
    print(f"  其中被门槛白扔的          {cnt['gate']} 条"
          f"   ← 调到合适阈值就能立刻捡回来")
    print(f"  其中排错位的              {cnt['rank']} 条"
          f"   ← 靠精排/多给一条解决（hit@ctx 已覆盖 {sum(1 for r in pos if r['hit2'])} 条）")
    print(f"  负例误放                  {false_pass}/{len(neg)}")
    return {"cnt": cnt, "pos": len(pos), "neg": len(neg),
            "false_pass": false_pass, "hit1": hit1, "acc": acc,
            "hit_ctx": hit_ctx, "recall": recall}


def threshold_tradeoff(rows, label):
    """同一批资料，只动阈值：救回多少该答的，代价是漏放多少不该答的。"""
    pos = [r for r in rows if r["kind"] == "hit"]
    neg = [r for r in rows if r["kind"] == "refuse"]
    print(f"\n【阈值权衡 · {label}】降阈值能救回多少、代价多大")
    print(f"  {'阈值':>5} {'该答答到':>9} {'负例漏放':>9} {'整体acc':>8}   {'救回':>4} {'多漏':>4}")
    base_loss = sum(1 for r in neg if r["false_pass"])
    base_get = sum(1 for r in pos if r["bucket"] == "ok")
    for t in (0.95, 0.94, 0.92, 0.90, 0.88, 0.85, 0.80):
        got = 0
        for r in pos:
            kept = app.pick_hits(r["ranking"], app.RERANK_TOP_K, t)
            if kept and kept[0]["topic"] == r["expect"]:
                got += 1
        leak = sum(1 for r in neg if app.pick_hits(r["ranking"], app.RERANK_TOP_K, t))
        acc = (got + len(neg) - leak) / (len(pos) + len(neg))
        print(f"  {t:5.2f} {got / len(pos) * 100:8.1f}% {leak / len(neg) * 100:8.1f}% "
              f"{acc * 100:7.1f}%   {got - base_get:+4d}  {leak - base_loss:+4d}")


def list_bucket(rows, bucket, limit=12):
    items = [r for r in rows if r["bucket"] == bucket]
    if not items:
        return
    print(f"\n--- {bucket} 桶明细（{len(items)} 条，列前 {min(limit, len(items))}）---")
    for r in items[:limit]:
        best = r.get("best")
        sc = f"{best['score']:.4f}" if best else "—"
        print(f"  [{sc}] {r['q']}")
        print(f"        期望 {r['expect']}   实得 {r['got'] or '（拒答）'}")


def main():
    which = sys.argv[sys.argv.index("--set") + 1] if "--set" in sys.argv else None
    scores = E.get_rerank_scores()
    by_id = {d["id"]: d for d in app.KB}
    all_rows = {}
    for label, path in SETS.items():
        if which and label != which:
            continue
        cases = json.load(open(path, encoding="utf-8"))
        rows = analyze(cases, scores, by_id)
        all_rows[label] = rows
        bucket_report(rows, label, len(cases))

    for label, rows in all_rows.items():
        threshold_tradeoff(rows, label)
    for label, rows in all_rows.items():
        print(f"\n########## {label} ##########")
        for b in ("gate", "rank", "miss"):
            list_bucket(rows, b)


if __name__ == "__main__":
    main()
