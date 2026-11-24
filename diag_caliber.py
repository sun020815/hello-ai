"""口径诊断：把话题级判错的题逐条摊开，看是真答错，还是答案在相邻话题里。

只看 rerank（默认链路）。走本地缓存，不花 token。

用法：
    .venv/Scripts/python.exe diag_caliber.py               # 综合集 200
    .venv/Scripts/python.exe diag_caliber.py --set oral    # 口语集 135
"""
import json
import sys

import main as app
import eval as E

SETS = {
    "main": ("eval_cases.json", E),
}


def load_cases(which):
    path = "eval_cases.json" if which == "main" else "eval_cases_colloquial.json"
    cases = json.load(open(path, encoding="utf-8"))
    return cases


def rerank_preds(questions):
    """与 eval.predict_rerank 同逻辑，但返回完整 ranking 方便看分数分布。"""
    scores = E.get_rerank_scores()
    by_id = {d["id"]: d for d in app.KB}
    out = []
    for q in questions:
        ranking = [{"id": i, "topic": by_id[i]["topic"], "score": s}
                   for i, s in scores[q].items()]
        ranking.sort(key=lambda x: x["score"], reverse=True)
        kept = app.pick_hits(ranking, app.RERANK_TOP_K, app.RERANK_THRESHOLD)
        out.append({"ranking": ranking, "kept": kept,
                    "top_score": ranking[0]["score"] if ranking else None})
    return out


def main():
    which = "main"
    if "--set" in sys.argv:
        which = sys.argv[sys.argv.index("--set") + 1]
    cases = load_cases(which)
    qs = [c["q"] for c in cases]
    preds = rerank_preds(qs)
    by_id = {d["id"]: d for d in app.KB}

    wrong = []
    for c, p in zip(cases, preds):
        if c["kind"] == "refuse":
            continue
        got_topics = [k["topic"] for k in p["kept"]]
        ok1 = bool(got_topics) and got_topics[0] == c["expect"]
        if not ok1:
            wrong.append((c, p, got_topics))

    print(f"集：{which}  n={len(cases)}  RERANK_TOP_K={app.RERANK_TOP_K} "
          f"阈值={app.RERANK_THRESHOLD}")
    print(f"判错 {len(wrong)} 条（含该答没答）\n")

    for i, (c, p, got) in enumerate(wrong, 1):
        print("=" * 78)
        print(f"[{i}] {c['q']}")
        print(f"    期望话题：{c['expect']}    note: {c.get('note', '')[:60]}")
        top3 = [(k["topic"], round(k["score"], 4)) for k in p["ranking"][:3]]
        print(f"    精排 top3：{top3}")
        print(f"    实际保留：{got or '（拒答）'}")
        # gold 话题下所有 chunk 的正文
        gold = [d for d in app.KB if d["topic"] == c["expect"]]
        print(f"    --- gold「{c['expect']}」共 {len(gold)} 条 ---")
        for d in gold[:3]:
            print(f"    [{d['id']}] {d['text'][:110]}")
        # 实际返回的第一条正文
        if p["kept"]:
            kid = p["kept"][0]["id"]
            print(f"    --- 实际返回 [{kid}] ---")
            print(f"    {by_id[kid]['text'][:110]}")

    # 汇总：判错的题里，gold 话题是否出现在 top3
    in_top3 = sum(1 for c, p, g in wrong
                  if any(k["topic"] == c["expect"] for k in p["ranking"][:3]))
    print("\n" + "=" * 78)
    print(f"判错 {len(wrong)} 条中，gold 话题仍在 top3 的：{in_top3} 条（排序问题）")
    print(f"gold 话题掉出 top3 的：{len(wrong) - in_top3} 条（召回问题）")


if __name__ == "__main__":
    main()
