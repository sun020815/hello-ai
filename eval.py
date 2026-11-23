# -*- coding: utf-8 -*-
"""检索评测：同一批问题，分别用「关键词」「向量」「向量+reranker」跑一遍，算命中率。

只测检索，不调大模型 —— 快、免费、每次结果一样（向量和 rerank 结果都本地缓存）。
用法：uv run python eval.py

扩库到 40 条之后有两处必须跟着改，不然测出来的东西是假的：
1. 命中判定按「话题」算，不按 chunk 算 —— 一个话题可能有多条 chunk，评测集只标到话题这一层；
2. rerank 那段只精排「向量召回的那 12 条」，跟线上完全一致，不能直接拿全库去 rerank。
"""
import hashlib
import json
from pathlib import Path

import main as app

HERE = Path(__file__).parent
CASES = json.loads((HERE / "eval_cases.json").read_text(encoding="utf-8"))
Q_CACHE = HERE / ".eval_qvecs.json"
RERANK_CACHE = HERE / ".eval_reranks.json"
QUESTIONS = [c["q"] for c in CASES]

# 知识库指纹：正文或 id 一改，缓存就作废。
# 只比对条数是不够的 —— 改了某一条的正文，条数没变，缓存会拿旧分数骗人。
KB_FINGERPRINT = hashlib.md5(
    "|".join(f"{d['id']}::{d['text']}" for d in app.KB).encode("utf-8")
).hexdigest()


def chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def get_q_vectors():
    """问题向量缓存：同一批问题反复评测，没必要每次重算（每次都要花钱）。
    改了 eval_cases.json 会自动失效重算。"""
    if Q_CACHE.exists():
        try:
            c = json.loads(Q_CACHE.read_text(encoding="utf-8"))
            if c.get("model") == app.EMBED_MODEL and c.get("questions") == QUESTIONS:
                return c["vectors"]
        except Exception:
            pass

    vecs = []
    for part in chunked(QUESTIONS, 16):     # 批量嵌入：分几次调用，而不是一条一次
        vecs.extend(app.embed(part))
    Q_CACHE.write_text(
        json.dumps({"model": app.EMBED_MODEL, "questions": QUESTIONS, "vectors": vecs},
                   ensure_ascii=False), encoding="utf-8")
    return vecs


def vector_ranking(q_vec):
    """一条问题对全库的向量排名，返回 [(chunk, 分数)]，降序。"""
    kb_vecs = app.get_kb_vectors()
    pairs = [(d, round(app.cosine(q_vec, v), 4)) for d, v in zip(app.KB, kb_vecs)]
    pairs.sort(key=lambda p: p[1], reverse=True)
    return pairs


def predict_vector(questions):
    q_vecs = get_q_vectors()
    results = []
    for q_vec in q_vecs:
        pairs = vector_ranking(q_vec)
        ranking = [{"id": d["id"], "topic": d["topic"], "score": s} for d, s in pairs]
        # 跟线上接口用同一个函数挑，保证评测和线上不会各走一套逻辑
        kept = app.pick_hits(ranking, app.VEC_TOP_K, app.VEC_THRESHOLD)
        results.append({"topics": [r["topic"] for r in kept],
                        "ids": [r["id"] for r in kept],
                        "score": pairs[0][1]})
    return results


def predict_keyword(questions):
    results = []
    for q in questions:
        hits = app.search_kb_keyword(q)
        results.append({"topics": [h["topic"] for h in hits],
                        "ids": [h["id"] for h in hits], "score": None})
    return results


def get_rerank_scores():
    """每条问题：向量召回 RECALL_K 条 -> 只对这 12 条调 reranker -> 结果缓存。
    和线上 search_kb_rerank 走完全相同的路径。
    每调一次都要花钱，调参要反复跑很多遍，所以必须缓存。
    改了知识库（指纹变化）或换了模型会自动失效重算。"""
    cache = {}
    if RERANK_CACHE.exists():
        try:
            c = json.loads(RERANK_CACHE.read_text(encoding="utf-8"))
            if c.get("model") == app.RERANK_MODEL and c.get("fingerprint") == KB_FINGERPRINT:
                cache = c.get("scores", {})
        except Exception:
            pass

    todo = [q for q in QUESTIONS if q not in cache]
    if todo:
        print(f"  reranker 需新算 {len(todo)} 条（其余走缓存），每条精排 {app.RECALL_K} 段")
        q_vecs = dict(zip(QUESTIONS, get_q_vectors()))
        for i, q in enumerate(todo, 1):
            cand = [d for d, _ in vector_ranking(q_vecs[q])[: app.RECALL_K]]
            docs = [f"{d['topic']}：{d['text']}" for d in cand]
            out = app.rerank(q, docs)
            cache[q] = {cand[s["index"]]["id"]: s["score"] for s in out}
            print(f"    {i}/{len(todo)}", end="\r", flush=True)
        print()
        RERANK_CACHE.write_text(
            json.dumps({"model": app.RERANK_MODEL, "fingerprint": KB_FINGERPRINT,
                        "scores": cache}, ensure_ascii=False),
            encoding="utf-8",
        )
    return cache


def predict_rerank(questions):
    scores = get_rerank_scores()
    by_id = {d["id"]: d for d in app.KB}
    results = []
    for q in questions:
        ranking = [{"id": i, "topic": by_id[i]["topic"], "score": s}
                   for i, s in scores[q].items()]
        ranking.sort(key=lambda x: x["score"], reverse=True)
        kept = app.pick_hits(ranking, app.RERANK_TOP_K, app.RERANK_THRESHOLD)
        results.append({"topics": [r["topic"] for r in kept],
                        "ids": [r["id"] for r in kept],
                        "score": ranking[0]["score"] if ranking else None})
    return results


def predict_hybrid(questions):
    """向量 + BM25 混合。走 app.hybrid_rank，和线上一模一样的排序逻辑，
    只是问题向量从缓存拿，不重复调 embedding。"""
    q_vecs = get_q_vectors()
    results = []
    for q, q_vec in zip(questions, q_vecs):
        ranking = app.hybrid_rank(q, q_vec)
        kept = app.pick_hits(ranking, app.VEC_TOP_K, app.VEC_THRESHOLD)
        results.append({"topics": [r["topic"] for r in kept],
                        "ids": [r["id"] for r in kept],
                        "score": ranking[0]["score"] if ranking else None})
    return results


PREDICTORS = {
    "keyword": predict_keyword,
    "vector": predict_vector,
    "hybrid": predict_hybrid,
    "rerank": predict_rerank,
}


def evaluate(mode):
    questions = [c["q"] for c in CASES]
    preds = PREDICTORS[mode](questions)

    rows = []
    for case, pred in zip(CASES, preds):
        got = pred["topics"][0] if pred["topics"] else None
        if case["kind"] == "refuse":
            ok1 = got is None          # 该拒答，也确实拒了
            ok2 = ok1
        else:
            ok1 = got == case["expect"]        # hit@1：第一条就对了
            ok2 = case["expect"] in pred["topics"]  # hit@2：前两条里有对的
        rows.append({
            "q": case["q"], "expect": case["expect"], "kind": case["kind"],
            "got": got, "hit1": ok1, "hit2": ok2,
            "score": pred["score"], "note": case.get("note", ""),
        })
    return rows


def summarize(rows):
    hit_cases = [r for r in rows if r["kind"] == "hit"]
    ref_cases = [r for r in rows if r["kind"] == "refuse"]
    return {
        "n": len(rows),
        "hit1": sum(r["hit1"] for r in hit_cases) / len(hit_cases),
        "hit2": sum(r["hit2"] for r in hit_cases) / len(hit_cases),
        "refuse": (sum(r["hit1"] for r in ref_cases) / len(ref_cases)) if ref_cases else 1.0,
        "acc": sum(r["hit1"] for r in rows) / len(rows),
        "hit_n": len(hit_cases),
        "ref_n": len(ref_cases),
    }


def pct(x):
    return f"{x * 100:.0f}%"


def run_report():
    print(f"知识库：{len(app.KB)} 条 chunk，{len(app.TOPICS)} 个话题")
    print(f"评测集：{len(CASES)} 条（应命中 {sum(1 for c in CASES if c['kind']=='hit')} 条，"
          f"应拒答 {sum(1 for c in CASES if c['kind']=='refuse')} 条）\n")

    all_rows = {}
    for mode in ("keyword", "vector", "hybrid", "rerank"):
        rows = evaluate(mode)
        all_rows[mode] = rows
        m = summarize(rows)
        print(f"===== {mode} =====")
        print(f"  hit@1（第一条就对）  {pct(m['hit1'])}   {sum(r['hit1'] for r in rows if r['kind']=='hit')}/{m['hit_n']}")
        print(f"  hit@2（前两条有）    {pct(m['hit2'])}   {sum(r['hit2'] for r in rows if r['kind']=='hit')}/{m['hit_n']}")
        print(f"  拒答正确率           {pct(m['refuse'])}   {sum(r['hit1'] for r in rows if r['kind']=='refuse')}/{m['ref_n']}")
        print(f"  整体准确率           {pct(m['acc'])}   {sum(r['hit1'] for r in rows)}/{m['n']}")

        bad = [r for r in rows if not r["hit1"]]
        if bad:
            print(f"  --- 答错的 {len(bad)} 条 ---")
            for r in bad:
                if r["kind"] == "refuse":
                    print(f"    [应拒答却命中] {r['q']}  →  {r['got']}")
                else:
                    print(f"    [期望 {r['expect']}] {r['q']}  →  {r['got'] or '拒答'}")
        print()

    print("===== 逐条对比（只显示几种结论不完全一致的）=====")
    for k, v, h, r in zip(all_rows["keyword"], all_rows["vector"],
                          all_rows["hybrid"], all_rows["rerank"]):
        if not (k["got"] == v["got"] == h["got"] == r["got"]):
            print(f"  {k['q']:<22} 关键词={k['got'] or '拒答':<6} "
                  f"向量={v['got'] or '拒答':<6} 混合={h['got'] or '拒答':<6} "
                  f"rerank={r['got'] or '拒答'}")

    (HERE / "eval_results.json").write_text(
        json.dumps({m: {"summary": summarize(all_rows[m]), "rows": all_rows[m]} for m in all_rows},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\n明细已写入 eval_results.json")


if __name__ == "__main__":
    run_report()
