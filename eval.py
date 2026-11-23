# -*- coding: utf-8 -*-
"""检索评测：同一批问题，分别用「关键词」「向量」「向量+reranker」跑一遍，算命中率。

只测检索，不调大模型 —— 快、免费、每次结果一样（rerank 结果本地缓存）。
用法：uv run python eval.py
"""
import json
from pathlib import Path

import main as app

HERE = Path(__file__).parent
CASES = json.loads((HERE / "eval_cases.json").read_text(encoding="utf-8"))
Q_CACHE = HERE / ".eval_qvecs.json"
RERANK_CACHE = HERE / ".eval_reranks.json"
QUESTIONS = [c["q"] for c in CASES]


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
    for part in chunked(QUESTIONS, 16):     # 批量嵌入：46 条分 3 次调用，而不是 46 次
        vecs.extend(app.embed(part))
    Q_CACHE.write_text(
        json.dumps({"model": app.EMBED_MODEL, "questions": QUESTIONS, "vectors": vecs},
                   ensure_ascii=False), encoding="utf-8")
    return vecs


def predict_vector(questions):
    q_vecs = get_q_vectors()

    kb_vecs = app.get_kb_vectors()
    names = list(app.KB.keys())

    results = []
    for q_vec in q_vecs:
        ranking = [
            {"topic": name, "score": app.cosine(q_vec, kb_vec)}
            for name, kb_vec in zip(names, kb_vecs)
        ]
        ranking.sort(key=lambda x: x["score"], reverse=True)
        # 跟线上接口完全一致的规则：先过阈值，再取前 top_k 条
        kept = [r["topic"] for r in ranking if r["score"] >= app.VEC_THRESHOLD][: app.VEC_TOP_K]
        results.append({"topics": kept, "score": ranking[0]["score"]})
    return results


def predict_keyword(questions):
    results = []
    for q in questions:
        hits = app.search_kb_keyword(q)
        results.append({"topics": [h["topic"] for h in hits], "score": None})
    return results


def kb_docs():
    """reranker 的输入文本：和线上接口保持一致（话题名 + 正文）"""
    return [f"{name}：{text}" for name, text in app.KB.items()]


def get_rerank_scores():
    """每条问题调一次 reranker，结果缓存到本地。
    每调一次都要花钱，而调参要反复跑很多遍，所以必须缓存。
    改了知识库或换了模型会自动失效重算。"""
    docs = kb_docs()
    scores = {}
    if RERANK_CACHE.exists():
        try:
            c = json.loads(RERANK_CACHE.read_text(encoding="utf-8"))
            if c.get("model") == app.RERANK_MODEL and c.get("docs") == docs:
                scores = c.get("scores", {})
        except Exception:
            pass

    todo = [q for q in QUESTIONS if q not in scores]
    if todo:
        print(f"  reranker 需新算 {len(todo)} 条（其余走缓存）")
        for i, q in enumerate(todo, 1):
            scores[q] = app.rerank(q, docs)
            print(f"    {i}/{len(todo)}", end="\r", flush=True)
        print()
        RERANK_CACHE.write_text(
            json.dumps({"model": app.RERANK_MODEL, "docs": docs, "scores": scores},
                       ensure_ascii=False),
            encoding="utf-8",
        )
    return scores


def predict_rerank(questions):
    """向量召回全部 4 条（现在库里就 4 条）→ reranker 精排 → 按 rerank 分数过阈值"""
    scores = get_rerank_scores()
    names = list(app.KB.keys())
    results = []
    for q in questions:
        ranked = sorted(scores[q], key=lambda x: x["score"], reverse=True)
        kept = [names[r["index"]] for r in ranked if r["score"] >= app.RERANK_THRESHOLD]
        results.append({"topics": kept[: app.RERANK_TOP_K], "score": ranked[0]["score"]})
    return results


PREDICTORS = {
    "keyword": predict_keyword,
    "vector": predict_vector,
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
    print(f"评测集：{len(CASES)} 条（应命中 {sum(1 for c in CASES if c['kind']=='hit')} 条，"
          f"应拒答 {sum(1 for c in CASES if c['kind']=='refuse')} 条）\n")

    all_rows = {}
    for mode in ("keyword", "vector", "rerank"):
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

    # 三种方式逐条对比，只列出结论有变化的
    print("===== 逐条对比（只显示三种结论不完全一致的）=====")
    for k, v, r in zip(all_rows["keyword"], all_rows["vector"], all_rows["rerank"]):
        if not (k["got"] == v["got"] == r["got"]):
            print(f"  {k['q']:<22} 关键词={k['got'] or '拒答':<6} "
                  f"向量={v['got'] or '拒答':<6} rerank={r['got'] or '拒答'}")

    (HERE / "eval_results.json").write_text(
        json.dumps({m: {"summary": summarize(all_rows[m]), "rows": all_rows[m]} for m in all_rows},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\n明细已写入 eval_results.json")


if __name__ == "__main__":
    run_report()
