# -*- coding: utf-8 -*-
"""检索评测：同一批问题，分别用「关键词」和「向量」跑一遍，算命中率。

只测检索，不调大模型 —— 快、免费、每次结果一样。
用法：uv run python eval.py
"""
import json
from pathlib import Path

import main as app

HERE = Path(__file__).parent
CASES = json.loads((HERE / "eval_cases.json").read_text(encoding="utf-8"))
Q_CACHE = HERE / ".eval_qvecs.json"
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


def evaluate(mode):
    questions = [c["q"] for c in CASES]
    preds = predict_vector(questions) if mode == "vector" else predict_keyword(questions)

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
    for mode in ("keyword", "vector"):
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

    # 两次结果对比，只列出结论有变化的
    print("===== 逐条对比（只显示答案不同的）=====")
    for k, v in zip(all_rows["keyword"], all_rows["vector"]):
        if k["got"] != v["got"]:
            arrow = "✅" if (v["hit1"] and not k["hit1"]) else ("❌" if (k["hit1"] and not v["hit1"]) else "·")
            print(f"  {arrow} {k['q']:<22} 关键词={k['got'] or '拒答':<6} 向量={v['got'] or '拒答'}")

    (HERE / "eval_results.json").write_text(
        json.dumps({m: {"summary": summarize(all_rows[m]), "rows": all_rows[m]} for m in all_rows},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\n明细已写入 eval_results.json")


if __name__ == "__main__":
    run_report()
