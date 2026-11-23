"""换 reranker 横向对比：只换模型名，语料/召回/阈值逻辑一个字不动。

为什么不能直接比「过 0.50 的条数」：
  不同 reranker 的分数尺度完全不同（同一个问题 bge 给 0.99、0.6B 给 1.000、
  4B 给 0.9998），0.50 这个线在各模型上含义不一样。拿同一根尺子量不同模型，
  量出来的差异是尺子的差异，不是模型能力的差异。

所以主指标用 AUC（阈值无关）：
  正样本 = 库里确实有答案、只是问法口语的题（期望 top1 分高）
  负样本 = 库里真没有的题（期望 top1 分低）
  AUC = 随机取一正一负，正样本分数更高的概率。0.5 = 瞎猜，1.0 = 完美分开。
排序指标（hit@1 / 落对话题）本身就是阈值无关的，一起报。

用法：
  uv run python ab_rerank_models.py --limit 3      # 先小样本确认链路
  uv run python ab_rerank_models.py                # 全量
  uv run python ab_rerank_models.py --report       # 只读缓存出报告
"""

import json
import random
import sys
import time
from pathlib import Path

import main as app

HERE = Path(__file__).parent
CACHE = HERE / ".rerank_models.json"

MODELS = [
    "BAAI/bge-reranker-v2-m3",     # 现状，免费
    "Qwen/Qwen3-Reranker-0.6B",    # ¥0.07/M
    "Qwen/Qwen3-Reranker-4B",      # ¥0.14/M
    "Qwen/Qwen3-Reranker-8B",      # ¥0.28/M
]

N_REFUSE = 40          # 负样本抽多少条
SEED = 11              # 固定抽样，保证每个模型看到完全相同的题


def load_sets(limit=None):
    """三组样本：口语低分题（有正确答案 chunk）、口语集（有正确话题）、真拒答题（负样本）。"""
    oral = json.loads((HERE / ".phrasing_ab.json").read_text(encoding="utf-8"))
    oral20 = [{"q": k.split("|", 1)[1], "cid": k.split("|", 1)[0]} for k in oral]

    col = json.loads((HERE / "eval_cases_colloquial.json").read_text(encoding="utf-8"))
    col24 = [{"q": c["q"], "cid": None, "topic": c["expect"]} for c in col if c["kind"] == "hit"]

    cases = json.loads((HERE / "eval_cases.json").read_text(encoding="utf-8"))
    refuse = [c["q"] for c in cases if c["kind"] == "refuse"]
    random.Random(SEED).shuffle(refuse)
    refuse40 = refuse[:N_REFUSE]

    if limit:
        oral20, col24, refuse40 = oral20[:limit], col24[:limit], refuse40[:limit]
    return oral20, col24, refuse40


def run_model(model, oral20, col24, refuse40, cache):
    entry = cache.setdefault(model, {})
    topic_of = {d["id"]: d["topic"] for d in app.KB}

    todo = [s for s in oral20 + col24 if s["q"] not in entry or "error" in entry[s["q"]]] + \
           [{"q": q, "cid": None, "topic": None} for q in refuse40
            if q not in entry or "error" in entry[q]]
    if not todo:
        print(f"  {model}: 全部命中缓存")
        return

    orig = app.RERANK_MODEL
    app.RERANK_MODEL = model
    print(f"  {model}: 要跑 {len(todo)} 条")
    try:
        for i, s in enumerate(todo, 1):
            q = s["q"]
            t0 = time.time()
            try:
                ranking = app.rank_one(q)
                top = ranking[0]
                entry[q] = {
                    "top_id": top["id"], "top_topic": top["topic"],
                    "top_score": top["score"],
                    "ms": round((time.time() - t0) * 1000),
                }
            except Exception as e:
                entry[q] = {"error": f"{type(e).__name__}: {e}"[:120]}
            if i % 10 == 0 or i == len(todo):
                CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
                print(f"    {i}/{len(todo)}")
    finally:
        app.RERANK_MODEL = orig


def auc(pos, neg):
    """正样本分数高于负样本的比例。并列记 0.5，避免用 >= 抬高结果。"""
    if not pos or not neg:
        return None
    win = 0.0
    for p in pos:
        for n in neg:
            win += 1.0 if p > n else (0.5 if p == n else 0.0)
    return win / (len(pos) * len(neg))


def best_cut(pos, neg):
    """在各模型自己的候选阈值里找准确率最高的切点。
    这是「该模型最好能到多少」，不是「用统一 0.50 能到多少」——
    后者对分数尺度不同的模型不公平。"""
    if not pos or not neg:
        return 0.0, None
    cands = sorted(set(pos + neg))
    best = (0.0, None)
    for t in cands:
        acc = (sum(1 for p in pos if p >= t) + sum(1 for n in neg if n < t)) / (len(pos) + len(neg))
        if acc > best[0]:
            best = (acc, t)
    return best


def report(cache, oral20, col24, refuse40):
    topic_of = {d["id"]: d["topic"] for d in app.KB}
    oral_topic = {s["q"]: topic_of[s["cid"]] for s in oral20}
    col_topic = {s["q"]: s["topic"] for s in col24}

    print("\n" + "=" * 96)
    print(f"{'模型':<26}{'hit@1自己':>11}{'落对话题':>10}{'正样本均分':>11}"
          f"{'负样本均分':>11}{'AUC':>7}{'最优切点':>11}{'延迟':>9}")
    print("=" * 96)

    rows = []
    for m in MODELS:
        e = cache.get(m) or {}
        ok1 = ok_topic = n_oral = 0
        for s in oral20:
            r = e.get(s["q"])
            if not r or "error" in r:
                continue
            n_oral += 1
            ok1 += r["top_id"] == s["cid"]
            ok_topic += r["top_topic"] == oral_topic[s["q"]]
        n_col = ok_col_topic = 0
        for s in col24:
            r = e.get(s["q"])
            if not r or "error" in r:
                continue
            n_col += 1
            ok_col_topic += r["top_topic"] == col_topic[s["q"]]

        pos = [e[s["q"]]["top_score"] for s in oral20 + col24 if s["q"] in e and "error" not in e[s["q"]]]
        neg = [e[q]["top_score"] for q in refuse40 if q in e and "error" not in e[q]]
        a = auc(pos, neg)
        acc, t = best_cut(pos, neg)
        ms = [e[s["q"]]["ms"] for s in oral20 + col24 if s["q"] in e and "ms" in e[s["q"]]]

        tot_topic = n_oral + n_col
        auc_str = f"{a:>7.3f}" if a is not None else "     --"
        cut_str = f"{acc:.1%}@{t:.2f}".rjust(11) if t is not None else "         --"
        line = (f"{m.split('/')[-1]:<26}"
                f"{ok1}/{n_oral:<8}"
                f"{ok_topic + ok_col_topic}/{tot_topic:<6}"
                f"{sum(pos)/len(pos) if pos else 0:>11.3f}"
                f"{sum(neg)/len(neg) if neg else 0:>11.3f}"
                f"{auc_str}{cut_str}"
                f"{sum(ms)//len(ms) if ms else 0:>7}ms")
        print(line)
        rows.append((m, ok1, n_oral, ok_topic + ok_col_topic, tot_topic, a, acc, t))

    # 正样本按组拆开看：口语集是「本来就该答对」的，救不回来才要命
    print("\n分组明细（正样本均分 / 过自身最优切点比例）：")
    for m in MODELS:
        e = cache.get(m) or {}
        for name, s in (("20 极低分", oral20), ("口语集", col24)):
            sc = [e[x["q"]]["top_score"] for x in s if x["q"] in e and "error" not in e[x["q"]]]
            if not sc:
                continue
            print(f"  {m.split('/')[-1]:<22}{name:<10}均分 {sum(sc)/len(sc):.3f}  n={len(sc)}")

    print("\n结论看 AUC 与「落对话题」，不看最优切点准确率——后者是各模型自己挑的阈值，"
          "换到线上没有意义（线上只能定一个阈值）。")


if __name__ == "__main__":
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])

    oral20, col24, refuse40 = load_sets(limit)
    print(f"样本：极低分口语 {len(oral20)} 条 + 口语集 {len(col24)} 条 + 真拒答 {len(refuse40)} 条"
          f" = {len(oral20) + len(col24) + len(refuse40)} 题 × {len(MODELS)} 模型")

    cache = {}
    if CACHE.exists():
        cache = json.loads(CACHE.read_text(encoding="utf-8"))
    if "--report" not in sys.argv:
        for m in MODELS:
            run_model(m, oral20, col24, refuse40, cache)
        CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")

    errs = [(m, q, r["error"]) for m, e in cache.items() for q, r in e.items() if "error" in r]
    if errs:
        print(f"\n有 {len(errs)} 条调用失败，例如：{errs[0]}")
    report(cache, oral20, col24, refuse40)
