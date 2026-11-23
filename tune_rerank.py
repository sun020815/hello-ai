# -*- coding: utf-8 -*-
"""扫 rerank 阈值：用评测缓存（.eval_reranks.json）离线算，不花一分钱。

rerank 分数和余弦相似度不是一个量纲，0.55 那套数字完全不能套用，
必须单独扫一遍才知道该取多少。用法：uv run python tune_rerank.py

先跑 eval.py 把缓存生成出来，否则这个脚本没东西可读。

和上一版的区别：拒答样本从 10 条扩到 34 条，所以这一版不只回答「阈值取多少」，
还要回答三个更关键的问题：
  1. 34 条够不够？（Wilson 置信区间）
  2. 切不开的时候，错在哪几条上？（误判清单 -> 内容修改清单）
  3. 按难度分层的拒答率分别是多少？（一个总数会把难例掩盖掉）
"""
import json
from pathlib import Path

import main as app

HERE = Path(__file__).parent
CASES = json.loads((HERE / "eval_cases.json").read_text(encoding="utf-8"))
CACHE_PATH = HERE / ".eval_reranks.json"
SCORES = json.loads(CACHE_PATH.read_text(encoding="utf-8"))["scores"]   # {问题: {chunk_id: 分数}}
BY_ID = {d["id"]: d for d in app.KB}


def layer_of(case):
    """从 note 里读难度层：A 完全无关 / B 领域整块缺失 / C 邻近难例"""
    for L in ("A", "B", "C"):
        if f"层{L}" in case.get("note", ""):
            return L
    return "?"


def ranking_of(q):
    """把缓存里的 {id: 分数} 还原成排好序的列表"""
    rows = [{"id": i, "topic": BY_ID[i]["topic"], "score": s} for i, s in SCORES[q].items()]
    rows.sort(key=lambda x: x["score"], reverse=True)
    return rows


def wilson(k, n, z=1.96):
    """Wilson 置信区间。n=34 这种小样本不能用正态近似（会算出超过 100% 的区间），
    二项分布的 Wilson 区间在 k=0 或 k=n 时也不会退化成零宽度。"""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return max(0.0, center - half), min(1.0, center + half)


def score_at(th):
    """给定阈值，算 hit@1 / hit@2 / 拒答率 / 整体准确率，外加分层拒答率和误判清单"""
    h1 = h2 = 0
    hit_n = ref_ok = ref_n = 0
    by_layer = {}
    wrong_refuse = []      # 该拒却放行
    wrong_hit = []         # 该答却拦掉
    for c in CASES:
        # 和线上 search_kb_rerank 同一套挑法
        kept = [r["topic"] for r in app.pick_hits(ranking_of(c["q"]), app.RERANK_TOP_K, th)]
        got = kept[0] if kept else None
        top = ranking_of(c["q"])[0]
        if c["kind"] == "refuse":
            ref_n += 1
            ok = got is None
            ref_ok += ok
            L = layer_of(c)
            s = by_layer.setdefault(L, [0, 0])
            s[0] += ok
            s[1] += 1
            if not ok:
                wrong_refuse.append((top["score"], c["q"], got))
        else:
            hit_n += 1
            h1 += got == c["expect"]
            h2 += c["expect"] in kept
            if got != c["expect"]:
                wrong_hit.append((top["score"], c["q"], c["expect"], got))
    n = len(CASES)
    return {
        "th": th,
        "hit1": h1 / hit_n,
        "hit2": h2 / hit_n,
        "refuse": ref_ok / ref_n,
        "acc": (h1 + ref_ok) / n,
        "ref_ok": ref_ok, "ref_n": ref_n,
        "by_layer": by_layer,
        "wrong_refuse": wrong_refuse,
        "wrong_hit": wrong_hit,
    }


def main():
    print("【一、两类分数的分布：能不能用一条线切开】\n")
    hit_tops, ref_tops = [], []
    for c in CASES:
        top = ranking_of(c["q"])[0]
        (ref_tops if c["kind"] == "refuse" else hit_tops).append((top["score"], c["q"]))

    hit_tops.sort()
    ref_tops.sort(reverse=True)
    print(f"  应拒答 {len(ref_tops)} 条，最高分前五：")
    for s, q in ref_tops[:5]:
        print(f"    {s:.4f}  {q}")
    print(f"  该命中 {len(hit_tops)} 条，最低分前五：")
    for s, q in hit_tops[:5]:
        print(f"    {s:.4f}  {q}")
    print()
    overlap = sum(1 for s, _ in hit_tops if s < ref_tops[0][0])
    print(f"  → 有 {overlap} 条该命中的题，分数低于「该拒答的最高分 {ref_tops[0][0]:.4f}」")
    print("     这就是切不开的证据：任何阈值都会同时误伤和漏放。\n")

    # 扫描上限从数据里推出来，不写死。
    # 上一版写死 0.20（v1 的分数落在那），换语料后分数整体上移，
    # 工具停在真最优之前，报了个假的"最优"。
    upper = min(1.0, max(ref_tops[0][0], hit_tops[-1][0]) + 0.05)

    print("【二、扫一遍：只打印结果真的变了的转折点】\n")
    print(f"{'阈值':>8}{'hit@1':>9}{'hit@2':>9}{'拒答率':>9}{'整体准确':>10}")
    print("-" * 46)
    best = None
    th = 0.0000
    prev = None
    while th <= upper:
        m = score_at(th)
        cur = (m["hit1"], m["hit2"], m["refuse"], m["acc"])
        if cur != prev:
            print(f"{th:>8.4f}{m['hit1']*100:>8.0f}%{m['hit2']*100:>8.0f}%"
                  f"{m['refuse']*100:>8.0f}%{m['acc']*100:>9.0f}%")
            prev = cur
        if best is None or m["acc"] > best["acc"] or (
            abs(m["acc"] - best["acc"]) < 1e-9 and m["refuse"] > best["refuse"]
        ):
            best = m
        th = round(th + 0.005, 4)

    print("-" * 46)
    print(f"\n  整体准确率最高：阈值 {best['th']:.3f} → "
          f"hit@1 {best['hit1']*100:.0f}% / 拒答 {best['refuse']*100:.0f}% / "
          f"整体 {best['acc']*100:.0f}%")

    print("\n【三、这个拒答率有多可信（Wilson 95% 区间）】\n")
    for th_x in sorted({0.40, 0.45, 0.50, 0.535, 0.60, best["th"]}):
        m = score_at(th_x)
        lo, hi = wilson(m["ref_ok"], m["ref_n"])
        print(f"  阈值 {th_x:.3f}: 拒答 {m['ref_ok']}/{m['ref_n']} = {m['refuse']*100:.0f}%"
              f"   95% 区间 {lo*100:.0f}%~{hi*100:.0f}%")

    print("\n【四、分层拒答率（阈值 0.50）：难例被总数掩盖了】\n")
    m50 = score_at(0.50)
    name = {"A": "A 完全无关", "B": "B 领域整块缺失", "C": "C 邻近难例"}
    for L in ("A", "B", "C"):
        if L not in m50["by_layer"]:
            continue
        ok, n = m50["by_layer"][L]
        print(f"  {name[L]:<14} {ok:>2}/{n:<2} = {ok/n*100:>5.0f}%")

    print("\n【五、0.50 下判错的拒答题（这些是内容问题，不是阈值问题）】\n")
    if m50["wrong_refuse"]:
        for s, q, got in sorted(m50["wrong_refuse"], reverse=True):
            print(f"  {s:.4f}  {q:<22} -> 被当成「{got}」")
    else:
        print("  无")

    print("\n【六、五折交叉验证：最优点稳不稳】\n")
    import random
    random.seed(42)
    idx = list(range(len(CASES)))
    random.shuffle(idx)
    folds = [idx[i::5] for i in range(5)]
    picked, oos = [], []
    for k in range(5):
        hold = set(folds[k])
        train = [CASES[i] for i in idx if i not in hold]
        test = [CASES[i] for i in idx if i in hold]
        best_tr, best_acc = 0.0, -1
        for t in [round(i * 0.005, 4) for i in range(201)]:
            h1 = rt = hn = rn = 0
            for c in train:
                kept = [r["topic"] for r in app.pick_hits(ranking_of(c["q"]), app.RERANK_TOP_K, t)]
                got = kept[0] if kept else None
                if c["kind"] == "refuse":
                    rn += 1
                    rt += got is None
                else:
                    hn += 1
                    h1 += got == c["expect"]
            acc = (h1 + rt) / len(train)
            if acc > best_acc:
                best_acc, best_tr = acc, t
        h1 = rt = 0
        for c in test:
            kept = [r["topic"] for r in app.pick_hits(ranking_of(c["q"]), app.RERANK_TOP_K, best_tr)]
            got = kept[0] if kept else None
            if c["kind"] == "refuse":
                rt += got is None
            else:
                h1 += got == c["expect"]
        acc_te = (h1 + rt) / len(test)
        picked.append(best_tr)
        oos.append(acc_te)
        print(f"  折{k+1}: 训练集选出 {best_tr:.3f}（内部 {best_acc*100:.0f}%） → "
              f"留出集 {acc_te*100:.0f}%")
    print(f"  选出阈值 {[round(x,3) for x in picked]}")
    print(f"  极差 {max(picked)-min(picked):.3f}   留出平均 {sum(oos)/len(oos)*100:.0f}%")

    # 对照纯向量
    try:
        vec = json.loads((HERE / "eval_results.json").read_text(encoding="utf-8"))["vector"]["summary"]
        print(f"\n对照纯向量：hit@1 {vec['hit1']*100:.0f}% / 拒答 {vec['refuse']*100:.0f}% / "
              f"整体 {vec['acc']*100:.0f}%")
    except KeyError:
        print("\n对照纯向量：（eval_results.json 里没有 vector）")


if __name__ == "__main__":
    main()
