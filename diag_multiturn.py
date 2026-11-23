"""多轮会话到底有没有用：单轮 vs 多轮，同一批对话直接比。

口径上做了一件关键的事：**第二轮的成绩只在「第一轮就答对了」的对话上统计**。
第一轮答错时，上文本身就是错的，第二轮的正确答案不该由它负责 ——
把那些算进来，等于用一个错误去量另一个错误。

三个集合：
  follow  追问（「那上斜呢」）—— 多轮该赢的地方
  control 第二轮本身就是完整问题 —— 多轮不该变差（变差说明补全在乱改）
  refuse  出界 / 康复 —— 多轮最容易漏放的地方（补全可能把出界问题补成健身问题）

跑法：uv run python diag_multiturn.py
      uv run python diag_multiturn.py --fresh     # 忽略检索缓存，全部重算
"""
import argparse
import json
import statistics as st
import time
from pathlib import Path

import main as app

CACHE = Path(".diag_multiturn_reranks.json")


def load_cases():
    return json.loads(Path("eval_cases_multiturn.json").read_text(encoding="utf-8"))


def kind_of(cid: str) -> str:
    if cid.startswith("ctl"):
        return "control"
    if cid in ("mt-22", "mt-23", "mt-24"):
        return "refuse"
    return "follow"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true", help="忽略检索缓存")
    args = ap.parse_args()

    memo = {}
    if CACHE.exists() and not args.fresh:
        memo = json.loads(CACHE.read_text(encoding="utf-8"))

    def rerank(q):
        """只做检索判定，不做生成 —— 要量的是「找没找对话题」，
        生成那一步只是在既定的几段资料里组织语言，跟多轮无关。"""
        if q in memo:
            return memo[q]
        hits, ranking = app.search_kb_rerank(q)
        top = ranking[0]["score"] if ranking else 0.0
        memo[q] = {"topic": hits[0]["topic"] if hits else None,
                   "score": round(float(top), 4)}
        CACHE.write_text(json.dumps(memo, ensure_ascii=False, indent=1), encoding="utf-8")
        return memo[q]

    cases = load_cases()
    rows = []
    for c in cases:
        cid, turns = c["id"], c["turns"]
        kind = kind_of(cid)

        # ---- 单轮：每句话都当新问题 ----
        single = [rerank(t["q"])["topic"] for t in turns]

        # ---- 多轮：走生产代码同一条路径（resolve_question + session_append）----
        sid = f"diag:{cid}"
        app.session_clear(sid)
        multi, resolved, condensed, costs = [], [], [], []
        for t in turns:
            t0 = time.perf_counter()
            q, _hist_n, did = app.resolve_question(t["q"], sid)
            if did:
                # 只有真补全的那轮才计成本；完整问题不走这条路
                costs.append(time.perf_counter() - t0)
            else:
                costs.append(None)
            r = rerank(q)
            topic = r["topic"]
            app.session_append(sid, t["q"], q, topic)
            multi.append(topic)
            resolved.append(q)
            condensed.append(did)
        app.session_clear(sid)

        def ok(topic, expect):
            return (topic is None) if expect == "refuse" else (topic == expect)

        rows.append({
            "id": cid, "kind": kind, "turns": turns,
            "single": single, "multi": multi,
            "resolved": resolved, "condensed": condensed,
            "ok1_single": ok(single[0], turns[0]["expect"]),
            "ok1_multi": ok(multi[0], turns[0]["expect"]),
            "ok2_single": ok(single[1], turns[1]["expect"]),
            "ok2_multi": ok(multi[1], turns[1]["expect"]),
            "score2": rerank(resolved[1])["score"],
            "costs": costs,
        })

    # ---------- 汇总 ----------
    print(f"\n共 {len(rows)} 段对话（follow {sum(r['kind']=='follow' for r in rows)} / "
          f"control {sum(r['kind']=='control' for r in rows)} / "
          f"refuse {sum(r['kind']=='refuse' for r in rows)}）")

    print("\n【第一轮：两种模式必须一致】")
    same = sum(1 for r in rows if r["ok1_single"] == r["ok1_multi"])
    correct = sum(1 for r in rows if r["ok1_multi"])
    print(f"  第一轮判对 {correct}/{len(rows)}，两模式结论一致 {same}/{len(rows)}")
    if same != len(rows):
        for r in rows:
            if r["ok1_single"] != r["ok1_multi"]:
                print(f"    不一致 {r['id']}: single={r['single'][0]} multi={r['multi'][0]}")

    def report(kind, label):
        sel = [r for r in rows if r["kind"] == kind]
        # 只保留第一轮答对的：上文错了不是第二轮的锅
        gate = [r for r in sel if r["ok1_multi"]]
        a = sum(1 for r in gate if r["ok2_single"])
        b = sum(1 for r in gate if r["ok2_multi"])
        n = len(gate)
        print(f"\n【{label}】{n} 段（共 {len(sel)} 段，剔除第一轮就答错的 {len(sel)-n} 段）")
        print(f"  单轮 {a}/{n} = {a/n*100 if n else 0:.0f}%   多轮 {b}/{n} = {b/n*100 if n else 0:.0f}%   "
              f"差 {b-a:+d} 段")
        return sel, gate

    follow, follow_gate = report("follow", "追问：多轮该赢的地方")
    control, control_gate = report("control", "对照：第二轮本身完整，多轮不该变差")
    refuse, refuse_gate = report("refuse", "出界/康复：多轮最容易漏放")

    print("\n【补全的成本】")
    all_costs = [c for r in rows for c in r["costs"] if c]
    if all_costs:
        print(f"  触发 {len(all_costs)} 次、共 {len(rows)*2} 轮，"
              f"占 {len(all_costs)/(len(rows)*2)*100:.0f}% 的轮次")
        print(f"  单次耗时 中位 {st.median(all_costs)*1000:.0f}ms  "
              f"P90 {sorted(all_costs)[int(len(all_costs)*0.9)]*1000:.0f}ms  "
              f"最长 {max(all_costs)*1000:.0f}ms")
        print(f"  即：完整问题零成本，只有省略的追问多花一次调用")

    print("\n【补全行为】")
    turned = [(r, i) for r in rows for i, d in enumerate(r["condensed"]) if d]
    changed = [(r, i) for r, i in turned if r["resolved"][i] != r["turns"][i]["q"]]
    print(f"  触发补全 {len(turned)} 次，其中真的改写了 {len(changed)} 次"
          f"（{len(turned)-len(changed)} 次原样返回 = 白花一次调用）")
    for r, i in turned:
        mark = "改" if r["resolved"][i] != r["turns"][i]["q"] else "原"
        hit = "✓" if r["ok2_multi"] else "✗"
        print(f"    [{mark}] {r['id']} 轮{i+1}  {r['turns'][i]['q']}  →  {r['resolved'][i]}  {hit}")

    print("\n【逐条明细】（第二轮）")
    print(f"  {'id':8} {'模式':4} {'问法':22} {'单轮判给':12} {'多轮判给':12} 分   ")
    for r in rows:
        t2 = r["turns"][1]
        print(f"  {r['id']:8} {r['kind'][:4]:4} {t2['q'][:20]:22} "
              f"{str(r['single'][1])[:10]:12} {str(r['multi'][1])[:10]:12} "
              f"{r['score2']:.3f}")

    out = Path(".diag_multiturn.json")
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n明细写 {out}")


if __name__ == "__main__":
    main()
