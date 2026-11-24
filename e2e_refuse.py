# -*- coding: utf-8 -*-
"""端到端拒答率：检索放行之后，生成层到底有没有守住。

为什么这个指标比检索口径重要：检索的「拒答率」只是阈值开关的统计。
真正决定用户会不会拿到错误建议的，是生成层看到错误资料后编不编。

上一轮实测发现边界题在 0.50 下生成层已经全部认怂 —— 那就得量一下
这个「第二道闸」到底有多可靠：34 条拒答题全部放行给生成层，数它守住几条。

用法：uv run python e2e_refuse.py [--th 0.50]
"""
import re
import sys

import main as app
from main import AskRequest

# 生成层认怂的话术特征。只做粗判，最终以人工看输出为准（脚本会把原文打出来）。
ADMIT = ("资料里没有", "资料中没有", "没有这部分", "没有相关", "无法", "不涉及",
         "没有提到", "未提及", "没有讲", "不在资料",
         # 2026-11-24 主生成提示词改成「这个我还答不上来」
         "答不上来", "不知道", "没法回答")


def admits(ans):
    return any(k in ans for k in ADMIT)


def main():
    th = 0.50
    if "--th" in sys.argv:
        th = float(sys.argv[sys.argv.index("--th") + 1])

    import json
    from pathlib import Path
    cases = json.loads((Path(__file__).parent / "eval_cases.json").read_text(encoding="utf-8"))
    ref = [c for c in cases if c["kind"] == "refuse"]

    old = app.RERANK_THRESHOLD
    app.RERANK_THRESHOLD = th
    n_ret_pass = n_e2e_ok = 0
    rows = []
    try:
        for i, c in enumerate(ref, 1):
            q = c["q"]
            r = app.rag(AskRequest(question=q))
            ret_pass = bool(r["sources"])          # 检索有没有放行
            ok = (not ret_pass) or admits(r["answer"])
            n_ret_pass += ret_pass
            n_e2e_ok += ok
            rows.append({"q": q, "ret": ret_pass, "ok": ok,
                         "topic": r["sources"][0]["topic"] if ret_pass else None,
                         "ans": r["answer"].replace("\n", " ").strip()})
            flag = "守住" if ok else "**漏了**"
            src = f"放行->{rows[-1]['topic']}" if ret_pass else "检索拦截"
            print(f"  {i:>2}/{len(ref)}  {flag:<8} {src:<18} {q[:26]:<28}"
                  f"{rows[-1]['ans'][:44]}", flush=True)
    finally:
        app.RERANK_THRESHOLD = old

    n = len(ref)
    print()
    print("=" * 72)
    print(f"拒答题 {n} 条，阈值 {th:.2f}")
    print(f"  检索自己拦下        {n - n_ret_pass}/{n} = {(n-n_ret_pass)/n*100:.0f}%")
    print(f"  检索放行            {n_ret_pass}/{n} = {n_ret_pass/n*100:.0f}%")
    print(f"  端到端守住（含生成层）{n_e2e_ok}/{n} = {n_e2e_ok/n*100:.0f}%")
    print(f"  → 生成层额外兜住    {n_e2e_ok - (n - n_ret_pass)} 条")
    print("=" * 72)

    missed = [r for r in rows if not r["ok"]]
    if missed:
        print("\n真正漏掉的（检索放行 + 生成层也答了）：")
        for r in missed:
            print(f"  {r['q']}  -> {r['topic']}")
            print(f"    答：{r['ans'][:150]}")


if __name__ == "__main__":
    main()
