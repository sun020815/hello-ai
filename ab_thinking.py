"""思维链开不开：成本砍一半，质量代价是多少。

背景：cost_report.py 实测输出占比不高（174 token vs rerank 2097），但输出单价是输入的
4 倍，按示例价折算**输出占总成本 70%，其中思维链又占输出的 75%** —— 一半的钱花在「想」上。

补全调用（condense_llm）早就关了推理（开 2.7~4.6 秒 / 190~219 token，关 0.7~1.9 秒 / 48~49 token）。
主生成一直没关，因为答案是用户真正要看的东西，关掉可能掉质量。

这个脚本就是来量「掉多少」的。同一批问题、同一份资料、temperature=0，
只差 enable_thinking 一个布尔，各跑一遍，把答案并排打出来。

用法：
    uv run python ab_thinking.py            # 默认 8 条，跑两轮
    uv run python ab_thinking.py -n 4       # 少跑几条快速看
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from pathlib import Path

from fastapi.testclient import TestClient

import main

# 覆盖四类：能答的好题、需要综合的宽问题、贴门线的、库里没有的
QUESTIONS = [
    "减脂期间，推荐吃什么呢",       # 宽问题，要多条资料综合
    "一周瘦几斤算正常",             # 数字题，最容易被思维链影响精度
    "深蹲膝盖疼怎么办",             # 需要给出边界（库里未必有康复内容）
    "推日之后二头还练吗",           # 训练安排的因果推理
    "坚果一天吃多少",               # 单点数字
    "有氧一般要做多久",
    "我妈妈血压高能练深蹲吗",        # 应该拒答的医疗类
    "帮我写个增肌计划表",           # 应该拒答的生成类
]

NUM_RE = re.compile(r"\d")


def metrics(text: str) -> dict:
    return {
        "chars": len(text),
        "nums": len(NUM_RE.findall(text)),
        "admits_gap": ("资料" in text and "没有" in text) or "库里没有" in text,
    }


def run_round(client, headers, questions, thinking):
    main.GEN_THINKING = thinking
    rows = []
    for q in questions:
        main._cache.clear()
        before = dict(main.COST)
        t0 = time.perf_counter()
        r = client.post("/rag", json={"question": q}, headers=headers)
        wall = (time.perf_counter() - t0) * 1000
        if r.status_code != 200:
            print(f"  !! {q} HTTP {r.status_code}")
            continue
        d = r.json()
        used = {k: main.COST[k] - before.get(k, 0) for k in main.COST}
        ans = d.get("answer") or ""
        rows.append({
            "q": q, "wall_ms": round(wall),
            "refused": bool(d.get("refused")),
            "sources": len(d.get("sources") or []),
            "completion_tokens": used["completion_tokens"],
            "reasoning_tokens": used["reasoning_tokens"],
            "answer": ans,
            **metrics(ans),
        })
    return rows


def main_run():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=8)
    ap.add_argument("--json-out", default="ab_thinking.json")
    args = ap.parse_args()

    main.QUERY_LOG_ON = False
    main.RATE_LIMIT_PER_MIN = 10_000
    client = TestClient(main.app)
    headers = {"x-access-code": main.ACCESS_CODE} if main.ACCESS_CODE else {}
    qs = QUESTIONS[: args.n]

    print("预热…")
    client.post("/rag", json={"question": "热身"}, headers=headers)

    print(f"\n【A】不传 enable_thinking（供应商默认，线上现状）")
    a_rows = run_round(client, headers, qs, None)
    print(f"\n【B】显式关闭 enable_thinking")
    b_rows = run_round(client, headers, qs, False)

    def agg(rows, key):
        vals = [r[key] for r in rows]
        return statistics.mean(vals) if vals else 0

    print("\n" + "=" * 74)
    print(f"{'指标':<22}{'A 默认':>14}{'B 关推理':>14}{'变化':>16}")
    print("-" * 74)

    def line(label, ka, kb, unit="", pct=True):
        va, vb = agg(a_rows, ka), agg(b_rows, kb)
        if pct and va:
            chg = f"{(vb - va) / va * 100:+.1f}%"
        else:
            chg = f"{vb - va:+.0f}"
        print(f"{label:<22}{va:>12.1f}{unit}{vb:>12.1f}{unit}{chg:>16}")

    line("耗时 ms", "wall_ms", "wall_ms", "", True)
    line("输出 token", "completion_tokens", "completion_tokens")
    line("其中思维链", "reasoning_tokens", "reasoning_tokens")
    line("答案字数", "chars", "chars")
    line("答案里的数字个数", "nums", "nums")
    print(f"{'拒答条数':<22}{sum(r['refused'] for r in a_rows):>14}"
          f"{sum(r['refused'] for r in b_rows):>14}")
    print(f"{'承认资料不足':<22}{sum(r['admits_gap'] for r in a_rows):>14}"
          f"{sum(r['admits_gap'] for r in b_rows):>14}")

    print("\n" + "=" * 74)
    print("逐条答案对比（自己看质量，别只看数字）")
    for ra, rb in zip(a_rows, b_rows):
        print(f"\n【{ra['q']}】  来源 {ra['sources']} 条")
        print(f"  A（默认，{ra['completion_tokens']}t/思维链{ra['reasoning_tokens']}t，{ra['wall_ms']}ms）")
        print("    " + ra["answer"].replace("\n", "\n    ")[:420])
        print(f"  B（关推理，{rb['completion_tokens']}t/思维链{rb['reasoning_tokens']}t，{rb['wall_ms']}ms）")
        print("    " + rb["answer"].replace("\n", "\n    ")[:420])

    Path(args.json_out).write_text(
        json.dumps({"A_default": a_rows, "B_no_thinking": b_rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(f"\n明细已写入 {args.json_out}")


if __name__ == "__main__":
    main_run()
