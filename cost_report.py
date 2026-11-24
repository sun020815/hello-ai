"""成本核算：一条真实提问到底花多少 token、多少钱。

为什么单独写一个：
  1. 面试必问「成本怎么控」，答不出具体数字就是空话；
  2. 运营必看 —— 粉丝涨十倍，账单也涨十倍，得知道钱花在 embed、rerank 还是生成；
  3. 缓存到底省了多少，必须有对比数，不能靠感觉。

只统计 token，不硬编码单价（单价会变、不同账号有折扣）。要钱就填 prices.json：
    {
      "embed_per_m": 0,          # 每百万 token 价格，0 表示这个模型免费
      "rerank_per_m": 0.1,
      "prompt_per_m": 2,
      "completion_per_m": 8,
      "currency": "CNY"
    }

用法：
    uv run python cost_report.py                 # 只出 token，不出钱
    uv run python cost_report.py --prices prices.json
    uv run python cost_report.py -n 20           # 跑 20 条
    uv run python cost_report.py --mode rerank   # 指定检索模式
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from fastapi.testclient import TestClient

import main

# 真实粉丝问法：取自公众号后台日志里出现过的原话，不是对着正文反推的题。
QUESTIONS = [
    "减脂期间，推荐吃什么呢",
    "一周瘦几斤算正常",
    "有氧一般要做多久",
    "空腹有氧更减脂吗",
    "推日之后二头还练吗",
    "坐姿划船要不要夹肩胛骨",
    "坚果一天吃多少",
    "肯德基能吃吗",
    "肩宽怎么练",
    "深蹲膝盖疼怎么办",
    "增肌期一天吃几顿",
    "卧推手腕疼",
]


def diff_cost(before: dict, after: dict) -> dict:
    return {k: after.get(k, 0) - before.get(k, 0) for k in after}


def main_run():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=10, help="跑多少条（默认 10）")
    ap.add_argument("--mode", default="", help="检索模式，默认用线上默认值")
    ap.add_argument("--prices", default="prices.json", help="单价文件，缺省不出钱")
    ap.add_argument("--json-out", default="cost_report.json")
    args = ap.parse_args()

    # 跑真实链路，但不能污染真实日志（日志是内容缺口清单，自测数据混进去就废了）
    main.QUERY_LOG_ON = False
    main.RATE_LIMIT_PER_MIN = 10_000      # 连跑十条不想被自己的限流挡住
    client = TestClient(main.app)
    headers = {"x-access-code": main.ACCESS_CODE} if main.ACCESS_CODE else {}

    # 预热：第一次请求要把 119 条正文嵌进内存，那笔钱不该摊到单次提问头上
    print("预热中（加载知识库向量）…")
    t0 = time.perf_counter()
    client.post("/rag", json={"question": "热身问题"}, headers=headers)
    warm_ms = (time.perf_counter() - t0) * 1000
    print(f"预热完成 {warm_ms:.0f} ms（这段是启动成本，不计入单次）\n")

    rows = []
    for q in QUESTIONS[: args.n]:
        if main._cache and hasattr(main._cache, "clear"):
            main._cache.clear()          # 清了才测得到「真的走了一遍模型」
        before = dict(main.COST)
        t0 = time.perf_counter()
        url = "/rag" + (f"?mode={args.mode}" if args.mode else "")
        r = client.post(url, json={"question": q}, headers=headers)
        wall = (time.perf_counter() - t0) * 1000
        if r.status_code != 200:
            print(f"  !! {q} HTTP {r.status_code} {r.text[:60]}")
            continue
        d = r.json()
        used = diff_cost(before, dict(main.COST))
        rows.append({
            "q": q, "wall_ms": round(wall, 1),
            "server_ms": (d.get("timings") or {}).get("total_ms"),
            "refused": bool(d.get("refused")),
            "sources": len(d.get("sources") or []),
            **used,
        })
        print(f"  {q:<18} {wall:7.0f}ms  embed {used['embed_tokens']:>4}t  "
              f"rerank {used['rerank_tokens']:>5}t  "
              f"prompt {used['prompt_tokens']:>4}t  out {used['completion_tokens']:>4}t"
              f"（含思维链 {used['reasoning_tokens']}）  {'拒' if d.get('refused') else '答'}")

    if not rows:
        sys.exit("一条都没跑成")

    def avg(key):
        return round(statistics.mean(r[key] for r in rows), 1)

    def total(key):
        return sum(r[key] for r in rows)

    n = len(rows)
    summary = {
        "n": n,
        "avg_wall_ms": avg("wall_ms"),
        "p95_wall_ms": round(sorted(r["wall_ms"] for r in rows)[int(n * 0.95) - 1], 1) if n > 1 else rows[0]["wall_ms"],
        "answered": sum(1 for r in rows if not r["refused"]),
        "per_question": {
            "embed_tokens": avg("embed_tokens"),
            "rerank_tokens": avg("rerank_tokens"),
            "prompt_tokens": avg("prompt_tokens"),
            "completion_tokens": avg("completion_tokens"),
            "reasoning_tokens": avg("reasoning_tokens"),
            "embed_calls": avg("embed_calls"),
            "rerank_calls": avg("rerank_calls"),
            "chat_calls": avg("chat_calls"),
        },
        "total": {
            "embed_tokens": total("embed_tokens"),
            "rerank_tokens": total("rerank_tokens"),
            "prompt_tokens": total("prompt_tokens"),
            "completion_tokens": total("completion_tokens"),
            "reasoning_tokens": total("reasoning_tokens"),
        },
        "models": {
            "embed": main.EMBED_MODEL, "rerank": main.RERANK_MODEL, "chat": main.LLM_MODEL,
        },
        "query_cache_size": main.CACHE_MAX,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    print(f"\n{'='*66}\n{n} 条平均（已清查询缓存，每条都真走了一遍模型）")
    print(f"  端到端 {summary['avg_wall_ms']:.0f} ms   P95 {summary['p95_wall_ms']:.0f} ms")
    print(f"  调用次数  embed {summary['per_question']['embed_calls']:.1f} 次"
          f"  rerank {summary['per_question']['rerank_calls']:.1f} 次"
          f"  chat {summary['per_question']['chat_calls']:.1f} 次")
    print(f"  token     embed {summary['per_question']['embed_tokens']:.0f}"
          f"  rerank {summary['per_question']['rerank_tokens']:.0f}"
          f"  prompt {summary['per_question']['prompt_tokens']:.0f}"
          f"  output {summary['per_question']['completion_tokens']:.0f}"
          f"（其中思维链 {summary['per_question']['reasoning_tokens']:.0f}）")

    # ---- 有单价就算钱 ----
    pf = Path(args.prices)
    if pf.exists():
        pr = json.loads(pf.read_text(encoding="utf-8"))
        cur = pr.get("currency", "CNY")
        sym = {"CNY": "¥", "USD": "$"}.get(cur, cur + " ")

        def money(per_m, tokens):
            return per_m * tokens / 1_000_000

        pq = summary["per_question"]
        c_embed = money(pr.get("embed_per_m", 0), pq["embed_tokens"])
        c_rerank = money(pr.get("rerank_per_m", 0), pq["rerank_tokens"])
        c_in = money(pr.get("prompt_per_m", 0), pq["prompt_tokens"])
        c_out = money(pr.get("completion_per_m", 0), pq["completion_tokens"])
        each = c_embed + c_rerank + c_in + c_out
        summary["cost"] = {
            "currency": cur,
            "per_question": {
                "embed": round(c_embed, 6), "rerank": round(c_rerank, 6),
                "prompt": round(c_in, 6), "completion": round(c_out, 6),
                "total": round(each, 6),
            },
            "monthly": {
                f"dau_{d}": round(each * d * 1.5 * 30, 2) for d in (10, 100, 1000)
            },
            "prices_used": pr,
        }
        print(f"\n  单次成本 {sym}{each:.5f}"
              f"（embed {sym}{c_embed:.5f} + rerank {sym}{c_rerank:.5f}"
              f" + 输入 {sym}{c_in:.5f} + 输出 {sym}{c_out:.5f}）")
        print("  换算（每人每天 1.5 问 × 30 天）：")
        for d in (10, 100, 1000):
            print(f"    DAU {d:<5} → {sym}{each * d * 1.5 * 30:.1f}/月")
        print("\n  注意：这里算的是「全部未命中缓存」的上限。"
              f"\n  缓存命中时不打任何模型，成本为 0 —— 命中率越高，实际越接近 0。")
    else:
        print(f"\n  （没有 {pf}，只出 token。填好单价再跑一次就有钱了）")

    Path(args.json_out).write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(f"\n明细已写入 {args.json_out}")


if __name__ == "__main__":
    main_run()
