"""真实模型跑一次排计划，把 trace 打出来。不经过 HTTP，方便调。

跑法：
  python run_plan.py                                  # 默认：增肌 3 天 健身房 新手
  python run_plan.py --days 4 --injuries 膝盖 --level 有基础
  python run_plan.py --equipment 家里哑铃 --goal 减脂

为什么要看 trace 而不只看计划：计划好看不代表链路对。
trace 能看出它是「真查了三个部位」还是「一次都没查就凭印象写」——
后者会被 normalize 剔除掉大半动作，最后计划是残的。
"""

import argparse
import json
import sys

import main
import plan_agent


def main_():
    ap = argparse.ArgumentParser()
    ap.add_argument("--goal", default="增肌")
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--equipment", default="健身房")
    ap.add_argument("--session-min", type=int, default=60)
    ap.add_argument("--level", default="新手")
    ap.add_argument("--injuries", nargs="*", default=[])
    ap.add_argument("--note", default="")
    ap.add_argument("--save", default=".plan_last.json")
    a = ap.parse_args()

    prof = {"goal": a.goal, "days": a.days, "equipment": a.equipment,
            "session_min": a.session_min, "level": a.level,
            "injuries": a.injuries, "note": a.note}

    def _llm(messages, tools=None):
        r = main.call_llm(messages, tools=tools)
        main.bill_chat(r)
        return r

    print(f"档案：{json.dumps(prof, ensure_ascii=False)}")
    c0 = dict(main.COST)
    out = plan_agent.plan_week(prof, _llm)
    delta = {k: main.COST.get(k, 0) - c0.get(k, 0) for k in main.COST}

    print(f"\n=== trace（{len(out['trace'])} 次工具调用，{out['steps']} 步，"
          f"{out['tokens']} token，{out['elapsed_ms']} ms）===")
    for t in out["trace"]:
        flag = "ok " if t["ok"] else "ERR"
        print(f"  {flag} step{t['step']} {t['tool']}({json.dumps(t['args'], ensure_ascii=False)}) "
              f"{t['ms']}ms → {t['brief']}")

    print(f"\n=== 计划（{len(out['plan'])} 天）===\n{out['summary']}")
    for d in out["plan"]:
        print(f"\n第 {d['day']} 天 · {d['focus']}")
        for e in d["exercises"]:
            print(f"  - {e['name']}（{e['part']}/{e['sub']}）{e['sets']} × {e['reps']}，"
                  f"间歇 {e['rest']}s  [ref {e['ref']}]")

    rep = out["audit"]
    print(f"\n=== 体检 ===")
    print(f"  ok={rep['ok']}  needs_review={out['needs_review']}  stopped={out['stopped'] or '-'}")
    for e in rep["errors"]:
        print("  ERROR", e.get("detail") or e["type"])
    for w in rep["warnings"]:
        print("  warn ", w.get("detail") or w["type"])
    if out["dropped"]:
        print("  被剔除的动作：", out["dropped"])

    # 成本：token 分输入/输出记，钱按 prices.json 的单价算（那个文件是本地的，不进仓库）。
    print(f"\n=== 这次花了多少 ===")
    print(f"  输入 {delta['prompt_tokens']} token（{delta['chat_calls']} 次调用）")
    print(f"  输出 {delta['completion_tokens']} token，其中思维链 {delta['reasoning_tokens']}")
    try:
        pr = json.load(open("prices.json", encoding="utf-8"))
        cny = (delta["prompt_tokens"] * pr["prompt_per_m"]
               + delta["completion_tokens"] * pr["completion_per_m"]) / 1_000_000
        print(f"  按示例单价 ≈ ¥{cny:.4f} / 次")
    except FileNotFoundError:
        print("  （没有 prices.json，跳过金额换算）")

    with open(a.save, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\n完整结果写到 {a.save}")
    return 0 if not out["needs_review"] else 2


if __name__ == "__main__":
    sys.exit(main_())
