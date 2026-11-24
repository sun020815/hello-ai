"""排计划的实跑诊断：不是看「计划好不好看」，是看**链路的硬指标稳不稳**。

三件事：
1. 动作合法率 —— 模型写的动作名，有多少能被动作库认出来（不是编的）。
2. 约束遵守 —— 说有膝伤，计划里还会不会出现 contraindication 命中的动作。
3. 波动 —— 同一份档案连跑几次，天数、覆盖、耗时、token 抖多少。

为什么不做「答案质量打分」：那需要人来判，而且每份计划都能挑出毛病。
工程上能客观量的是上面三条，它们也是上线后真正会出问题的三条。

跑法：
  python diag_plan.py                 # 4 个不同档案各跑一次
  python diag_plan.py --repeat 3      # 第一个档案连跑 3 次看波动
  python diag_plan.py --only A        # 只跑某个档案

结果写 .diag_plan.json，同时打到 stdout。
"""

import argparse
import json
import statistics
import sys
import time

import main
import plan_agent

CASES = {
    "A": {"name": "新手增肌 3 天", "profile": {
        "goal": "增肌", "days": 3, "equipment": "健身房", "session_min": 60,
        "level": "新手", "injuries": [], "note": ""}},
    "B": {"name": "有基础减脂 4 天", "profile": {
        "goal": "减脂", "days": 4, "equipment": "健身房", "session_min": 75,
        "level": "有基础", "injuries": [], "note": "想保留肌肉，力量训练为主"}},
    "C": {"name": "家里哑铃 + 膝盖痛", "profile": {
        "goal": "增肌", "days": 3, "equipment": "哑铃", "session_min": 50,
        "level": "新手", "injuries": ["膝盖"], "note": "只有一对可调哑铃和一张凳子"}},
    "D": {"name": "新手 2 天 45 分钟", "profile": {
        "goal": "增肌", "days": 2, "equipment": "健身房", "session_min": 45,
        "level": "新手", "injuries": [], "note": "时间很紧"}},
}


def run_one(label, case):
    prof = case["profile"]
    t0 = time.time()

    def _llm(messages, tools=None):
        r = main.call_llm(messages, tools=tools)
        main.bill_chat(r)
        return r

    out = plan_agent.plan_week(prof, _llm)
    n_actions = sum(len(d["exercises"]) for d in out["plan"])
    illegal = len(out["dropped"])
    est = out["audit"].get("est_minutes") or {}
    row = {
        "case": label, "name": case["name"],
        "days": len(out["plan"]), "want_days": prof["days"],
        "actions": n_actions, "illegal": illegal,
        "review": out["needs_review"], "stopped": out["stopped"] or "",
        "tools": len(out["trace"]), "steps": out["steps"],
        "sec": round(time.time() - t0, 1), "tokens": out["tokens"],
        "est_minutes": list(est.values()), "session_min": prof["session_min"],
        "parts": out["audit"]["by_part"],
        "errors": [e.get("detail") or e["type"] for e in out["audit"]["errors"]],
        "warnings": [w.get("detail") or w["type"] for w in out["audit"]["warnings"]],
        "dropped": out["dropped"],
    }
    # 膝伤这条单独验：计划里出现了 contraindication 命中的动作就是硬失败
    inj = prof.get("injuries") or []
    if inj:
        lib = plan_agent.Library.load()
        tags = {plan_agent.INJURY_MAP.get(i, i) for i in inj}
        hit = []
        for d in out["plan"]:
            for e in d["exercises"]:
                if set(tags) & set(e.get("contra", [])):
                    hit.append(e["name"])
        row["injury_violations"] = hit
    return row, out


def fmt(rows):
    head = f"{'case':<5}{'档案':<20}{'天':>3}{'动作':>4}{'编的':>4}{'工具':>4}{'步':>3}{'秒':>6}{'token':>7}  时长估算"
    print(head)
    print("-" * len(head.encode("gbk", errors="ignore")))
    for r in rows:
        est = "/".join(str(x) for x in r["est_minutes"])
        flag = " ⚠" if r["review"] else ""
        print(f"{r['case']:<5}{r['name']:<20}{r['days']:>3}{r['actions']:>4}{r['illegal']:>4}"
              f"{r['tools']:>4}{r['steps']:>3}{r['sec']:>6}{r['tokens']:>7}  {est}"
              f"（预算 {r['session_min']}）{flag}")
    for r in rows:
        if r.get("injury_violations") is not None:
            v = r["injury_violations"]
            print(f"\n[{r['case']}] 膝伤禁忌冲突：{'无' if not v else v}")
        if r["errors"]:
            print(f"[{r['case']}] 硬问题：{r['errors']}")
        if r["warnings"]:
            print(f"[{r['case']}] 提醒：{r['warnings']}")


def main_():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=1, help="对 A 重复几次，看波动")
    ap.add_argument("--only", default="", help="只跑某个档案，如 A")
    a = ap.parse_args()

    if a.only:
        keys = [k for k in a.only.upper() if k in CASES]
    else:
        keys = list(CASES)

    rows, store = [], {}
    for k in keys:
        for i in range(a.repeat if k == "A" else 1):
            label = k if a.repeat == 1 or k != "A" else f"{k}{i + 1}"
            print(f"\n>>> 跑 {label} {CASES[k]['name']} ...", flush=True)
            row, out = run_one(label, CASES[k])
            rows.append(row)
            store[label] = out

    print("\n=== 汇总 ===")
    fmt(rows)

    if a.repeat > 1:
        same = [r for r in rows if r["case"].startswith("A")]
        if len(same) > 1:
            print("\n=== 同一档案的波动 ===")
            for k, unit in (("sec", "秒"), ("tokens", "token"), ("tools", "次工具"),
                            ("steps", "步"), ("actions", "个动作")):
                vs = [r[k] for r in same]
                print(f"  {k:<8} {vs}   中位 {statistics.median(vs)}  极差 {max(vs) - min(vs)}{unit}")
            covs = {tuple(sorted(r["parts"])) for r in same}
            print(f"  部位覆盖 {'一致' if len(covs) == 1 else '不一致：' + str(covs)}")

    with open(".diag_plan.json", "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False, indent=1)
    print("\n明细写到 .diag_plan.json")
    bad = [r for r in rows if r["review"] or r.get("injury_violations")]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main_())
