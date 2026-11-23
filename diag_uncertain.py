"""验证 uncertain 标记到底有没有用（离线，零成本）。

问题：加一个 `uncertain: true` 很容易，但它得真有判别力才行 —— 如果带内和带外的
正确率一样，这个标记只是噪声，前端标黄反而增加读者负担。

做法：拿评测集的逐条结果（eval_results.json 里有每条的 top1 分数），把「离门线
< band」的题挑出来，比较带内 / 带外的正确率。带内正确率明显更低，标记才有意义。

用法：
    uv run python diag_uncertain.py                      # 默认读 200 题、rerank 模式
    uv run python diag_uncertain.py --band 0.01          # 换带宽看敏感性
    uv run python diag_uncertain.py --results eval_results.oral135.json --mode oral
"""
import json
import sys
from pathlib import Path

import main as app

HERE = Path(__file__).parent


def pick_rows(data: dict, mode: str):
    block = data.get(mode)
    if not block:
        raise SystemExit(f"结果里没有 {mode} 这一路，已有：{[k for k in data if not k.startswith('__')]}")
    return block["rows"]


def main():
    args = sys.argv[1:]

    def opt(name, default=None):
        if name in args:
            return args[args.index(name) + 1]
        return default

    path = HERE / opt("--results", "eval_results.json")
    mode = opt("--mode", "rerank")
    band = float(opt("--band", app.UNCERTAIN_BAND))
    gate = float(opt("--gate", app.RERANK_THRESHOLD))

    data = json.loads(path.read_text(encoding="utf-8"))
    rows = [r for r in pick_rows(data, mode) if r.get("score") is not None]

    inside = [r for r in rows if abs(r["score"] - gate) < band]
    outside = [r for r in rows if abs(r["score"] - gate) >= band]

    def acc(rs):
        return sum(r["hit1"] for r in rs) / len(rs) if rs else 0.0

    print(f"结果文件 {path.name} / 模式 {mode} / 门线 {gate} / 带宽 ±{band}")
    print(f"有分数的题 {len(rows)} 条\n")

    print(f"  带内（贴门线）  {len(inside):>4} 条   正确率 {acc(inside) * 100:5.1f}%"
          f"   其中错 {sum(not r['hit1'] for r in inside)} 条")
    print(f"  带外（离门线远）{len(outside):>4} 条   正确率 {acc(outside) * 100:5.1f}%"
          f"   其中错 {sum(not r['hit1'] for r in outside)} 条")

    all_acc = acc(rows)
    print(f"  全部            {len(rows):>4} 条   正确率 {all_acc * 100:5.1f}%")

    if inside:
        lift = all_acc - acc(inside)
        print(f"\n  带内比整体低 {lift * 100:.1f} 个百分点"
              f"（带宽只要覆盖 {len(inside) / len(rows) * 100:.1f}% 的题）")

        # 这条才是「标记值不值得做」的实际口径：如果只复核带内这几条，
        # 能覆盖多少条错误，代价是复核多少条。
        total_err = sum(not r["hit1"] for r in rows)
        covered = sum(not r["hit1"] for r in inside)
        print(f"  复核带内 {len(inside)} 条可覆盖 {covered}/{total_err} 条错误"
              f"（{covered / total_err * 100:.0f}%），精度 {covered / len(inside) * 100:.0f}%")

        print("\n  --- 带内明细 ---")
        for r in sorted(inside, key=lambda x: x["score"]):
            flag = "错" if not r["hit1"] else "对"
            exp = r["expect"] or "（该拒答）"
            print(f"    {r['score']:.4f}  差 {r['score'] - gate:+.4f}  [{flag}] "
                  f"{r['q']}  →  {r['got'] or '拒答'}  (期望 {exp})")

    # 敏感性：带宽换几档，看结论会不会翻。如果只有某个特定带宽才好看，
    # 那个带宽就是拟合出来的，不能用。
    print("\n  --- 带宽敏感性 ---")
    print("    带宽     带内    带内正确率   带外正确率")
    for b in (0.002, 0.004, 0.006, 0.010, 0.020, 0.050):
        i = [r for r in rows if abs(r["score"] - gate) < b]
        o = [r for r in rows if abs(r["score"] - gate) >= b]
        print(f"    ±{b:<5}  {len(i):>4}    {acc(i) * 100:6.1f}%      {acc(o) * 100:6.1f}%")


if __name__ == "__main__":
    main()
