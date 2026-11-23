"""口语改写路径的门槛该定多少 —— 离线扫，零成本。

背景：135 条口语集（110 该答 / 25 该拒）上，oral 模式的拒答率从 100% 掉到 32%，
而命中只多救了 2 条。原因不是改写不好，是 ORAL_THRESHOLD=0.50 这个门线本身
是 bge 时代的刻度 —— 4B 给「像但不是答案」的分数就在 0.4~0.6，
0.50 等于把「半相关」全放进来当答案。

这个脚本不调接口，全部吃 .eval_reranks.json / .rewrites.json / .eval_oral.json
三份缓存，扫一遍门槛看代价曲线：
  门槛放宽 -> 救回更多被拒的正确答案，同时放进更多不该答的
  门槛收紧 -> 拒答率回来，但改写就白做了

用法：uv run python diag_oral_gate.py [--cases eval_cases_colloquial.json]
"""
import json
import sys
from pathlib import Path

import main as app

HERE = Path(__file__).parent


def load(name):
    p = HERE / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def main():
    cases_name = "eval_cases_colloquial.json"
    if "--cases" in sys.argv:
        cases_name = sys.argv[sys.argv.index("--cases") + 1]
    cases = json.loads((HERE / cases_name).read_text(encoding="utf-8"))

    base = load(".eval_reranks.json").get("scores", {})
    rewrites = load(".rewrites.json").get("items", {})
    oral = load(".eval_oral.json").get("scores", {})
    by_id = {d["id"]: d for d in app.KB}

    def rank_from(scores):
        r = [{"id": i, "topic": by_id[i]["topic"], "score": s}
             for i, s in scores.items() if i in by_id]
        r.sort(key=lambda x: x["score"], reverse=True)
        return r

    # 先把「原问法的判定」和「改写候选」都算好，扫门槛时只重算最后一步
    prepared = []
    for c in cases:
        q = c["q"]
        if q not in base:
            prepared.append((c, None, None))
            continue
        br = rank_from(base[q])
        kept = app.pick_hits(br, app.RERANK_TOP_K, app.RERANK_THRESHOLD)
        best_r2 = None
        if not kept:                      # 只有被拒答的才看改写
            vs = [v for v in rewrites.get(q, [])[: app.ORAL_N] if v in oral]
            if vs:
                best_v = max(vs, key=lambda v: max(oral[v].values()))
                r2 = rank_from(oral[best_v])
                if r2 and r2[0]["score"] > br[0]["score"]:
                    best_r2 = r2
        prepared.append((c, kept, best_r2))

    missing = sum(1 for c, k, r in prepared if k is None and c["q"] not in base)
    print(f"评测集 {cases_name}：{len(cases)} 条"
          f"{f'（{missing} 条没有基座缓存，已跳过）' if missing else ''}")
    print(f"原问法门线 {app.RERANK_THRESHOLD}，改写候选门槛待扫\n")
    base_hit = sum(1 for c, kept, _ in prepared
                   if c["kind"] == "hit" and kept and kept[0]["topic"] == c["expect"])
    print(f"  改写门槛   命中       拒答       整体      改写生效（相对基座 {base_hit} 条）")
    for gate in (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.985):
        hit = hit_n = ref = ref_n = used = 0
        for c, kept, r2 in prepared:
            final = kept
            if not final and r2 is not None:
                cand = app.pick_hits(r2, app.RERANK_TOP_K, gate)
                if cand:
                    final, used = cand, used + 1
            got = final[0]["topic"] if final else None
            if c["kind"] == "refuse":
                ref_n += 1
                ref += got is None
            else:
                hit_n += 1
                hit += got == c["expect"]
        print(f"  {gate:<9}  {hit / hit_n * 100:5.1f}%   {ref / ref_n * 100:6.1f}%   "
              f"{(hit + ref) / (hit_n + ref_n) * 100:5.1f}%   {used:>2} 条（{hit - base_hit:+d}）")


if __name__ == "__main__":
    main()
