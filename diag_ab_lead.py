"""旧版 vs 新版知识库，同一批题逐条对照。

改主题句有个已知副作用（README 记着）：新词会引入新的字面重叠，把相邻话题的题
拉走。所以不能只看「目标题涨了」，要全量对比，看有没有别处掉下来。

做法：动态替换 main.KB + 清空向量缓存 + 各自独立缓存文件，跑同一批题
（200 集 + 135 口语集），两条都跑 rerank 单路，输出：
  - 由拒变答 / 由答变拒 的条目
  - 分数变化最大的 top N
"""
import json
from pathlib import Path

import main

TH = main.RERANK_THRESHOLD
CASES = []
for f, tag in (("eval_cases.json", "main"), ("eval_cases_colloquial.json", "oral")):
    for c in json.loads(Path(f).read_text(encoding="utf-8")):
        CASES.append({"q": c["q"], "kind": c["kind"], "tag": tag,
                      "expect": c.get("expect")})


def run(kb_path):
    main.KB = json.loads(Path(kb_path).read_text(encoding="utf-8"))
    main._kb_vectors = None
    main.CACHE_PATH = Path(kb_path).with_suffix(".vectors.json")
    out = {}
    for i, c in enumerate(CASES):
        r = main.rank_one(c["q"])
        t = r[0]
        out[c["q"]] = {"score": t["score"], "id": t["id"], "topic": t["topic"],
                       "pass": t["score"] >= TH}
        if (i + 1) % 60 == 0:
            print(f"    {i + 1}/{len(CASES)}", flush=True)
    return out


print("跑旧版…")
old = run(".kb_old.json")
print("跑新版…")
new = run(".kb_new.json")

changed = [(q, old[q], new[q]) for q in old
           if old[q]["pass"] != new[q]["pass"]]
print("\n" + "=" * 70)
print(f"判定翻转 {len(changed)} 条")
print("=" * 70)
kinds = {c["q"]: c for c in CASES}
for q, o, n in changed:
    k = kinds[q]
    direction = "拒→答" if n["pass"] else "答→拒"
    good = (direction == "拒→答") == (k["kind"] == "hit")
    print(f"  [{direction}] {k['tag']}/{k['kind']:<6} {o['score']:.4f}->{n['score']:.4f}  {q}")
    print(f"      {'✓ 符合预期' if good else '✗ 不符合预期'}  {o['id']} -> {n['id']}")

# 判定没变但分数动了的
moved = [(q, old[q]["score"], new[q]["score"]) for q in old
         if old[q]["pass"] == new[q]["pass"] and abs(new[q]["score"] - old[q]["score"]) > 0.05]
moved.sort(key=lambda x: abs(x[2] - x[1]), reverse=True)
print(f"\n判定未变但分数波动 >0.05：{len(moved)} 条")
for q, a, b in moved[:15]:
    print(f"  {a:.4f} -> {b:.4f} ({b - a:+.4f})  {q}")


def summary(res, kind):
    sub = [(q, res[q]) for q in res if kinds[q]["kind"] == kind]
    hit = sum(1 for _, r in sub if r["pass"])
    return hit, len(sub)


print("\n" + "=" * 70)
for name, res in (("旧版", old), ("新版", new)):
    hp, hn = summary(res, "hit")
    rp, rn = summary(res, "refuse")
    print(f"{name}  正例 hit@1 {hp}/{hn} = {hp / hn * 100:.1f}%   "
          f"负例拒答 {rp}/{rn} = {rp / rn * 100:.1f}%   "
          f"整体 {(hp + rp) / (hn + rn) * 100:.1f}%")
