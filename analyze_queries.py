"""分析真实问法日志：粉丝在问什么、哪些答不出来。

这份日志存在的唯一理由是修掉评测集的先天缺陷 —— 那 86 条口语题是
**对着知识库正文反推**出来的，用词必然重叠，跑出来 98% 命中是自己骗自己。
只有真人写的问法才知道真实命中率。

用法：
    uv run python analyze_queries.py                              # 读本地 .qa_log.jsonl
    uv run python analyze_queries.py --src mp                     # 只看公众号来的
    uv run python analyze_queries.py --remote <网址> --code <口令>  # 从线上拉（容器重启会丢文件）
    uv run python analyze_queries.py --export                     # 导出候选评测题

输出四块：
    1. 总量 / 拒答率 / 来源分布
    2. 拒答问题清单 —— 不是错误日志，是**内容选题清单**
    3. 高频问题（同一个问题被问了几次）
    4. 与现有 135 条口语集的重合度 —— 有多少问法是评测集没覆盖的
"""
import argparse
import json
import re
import sys
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

LOG = Path(".qa_log.jsonl")
ORAL_CASES = Path("eval_cases_colloquial.json")
OUT = Path(".query_candidates.json")

PUNCT = re.compile(r"[\s，。！？、,.!?;:；：\"'“”‘’()（）\[\]【】…—~-]")


def norm(s: str) -> str:
    return PUNCT.sub("", s or "").lower()


def bigrams(s: str) -> set[str]:
    s = norm(s)
    return {s[i:i + 2] for i in range(len(s) - 1)} if len(s) > 1 else {s}


def jaccard(a: str, b: str) -> float:
    A, B = bigrams(a), bigrams(b)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def load_local() -> list[dict]:
    if not LOG.exists():
        sys.exit(f"找不到 {LOG}。要么先跑几次真实提问，要么用 --remote 从线上拉。")
    rows = []
    for line in LOG.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def load_remote(base: str, code: str, sharecode: str) -> list[dict]:
    """从线上 /queries 拉。反代要求 sharecode，所以拼在 query 上。"""
    q = {"fmt": "jsonl", "limit": "2000"}
    if sharecode:
        q["sharecode"] = sharecode
    url = base.rstrip("/") + "/queries?" + urllib.parse.urlencode(q)
    req = urllib.request.Request(url, headers={"X-Access-Code": code})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read().decode("utf-8")
    except Exception as e:
        sys.exit(f"拉取失败：{type(e).__name__}: {str(e)[:160]}\n"
                 f"（401 说明口令不对；返回 HTML 说明 sharecode 没带或不对）")
    rows = []
    for line in raw.splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def main():
    global LOG          # --file 会改它；必须放在任何引用之前，否则 SyntaxError
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote", help="线上地址，如 https://fitness-kb-rag.app.workbuddy.host")
    ap.add_argument("--code", default="", help="访问口令")
    ap.add_argument("--sharecode", default="", help="反代的 sharecode")
    ap.add_argument("--src", default="", help="只看某个来源（mp / h5 / api）")
    ap.add_argument("--file", default=str(LOG), help="日志文件路径（默认 .qa_log.jsonl）")
    ap.add_argument("--export", action="store_true", help="导出评测集没覆盖的拒答问法")
    args = ap.parse_args()

    if args.file != str(LOG):
        LOG = Path(args.file)
    rows = (load_remote(args.remote, args.code, args.sharecode)
            if args.remote else load_local())
    if args.src:
        rows = [r for r in rows if (r.get("src") or "api") == args.src]
    if not rows:
        sys.exit("没有可分析的记录。")

    n = len(rows)
    # 用 ret 而不是 ok：兜底救回来的题，检索层其实也没命中，同样属于知识库缺口。
    # 只看 ok 会漏掉这一整批 —— 而它们恰恰是「模型在硬撑」的最危险的一类。
    gap = [r for r in rows if not r.get("ret", r.get("ok"))]
    print(f"总计 {n} 条提问，知识库没接住 {len(gap)} 条（{len(gap) / n * 100:.1f}%）"
          f"，其中被兜底救回 {sum(1 for r in gap if r.get('fb'))} 条\n")

    # ---- 1. 来源与时间分布 ----
    print("【来源】")
    for k, v in Counter(r.get("src") or "api" for r in rows).most_common():
        print(f"  {k:<8} {v:>5}")
    days = Counter((r.get("t") or "")[:10] for r in rows)
    if len(days) > 1:
        print("【按天】")
        for d in sorted(days):
            print(f"  {d}  {days[d]:>4}")

    # ---- 2. 高频问题：同一个问题被问了几次 ----
    print("\n【高频问题】Top 10（同一个问题被反复问，说明这条内容最该做透）")
    cq = Counter(norm(r.get("q")) for r in rows)
    first = {}
    for r in rows:
        first.setdefault(norm(r.get("q")), r.get("q"))
    for key, cnt in cq.most_common(10):
        mark = "  " if cnt > 1 else "  "
        print(f"  {cnt:>3}× {mark}{first.get(key, key)}")

    # ---- 3. 缺口清单 = 选题清单 ----
    print(f"\n【知识库没接住的 {len(gap)} 条】—— 这不是错误日志，是内容缺口")
    if not gap:
        print("  （没有，说明当前知识库覆盖得住现有提问）")
    for r in gap[-40:]:
        top = f"{r['top']:.2f}" if r.get("top") is not None else "—"
        tag = "  【兜底救回，模型在硬撑】" if r.get("fb") else ""
        print(f"  [{top}] {r.get('q')}{tag}")
        if r.get("rq"):
            print(f"        被理解成：{r['rq']}")

    # ---- 4. 与现有口语集的重合度 ----
    if ORAL_CASES.exists():
        cases = json.load(open(ORAL_CASES, encoding="utf-8"))
        case_qs = [c["q"] for c in cases]
        print(f"\n【与 {len(case_qs)} 条口语评测集的重合度】")
        exact = near = fresh = 0
        fresh_rows = []
        seen = set()
        for r in rows:
            q = r.get("q") or ""
            k = norm(q)
            if k in seen:
                continue
            seen.add(k)
            best = max((jaccard(q, cq_) for cq_ in case_qs), default=0.0)
            if best >= 0.99:
                exact += 1
            elif best >= 0.55:
                near += 1
            else:
                fresh += 1
                fresh_rows.append((r, best))
        uniq = exact + near + fresh
        print(f"  完全一样        {exact:>4}  ({exact / uniq * 100:.0f}%)")
        print(f"  近似（Jaccard≥.55）{near:>3}  ({near / uniq * 100:.0f}%)")
        print(f"  **评测集没覆盖** {fresh:>4}  ({fresh / uniq * 100:.0f}%)"
              f"   ← 这些才是能补进评测集的真实问法")

        if args.export:
            cands = []
            for r, _ in fresh_rows:
                if r.get("ret", r.get("ok")):
                    continue          # 只导出知识库没接住的：既有缺口、又是真实问法
                cands.append({
                    "q": r.get("q"),
                    "expect": "",          # 人工填：本该命中哪个话题
                    "kind": "hit",
                    "note": f"真实提问（{r.get('src', '?')}，{r.get('t', '')[:10]}），"
                            f"精排最高 {r.get('top')}"
                            + ("，兜底救回" if r.get("fb") else ""),
                    "tier": "real",
                })
            OUT.write_text(json.dumps(cands, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"\n  已导出 {len(cands)} 条候选到 {OUT}，expect 待人工填。"
                  f"\n  注意：这批是「知识库没接住」的，填之前先判断是「该补内容」还是「该补标注」。")

    print(f"\n提示：稿子里的 src=mp 才是公众号来的真实提问；h5/api 可能包含自测。")


if __name__ == "__main__":
    main()
