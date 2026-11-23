"""候选拒答题的探针：先算分数再定稿，不靠想象判断「库里有没有」。

为什么必须先测：# C 层（邻近难例）问的是同一个动作，向量会把它们拉向
对应话题。如果分数远高于 0.50，说明库里其实写了相关内容，那它就不是
拒答题，硬塞进去等于给评测集注入标签错误——这比少几条拒答题危险得多。

用法：
    uv run python probe_refuse.py            # 只算分并分组打印
    uv run python probe_refuse.py --merge    # 通过复核的写入 eval_cases.json
"""

import json
import sys
from pathlib import Path

import main as app

HERE = Path(__file__).parent

# (问题, 层, 库里为什么没有 / 标注依据)
CANDIDATES = [
    # ---------- A 完全无关：测下限，应该分数极低 ----------
    ("帮我写一首诗", "A", "完全无关"),
    ("推荐一部健身题材的电影", "A", "完全无关"),
    ("今天杭州天气怎么样", "A", "完全无关"),
    ("明天A股会涨吗", "A", "完全无关"),
    ("帮我改一下这段 Python 代码", "A", "完全无关"),
    ("杭州有哪些好吃的火锅店", "A", "完全无关"),

    # ---------- B 健身领域但整块缺失 ----------
    ("肌酸一天吃多少", "B", "新库无补剂内容"),
    ("肌酸要不要冲击期", "B", "新库无补剂内容"),
    ("蛋白粉买哪个牌子", "B", "新库不涉及品牌"),
    ("生酮饮食适合增肌吗", "B", "新库无生酮 / 低碳"),
    ("减脂期每天吃多少卡", "B", "只讲增肌盈余，无减脂缺口"),
    ("空腹有氧更减脂吗", "B", "无有氧与减脂内容"),
    ("减脂会不会掉肌肉", "B", "无减脂期营养"),
    ("膝盖半月板损伤怎么康复", "B", "无康复医学，只有换动作建议"),
    ("BCAA 训练前喝有用吗", "B", "无补剂内容"),
    ("氮泵能不能天天喝", "B", "无补剂 / 咖啡因内容"),
    ("促睾补剂有用吗", "B", "无补剂内容"),
    ("一周减几斤算正常", "B", "无减脂进度内容"),
    ("减脂期蛋白质要吃更多吗", "B", "diet-02 只给增肌期 1.6-2.2g，未涉减脂"),
    ("肩峰撞击综合征怎么治", "B", "无康复医学，warm-01 只讲预防性激活"),
    ("腰椎间盘突出能练硬拉吗", "B", "无医学禁忌症内容"),
    ("生理期能不能练腿", "B", "无女性 / 周期内容"),

    # ---------- C 邻近难例：同动作 / 同话题，但库里没答那个点 ----------
    ("卧推手腕疼怎么办", "C", "bench-02 只讲腕中立是预防，未讲疼痛处理"),
    ("硬拉用正反握还是双正握", "C", "dead-01/02 讲髋铰链，零握法内容"),
    ("引体向上一个都做不了怎么练", "C", "lat / back 只把引体当名词提过，无进阶教学"),
    ("练完要不要拉伸", "C", "warm-* 全是训练前热身，无训练后放松"),
    ("卧推要不要戴护腕", "C", "库内零护具 / 装备内容"),
    ("深蹲穿什么鞋", "C", "库内零装备内容"),
    ("硬拉要不要用助力带", "C", "库内零护具内容"),
    ("相扑硬拉和常规硬拉有什么区别", "C", "只有罗马尼亚 / 单腿 / 山羊挺身，无相扑"),
    ("史密斯机卧推能替代自由卧推吗", "C", "只出现史密斯实力推 / 史密斯负重，无卧推对比"),
    ("腹肌怎么练", "C", "全库无腹肌动作，仅 legs-05 提到侧腹"),
    ("增肌期能不能做有氧", "C", "库内零有氧内容"),
    ("组间休息时该坐着还是走动", "C", "param-06 只讲时长，未讲休息期做什么"),
]

# 探针推翻了两条直觉判断：本想当拒答题，实际库里答了。
# 保留在这里是为了留下证据 —— 拒答题不能靠「我觉得库里没有」来定。
RECLASSIFIED = [
    # (问题, 应命中的话题, 依据)
    ("硬拉每次都要从地面拉起来吗", "硬拉",
     "dead-01 原文「它不是把杠铃从地上拉起来，而是靠屈髋把重量送下去再收回来」，直接答了"),
    ("侧平举用多重的哑铃", "侧平举",
     "lateral-02「控制得住的重量才叫有效重量」+ 12-15 次不耸肩，是完整的标准答案"),
]

# 复核标准：分数高于这个值说明库里可能真有答案，必须人工看原文
SUSPECT_SCORE = 0.75

# 人工复核结论：分数高但仍判为拒答的，理由必须写在下面。
# 这三条是整份评测集里最难的负样本 —— 库里出现过同样的关键词，
# 但只讲了「怎么预防」或「相邻人群」，没有回答用户问的那件事。
# 它们存在的意义是证明：reranker 会被字面重叠骗。
REVIEWED_REFUSE = {
    "肩峰撞击综合征怎么治":
        "shoulder-05 只说「正手前平举容易产生肩峰撞击」→ 讲的是预防不是治疗。问「怎么治」，库内零康复内容",
    "腰椎间盘突出能练硬拉吗":
        "dead-03 只给「腰痛换臀桥」的症状级换动作建议。腰突是诊断，能不能练属医学禁忌症判断，资料够不着",
    "卧推手腕疼怎么办":
        "warm-01 提的是「卧推肩痛」不是腕痛；bench-02「腕中立不折腕」是预防。字面重叠（卧推 + 痛），问题不同",
}


def top_score(q):
    """和线上 search_kb_rerank 相同路径：向量召回 12 条 -> 精排 -> 取最高分。"""
    hits, ranking = app.search_kb_rerank(q)
    top = ranking[0]
    return top["score"], top["topic"], top["id"], bool(hits)


def main():
    merge = "--merge" in sys.argv
    rows = []
    for q, layer, why in CANDIDATES:
        score, topic, cid, passed = top_score(q)
        reviewed = q in REVIEWED_REFUSE
        rows.append({"q": q, "layer": layer,
                     "why": REVIEWED_REFUSE.get(q, why),
                     "score": score, "topic": topic, "id": cid, "passed": passed,
                     "ok": score < SUSPECT_SCORE or reviewed, "reviewed": reviewed})

    rows.sort(key=lambda r: -r["score"])

    print(f"{'分数':>7}  {'层':<3} {'检索结果':<20} 问题")
    print("-" * 84)
    for r in rows:
        mark = ""
        if r["reviewed"]:
            mark = "  <== 复核后仍拒答"
        elif not r["ok"]:
            mark = "  <== 存疑，需复核"
        got = f"{r['topic']}({r['id']})" if r["passed"] else "阈值拦截"
        print(f"{r['score']:>7.4f}  {r['layer']:<3} {got:<20} {r['q']}{mark}")

    print()
    for layer in ["A", "B", "C"]:
        sub = [r for r in rows if r["layer"] == layer]
        if not sub:
            continue
        hi, lo = max(r["score"] for r in sub), min(r["score"] for r in sub)
        n_pass = sum(1 for r in sub if r["passed"])
        print(f"  层 {layer}: {len(sub)} 条，分数 {lo:.4f} ~ {hi:.4f}，"
              f"被阈值放行 {n_pass} 条")

    suspects = [r for r in rows if not r["ok"]]
    print()
    if suspects:
        print(f"仍需人工复核（分数 >= {SUSPECT_SCORE} 且未登记复核结论）：")
        for r in suspects:
            print(f"  {r['score']:.4f}  {r['q']}  -> {r['topic']}")
    else:
        print(f"分数 >= {SUSPECT_SCORE} 的候选都有复核结论，无待办。")

    if not merge:
        print("\n（未写入。加 --merge 合并进 eval_cases.json）")
        return

    kept = [r for r in rows if r["ok"]]
    path = HERE / "eval_cases.json"
    cases = json.loads(path.read_text(encoding="utf-8"))
    have = {c["q"] for c in cases}
    added = added_hit = 0

    for r in kept:
        if r["q"] in have:
            continue
        cases.append({"q": r["q"], "expect": None, "kind": "refuse",
                      "note": f"{r['why']}（层{r['layer']}，探针分 {r['score']:.4f}）"})
        added += 1

    topics = {d["topic"] for d in app.KB}
    for q, topic, why in RECLASSIFIED:
        assert topic in topics, f"话题不存在：{topic}"
        if q in have:
            continue
        cases.append({"q": q, "expect": topic, "kind": "hit", "note": f"探针改判：{why}"})
        added_hit += 1

    path.write_text(json.dumps(cases, ensure_ascii=False, indent=1), encoding="utf-8")
    n_ref = sum(1 for c in cases if c["kind"] == "refuse")
    print(f"\n新增拒答 {added} 条、命中 {added_hit} 条，现在共 {len(cases)} 题"
          f"（命中 {len(cases) - n_ref} / 拒答 {n_ref}）")


if __name__ == "__main__":
    main()
