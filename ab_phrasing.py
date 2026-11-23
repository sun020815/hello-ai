"""同一批问题，口语版 vs 书面版，问同一份语料，看分数差多少。

起因：自问自答体检抓出 57 条「病」，抽检 6 条发现正文全都答得好好的。
    「为什么每个人合适的重量都不一样」正文最后一句就是「同样写 10RM，不同人的重量完全不同」，给 0.031
    「站着往两边抬手臂为啥底下没感觉」正文第一句就是「底部几乎没有张力」，给 0.065
两条都低到这个程度，不像是「没答」，像是 reranker 根本不认这种问法。

于是造一个对照：同一个问题两种问法，一个字都不改语料。
  口语版：用户真会这么问（来自 doc2query）
  书面版：健身圈术语写法（人工按同一含义重写，不抄正文）

如果书面版分数远高于口语版，病根就在**问法与语料的语域落差**，不在语料内容。
修法随之改变：
  改语料  -> 105 条全都要动，改一条亏一条，还引入新副作用
  改查询  -> 只在入口加一层「口语转书面」，语料一个字不动

局限：书面版是照着健身术语写的，天然更接近语料用词 —— 这正是被测量的变量，
不是作弊，但也意味着它只能证明「落差存在」，不能证明「改写一定好使」。
改写会不会把语义拧歪，得另外用评测集验。
"""

import json
from pathlib import Path

import main as app

HERE = Path(__file__).parent
OUT = HERE / ".phrasing_ab.json"

# (chunk_id, 口语版, 书面版)  书面版按同一含义重写，不照抄正文
PAIRS = [
    ("bench-06", "每次吸气要吸到多满才算够？", "卧推呼吸时胸腔充气的程度"),
    ("diet-02", "一天吃的肉蛋奶怎么分到几顿？", "每日蛋白质来源在餐次间的分配"),
    ("upper-01", "靠背调低后重量要减多少？", "上斜卧推角度降低后的负重调整"),
    ("newbie-04", "疼得睡不着要少练多少？", "严重肌肉酸痛时的训练量削减"),
    ("param-01", "为什么每个人合适的重量都不一样？", "RM 相同而负重不同的人体差异"),
    ("shoulder-01", "肩膀后面那块每次做几组合适？", "三角肌后束训练的组数安排"),
    ("rear-03", "肩膀后面那块肌肉有啥用？", "三角肌后束的解剖功能"),
    ("newbie-02", "空手找感觉一般做多少下合适？", "空手动作建立念动一致的次数"),
    ("pull-03", "练背那天开始前都热什么？", "背部训练前的热身内容"),
    ("bench-07", "卧推时腰下面空多少算正常？", "卧推腰部反弓的合理幅度"),
    ("iso-01", "趴着练腿后侧时，脚尖朝里朝外有啥区别？", "俯身腿弯举内旋位与外旋位的区别"),
    ("lateral-03", "站着往两边抬手臂为啥底下没感觉？", "站姿侧平举底部张力缺失的原因"),
    ("diet-01", "吃太多为什么容易长肥肉？", "热量盈余与脂肪堆积的关系"),
    ("push-02", "为什么卧推越往后每组次数越少？", "金字塔组次数递减的原理"),
    ("push-05", "最后练手臂做几组就够了？", "手臂作为收尾动作的组数"),
    ("param-02", "每组快没劲了还要不要再硬撑？", "接近力竭时是否继续完成次数"),
    ("single-04", "为什么深蹲总感觉一边更用力？", "深蹲时左右腿发力不对称的原因"),
    ("iso-01", "脚尖往里转着练，腰背怎么才不酸？", "内旋位腿弯举的腰椎保护"),
    ("single-02", "做的时候小腿要一直竖着吗？", "保加利亚分腿蹲小腿垂直地面的要求"),
    ("dead-04", "练腰和屁股中间那块有啥好处？", "山羊挺身对下背与臀部衔接的作用"),
]


def probe(q):
    """一条问法走完整链路：全库向量取 12 -> 精排 -> 返回 top1 和分数字典。"""
    out = app.search_kb_rerank(q, top_k=app.RERANK_TOP_K)
    ranking = out[1]
    scores = {r["id"]: r["score"] for r in ranking}
    top = ranking[0] if ranking else {"id": None, "topic": None, "score": 0.0}
    return {"top_id": top["id"], "top_topic": top["topic"], "top_score": top["score"],
            "scores": scores}


def main():
    topic_of = {d["id"]: d["topic"] for d in app.KB}
    cache = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}

    rows = []
    print(f"{'chunk':<11}{'口语':>7}{'书面':>7}{'差':>8}  说明")
    print("-" * 84)
    for cid, col, wr in PAIRS:
        key = f"{cid}|{col}"
        if key not in cache:
            cache[key] = {"colloquial": probe(col), "written": probe(wr)}
            OUT.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        c, w = cache[key]["colloquial"], cache[key]["written"]
        d = w["top_score"] - c["top_score"]
        note = []
        if w["top_id"] == cid and c["top_id"] != cid:
            note.append("书面版把答案抢回来了")
        if c["top_id"] != cid and w["top_id"] != cid:
            note.append("两版都没回到自己")
        rows.append({"cid": cid, "col": col, "wr": wr,
                     "c": c["top_score"], "w": w["top_score"],
                     "c_top": c["top_id"], "w_top": w["top_id"],
                     "c_topic_ok": c["top_topic"] == topic_of[cid],
                     "w_topic_ok": w["top_topic"] == topic_of[cid]})
        print(f"{cid:<11}{c['top_score']:>7.3f}{w['top_score']:>7.3f}{d:>+8.3f}  "
              f"{'，'.join(note)}")

    n = len(rows)
    avg_c = sum(r["c"] for r in rows) / n
    avg_w = sum(r["w"] for r in rows) / n
    pass_c = sum(1 for r in rows if r["c"] >= app.RERANK_THRESHOLD)
    pass_w = sum(1 for r in rows if r["w"] >= app.RERANK_THRESHOLD)
    better = sum(1 for r in rows if r["w"] > r["c"] + 0.05)
    worse = sum(1 for r in rows if r["c"] > r["w"] + 0.05)

    print("-" * 84)
    print(f"  平均分：口语 {avg_c:.3f}  ->  书面 {avg_w:.3f}   （{avg_w - avg_c:+.3f}）")
    print(f"  过门槛 {app.RERANK_THRESHOLD}：口语 {pass_c}/{n}  ->  书面 {pass_w}/{n}")
    print(f"  书面明显更高（+0.05 以上）：{better} 条   明显更低：{worse} 条")
    print(f"  top1 落对自己话题：口语 {sum(r['c_topic_ok'] for r in rows)}/{n}  "
          f"->  书面 {sum(r['w_topic_ok'] for r in rows)}/{n}")
    print("\n  结论口径：差得越大，越说明病在「问法语域」而不是「语料内容」。")
    print("  但这只证明落差存在，改写能不能用，要拿评测集验（别在改查询时把语义拧歪）。")


if __name__ == "__main__":
    main()
