"""把 chunk 的主题句改成「用户会问的那样」。

为什么做这件事，以及为什么不做 doc2query，两个实验都留了证据：

  doc2query（给 105 条各加 4 个问法，拼在正文后面）：
    综合集 hit@1 92.6% -> 88.4%，拒答率 74% -> 87%。整体好看，
    但那是「拒答换命中」——放行的错答本来会被生成层兜住，被拦的真答案没人救。
    口语集 +1 条，样本太少不构成结论。判定：不启用。

  主题句改写（只改开头一句话，5 条 chunk）：
    练了两个月不涨了      0.3254 -> 0.9624
    胸练完一点感觉都没有   0.2413 -> 0.9593（第 4 名升到第 1）
    下拉感觉不到背        0.2571 -> 0.9611（第 8 名升到第 1）
    胸上沿总是不长        0.0787 -> 0.5377（第 3 名升到第 1）
    哥本哈根支撑有什么用   0.5404 -> 0.9917

机制：rerank 判断的是「这段能不能回答这个问题」，它第一眼看开头。
newbie-05 开头写「这套计划能练多久」，用户问「练了两个月不涨了」，
精排判成「在讲计划时长」，给 0.32 —— 而正文完整答了这个问题。
问题不在缺问法，在主题句没对着用户说话。

代价：一条题从 0.976 掉到 0.861（「推日练完怎么判断练到位了」，
因为新的主题句把语义空间往「没感觉」那边挪了）。仍远在门槛之上。

约束：改写只动开头那半句，正文一个字不删。信息不能丢 ——
所以 newbie-05 不能直接把「能练多久」换掉，得保留在后面。
"""

import json
from pathlib import Path

HERE = Path(__file__).parent
KB = HERE / "kb.json"

# chunk_id -> 新的开头半句（替换正文里旧的开头部分）
LEADS = {
    "newbie-05": {
        "old": "这套计划能练多久：一直练到无法再增加重量为止。",
        "new": "练了几个月突然不涨了、重量上不去怎么办：这套计划一直练到无法再增加重量为止。",
    },
    "legs-05": {
        "old": "哥本哈根支撑强烈推荐。",
        "new": "哥本哈根支撑是练什么的、有什么用：强烈推荐。",
    },
    "push-04": {
        "old": "推日的质量不看推了多少公斤。",
        "new": "练完胸没感觉、说不出胸在哪里怎么办：推日的质量不看推了多少公斤。",
    },
    "lat-01": {
        "old": "高位下拉的握距选对握和反握之间的那个位置，",
        "new": "高位下拉感觉不到背、握距该多宽：选对握和反握之间的那个位置，",
    },
    "upper-01": {
        # 第一版写成「凳子该调多少度」，结果和「器械推胸座椅调多高」字面撞车，
        # 把那条题从「中胸」拉到了「上胸」（0.789 -> 0.916，答错了）。
        # 换成「翘多少度」避开「凳 / 座椅」，副作用消失，原来的两条仍在 0.99。
        # 这是改主题句的真实代价：新词会引入新的字面重叠，必须回头检查。
        "old": "上胸不长、上斜卧推的凳子该调多少度：",
        "new": "上胸不长、上斜卧推要翘多少度：",
    },
}


def main():
    kb = json.loads(KB.read_text(encoding="utf-8"))
    by_id = {d["id"]: d for d in kb}
    changed = 0

    for cid, spec in LEADS.items():
        chunk = by_id[cid]
        text = chunk["text"]
        if text.startswith(spec["new"]):
            print(f"  {cid} 已是新开头，跳过")
            continue
        if not text.startswith(spec["old"]):
            print(f"  !! {cid} 开头对不上，跳过。现在开头：{text[:30]}")
            continue
        new_text = spec["new"] + text[len(spec["old"]):]
        # 信息不能丢：正文其余部分必须原样保留；长度只允许小幅精简（改措辞时会出现）
        rest = text[len(spec["old"]):]
        assert rest in new_text, f"{cid} 改写后丢了正文"
        assert len(new_text) >= len(text) * 0.95, f"{cid} 改写后短了太多"
        chunk["text"] = new_text
        changed += 1
        print(f"  {cid} 改写完成：{len(text)} -> {len(new_text)} 字")

    if changed:
        KB.write_text(json.dumps(kb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n改了 {changed} 条，共 {len(kb)} 条")

    # 打印改动后的开头，肉眼复核一遍语义是否通顺
    print("\n改写后：")
    for cid in LEADS:
        print(f"  [{cid}] {by_id[cid]['text'][:52]}...")


if __name__ == "__main__":
    main()
