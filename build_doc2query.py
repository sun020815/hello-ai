"""给每条 chunk 生成「用户会怎么问」的口语问法（Doc2Query）。

为什么需要：chunk 是书面语，用户是口语。newbie-05 开头写「这套计划能练多久」，
正文才提到「连续两周无法增加重量就是平台期」——用户问「练了两个月不涨了怎么办」，
向量能排到第 1（bge-m3 抓得住语义），但 reranker 看到主题句就判成「讲计划时长」，
给 0.3222。精排看的是「这段能不能回答这个问题」，主题句一偏，整条就废了。

做法：把「这条资料能回答的问题」拼进 rerank 的候选文本，让精排同时看到
正文和它对应的问法。检索用的向量不动 —— 诊断显示召回路 0 失败，动它只会引入风险。

输出存 kb_queries.json，不写 kb.json：kb.json 是语料真源，问法是派生物，
分开放才能在改问法时不动正文，也才能一键关掉。
"""

import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import main as app

HERE = Path(__file__).parent
OUT = HERE / "kb_queries.json"
N_QUERIES = 4

PROMPT = """下面是健身知识库里的一条资料。请写出 {n} 个普通健身爱好者会用来问这条资料的问题。

要求：
1. 用日常口语，像微信里随口问朋友。**但不要用方言、土话或网络黑话**——
   「哑铃」就叫哑铃，不要叫「铁疙瘩」；「重量」就叫重量，不要叫「片数」「几把」
2. 不要复述资料里的书面原词，换成用户自己的说法。比如资料写「平台期」，
   用户会说「练了两个月不涨了」；资料写「渐进超负荷」，用户会说「重量怎么往上加」
3. 每个问题 8 到 20 个字，一句话，句末加问号
4. 不要出现「资料」「上面说」「这段」这类指代本文的词，用户不知道有这份资料
5. 覆盖不同角度：有人问是什么、有人问怎么做、有人问为什么、有人问做多少

资料话题：{topic}
资料正文：{text}

只输出一个 JSON 数组，例如 ["问题一", "问题二"]，不要任何解释文字。"""


def parse_array(text):
    """模型偶尔会包一层 ```json 或者加前言，捞第一个 [ ... ]。"""
    m = re.search(r"\[[\s\S]*\]", text)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    return [s.strip() for s in arr if isinstance(s, str) and 4 <= len(s.strip()) <= 40]


def gen_one(chunk):
    resp = app.call_llm([
        {"role": "user",
         "content": PROMPT.format(n=N_QUERIES, topic=chunk["topic"], text=chunk["text"])},
    ])
    qs = parse_array(resp.choices[0].message.content)
    return chunk["id"], qs[:N_QUERIES]


def main():
    force = "--force" in sys.argv
    cache = {}
    if OUT.exists() and not force:
        cache = json.loads(OUT.read_text(encoding="utf-8")).get("queries", {})

    todo = [c for c in app.KB if c["id"] not in cache]
    print(f"知识库 {len(app.KB)} 条，已有 {len(cache)} 条，需生成 {len(todo)} 条")

    if todo:
        done = 0
        with ThreadPoolExecutor(max_workers=6) as pool:
            for cid, qs in pool.map(gen_one, todo):
                cache[cid] = qs
                done += 1
                if done % 10 == 0 or done == len(todo):
                    print(f"  {done}/{len(todo)}", flush=True)
        OUT.write_text(
            json.dumps({"n_per_chunk": N_QUERIES, "model": app.LLM_MODEL,
                        "queries": cache}, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )

    # 体检：空的、太少的、互相重复的
    empty = [k for k, v in cache.items() if not v]
    short = [(k, len(v)) for k, v in cache.items() if 0 < len(v) < N_QUERIES]
    print(f"\n写入 {OUT.name}：{len(cache)} 条")
    if empty:
        print(f"  !! 生成失败 {len(empty)} 条：{empty[:5]}")
    if short:
        print(f"  !! 少于 {N_QUERIES} 个问法：{short[:5]}")
    print(f"  平均每个 chunk {sum(len(v) for v in cache.values()) / max(1, len(cache)):.1f} 个问法")

    print("\n抽样 6 条看质量：")
    for cid in list(cache)[:6]:
        chunk = next(c for c in app.KB if c["id"] == cid)
        print(f"\n  [{cid}] {chunk['text'][:34]}...")
        for q in cache[cid]:
            print(f"      问：{q}")


if __name__ == "__main__":
    main()
