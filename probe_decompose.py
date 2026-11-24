"""复合问题拆解的探针：先把「模型能不能拆对」量出来，再决定要不要进主链路。

为什么先探针：拆解是**有代价**的一层 —— 每拆一次多一次 LLM 调用、多一路检索、
多一路 rerank。如果拆得不准（该拆的不拆、不该拆的乱拆），
那这一层的净收益是负的，跟之前「口语改写」那次一样（见 ab_oral_prompt 的结论）。

三条必须测到的事：
1. 该拆的拆得出，且**子问题里没有原句没有的新对象**（幻觉式拆解会把拒答打开后门）；
2. 不该拆的原样返回（单意图问题被拆碎 = 白花钱 + 上下文被稀释）；
3. 负例（完全无关的问题）不能拆出任何「健身话题」的样子 —— 否则拒答直接被绕过。

用法：uv run python probe_decompose.py [--round 2]
"""
import json
import sys
import time
from pathlib import Path

import main as app

HERE = Path(__file__).parent
OUT = HERE / ".probe_decompose.json"

# ---------- 待测样本 ----------
# 复合题：一句话里有两个以上**独立的检索目标**（不是「一个问题的两个侧面」）。
# 括号里写的是期望被覆盖的话题，用来事后核对「拆出来的路是不是真的打到了」。
COMPOUND = [
    ("推日练完胸第二天胳膊酸得厉害，是不是练太多了，要隔几天再练", ["推日", "分化与频率"]),
    ("卧推的时候手该放多宽，高位下拉又该握多宽", ["卧推技术", "高位下拉"]),
    ("想练上胸和下胸，一天全练还是分两天练", ["上胸", "下胸与中缝"]),
    ("练完腿第二天疼得走不了路，还能继续练吗，练完吃点什么好", ["新手上路", "饮食配合"]),
    ("增肌一天要吃多少蛋白，一周练几次比较合适", ["饮食配合", "分化与频率"]),
    ("肩膀侧面很扁，后束也没劲，到底先练哪个", ["侧平举", "后束"]),
    ("硬拉的时候腰疼还能练吗，有没有别的动作能代替", ["硬拉"]),
    ("新手第一次去健身房，重量从多少开始，一次练几个动作", ["新手上路"]),
    ("练胸那天先练上胸还是先练中胸，中间休息多久", ["上胸", "中胸", "训练参数"]),
    ("深蹲膝盖能不能过脚尖，蹲的时候脚跟老抬起来怎么办", ["深蹲"]),
    ("卧推往下放要数几秒，杠铃碰到胸口要不要停一下再推", ["卧推技术"]),
    ("增肌期和减脂期蛋白吃的一样多吗，训练量要不要跟着变", ["饮食配合"]),
]

# 对照组：单意图。拆了就是错（拆碎会让上下文被同一话题占满，还多花钱）。
SINGLE = [
    "深蹲膝盖要不要过脚尖",
    "高位下拉握距多宽",
    "卧推的时候脚和屁股到底该怎么放",
    "练腿日隔多久练一次比较好",
    "一周练几次比较合适",
    "卧推手肘该收着还是打开",
]

# 负例：库里没有 / 完全无关。拆出「健身话题」样子就等于给拒答开后门。
NEGATIVE = [
    "帮我写一首关于春天的诗",
    "今天杭州天气怎么样",
    "肌酸一天吃多少",
    "推荐一部健身题材的电影",
]

PROMPT = """用户问了一句话。判断这句话里是不是**同时问了两个以上互不相干的问题**。

如果是，把它拆成 2~3 个能各自独立去检索的子问题；如果不是，原样输出这一句。

判断标准（很重要，别搞错）：
- 两个问题各自有**不同的对象**（比如「卧推握多宽」和「下拉握多宽」），才算复合。
- 同一个对象的两个侧面（比如「卧推下放几秒」和「卧推要不要停」）**不算**复合，不要拆。
- 只是问得啰嗦、加了语气词、举了例子，都不算复合。

硬约束：
1. 子问题里**不许出现原句没有的对象、部位、数字**。原句没说「哑铃」，就不许在子问题里加「哑铃」。
2. 每条子问题都要能被单独看懂，不许用「它」「这个」指代。
3. 不改写原句的意思，不扩范围，不回答问题。
4. 每行一条，行首写「- 」。最多 3 行。
5. 如果不需要拆，只输出一行「- 原句」（把原句一个字不改抄一遍）。

例一：原句「自由泳怎么换气，骑行的时候又该怎么呼吸」
- 自由泳怎么换气
- 骑行的时候怎么呼吸

例二：原句「自由泳换气要练多久才能顺」
- 自由泳换气要练多久才能顺

例三：原句「帮我看看这段代码哪里错了」
- 帮我看看这段代码哪里错了

【原句】
{question}"""


def parse_lines(text: str) -> list[str]:
    """解析「- xxx」行。解析不出东西就返回空列表，由上层决定怎么退。"""
    out = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        for pre in ("- ", "-", "·", "•", "1.", "2.", "3."):
            if line.startswith(pre):
                line = line[len(pre):].strip()
                break
        line = line.strip("「」\"'。 ")
        if line:
            out.append(line)
    return out


def decompose_v1(question: str, timeout: float = 4.0) -> list[str] | None:
    """返回子问题列表；None = 失败（超时/上游抖）。长度 1 表示「不用拆」。"""
    try:
        resp = app.condense_llm([
            {"role": "system", "content": "你只做句子拆分，不回答问题。"},
            {"role": "user", "content": PROMPT.format(question=question)},
        ])
    except Exception as e:
        print(f"    [fail] {type(e).__name__}: {str(e)[:70]}")
        return None
    subs = parse_lines(resp.choices[0].message.content or "")
    if not subs or len(subs) > 3:
        return None
    return subs


GAP = 1.2          # 两次调用之间的间隔：连续快打会被上游排队，测出来的超时率是假的


def run(samples, label):
    print(f"\n===== {label} =====")
    rows = []
    for item in samples:
        q, expect = (item, []) if isinstance(item, str) else item
        t0 = time.perf_counter()
        subs = decompose_v1(q)
        ms = (time.perf_counter() - t0) * 1000
        time.sleep(GAP)
        rows.append({"q": q, "expect": expect, "subs": subs, "ms": round(ms)})
        flag = "ERR " if subs is None else ("SAME" if subs == [q] else f"{len(subs)}路")
        print(f"  [{flag}] {ms:5.0f}ms  {q}")
        if subs and subs != [q]:
            for s in subs:
                print(f"          -> {s}")
        elif subs == [q]:
            print(f"          -> （未拆）")
    return rows


def main():
    rnd = 1
    if "--round" in sys.argv:
        rnd = int(sys.argv[sys.argv.index("--round") + 1])
    if "--gap" in sys.argv:
        global GAP
        GAP = float(sys.argv[sys.argv.index("--gap") + 1])
    # 拆解要输出 2~3 行，64 token 正好卡在边界上，给到 96
    app.CONDENSE_MAX_TOKENS = 96
    print(f"gap={GAP}s  max_tokens={app.CONDENSE_MAX_TOKENS}")

    all_rows = {}
    all_rows["compound"] = run(COMPOUND, "一、复合题：应该拆成 2~3 路")
    all_rows["single"] = run(SINGLE, "二、单意图：应该原样返回")
    all_rows["negative"] = run(NEGATIVE, "三、负例：不许拆出健身话题")

    # ---- 自动核对 ----
    print("\n===== 核对 =====")
    c_ok = sum(1 for r in all_rows["compound"]
               if r["subs"] and 2 <= len(r["subs"]) <= 3)
    s_ok = sum(1 for r in all_rows["single"]
               if r["subs"] == [r["q"]])
    n_ok = sum(1 for r in all_rows["negative"]
               if r["subs"] == [r["q"]])
    print(f"复合题拆出 2~3 路：{c_ok}/{len(COMPOUND)}")
    print(f"单意图原样返回：{s_ok}/{len(SINGLE)}")
    print(f"负例原样返回：{n_ok}/{len(NEGATIVE)}")

    errs = [r for rows in all_rows.values() for r in rows if r["subs"] is None]
    print(f"解析失败：{len(errs)} 条")
    ms = [r["ms"] for rows in all_rows.values() for r in rows]
    ms.sort()
    print(f"延迟：中位 {ms[len(ms)//2]}ms / P90 {ms[int(len(ms)*0.9)]}ms / 最大 {ms[-1]}ms")

    OUT.write_text(json.dumps({"round": rnd, "rows": all_rows}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"\n明细写入 {OUT.name}")


if __name__ == "__main__":
    main()
