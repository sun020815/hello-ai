# -*- coding: utf-8 -*-
"""把公众号文章原稿切成 chunk，合并进 kb.json。

用法：
    uv run python import_articles.py --dry-run    # 只看切出来什么样，不写文件
    uv run python import_articles.py              # 真的合并进 kb.json

三个设计决定，都不是随便定的：

1. **按小节切，不按字数切。** 一小节本来就是一个能单独回答的问题，
   和 kb.json 里手工写的那 40 条是同一个粒度。按 300 字硬切会把一个动作的
   要领切断成两半，两半都答不了问题。

2. **小节标题留在正文里做前缀。** `为什么低碳到后来不管用。碳水一降再降……`
   之前验证过「带话题名的文本」比「只有正文」命中率高得多，同一个道理。

3. **按文章去重，不按 chunk 去重。** 同一个 id 已经存在就跳过，所以脚本可以
   反复跑，不会把同一篇文章导入两遍。
"""
import argparse
import json
import re
from pathlib import Path

HOME = Path.home()
KB_PATH = Path(__file__).parent / "kb.json"

# 每篇文章配一个话题。话题必须是 kb.json 里已有的，或者这次新增后要一起维护的。
# 新增了「胸肌 / 增肌 / 动作技术」三个话题，评测集要跟着补题，否则新话题没人测。
ARTICLES = [
    {"file": "健身公众号-三倍碳水减脂法.md", "topic": "减脂"},
    {"file": "新手减脂全攻略.md", "topic": "减脂"},
    {"file": "健身公众号-减脂全攻略-精简版.md", "topic": "减脂"},
    {"file": "减脂高碳还是低碳-增肌要不要有氧.md", "topic": "减脂"},
    {"file": "健身公众号-第一篇-深蹲.md", "topic": "深蹲"},
    {"file": "胸肌不涨的真相：你一直漏掉了三头.md", "topic": "胸肌"},
    {"file": "WorkBuddy/2026-10-19-15-31-00/健身焚诀9句意念_成稿_2026-10-24.md", "topic": "动作技术"},
    {"file": "WorkBuddy/2026-10-19-15-31-00/增肌减脂平衡_成稿_2026-10-23.md", "topic": "增肌"},
]

# 每块的字数区间。太短撑不起一个答案，太长会把两个问题混进一个块。
# 下限给到 60：很多小节（比如「机制四：给食欲一个出口」）本身就短，但它是完整的知识点，
# 硬并到别的小节里反而把两件事搅在一起。
MIN_CHARS, MAX_CHARS = 60, 400


def clean_md(s: str) -> str:
    """把排版标记清掉，只留文字。图片整个删掉，因为 RAG 读不了图。"""
    s = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", s)        # 图片
    s = re.sub(r"<!--.*?-->", "", s, flags=re.S)      # HTML 注释
    s = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", s)    # 链接只留文字
    s = re.sub(r"[*_`>]", "", s)                      # 加粗/斜体/引用/代码符号
    s = re.sub(r"^\s*\|[\s:\-|]+\|\s*$", "", s, flags=re.M)  # 表格的 |---|---| 分隔行
    s = re.sub(r"^\s*[-—]{2,}\s*$", "", s, flags=re.M)  # 分隔线
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def text_len(s: str) -> int:
    """按非空白字符算长度。中文按字算，比 len() 更接近直觉。"""
    return len(re.sub(r"\s", "", s))


def split_to_paragraphs(block: str):
    """一块内容太长时，按空行切成段落，再尽量均分成几块。"""
    paras = [p.strip() for p in re.split(r"\n\s*\n", block) if p.strip()]
    return paras


def pack(paras):
    """把段落贪心装进块里，每块不超过 MAX_CHARS。"""
    out, buf = [], ""
    for p in paras:
        candidate = (buf + "\n" + p).strip() if buf else p
        if text_len(candidate) > MAX_CHARS and buf:
            out.append(buf)
            buf = p
        else:
            buf = candidate
    if buf:
        out.append(buf)

    # 太短的块并到前一块去（前提是不超过上限）
    merged = []
    for chunk in out:
        if merged and text_len(chunk) < MIN_CHARS and text_len(merged[-1]) + text_len(chunk) <= MAX_CHARS:
            merged[-1] = merged[-1] + "\n" + chunk
        else:
            merged.append(chunk)
    return merged


def join_head(head: str, body: str) -> str:
    """把小节标题接到正文前面当语义锚点。
    标题本身已经带标点（「为什么低碳到后来，不灵了？」）时不再补句号，
    不然会拼出「了？。低碳减脂」这种双标点。"""
    if not head or head == "开头":
        return body
    if head[-1] in "。！？，、：；!?":
        return f"{head}{body}"
    return f"{head}。{body}"


def merge_short(pieces):
    """把太短的块并到前一块里。
    跨子小节的合并在 pack 里做不到（pack 只管一个小节内部），
    但短块进库反而有害：几十个字的向量撑不起语义，问什么都匹配不准。
    合并时把后一块的小节标题留在正文里，别让「机制四」这个锚点丢了。"""
    out = []
    for head, body in pieces:
        if out and text_len(body) < MIN_CHARS:
            ph, pb = out[-1]
            joined = pb + "\n" + join_head(head, body)
            if text_len(joined) <= MAX_CHARS:
                out[-1] = (ph, joined)
                continue
        out.append((head, body))
    return out


def split_article(md: str):
    """按 ## 切小节，小节内再按 ### 切子小节，最后按长度打散。
    返回 [(小节标题, 正文)]。"""
    md = clean_md(md)
    md = re.sub(r"^#\s+.*$", "", md, flags=re.M)      # 去掉文章一级标题，标题已经存进 source

    # 按二级标题切。前面补一个 \n 保证行首的 ## 也能匹配上。
    raw = re.split(r"\n##\s+", "\n" + md)
    intro, sections = raw[0].strip(), raw[1:]

    pieces = []
    if text_len(intro) >= MIN_CHARS:
        pieces.append(("开头", intro))

    for sec in sections:
        lines = sec.split("\n", 1)
        sec_title = lines[0].strip()
        sec_body = (lines[1] if len(lines) > 1 else "").strip()

        # 小节里有 ### 子标题：每个子标题是一个独立单元（比如「动作1」「动作2」）
        subs = re.split(r"\n###\s+", "\n" + sec_body)
        sub_head, sub_rest = subs[0].strip(), subs[1:]
        if sub_rest:
            if text_len(sub_head) >= MIN_CHARS:
                for body in pack(split_to_paragraphs(sub_head)):
                    pieces.append((sec_title, body))
            for s in sub_rest:
                sl = s.split("\n", 1)
                st = sl[0].strip()
                sb = (sl[1] if len(sl) > 1 else "").strip()
                for body in pack(split_to_paragraphs(sb)):
                    pieces.append((st or sec_title, body))
        else:
            # 同一个小节被长度切成多块时，只有第一块带标题。
            # 每块都带的话，三块的向量前缀一模一样，检索时互相抢位置，
            # 而且后来那块根本不知道自己讲的是第几部分。
            for k, body in enumerate(pack(split_to_paragraphs(sec_body))):
                pieces.append((sec_title if k == 0 else "", body))

    return merge_short(pieces)


def jaccard(a: str, b: str, n: int = 3) -> float:
    ga = {a[i:i + n] for i in range(len(a) - n + 1)}
    gb = {b[i:i + n] for i in range(len(b) - n + 1)}
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


def body_key(s: str) -> str:
    """只留中文数字字母，用来做近似重复判断"""
    return re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9]", "", s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="只打印，不写 kb.json")
    args = ap.parse_args()

    kb = json.loads(KB_PATH.read_text(encoding="utf-8"))
    existing = {d["id"]: d for d in kb}
    existing_bodies = [(d["id"], body_key(d["text"])) for d in kb]

    added, skipped, duplicated = [], 0, []

    for ai, art in enumerate(ARTICLES, 1):
        path = HOME / art["file"]
        if not path.exists():
            print(f"[跳过] 文件不存在：{path}")
            continue
        md = path.read_text(encoding="utf-8")
        title = next((l[2:].strip() for l in md.splitlines() if l.startswith("# ")), path.stem)
        pieces = split_article(md)

        print(f"\n【{ai}】{title}  →  话题「{art['topic']}」，{len(pieces)} 块")
        for bi, (head, body) in enumerate(pieces, 1):
            cid = f"art{ai:02d}-{bi:02d}"
            text = join_head(head, body)
            if cid in existing:
                skipped += 1
                print(f"   {cid} 已存在，跳过")
                continue

            key = body_key(body)
            dup = next((i for i, k in existing_bodies if jaccard(key, k) > 0.75), None)
            if dup:
                duplicated.append((cid, dup))
                print(f"   {cid} 与 {dup} 高度重复（>{0.75:.0%}），跳过")
                continue

            print(f"   {cid}  [{text_len(text):3d} 字] {text[:46]}…")
            added.append({
                "id": cid,
                "topic": art["topic"],
                "text": text,
                "source": f"公众号《{title}》",
            })
            existing_bodies.append((cid, key))

    print(f"\n合计：新增 {len(added)} 块，跳过已存在 {skipped} 块，跳过重复 {len(duplicated)} 块")
    print(f"知识库：{len(kb)} → {len(kb) + len(added)} 块")

    if args.dry_run:
        print("\n--dry-run：没有写文件")
        return
    if not added:
        print("没有新内容，kb.json 未改动")
        return

    kb.extend(added)
    KB_PATH.write_text(json.dumps(kb, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已写入 {KB_PATH}")
    print("下一步：删掉 .kb_vectors.json 让向量重算（uv run python eval.py 会自动重建）")


if __name__ == "__main__":
    main()
