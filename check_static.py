"""离线自检：不调任何模型、不需要 API key，30 毫秒跑完。

为什么需要它：这个项目最贵的一类错误不是代码写错，而是**数据与代码不同步**——
改了 kb.json 忘了重建 .kb_vectors.json，线上就会用旧向量检索新文本，
而且不会报错，只会「答案莫名其妙不对」。这种坑靠人记是靠不住的。

所以把能静态检的全检掉，CI 每次 push 都跑（.github/workflows/ci.yml）。
检测项目：

  1. kb.json 能解析、id 唯一、字段齐、正文长度在区间内、符合文风约束
  2. .kb_vectors.json 与 kb.json **逐条对齐**（条数、维度、模型、拼出来的文本）
  3. eval_cases.json 标签合法、题目不重复、gold 话题真实存在
  4. main.py 的关键常量在合理范围内
  5. 所有 .py 能编译（语法错在 CI 就暴露，不用等部署）

用法：python check_static.py        # 零依赖，标准库即可
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent
ERRORS: list[str] = []
WARNS: list[str] = []


def err(msg: str):
    ERRORS.append(msg)


def warn(msg: str):
    WARNS.append(msg)


def load(name: str, default=None):
    p = ROOT / name
    if not p.exists():
        if default is None:
            err(f"{name} 不存在")
        return default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        err(f"{name} 不是合法 JSON：{e}")
        return default


# ---------- 1. 知识库 ----------

kb = load("kb.json")
if isinstance(kb, list):
    if len(kb) < 50:
        warn(f"知识库只有 {len(kb)} 条，少于 50 条时评测结论不稳")
    ids = [d.get("id") for d in kb]
    dup = {i for i in ids if ids.count(i) > 1}
    if dup:
        err(f"kb.json 有重复 id：{sorted(dup)[:5]}")
    need = {"id", "topic", "text", "source"}
    for i, d in enumerate(kb):
        miss = need - set(d)
        if miss:
            err(f"kb[{i}] 缺字段 {miss}")
            continue
        n = len(d["text"])
        if n < 100:
            # 太短的 chunk 会被向量阈值滤掉，build 脚本本身也有这条校验
            err(f"{d['id']} 正文只有 {n} 字（<100，会被阈值滤掉）")
        elif n > 400:
            warn(f"{d['id']} 正文 {n} 字偏长，精排 token 会涨")
        for bad in ("！", "我们", "大家"):
            if bad in d["text"]:
                err(f"{d['id']} 出现文风违禁词「{bad}」")
    topics = {d.get("topic") for d in kb}
    if len(topics) < 20:
        warn(f"只有 {len(topics)} 个话题，可能过细或过粗")

# ---------- 2. 向量缓存与知识库对齐（最容易出错的地方）----------

# 向量缓存是本地产物（.gitignore 里），CI 上没有属于正常情况，只提醒不报错
if not (ROOT / ".kb_vectors.json").exists():
    warn(".kb_vectors.json 不存在（本地产物，CI 正常如此），跳过向量对齐校验")
    vec = None
else:
    vec = load(".kb_vectors.json")
if isinstance(vec, dict) and isinstance(kb, list):
    n_vec = len(vec.get("vectors") or [])
    if n_vec != len(kb):
        err(f"向量缓存 {n_vec} 条 ≠ 知识库 {len(kb)} 条 —— 改完 kb.json 没重建向量")
    else:
        expect = [f"{d['topic']}：{d['text']}" for d in kb]
        if vec.get("texts") != expect:
            err("向量缓存里的文本与 kb.json 对不上（文本变了但没重建）—— "
                "跑一次 uv run python -c 'import main; main.get_kb_vectors()'")
        if vec.get("model") != "BAAI/bge-m3":
            warn(f"向量模型是 {vec.get('model')}，与代码默认值不同，确认是有意为之")
        dims = {len(v) for v in vec["vectors"][:5]}
        if len(dims) > 1:
            err(f"向量维度不一致：{dims}")

# ---------- 3. 评测集 ----------

for fname in ("eval_cases.json", "eval_cases_colloquial.json", "eval_cases_compound.json"):
    cases = load(fname, default=[])
    if not isinstance(cases, list):
        continue
    qs = [c.get("q") for c in cases]
    dupq = {q for q in qs if qs.count(q) > 1}
    if dupq:
        err(f"{fname} 有重复题目：{list(dupq)[:3]}")
    kinds = {c.get("kind") for c in cases}
    unknown = kinds - {"hit", "refuse", "single", "compound", "neg", None}
    if unknown:
        err(f"{fname} 有未知 kind：{unknown}")
    if isinstance(kb, list):
        topics = {d.get("topic") for d in kb}
        for c in cases:
            exp = c.get("expect")
            # expect 可能是单个话题名，也可能是一组（复合题要求覆盖多个话题）
            exps = [exp] if isinstance(exp, str) else (exp or [])
            for one in exps:
                if isinstance(one, str) and one and one not in topics:
                    err(f"{fname} 里「{c.get('q')}」的 gold 话题「{one}」在知识库里不存在")

# ---------- 4. 关键常量 ----------

mp = ROOT / "main.py"
src = mp.read_text(encoding="utf-8") if mp.exists() else ""
for name, lo, hi in [("VEC_THRESHOLD", 0.3, 0.95), ("RERANK_THRESHOLD", 0.5, 0.999),
                     ("RECALL_K", 4, 50), ("RERANK_TOP_K", 1, 10)]:
    m = re.search(rf"^{name}\s*=\s*([0-9.]+)", src, re.M)
    if not m:
        # 有些常量写成 float(os.getenv("X", "0.95"))，默认值藏在括号里
        m = re.search(rf'{name}\s*=\s*(?:int|float)\(os\.getenv\("[^"]+",\s*"([0-9.]+)"\)\)', src)
    if not m:
        warn(f"main.py 里没找到常量 {name}")
        continue
    v = float(m.group(1))
    if not (lo <= v <= hi):
        err(f"{name} = {v}，不在合理区间 [{lo}, {hi}]")

# ---------- 5. 语法 ----------

# 只编当前目录的 .py，手动遍历而不是 compile_dir —— 后者会走进 .venv，
# 把几千个第三方包也编一遍，CI 上慢且噪音大。
import py_compile
import tempfile
for f in sorted(ROOT.glob("*.py")):
    try:
        py_compile.compile(str(f), cfile=str(Path(tempfile.gettempdir()) / (f.stem + ".pyc")),
                           doraise=True)
    except py_compile.PyCompileError as e:
        err(f"{f.name} 语法错误：{str(e)[:120]}")

# ---------- 输出 ----------

if WARNS:
    print("提醒：")
    for w in WARNS:
        print(f"  · {w}")
if ERRORS:
    print(f"\n失败 {len(ERRORS)} 项：")
    for e in ERRORS:
        print(f"  ✗ {e}")
    sys.exit(1)
print(f"\n自检通过（知识库 {len(kb) if isinstance(kb, list) else '?'} 条）")
