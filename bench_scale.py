"""规模实验：把「数据量小」这件事，换成「数据量大了会怎样」。

为什么做这个：面试官问「10 万条文档怎么办」时，答「我只有 119 条」是掉分的，
答「我测过」才是加分。但造 10 万条假健身内容没意义（一问来源就崩，而且污染检索质量）。
所以只测工程指标 —— 延迟、内存、召回，这些只跟「条数 × 维度 × 实现方式」有关，
跟文本内容是否真实无关。

做法：以真实的 119 条 bge-m3 向量（1024 维）为基底，随机采样 + 加小噪声合成大库，
这样向量分布接近真实（同模型、同维度、同归一化），测出来的延迟和内存是准的。

对比三种实现：
  A  当前实现      sum(x*y for x,y in zip(a,b))，每条一算、全排序
  B  矩阵化        L2 归一化后一次点积，仍全排序
  C  矩阵化 + topk  argpartition 只取前 k，不排全量

用法：
    uv run --with numpy python bench_scale.py
    uv run --with numpy python bench_scale.py --sizes 119,1000,10000
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

VEC_FILE = Path(".kb_vectors.json")
DIM_DEFAULT = 1024


# ---------- 三个实现 ----------

def cosine_py(a, b):
    """当前 main.py 里的手写实现，原样搬过来。"""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    return dot / (norm_a * norm_b)


def search_a_python(q_vec, mat_list, top_k):
    """A：逐条 Python 循环 + round + 排序（复刻线上逻辑）"""
    scores = [(i, round(cosine_py(q_vec, v), 4)) for i, v in enumerate(mat_list)]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


def search_b_matrix(q, mat_norm, top_k):
    """B：矩阵化全排序。mat_norm 已 L2 归一化，q 也归一化，点积即余弦。"""
    scores = mat_norm @ q
    order = np.argsort(-scores)
    return [(int(i), float(scores[i])) for i in order[:top_k]]


def search_c_topk(q, mat_norm, top_k):
    """C：矩阵化 + argpartition 只取前 k，O(n) 而非 O(n log n)"""
    scores = mat_norm @ q
    k = min(top_k, scores.shape[0])
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return [(int(i), float(scores[i])) for i in idx]


# ---------- 合成数据 ----------

def build_synth(base: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """从真实向量里采样 + 加噪，扩到 n 条。
    噪声压得很小（0.02），保证向量还在同一个语义邻域内，
    不会出现「10 万条里全是随机噪声，暴力检索反而更慢/更快」的失真。"""
    m = base.shape[0]
    if n <= m:
        return base[:n].copy()
    pick = rng.integers(0, m, size=n)
    out = base[pick].copy()          # fancy indexing 已返回新数组，不用再 copy 一次
    # 分块加噪：10 万条整块加噪会同时驻留两份 410MB，容易把小机器顶爆
    step = 20000
    for i in range(0, n, step):
        blk = out[i:i + step]
        blk += rng.normal(0.0, 0.02, size=blk.shape).astype(np.float32)
    # 原地归一化，让点积等于余弦（归一化不改余弦值，所以 A 也能用同一份矩阵）
    out /= np.linalg.norm(out, axis=1, keepdims=True)
    return out


def bench(fn, repeats: int, warmup: int = 2):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    p50 = statistics.median(ts)
    p95 = ts[int(len(ts) * 0.95) - 1] if len(ts) > 1 else ts[0]
    return p50, p95


def fmt_ms(v):
    if v >= 1000:
        return f"{v/1000:.2f} s"
    return f"{v:.1f} ms"


def fmt_mb(nbytes):
    mb = nbytes / 1024 / 1024
    if mb >= 1024:
        return f"{mb/1024:.2f} GB"
    return f"{mb:.1f} MB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="119,1000,10000,100000")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--top-k", type=int, default=12, help="对应 main.RECALL_K")
    ap.add_argument("--python-max", type=int, default=10000,
                    help="纯 Python 实现在多大以上只跑一次（太慢）")
    args = ap.parse_args()

    if not VEC_FILE.exists():
        sys.exit(f"找不到 {VEC_FILE}，先在项目根目录跑一次服务让它生成向量缓存")
    data = json.loads(VEC_FILE.read_text(encoding="utf-8"))
    base = np.asarray(data["vectors"], dtype=np.float32)
    dim = base.shape[1]
    print(f"基底：真实 {base.shape[0]} 条 × {dim} 维（{data.get('model')}）")
    print(f"机器：{platform.processor() or platform.machine()} / Python {platform.python_version()}"
          f" / numpy {np.__version__}")
    print(f"参数：每档重复 {args.repeats} 次，top_k={args.top_k}\n")

    rng = np.random.default_rng(20261124)
    q = base[0].copy()
    q /= np.linalg.norm(q)

    sizes = [int(s) for s in args.sizes.split(",")]
    rows = []

    header = f"{'条数':>8} {'内存(向量)':>11} {'A 纯Python':>13} {'B 矩阵全排':>12} {'C 矩阵topk':>12} {'B 加速':>8} {'C 加速':>8}"
    print(header)
    print("-" * len(header))

    for n in sizes:
        mat = build_synth(base, n, rng)          # 已原地归一化
        mat_norm = mat                            # 同一份，不再复制一份省内存
        mem_vec = mat.nbytes
        # 加上 Python list 容器的开销（每条 float 对象约 32B + 指针 8B）
        mem_py = n * (dim * 32 + 8) if n <= args.python_max else None

        rep_a = args.repeats if n <= args.python_max else 1
        a50, a95 = bench(lambda: search_a_python(q, mat, args.top_k), rep_a)
        b50, b95 = bench(lambda: search_b_matrix(q, mat_norm, args.top_k), args.repeats)
        c50, c95 = bench(lambda: search_c_topk(q, mat_norm, args.top_k), args.repeats)

        sp_b = a50 / b50 if b50 else 0
        sp_c = a50 / c50 if c50 else 0
        memtxt = fmt_mb(mem_vec) + (f" (py列表≈{fmt_mb(mem_py)})" if mem_py else "")
        print(f"{n:>8} {memtxt:>11} {fmt_ms(a50):>13} {fmt_ms(b50):>12} {fmt_ms(c50):>12} "
              f"{sp_b:>7.1f}x {sp_c:>7.1f}x")

        rows.append({
            "n": n, "dim": dim,
            "mem_matrix_mb": round(mem_vec / 1024 / 1024, 2),
            "mem_pylist_mb": round(mem_py / 1024 / 1024, 2) if mem_py else None,
            "a_p50_ms": round(a50, 2), "a_p95_ms": round(a95, 2),
            "b_p50_ms": round(b50, 2), "b_p95_ms": round(b95, 2),
            "c_p50_ms": round(c50, 2), "c_p95_ms": round(c95, 2),
            "speedup_b": round(sp_b, 1), "speedup_c": round(sp_c, 1),
        })

    # 核实 C 的结果和 A 一致（不能为了快把结果搞错）
    n = sizes[-1]
    mat = build_synth(base, n, np.random.default_rng(20261124))
    a_top = search_a_python(q, mat, args.top_k) if n <= args.python_max else None
    c_top = search_c_topk(q, mat, args.top_k)
    print("\n正确性核对：")
    if a_top:
        print(f"  A 的 top1={a_top[0][0]} 分={a_top[0][1]:.4f} ｜ C 的 top1={c_top[0][0]} 分={c_top[0][1]:.4f}")
        sa = sorted(s for _, s in a_top)
        sc = sorted(round(s, 4) for _, s in c_top)
        print(f"  top{args.top_k} 分数多重集一致：{sa == sc}")
        ids_a = {i for i, _ in a_top}
        ids_c = {i for i, _ in c_top}
        print(f"  top{args.top_k} 的 id 集合一致：{ids_a == ids_c}"
              + ("" if ids_a == ids_c else "  ← 分数相同、顺序不同（下面对比 A 用 round 后排序造成的并列）"))
        # 找出并列，证明「顺序不同」不是实现错误
        dup = {}
        for i, s in a_top:
            dup.setdefault(s, []).append(i)
        tie = {s: ids for s, ids in dup.items() if len(ids) > 1}
        if tie:
            print(f"  确实存在并列：{list(tie.values())[:2]} 分数都约 {list(tie)[0]:.4f}"
                  f" —— A 把分数 round 到 4 位再排序，并列时按原序；B/C 用原始分数排，"
                  f"所以边界处顺序会换人。")
    # B 与 C 必须完全一致
    b_top = search_b_matrix(q, mat, args.top_k)
    print(f"  B 与 C 的 top{args.top_k} 完全一致：{[i for i,_ in b_top] == [i for i,_ in c_top]}")

    out = Path("bench_scale.json")
    out.write_text(json.dumps({
        "meta": {
            "base_chunks": int(base.shape[0]), "dim": dim,
            "model": data.get("model"), "top_k": args.top_k,
            "repeats": args.repeats,
            "python": platform.python_version(),
            "machine": platform.processor() or platform.machine(),
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "rows": rows,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n结果已写入 {out}")


if __name__ == "__main__":
    main()
