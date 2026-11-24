"""并发压测：找出这份代码在单机上的真实上限，以及上限由什么决定。

为什么不用 ab / wrk / hey：
它们的数字好看，但对这个项目没用 —— 压测要回答的是「并发涨上去，谁先垮」，
而这个问题在不同路径上答案完全不同：

    /meta      纯内存，读几个常量      -> 测框架本身（uvicorn + FastAPI）能扛多少
    缓存命中    /rag 的 LRU 命中       -> 测「真实链路上除上游调用以外的全部开销」
    缓存未命中  /rag 打上游模型        -> 测线程池容量，以及上游并发下的失败
    /plan      4 次调用 + 工具循环     -> 测长请求占着线程时对短请求的影响

分开压才有诊断价值。混在一起只知道「慢了」，不知道「为什么慢」。

三个必须说清的口径（不然数字是假的）：

1. **限流必须显式放开**。默认每人每分钟 12 次、全站每天 300 次，直接压会全是 429，
   测到的是限流器而不是性能。所以脚本自己带 `X-Forwarded-For` 轮换模拟不同来源，
   并且明确报告 429 有几个 —— 如果有 429，这一档的数字就不作数。
2. **客户端和服务端抢同一台机器的 CPU**。20 核机器上抓 1~2 核给客户端，剩下的给服务端。
   所以 RPS 是「同机压测」的下界，不是上线后的上限。
3. **closed-loop**：每个虚拟用户发完一个再发下一个。它测的是吞吐饱和，不是「固定速率下
   能扛多久」——后者要 open-loop，对判断「上线会不会雪崩」更有意义，但这里先不做。

跑法（httpx 不进部署依赖，用 uv 临时挂上）：
    uv run --with httpx python load_test.py --mode meta --levels 1,2,4,8,16,32,64
    uv run --with httpx python load_test.py --mode rag-hit --levels 4,16,48
    uv run --with httpx python load_test.py --mode rag-miss --levels 1,4,12 --per 4
    uv run --with httpx python load_test.py --mode plan --levels 1,3 --per 2
"""
from __future__ import annotations

import argparse
import asyncio
import json
import secrets
import statistics
import sys
import time

import httpx

# 压测用的固定问题。选它是因为结果会被缓存（rag-hit 档就是靠这个），
# 而且长度接近真实问法的中位数，不特别长也不特别短。
Q_HIT = "减脂期蛋白质吃多少"
Q_MISS = "肩袖损伤怎么练"          # 库里没有，必然走完整链路（会拒答，但链路跑满）

MODES = {
    "meta":     ("GET", "/meta", None, "纯内存：测框架 RPS 上限"),
    "kb":       ("GET", "/kb", None, "纯内存但返回体大：测序列化开销"),
    "rag-hit":  ("POST", "/rag", {"question": Q_HIT}, "缓存命中：真实链路去掉上游调用"),
    "rag-miss": ("POST", "/rag", {"question": Q_MISS}, "缓存未命中：打上游模型，测线程池"),
    "plan":     ("POST", "/plan", {"goal": "增肌", "days": 3, "equipment": "健身房",
                                   "session_min": 60, "level": "新手", "injuries": []},
                 "多步工具调用：长请求"),
}


def pct(xs, p):
    """最近秩百分位。样本少的时候不做插值 —— 20 个样本算出来的 p99 就是最大值，
    插值会把它美化成一个不存在的中间值。"""
    if not xs:
        return 0.0
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round(p / 100 * len(s) + 0.5)) - 1))
    return s[k]


class Runner:
    def __init__(self, base: str, code: str, timeout: float, query: str = ""):
        self.base = base.rstrip("/")
        self.code = code
        self.timeout = timeout
        # 线上那道平台鉴权走 query（?sharecode=...），不在请求头里。
        # 不带它连 /meta 都是 403 —— 这也是「别拿压测器直接打公网」的原因之一：
        # 打到的第一层根本不是我的代码。
        self.query = query.lstrip("?")
        # 连接池要大于最大并发，否则 httpx 内部排队会把延迟算到「服务端」头上。
        # limits 是 httpx 客户端级的，每个档位重建一次客户端更干净。
        self._ip = 0

    def client(self, concurrency: int) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout),
            limits=httpx.Limits(max_connections=concurrency + 8,
                                max_keepalive_connections=concurrency + 8),
            headers={"x-access-code": self.code},
            follow_redirects=False,
        )

    def next_ip(self) -> str:
        """轮换来源 IP。压测时每个请求算不同的人，否则第二分钟就被自己的限流挡住。"""
        self._ip += 1
        n = self._ip
        return f"10.{(n >> 16) & 255}.{(n >> 8) & 255}.{n & 255}"

    async def one(self, c: httpx.AsyncClient, method: str, path: str, body):
        t0 = time.perf_counter()
        status, err, size = 0, "", 0
        try:
            url = self.base + path + (("?" + self.query) if self.query else "")
            if method == "GET":
                r = await c.get(url, headers={"x-forwarded-for": self.next_ip()})
            else:
                r = await c.post(url, json=body,
                                 headers={"x-forwarded-for": self.next_ip()})
            status = r.status_code
            size = len(r.content)
            if status >= 500:
                err = (r.text or "")[:120].replace("\n", " ")
        except Exception as e:                      # noqa: BLE001
            err = f"{type(e).__name__}: {str(e)[:100]}"
        dt = (time.perf_counter() - t0) * 1000
        return dt, status, err, size

    async def level(self, mode: str, concurrency: int, total: int, vary: bool = False) -> dict:
        method, path, body, _ = MODES[mode]
        c = self.client(concurrency)
        sem = asyncio.Semaphore(concurrency)
        lat, codes, errs, sizes = [], {}, {}, []

        async def worker():
            async with sem:
                b = body
                # 缓存键就是问题原文，同一个问题第二次问必然命中。
                # 要测「打上游」就必须每次换一个键，否则压的是缓存不是链路。
                # 加后缀而不是换整个问题：检索部分的行为不变，只有缓存键不同。
                if vary and body:
                    b = {**body, "question": f"{body['question']} {secrets.token_hex(4)}"}
                return await self.one(c, method, path, b)

        # 先打一个请求预热。第一次要加载 jieba 词典、建向量矩阵，
        # 不预热的话这一档的 p99 里永远混着一次 2 秒的冷启动，读起来像服务端抖动。
        await self.one(c, method, path, body)

        wall0 = time.perf_counter()
        results = await asyncio.gather(*[worker() for _ in range(total)])
        wall = time.perf_counter() - wall0
        await c.aclose()

        for dt, st, err, size in results:
            lat.append(dt)
            codes[st] = codes.get(st, 0) + 1
            if err:
                errs[err] = errs.get(err, 0) + 1
            sizes.append(size)

        return {
            "concurrency": concurrency,
            "total": total,
            "wall_s": round(wall, 3),
            "rps": round(total / wall, 1) if wall > 0 else 0,
            "p50": round(pct(lat, 50), 1),
            "p95": round(pct(lat, 95), 1),
            "p99": round(pct(lat, 99), 1),
            "max": round(max(lat), 1) if lat else 0,
            "codes": {str(k): v for k, v in sorted(codes.items())},
            "errors": dict(sorted(errs.items(), key=lambda kv: -kv[1])[:3]),
            "resp_kb": round(statistics.mean(sizes) / 1024, 1) if sizes else 0,
        }


async def fetch_stats(base, code):
    """拉服务端的自报数。压测器算出来 200 RPS，服务端说只处理了 40 个请求，
    那说明中间有东西在替它应答（CDN / 缓存 / 连接被拒），数字就不能信。"""
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(base.rstrip("/") + "/stats",
                            headers={"x-access-code": code})
            if r.status_code != 200:
                return None
            d = r.json()
            return {k: d.get(k) for k in
                    ("requests", "cache_hits", "rate_limited", "llm_calls")}
    except Exception:                               # noqa: BLE001
        return None


def delta(before, after):
    if not before or not after:
        return ""
    return " ".join(f"{k}+{after[k] - before[k]}" for k in before
                    if isinstance(after.get(k), int) and after[k] != before[k]) or "无变化"


async def run(args):
    method, path, body, desc = MODES[args.mode]
    r = Runner(args.url, args.code, args.timeout, args.query)
    levels = [int(x) for x in args.levels.split(",") if x.strip()]

    print(f"目标    {args.url}{path}  [{args.mode}] {desc}")
    print(f"方法    {method}   并发档位 {levels}  每档请求数 {'(自动)' if not args.total else args.total}")
    print(f"口径    closed-loop / 每请求轮换来源 IP（绕过 per-IP 限流）/ 超时 {args.timeout}s"
          + ("  / 每请求换问题（绕开缓存）" if args.vary else ""))
    print()

    head = f"{'并发':>4} {'请求':>5} {'耗时':>7} {'RPS':>8} {'p50':>8} {'p95':>8} {'p99':>8} {'max':>8} {'返回':>6}  状态码"
    print(head)
    print("-" * len(head))

    rows = []
    for lv in levels:
        total = args.total or max(lv * args.per, lv)
        before = await fetch_stats(args.url, args.code) if args.verify else None
        row = await r.level(args.mode, lv, total, vary=args.vary)
        after = await fetch_stats(args.url, args.code) if args.verify else None

        row["server_delta"] = delta(before, after) if args.verify else ""
        rows.append(row)

        codes = "/".join(f"{k}×{v}" for k, v in row["codes"].items())
        warn = ""
        if "429" in row["codes"]:
            warn = "  <- 有 429，这一档测的是限流器，不作数"
        print(f"{lv:>4} {total:>5} {row['wall_s']:>7.2f} {row['rps']:>8.1f} "
              f"{row['p50']:>7.0f}m {row['p95']:>7.0f}m {row['p99']:>7.0f}m "
              f"{row['max']:>7.0f}m {row['resp_kb']:>5.1f}K  {codes}{warn}")
        if row["errors"]:
            for e, n in row["errors"].items():
                print(f"{'':>4}   错误 ×{n}: {e}")
        if row["server_delta"]:
            print(f"{'':>4}   服务端自报: {row['server_delta']}")

    # 吞吐曲线上一档的「每加一个并发，多拿多少 RPS」。这个数掉到接近 0 就是饱和点，
    # 再往上加并发只会把延迟推高（排队变长），不会提高吞吐 —— 这是判断「该扩容还是该优化」的分水岭。
    print()
    print("并发收益（每多 1 个并发换来多少 RPS）：")
    for a, b in zip(rows, rows[1:]):
        dc = b["concurrency"] - a["concurrency"]
        dr = b["rps"] - a["rps"]
        print(f"  {a['concurrency']:>3} -> {b['concurrency']:<3} "
              f"+{dr:>7.1f} RPS  ({dr / dc:>6.1f} RPS/并发)  p95 {a['p95']:.0f}m -> {b['p95']:.0f}m")

    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump({"mode": args.mode, "url": args.url, "rows": rows},
                      f, ensure_ascii=False, indent=1)
        print(f"\n已存 {args.save}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8031")
    ap.add_argument("--code", default="kanwo-2026", help="x-access-code")
    ap.add_argument("--mode", default="meta", choices=list(MODES) + ["custom"])
    ap.add_argument("--levels", default="1,2,4,8,16,32,64")
    ap.add_argument("--per", type=int, default=10, help="每档每人发几个请求")
    ap.add_argument("--total", type=int, default=0, help="覆盖每档总请求数（0=并发×per）")
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--vary", action="store_true",
                    help="每个请求换一个问题（加随机后缀），绕开缓存去打上游。测链路用这个")
    ap.add_argument("--verify", action="store_true", help="压测前后拉 /stats 对账")
    ap.add_argument("--query", default="", help="附加到每个请求的 query，如 sharecode=xxx")
    ap.add_argument("--save", default="")
    # 通用模式：压任意路径。验证「同步路由在线程池里跑」这类框架行为时用得上，
    # 不想为了压一个 /slow 再写一个脚本。
    ap.add_argument("--path", default="", help="mode=custom 时的路径")
    ap.add_argument("--method", default="GET")
    ap.add_argument("--body", default="", help="mode=custom 时的 JSON 请求体")
    a = ap.parse_args()
    if a.mode == "custom":
        MODES["custom"] = (a.method.upper(), a.path, json.loads(a.body) if a.body else None,
                           "自定义")
    t0 = time.time()
    asyncio.run(run(a))
    print(f"\n总耗时 {time.time() - t0:.1f}s")


if __name__ == "__main__":
    sys.exit(main())
