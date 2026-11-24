"""多 worker 下的共享状态。

## 为什么需要它

uvicorn 每个 worker 是独立进程，模块级变量（缓存、限流表、日额度）各进程一份。
4 个 worker 就等于：4 份缓存（命中率掉到 1/4）、每人每分钟能问 48 次、
日额度变成 1200。压测量出多 worker 有 1.8 倍吞吐收益（docs/load_test.md），
但那 1.8 倍在状态分裂面前是不能要的 —— **限流失效比变快严重得多**。

## 为什么是 SQLite 不是 Redis

这个服务跑在一个容器里，Redis 要多一个进程、一套运维、一个故障点。
而真实负载只有 2 RPS（见 load_test.md 的业务换算），SQLite 的写吞吐
（单机几千次/秒，WAL 模式）差三个数量级，够用。换 Redis 是 DAU 上十万以后的事，
现在上只是给自己加依赖 —— 面试时也说不清「为什么非它不可」。

## 两种后端

- memory：单进程，行为跟改造前**完全一样**（默认）。main.py 里走原来的代码路径，
  一行都没变，所以现有测试和线上行为不受影响。
- sqlite：多进程共享一个 db 文件。用 WAL + busy_timeout，写并发交给 SQLite 排队。

## 代价（必须说清楚）

每请求多 1~3 次写库。热缓存路径原本是纯内存（810 RPS，p50 40ms），
换成 sqlite 之后会掉到多少，是 docs/load_test.md 里实测的数字。
**这就是水平扩展的账单**：不是免费的，要知道自己付了多少。
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS cache (
    k    TEXT PRIMARY KEY,
    body TEXT NOT NULL,
    ts   REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS rate (
    ip TEXT NOT NULL,
    ts REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_rate ON rate(ip, ts);
CREATE TABLE IF NOT EXISTS daily (
    day  TEXT PRIMARY KEY,
    used INTEGER NOT NULL
);
"""


class SqliteStore:
    """多进程共享的缓存 + 滑动窗口限流 + 日额度。

    每个线程一个连接：FastAPI 的同步路由跑在线程池里，sqlite3 的连接默认不能跨线程，
    用 threading.local 是最省事的正确做法（比 check_same_thread=False + 全局锁好，
    后者会把所有请求又串到一把锁上，等于把刚修掉的日志锁问题重新犯一遍）。
    """

    def __init__(self, path: str | Path, cache_max: int = 500,
                 expire_s: float = 86400.0):
        self.path = str(path)
        self.cache_max = cache_max
        self.expire_s = expire_s
        self._local = threading.local()
        self._rate_ops = 0          # 每 N 次才做一次全表过期清理，别每个请求都扫
        self._put_ops = 0           # 每 N 次写才检查一次条数/淘汰，缩短写事务
        self._init_db()

    # ---------- 连接 ----------
    def _init_db(self) -> None:
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        c = sqlite3.connect(self.path, timeout=5)
        try:
            c.executescript(SCHEMA)
            c.commit()
        finally:
            c.close()

    def _retry(self, fn, tries: int = 4):
        """WAL 模式下多个连接同时写会遇到 SQLITE_BUSY，而且**读事务想升级成写事务时
        busy_timeout 不会生效**（SQLITE_BUSY_SNAPSHOT），只能自己重试。
        压测里 8 个线程同时写就复现了：『database is locked』。
        重试 4 次、间隔 10/20/30ms，够扛住 4 个 worker 的写竞争；
        真扛不住就让异常冒出去由调用方降级（放行 / 跳过缓存）。"""
        for i in range(tries):
            try:
                return fn()
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower() and i < tries - 1:
                    time.sleep(0.01 * (i + 1))
                    continue
                raise

    def _c(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=5, isolation_level=None)
            # WAL：读不挡写。没有它，4 个 worker 会在同一把文件锁上互等。
            conn.execute("PRAGMA journal_mode=WAL")
            # NORMAL：断电可能丢最后几条（缓存丢了就重算，无所谓），
            # 但比 FULL 少一次 fsync，写吞吐差一个数量级。
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=5000")
            self._local.conn = conn
        return conn

    # ---------- 缓存 ----------
    def cache_get(self, key: str) -> dict | None:
        try:
            row = self._c().execute("SELECT body FROM cache WHERE k=?", (key,)).fetchone()
        except sqlite3.Error:
            return None
        if not row:
            return None
        try:
            return json.loads(row[0])
        except (TypeError, ValueError):
            return None

    def cache_put(self, key: str, body: dict) -> None:
        payload = json.dumps(body, ensure_ascii=False)
        self._put_ops += 1
        check_evict = self._put_ops % 20 == 0

        def _do():
            c = self._c()
            c.execute(
                "INSERT INTO cache(k, body, ts) VALUES(?,?,?) "
                "ON CONFLICT(k) DO UPDATE SET body=excluded.body, ts=excluded.ts",
                (key, payload, time.time()))
            if not check_evict:
                return
            # 淘汰按 ts（最久没被写的先走）。不是严格 LRU —— 真 LRU 要在每次读时
            # 更新 ts，等于把读变成写，读多写少不划算。缓存失效只影响命中率，
            # 不影响正确性，近似就够。
            # 条数检查每 20 次写才做一次：COUNT + DELETE 会让写事务变长，
            # 多进程下长事务就是冲突来源。缓存条数短暂超一点没有实际危害。
            n = c.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
            if n > self.cache_max:
                c.execute(
                    "DELETE FROM cache WHERE k IN "
                    "(SELECT k FROM cache ORDER BY ts ASC LIMIT ?)", (n - self.cache_max,))

        try:
            self._retry(_do)
        except sqlite3.Error:
            pass

    def cache_size(self) -> int:
        try:
            return self._c().execute("SELECT COUNT(*) FROM cache").fetchone()[0]
        except sqlite3.Error:
            return 0

    # ---------- 限流 ----------
    def rate_wait(self, ip: str, limit: int, window: float = 60.0) -> float:
        """返回 0 表示放行，正数表示要等几秒。**所有 worker 共用同一张表。**

        仍然是滑动窗口，不是固定窗口：固定窗口在整分钟边界会放过 2 倍流量，
        这对「防刷」正好是最坏的形状（改造前的注释里就写过，理由没变）。
        """
        now = time.time()
        try:
            c = self._c()
            # 过期清理不必每个请求都做。每 200 次清一次，表大小就有界了：
            # 一分钟内的请求行本来就会被下一次清理带走，多留几百行不影响判定。
            self._rate_ops += 1
            if self._rate_ops % 200 == 0:
                c.execute("DELETE FROM rate WHERE ts < ?", (now - window * 2,))

            n = c.execute("SELECT COUNT(*) FROM rate WHERE ip=? AND ts>=?",
                          (ip, now - window)).fetchone()[0]
            if n >= limit:
                oldest = c.execute(
                    "SELECT MIN(ts) FROM rate WHERE ip=? AND ts>=?", (ip, now - window)
                ).fetchone()[0]
                return max(0.1, window - (now - oldest))
            c.execute("INSERT INTO rate(ip, ts) VALUES(?,?)", (ip, now))
            return 0.0
        except sqlite3.Error:
            # 库坏了不能把所有用户挡在门外 —— 放行，并在 /stats 里能看出来。
            return 0.0

    # ---------- 日额度 ----------
    def daily_take(self, quota: int) -> bool:
        """占一个名额，返回是否还有额度。一条 UPSERT + RETURNING 搞定，
        不会出现「两个 worker 同时读到 299 都放行」的竞态。"""
        day = time.strftime("%Y-%m-%d")
        try:
            c = self._c()
            row = c.execute(
                "INSERT INTO daily(day, used) VALUES(?, 1) "
                "ON CONFLICT(day) DO UPDATE SET used = used + 1 "
                "RETURNING used", (day,)).fetchone()
            used = row[0] if row else 1
            return used <= quota
        except sqlite3.Error:
            return True

    def daily_used(self) -> int:
        day = time.strftime("%Y-%m-%d")
        try:
            row = self._c().execute("SELECT used FROM daily WHERE day=?", (day,)).fetchone()
            return int(row[0]) if row else 0
        except sqlite3.Error:
            return 0
