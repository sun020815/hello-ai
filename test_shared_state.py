"""共享状态层的离线测试：不联网、不花钱。

钉住四件事：
    1. 缓存能跨"连接"读到（模拟跨进程）
    2. 限流是滑动窗口，不是固定窗口，不同 IP 互不影响
    3. **并发占额度不超发** —— 这才是状态外置最容易错的地方：
       两个 worker 同时读到 used=299，都放行，实际发了 301 个
    4. 库被写坏/文件被删时降级放行，不能把所有人挡在门外

跑法：python test_shared_state.py
"""
import os
import tempfile
import threading
import time
from pathlib import Path

import shared_state

fails = []


def check(name, cond, extra=""):
    print(("  ok   " if cond else "  FAIL ") + name + ("" if cond else f"   -> {extra}"))
    if not cond:
        fails.append(name)


TMP = Path(tempfile.mkdtemp()) / "state.db"


def fresh():
    """换一个新库文件，而不是删旧的。

    Windows 上删不掉仍被连接占用的 db（WinError 32），而线程本地的连接
    在这个测试里本来就不会关。换个文件更简单，也能顺便验证
    「多个库文件互不干扰」。
    """
    global TMP
    TMP = Path(tempfile.mkdtemp()) / "state.db"
    return TMP


print("【一、缓存跨连接可读】")
fresh()
a = shared_state.SqliteStore(TMP, cache_max=3)
a.cache_put("k1", {"answer": "深蹲膝盖别内扣", "tokens": 12})
# 换一个实例，相当于换一个 worker 进程
b = shared_state.SqliteStore(TMP, cache_max=3)
got = b.cache_get("k1")
check("另一个实例读得到", got and got.get("answer") == "深蹲膝盖别内扣", got)
check("不存在的键返回 None", b.cache_get("nope") is None)
a.cache_put("k1", {"answer": "改了"})
check("同键覆盖", b.cache_get("k1")["answer"] == "改了")

print("\n【二、缓存上限（近似，不是硬约束）】")
# 淘汰每 20 次写才跑一次（长事务会加剧多进程冲突），所以条数会在
# 「上限」和「上限+19」之间浮动。缓存条数多十几条没有实际危害，
# 断言就按这个语义写 —— 否则测试会去要求一个实现故意不提供的保证。
for i in range(60):
    a.cache_put(f"x{i}", {"n": i})
check("条数收敛到上限附近", a.cache_size() <= 3 + 20, a.cache_size())
check("最新的还在", a.cache_get("x59") is not None)

print("\n【三、限流：滑动窗口】")
fresh()
s = shared_state.SqliteStore(TMP)
waits = [s.rate_wait("1.1.1.1", limit=3) for _ in range(3)]
check("前 3 次放行", all(w == 0 for w in waits), waits)
w4 = s.rate_wait("1.1.1.1", limit=3)
check("第 4 次被挡", w4 > 0, w4)
check("等待时间不超过窗口", 0 < w4 <= 60, w4)
check("别的 IP 不受影响", s.rate_wait("2.2.2.2", limit=3) == 0)

print("\n【四、并发占额度不超发】")
fresh()
s2 = shared_state.SqliteStore(TMP)
QUOTA = 5
got = []
lock = threading.Lock()


def take():
    ok = s2.daily_take(QUOTA)
    with lock:
        got.append(ok)


ths = [threading.Thread(target=take) for _ in range(24)]
for t in ths:
    t.start()
for t in ths:
    t.join()
passed = sum(1 for x in got if x)
check(f"24 个线程抢 {QUOTA} 个额度，放行了 {passed} 个", passed == QUOTA, passed)
check("没有超发", passed <= QUOTA, passed)

print("\n【五、多进程写同一个库】")
fresh()
s3 = shared_state.SqliteStore(TMP)
errors = []


def hammer(i):
    try:
        for j in range(30):
            s3.cache_put(f"p{i}-{j}", {"i": i, "j": j})
            s3.rate_wait(f"ip{i}", limit=1000)
            s3.daily_take(10000)
    except Exception as e:          # noqa: BLE001
        errors.append(f"{type(e).__name__}: {e}")


ths = [threading.Thread(target=hammer, args=(i,)) for i in range(8)]
t0 = time.perf_counter()
for t in ths:
    t.start()
for t in ths:
    t.join()
dt = time.perf_counter() - t0
check("8 线程 90 次混合写入不报错", not errors, errors[:2])
check("耗时合理（< 10 秒）", dt < 10, f"{dt:.2f}s")
print(f"       8 线程 × 30 轮 × 3 次写 = {8 * 30 * 3} 次写，用时 {dt * 1000:.0f}ms "
      f"({8 * 30 * 3 / dt:.0f} 写/秒)")

print("\n【六、坏了要放行，不能全挡】")
fresh()
bad = Path(tempfile.mkdtemp())     # 传一个目录当 db 文件，必炸
s4 = shared_state.SqliteStore.__new__(shared_state.SqliteStore)
s4.path = str(bad)
s4.cache_max = 10
s4.expire_s = 86400
s4._local = threading.local()
s4._rate_ops = 0
s4._put_ops = 0
check("限流炸了放行", s4.rate_wait("9.9.9.9", limit=1) == 0)
check("额度炸了放行", s4.daily_take(1) is True)
check("读缓存炸了返回 None", s4.cache_get("k") is None)
check("写缓存炸了不抛", s4.cache_put("k", {}) is None)

print()
if fails:
    print(f"{len(fails)} 项失败：{fails}")
    raise SystemExit(1)
print("全部通过")
