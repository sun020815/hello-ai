"""熔断器的状态机测试：全部离线，用假时钟，不用真等 30 秒。

为什么要专门测这个：
熔断器的 bug 基本都是「看着对、跑起来不对」——状态全都写对了，
但少了个 lock，或者探针忘了放掉，线上表现是「熔断之后再也没恢复」，
而这种故障不会报错，只会一直 503，查起来极慢。
所以这里把三个状态的每一条转移都钉死，包括最容易漏的两条：
  · 探针在飞的时候，第二个请求必须被挡住（否则「只放一个」形同虚设）
  · 401 这种「上游活着、我们错了」的错误不能计入熔断

跑：uv run python test_breaker.py
"""

import main
from main import CircuitBreaker, CircuitOpenError


class Clock:
    """假时钟。真等 30 秒的话这一套测试要跑好几分钟。"""

    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


class FakeStatus(Exception):
    def __init__(self, code):
        super().__init__(f"HTTP {code}")
        self.status_code = code


passed = 0
total = 0


def check(name, got, want):
    global passed, total
    total += 1
    ok = got == want
    passed += ok
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"         实际 {got} / 预期 {want}")
    return ok


def fresh(consecutive=5, cooldown=30.0, name="test", **kw):
    clock = Clock()
    return CircuitBreaker(name, consecutive=consecutive, cooldown=cooldown, clock=clock, **kw), clock


def test_state_machine():
    print("\n=== 1. 熔断阈值：连续 5 次才开 ===")
    b, _ = fresh()
    for i in range(4):
        b.on_failure()
    check("4 次失败后仍是 closed", b.state, "closed")
    b.on_failure()
    check("第 5 次失败后变 open", b.state, "open")
    check("熔断次数记了 1 次", b.trips, 1)

    print("\n=== 2. 中间成功一次就要清零 ===")
    b, _ = fresh()
    b.on_failure(); b.on_failure(); b.on_failure(); b.on_failure()
    check("4 次失败，计数 4", b.failures, 4)
    b.on_success()
    check("成功一次后计数归零", b.failures, 0)
    b.on_failure()
    check("再失败 1 次不该熔断（说明是连续计数）", b.state, "closed")

    print("\n=== 3. 4xx 不算上游故障 ===")
    b, _ = fresh()
    for _ in range(10):
        b.on_client_error()
    check("10 次 401 之后仍然 closed", b.state, "closed")
    check("失败计数没有被污染", b.failures, 0)

    print("\n=== 4. 熔断期间直接拒绝，不发请求 ===")
    b, clock = fresh(cooldown=30.0)
    for _ in range(5):
        b.on_failure()
    raised = None
    try:
        b.before_call()
    except CircuitOpenError as e:
        raised = str(e)
    check("熔断中抛 CircuitOpenError", raised is not None, True)
    print(f"         提示语：{raised}")

    print("\n=== 5. 冷却到期：只放一个探针 ===")
    b, clock = fresh(cooldown=30.0)
    for _ in range(5):
        b.on_failure()
    clock.advance(29)
    blocked = False
    try:
        b.before_call()
    except CircuitOpenError:
        blocked = True
    check("29 秒时仍然拒绝（差 1 秒也不行）", blocked, True)

    clock.advance(1.5)          # 现在 30.5 秒，冷却已过
    b.before_call()
    check("冷却后状态转 half_open", b.state, "half_open")
    second = False
    try:
        b.before_call()
    except CircuitOpenError:
        second = True
    check("探针在飞时，第二个请求被挡住", second, True)

    print("\n=== 6. 探针成功 -> 恢复 ===")
    b.on_success()
    check("状态回 closed", b.state, "closed")
    check("计数归零", b.failures, 0)
    b.before_call()             # 不该再抛
    check("恢复后正常放行", True, True)

    print("\n=== 7. 探针失败 -> 重新熔断，冷却重算 ===")
    b, clock = fresh(cooldown=30.0)
    for _ in range(5):
        b.on_failure()
    clock.advance(30.5)
    b.before_call()             # 探针出发
    check("探针阶段状态是 half_open", b.state, "half_open")
    b.on_failure()              # 探针也挂了
    check("探针失败后立刻回到 open", b.state, "open")
    check("熔断次数累加到 2", b.trips, 2)
    clock.advance(10)
    still_blocked = False
    try:
        b.before_call()
    except CircuitOpenError:
        still_blocked = True
    check("冷却从探针失败那刻重新计时（10 秒后仍拒绝）", still_blocked, True)

    print("\n=== 8. 每个上游一个熔断器，互不牵连 ===")
    main._breakers.clear()
    emb = main.get_breaker("embedding")
    chat = main.get_breaker("chat")
    for _ in range(5):
        emb.on_failure()
    check("embedding 已熔断", emb.state, "open")
    check("chat 不受影响", chat.state, "closed")
    main._breakers.clear()


def test_window_and_cooldown():
    """第十、十一组：两条新规则各自的存在理由，以及它们一起工作时的边界。"""
    from main import cooldown_for

    print("\n=== 10. 滑动窗口：间歇性失败也能熔断 ===")
    # 这是加窗口的**唯一理由**。失败-成功交替，连续失败次数永远是 1，
    # 老版本（纯连续计数）在这里永远不熔断，而上游其实已经在一半请求上失败了。
    b, _ = fresh()
    for i in range(1, 11):
        b.on_failure(FakeStatus(502))
        if b.state == "open":
            break
        b.on_success()
    check("交替失败的场景最终会熔断", b.state, "open")
    check("不是靠连续计数（它只有 1）", b.failures <= 1, True)
    check("熔断原因写明是窗口错误率", "窗口" in b.last_reason, True)
    print(f"         原因：{b.last_reason}")

    print("\n=== 11. 小样本不误判：1 次失败不熔断 ===")
    # 错误率 100%，但样本只有 1 个。不设 min_samples 的话，
    # 服务启动后第一个请求抖一下就直接熔断 30 秒。
    b, _ = fresh()
    b.on_failure(FakeStatus(502))
    check("1 次失败（错误率 100%）仍然 closed", b.state, "closed")

    print("\n=== 12. 窗口会滑动，旧样本过期 ===")
    b, clock = fresh(consecutive=1000)      # 关掉规则 B，只观察规则 A
    for _ in range(9):
        b.on_failure(FakeStatus(502))
    check("9 次失败：样本不足 10，不熔断", b.state, "closed")
    clock.advance(31)                        # 窗口滑过去
    b.on_success()
    total, fails = b._stats(clock())
    check("窗口过期后旧失败被清空", fails, 0)

    print("\n=== 13. 冷却按错误类型区分 ===")
    for exc, want, label in [
        (FakeStatus(429), 5.0, "429 限流"),
        (FakeStatus(408), 10.0, "408 超时"),
        (FakeStatus(503), 20.0, "503 过载"),
        (FakeStatus(502), 30.0, "502 网关错"),
        (ConnectionError("refused"), 15.0, "连接被拒"),
        (TimeoutError(), 10.0, "socket 超时"),
        (None, 30.0, "未知"),
    ]:
        check(f"{label} → {want:.0f}s", cooldown_for(exc), want)

    print("\n=== 14. 冷却真的生效（不只是算出来） ===")
    b, clock = fresh(cooldown=30.0)
    for _ in range(5):
        b.on_failure(FakeStatus(429))
    check("429 熔断后冷却被改成 5 秒", b.cooldown, 5.0)
    clock.advance(5.5)
    b.before_call()
    check("5.5 秒后就放探针了（不是等 30 秒）", b.state, "half_open")

    b2, clock2 = fresh(cooldown=30.0)
    for _ in range(5):
        b2.on_failure(FakeStatus(502))
    clock2.advance(5.5)
    blocked = False
    try:
        b2.before_call()
    except CircuitOpenError:
        blocked = True
    check("同样 5.5 秒，502 仍然拒绝（冷却 30 秒）", blocked, True)

    print("\n=== 15. 恢复时清空窗口，否则刚救活又被打死 ===")
    b, clock = fresh()
    for _ in range(5):
        b.on_failure(FakeStatus(502))
    check("已熔断", b.state, "open")
    clock.advance(30.5)
    b.before_call()                          # 放探针
    b.on_success()                           # 探针成功
    check("恢复后窗口清空", len(b.events), 0)
    # 不清空的话：5 条历史失败 + 这 1 条 = 100%，立刻二次熔断，永远恢复不了
    b.on_failure(FakeStatus(502))
    check("恢复后 1 次失败不该立刻再熔断", b.state, "closed")

    print("\n=== 16. 4xx 算健康样本，不该把自己（上游）熔断 ===")
    b, clock = fresh(consecutive=1000)
    for _ in range(9):
        b.on_failure(FakeStatus(502))
    for _ in range(20):
        b.on_client_error()                  # 上游一直在正常回 400，是我们请求写错了
    b.on_failure(FakeStatus(502))
    total, fails = b._stats(clock())
    check("20 次 4xx 把错误率稀释到 50% 以下", fails / total < 0.5, True)
    check("因此没有熔断", b.state, "closed")
    print(f"         窗口 {total} 条，失败 {fails} 条，错误率 {fails/total:.0%}")


def test_cooldown_wiring():
    """冷却时长是在 _trip 里算的，要证明真实调用链能把它带出来。"""
    print("\n=== 17. 接线：429 打到真实 embed 链路上，熔断后的冷却应该是 5 秒 ===")

    class FakeEmbeddings:
        def create(self, **kwargs):
            raise FakeStatus(429)

    class FakeClient:
        def __init__(self):
            self.embeddings = FakeEmbeddings()

    real_get_client, real_time, real_jitter = main.get_client, main.time, main.RETRY_JITTER
    main.get_client = lambda timeout: FakeClient()
    main.time = type("T", (), {"sleep": lambda self, s: None, "monotonic": real_time.monotonic})()
    main.RETRY_JITTER = 0
    main._breakers.clear()

    from fastapi import HTTPException

    try:
        for _ in range(6):
            try:
                main.embed(["测试"])
            except HTTPException as e:
                if e.status_code == 503:
                    break
        br = main.get_breaker("embedding")
        check("429 触发熔断后，冷却记录为 5 秒", br.cooldown, 5.0)
        check("熔断原因说明是窗口或连续失败", br.last_reason != "", True)
        snap = br.snapshot()
        check("快照能看出窗口样本数", snap["window_samples"] > 0, True)
        print(f"         快照：{snap}")
    finally:
        main.get_client, main.time, main.RETRY_JITTER = real_get_client, real_time, real_jitter
        main._breakers.clear()


def test_wiring():
    """离线测试只能证明状态机对，证明不了「embed 真的接上了熔断器」。
    这里把 get_client 换成一个永远 502 的假 client，走真实的 embed 调用链。"""
    print("\n=== 9. 接线：embed 连续失败会触发熔断并改报 503 ===")

    class FakeEmbeddings:
        def __init__(self):
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            raise FakeStatus(502)

    class FakeClient:
        def __init__(self):
            self.embeddings = FakeEmbeddings()

    fake = FakeClient()
    real_get_client, real_time, real_jitter = main.get_client, main.time, main.RETRY_JITTER
    main.get_client = lambda timeout: fake
    main.time = type("T", (), {"sleep": lambda self, s: None, "monotonic": real_time.monotonic})()
    main.RETRY_JITTER = 0
    main._breakers.clear()

    from fastapi import HTTPException

    codes = []
    try:
        for _ in range(6):
            try:
                main.embed(["测试"])
            except HTTPException as e:
                codes.append(e.status_code)
        check("前 5 次报 502（试过了确实失败）", codes[:5], [502] * 5)
        check("第 6 次报 503（熔断，没试）", codes[5], 503)
        check("一共只发出去 15 次请求（5 次 × 3 attempts）", fake.embeddings.calls, 15)
        check("熔断后一个网络包都没发", fake.embeddings.calls, 15)
    finally:
        main.get_client, main.time, main.RETRY_JITTER = real_get_client, real_time, real_jitter
        main._breakers.clear()


if __name__ == "__main__":
    test_state_machine()
    test_window_and_cooldown()
    test_wiring()
    test_cooldown_wiring()
    print(f"\n{'=' * 46}\n{passed}/{total} 通过\n{'=' * 46}")
    raise SystemExit(0 if passed == total else 1)
