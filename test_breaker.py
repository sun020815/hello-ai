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


def fresh(threshold=5, cooldown=30.0, name="test"):
    clock = Clock()
    return CircuitBreaker(name, threshold=threshold, cooldown=cooldown, clock=clock), clock


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
    test_wiring()
    print(f"\n{'=' * 46}\n{passed}/{total} 通过\n{'=' * 46}")
    raise SystemExit(0 if passed == total else 1)
