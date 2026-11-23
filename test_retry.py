"""重试策略的单元测试：不联网，靠注入假异常验证「该重试的重试、不该重试的不重试」。

为什么要专门写这个文件：
重试逻辑最容易自我欺骗 —— 代码看着有 for 循环有 sleep，就以为在工作了。
但它真正的价值全在判断上：502 要重试，401 不能重试。
这两条错了，症状是「所有请求都慢三秒」或者「key 过期了还在傻等」，线上很难看出来。
所以这里把 _should_retry 的判断表和 call_with_retry 的调用次数都钉死。

跑：uv run python test_retry.py
"""

import json
import time
import urllib.error

import main


class FakeStatus(Exception):
    """模拟 SDK 抛的 HTTP 状态错误。真实的 openai.APIStatusError 也带 status_code 字段，
    所以这里不需要真的构造一个 HTTP 响应。"""

    def __init__(self, code, msg="boom"):
        super().__init__(msg)
        self.status_code = code


class Timer:
    """替掉 main.time 里的 sleep，既记录退避时间，又不真的等。"""

    def __init__(self):
        self.sleeps = []

    def sleep(self, seconds):
        self.sleeps.append(round(seconds, 3))


def run_case(name, fn, expect_calls, expect_ok, patch_time=True):
    """跑一个用例，返回 (调用次数, 退避序列, 是否成功)。"""
    timer = Timer()
    real_time = main.time
    if patch_time:
        main.time = timer
    calls = {"n": 0}

    def wrapped():
        calls["n"] += 1
        return fn(calls["n"])

    ok = True
    try:
        main.call_with_retry(wrapped, label=name)
    except Exception:
        ok = False
    finally:
        main.time = real_time

    status = "PASS" if (calls["n"] == expect_calls and ok == expect_ok) else "FAIL"
    print(f"  [{status}] {name}")
    print(f"         调用 {calls['n']} 次 / 预期 {expect_calls} 次，"
          f"{'成功' if ok else '失败'} / 预期{'成功' if expect_ok else '失败'}")
    print(f"         退避序列 {timer.sleeps}")
    return status == "PASS"


def main_test():
    # 抖动归零，否则退避时间每次都不一样，没法断言
    main.RETRY_JITTER = 0
    passed = 0
    total = 0

    print("\n=== 1. _should_retry 判断表 ===")
    cases = [
        ("502 上游挂了", FakeStatus(502), True),
        ("503 服务不可用", FakeStatus(503), True),
        ("429 限流", FakeStatus(429), True),
        ("408 超时", FakeStatus(408), True),
        ("401 key 错", FakeStatus(401), False),
        ("403 没权限", FakeStatus(403), False),
        ("400 参数写错", FakeStatus(400), False),
        ("404 模型名错", FakeStatus(404), False),
        ("连接被重置", urllib.error.URLError("connection reset"), True),
        ("DNS 抽风", urllib.error.URLError("name resolution"), True),
        ("JSON 被截断（我们主动转成 URLError）", urllib.error.URLError("not json"), True),
        ("不合法的 key 格式", ValueError("bad key"), False),
    ]
    for name, exc, want in cases:
        got = main._should_retry(exc)
        status = "PASS" if got == want else "FAIL"
        print(f"  [{status}] {name:32s} 重试={got} 预期={want}")
        total += 1
        passed += status == "PASS"

    print("\n=== 2. call_with_retry 的调用次数 ===")
    total += 1
    passed += run_case(
        "502 两次后成功",
        lambda n: (_ for _ in ()).throw(FakeStatus(502)) if n < 3 else "ok",
        expect_calls=3, expect_ok=True,
    )
    total += 1
    passed += run_case(
        "502 三次全挂 -> 放弃",
        lambda n: (_ for _ in ()).throw(FakeStatus(502)),
        expect_calls=3, expect_ok=False,
    )
    total += 1
    passed += run_case(
        "401 第一次就该放弃",
        lambda n: (_ for _ in ()).throw(FakeStatus(401)),
        expect_calls=1, expect_ok=False,
    )
    total += 1
    passed += run_case(
        "一次就成功（不该有任何退避）",
        lambda n: "ok",
        expect_calls=1, expect_ok=True,
    )

    print("\n=== 3. HTTPException 不该被重试 ===")
    from fastapi import HTTPException

    total += 1
    passed += run_case(
        "自己抛的 500（key 没填）",
        lambda n: (_ for _ in ()).throw(HTTPException(status_code=500, detail="key 没填")),
        expect_calls=1, expect_ok=False,
    )

    print("\n=== 4. 退避时间是否按指数增长 ===")
    print("  期望：第 1 次重试等 1s，第 2 次等 2s（抖动已置 0）")
    timer = Timer()
    real_time = main.time
    main.time = timer
    try:
        main.call_with_retry(
            lambda: (_ for _ in ()).throw(FakeStatus(502)), label="backoff"
        )
    except Exception:
        pass
    finally:
        main.time = real_time
    ok = timer.sleeps == [1.0, 2.0]
    print(f"  [{'PASS' if ok else 'FAIL'}] 实际退避 {timer.sleeps}")
    total += 1
    passed += ok

    print(f"\n{'=' * 46}\n{passed}/{total} 通过\n{'=' * 46}")
    return passed == total


def live_dead_port_test():
    """把 base_url 指向一个没有服务监听的端口，看真实链路是不是真的重试了。

    单元测试只能证明「策略对」，证明不了「接线对」—— 忘了在 embed 里套 call_with_retry
    这种事，注入假异常是看不出来的。所以要真发一次请求。
    不发到外网、不花钱：9 号端口在本机注定连不上，报错是即时的。"""
    print("\n=== 5. 真实链路：连一个死端口 ===")
    import time as _time

    from fastapi import HTTPException

    saved = (main.LLM_BASE_URL, main.EMBED_TIMEOUT)
    main._clients.clear()
    main.LLM_BASE_URL = "http://127.0.0.1:9/v1"
    main.EMBED_TIMEOUT = 5
    t0 = _time.time()
    ok = False
    try:
        main.embed(["测试"])
    except HTTPException as e:
        ok = e.status_code == 502 and "已试 3 次" in str(e.detail)
        print(f"  状态码 {e.status_code}，detail 含「已试 3 次」：{ok}")
        print(f"  总耗时 {_time.time() - t0:.1f}s（其中退避约 3s）")
    except Exception as e:
        print(f"  [FAIL] 抛了意外的异常类型 {type(e).__name__}: {e}")
    finally:
        main.LLM_BASE_URL, main.EMBED_TIMEOUT = saved
        main._clients.clear()

    print(f"  [{'PASS' if ok else 'FAIL'}] 死端口 -> 重试 3 次 -> 502")
    return ok


def test_error_translation():
    """外部异常 -> HTTP 状态码的翻译。

    这一段是踩出来的：评测跑到一半崩了，报的是「502 调用 embedding 失败（已试 3 次）」，
    真实原因是上游 402 余额不足。502 会让人去查网络，402 才指向「去充值」——
    翻译错了，排障方向就全错了。顺带把「已试 3 次」改成真实次数：
    401 这种只试 1 次的错，报成 3 次同样是误导。"""
    print("\n=== 6. 错误翻译：别把计费错误说成上游挂了 ===")
    cases = [
        (FakeStatus(402), "余额不足", 500),
        (FakeStatus(401), "密钥无效", 500),
        (FakeStatus(404), "模型名不存在", 500),
        (FakeStatus(502), "试了 3 次", 502),
    ]
    ok_all = True
    for exc, want_text, want_code in cases:
        stat = {"attempts": 1 if want_code == 500 else 3}
        e = main.external_error(exc, "调用模型", stat)
        ok = e.status_code == want_code and want_text in str(e.detail)
        ok_all &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {exc.status_code} -> {e.status_code}：{e.detail[:40]}")
    return ok_all


if __name__ == "__main__":
    import sys

    ok = main_test()
    ok = test_error_translation() and ok

    # 默认只跑离线部分（2 秒）；加 --live 才发真实请求
    if "--live" in sys.argv:
        ok = live_dead_port_test() and ok
    else:
        print("\n（跳过真实链路测试，加 --live 运行）")

    raise SystemExit(0 if ok else 1)
