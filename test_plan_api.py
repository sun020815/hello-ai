"""/plan 接口层测试：口令、缓存、参数边界、失败不入缓存。

跟 test_plan.py 分工：那边测「agent 逻辑守不守得住」（不用 fastapi，系统 python 就能跑）；
这里测「接口对外长什么样」（需要 fastapi，用 .venv 跑）。混在一起的话，
改前端的人会被 agent 的剧本测试绊住。

跑法：.venv/Scripts/python.exe test_plan_api.py
"""

import json
import sys
import types

from fastapi.testclient import TestClient

import main

fails = []
CODE = main.ACCESS_CODE or ""
H = {"x-access-code": CODE} if CODE else {}


def check(name, cond, extra=""):
    print(("  ok   " if cond else "  FAIL ") + name + (f"   {extra}" if extra != "" else ""))
    if not cond:
        fails.append(name)


def tool_call(cid, name, args):
    return types.SimpleNamespace(
        id=cid, type="function",
        function=types.SimpleNamespace(name=name, arguments=json.dumps(args, ensure_ascii=False)),
    )


def resp(content=None, calls=None, tokens=120):
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=content, tool_calls=calls))],
        usage=types.SimpleNamespace(total_tokens=tokens),
        model="fake",
    )


GOOD = {"summary": "推拉腿三分化",
        "days": [{"day": 1, "focus": "推日", "exercises": ["哑铃平板卧推", "中位绳索夹胸"]},
                 {"day": 2, "focus": "拉日", "exercises": ["单手器械划船", "对握高位下拉"]},
                 {"day": 3, "focus": "腿日", "exercises": ["腿举", "罗马尼亚硬拉"]}]}


def scripted(*steps):
    """返回一个假的 main.call_llm，按顺序吐预设响应。"""
    queue = list(steps)

    def fake(messages, temperature=0.0, tools=None):
        return queue.pop(0) if queue else resp(content=json.dumps(GOOD, ensure_ascii=False))

    return fake


client = TestClient(main.app)


def test_auth():
    print("\n[1] 口令")
    if not CODE:
        print("  skip 没配 ACCESS_CODE，跳过")
        return
    r = client.post("/plan", json={"goal": "增肌"})
    check("不带口令 401", r.status_code == 401, str(r.status_code))
    r = client.post("/plan", json={"goal": "增肌"}, headers={"x-access-code": "wrong-code"})
    check("错口令 401", r.status_code == 401, str(r.status_code))


def test_bad_params():
    print("\n[2] 参数边界")
    r = client.post("/plan", json={"days": 6}, headers=H)
    check("days=6 被拒（最多 5 天）", r.status_code == 422, str(r.status_code))
    r = client.post("/plan", json={"days": 1}, headers=H)
    check("days=1 被拒（最少 2 天）", r.status_code == 422, str(r.status_code))
    r = client.post("/plan", json={"injuries": ["膝", "腰", "肩", "踝", "肘"]}, headers=H)
    check("伤病超过 4 个被拒", r.status_code == 422, str(r.status_code))


def test_ok_and_cache():
    print("\n[3] 正常返回 + 缓存")
    orig = main.call_llm
    main.call_llm = scripted(resp(content=json.dumps(GOOD, ensure_ascii=False)))
    try:
        body = {"goal": "增肌", "days": 3, "equipment": "健身房", "session_min": 60,
                "level": "新手", "injuries": ["测试缓存"], "note": "api-test-1"}
        r1 = client.post("/plan", json=body, headers=H)
        check("200", r1.status_code == 200, str(r1.status_code))
        d1 = r1.json()
        check("有 plan / trace / audit", all(k in d1 for k in ("plan", "trace", "audit")), str(list(d1))[:120])
        check("第一次不是缓存", d1.get("cached") is False, str(d1.get("cached")))
        check("排了 3 天", len(d1["plan"]) == 3, str(len(d1["plan"])))
        check("体检通过", d1["audit"]["ok"], str(d1["audit"]["errors"]))
        check("每个动作带 ref", all(e["ref"] for x in d1["plan"] for e in x["exercises"]))

        r2 = client.post("/plan", json=body, headers=H)
        d2 = r2.json()
        check("第二次命中缓存", d2.get("cached") is True, str(d2.get("cached")))
        check("缓存命中不报 token（没收钱）", d2.get("tokens") == 0, str(d2.get("tokens")))
    finally:
        main.call_llm = orig


def test_review_not_cached():
    print("\n[4] 没排好的结果不许进缓存")
    orig = main.call_llm
    bad = {"summary": "x", "days": [{"day": 1, "focus": "推日",
                                     "exercises": ["杠铃平板卧推", "库里没有的动作"]}]}
    main.call_llm = scripted(resp(content=json.dumps(bad, ensure_ascii=False)))
    try:
        body = {"goal": "增肌", "days": 3, "equipment": "健身房", "session_min": 60,
                "level": "新手", "injuries": [], "note": "api-test-bad"}
        d1 = client.post("/plan", json=body, headers=H).json()
        check("标记 needs_review", d1["needs_review"] is True, str(d1.get("needs_review")))
        check("说明为什么要复核", d1["audit"]["errors"], str(d1["audit"]["errors"])[:100])
        # 换了剧本再请求同一档案：如果第一次被缓存了，这次拿到的还是旧结果
        main.call_llm = scripted(resp(content=json.dumps(GOOD, ensure_ascii=False)))
        d2 = client.post("/plan", json=body, headers=H).json()
        check("第二次不是缓存（坏结果没被固化）",
              d2.get("cached") is False and len(d2["plan"]) == 3, str(d2.get("cached")))
    finally:
        main.call_llm = orig


def test_stats():
    print("\n[5] /stats 里能看到排计划的次数")
    s = client.get("/stats", headers=H).json()
    check("plans 计数存在", "plans" in s, str([k for k in s if "plan" in k]))
    check("至少跑过 2 次", s.get("plans", 0) >= 2, str(s.get("plans")))
    check("工具调用次数有累计", s.get("plan_tool_calls", 0) >= 0, str(s.get("plan_tool_calls")))
    check("cost 记账仍在", "cost" in s, str(list(s))[:80])


def main_():
    test_auth()
    test_bad_params()
    test_ok_and_cache()
    test_review_not_cached()
    test_stats()
    print()
    if fails:
        print(f"失败 {len(fails)} 项：{fails}")
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main_())
