"""/plan 的离线测试：不联网、不花钱，用一个「按剧本说话的假模型」。

为什么必须用假模型
------------------
这条链路真正要证明的不是「模型排得好不好看」，是**它排错了会不会被发现**：
编一个库里没有的动作、给膝盖疼的人排保加利亚蹲、组次数字自己拍。
这些事用真模型测是碰运气 —— 它这次没编，不代表代码能抓到编的时候。

所以这里让假模型**故意犯错**，看代码层接不接得住。
真模型的实跑结果另外记在 docs/plan_agent.md 里，两者测的东西不重叠：
假的测「守不守得住」，真的测「好不好用」。

跑法：uv run python test_plan.py
"""

import json
import sys
import types

import plan_agent as pa

fails = []


def check(name, cond, extra=""):
    print(("  ok   " if cond else "  FAIL ") + name + (f"   {extra}" if extra != "" else ""))
    if not cond:
        fails.append(name)


# ---------------------------------------------------------------- 假模型

def tool_call(cid, name, args):
    return types.SimpleNamespace(
        id=cid, type="function",
        function=types.SimpleNamespace(name=name, arguments=json.dumps(args, ensure_ascii=False)),
    )


def resp(content=None, calls=None, tokens=100):
    msg = types.SimpleNamespace(content=content, tool_calls=calls)
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=msg)],
        usage=types.SimpleNamespace(total_tokens=tokens),
        model="fake",
    )


class Scripted:
    """按顺序吐出预设响应。剧本用完之后一律返回「不调工具了」，避免测试挂死。"""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def __call__(self, messages, tools=None):
        self.calls.append({"messages": list(messages), "tools": tools})
        if not self.script:
            return resp(content="够了。")
        return self.script.pop(0)


LIB = pa.Library.load()
PROFILE = {"goal": "增肌", "days": 3, "session_min": 60, "equipment": "健身房",
           "level": "新手", "injuries": []}


# ---------------------------------------------------------------- 1. 正常链路

def test_happy_path():
    print("\n[1] 正常三步：查胸 → 查腿 → 自检 → 出计划")
    good = {"summary": "推拉腿三分化，复合在前孤立在后",
            "days": [
                {"day": 1, "focus": "推日", "exercises": ["杠铃平板卧推", "哑铃上斜卧推", "中位绳索夹胸"]},
                {"day": 2, "focus": "拉日", "exercises": ["单手钢线下拉", "对握高位下拉", "蝴蝶机反向飞鸟"]},
                {"day": 3, "focus": "腿日", "exercises": ["腿举", "罗马尼亚硬拉", "坐姿腿屈伸"]},
            ]}
    llm = Scripted([
        resp(calls=[tool_call("c1", "search_exercises", {"part": "胸"})]),
        resp(calls=[tool_call("c2", "search_exercises", {"part": "腿", "level_max": 2})]),
        resp(calls=[tool_call("c3", "check_plan", {"days": good["days"]})]),
        resp(content="```json\n" + json.dumps(good, ensure_ascii=False) + "\n```"),
    ])
    out = pa.plan_week(PROFILE, llm, lib=LIB)
    check("三步工具都记进 trace", [t["tool"] for t in out["trace"]] ==
          ["search_exercises", "search_exercises", "check_plan"], str([t["tool"] for t in out["trace"]]))
    check("计划 3 天", len(out["plan"]) == 3, str(len(out["plan"])))
    check("体检通过", out["audit"]["ok"], str(out["audit"]["errors"]))
    check("不需要复核", out["needs_review"] is False, str(out["needs_review"]))
    check("组次来自动作库而不是模型",
          out["plan"][0]["exercises"][0]["reps"] == "15 / 12 / 10 / 8 递减加重",
          out["plan"][0]["exercises"][0]["reps"])
    check("每个动作带得回正文的 ref",
          all(e["ref"] for d in out["plan"] for e in d["exercises"]))
    check("token 累计了 4 次调用", out["tokens"] == 400, str(out["tokens"]))


# ---------------------------------------------------------------- 2. 编造动作

def test_fake_exercise():
    print("\n[2] 模型编了一个库里没有的动作")
    bad = {"summary": "随便排的",
           "days": [{"day": 1, "focus": "推日",
                     "exercises": ["杠铃平板卧推", "杠铃俯身飞鸟", "上斜哑铃卧推"]}]}
    llm = Scripted([resp(content=json.dumps(bad, ensure_ascii=False))])
    out = pa.plan_week({**PROFILE, "days": 2}, llm, lib=LIB)
    names = [e["name"] for d in out["plan"] for e in d["exercises"]]
    check("编的动作被剔除", "杠铃俯身飞鸟" not in names, str(names))
    check("剔除了 1 个并留痕", len(out["dropped"]) == 1 and out["dropped"][0]["name"] == "杠铃俯身飞鸟",
          str(out["dropped"]))
    check("体检标红（unknown_exercise）",
          any(e["type"] == "unknown_exercise" for e in out["audit"]["errors"]),
          str(out["audit"]["errors"]))
    check("要求 2 天只排了 1 天，也被抓到",
          any(e["type"] == "day_mismatch" for e in out["audit"]["errors"]),
          str(out["audit"]["errors"]))
    check("要求人工复核", out["needs_review"] is True)


# ---------------------------------------------------------------- 3. 伤病冲突

def test_injury_conflict():
    print("\n[3] 膝盖疼，还排了保加利亚蹲")
    bad = {"summary": "腿日",
           "days": [{"day": 1, "focus": "腿日", "exercises": ["颈前深蹲", "保加利亚分腿蹲", "腿举"]},
                    {"day": 2, "focus": "推日", "exercises": ["杠铃平板卧推", "中位绳索夹胸"]}]}
    llm = Scripted([resp(content=json.dumps(bad, ensure_ascii=False))])
    out = pa.plan_week({**PROFILE, "days": 2, "injuries": ["膝盖"]}, llm, lib=LIB)
    errs = [e for e in out["audit"]["errors"] if e["type"] == "injury_conflict"]
    got = {e["name"] for e in errs}
    check("颈前深蹲和保加利亚都被抓到", {"颈前深蹲", "保加利亚分腿蹲"} <= got, str(got))
    check("给出了替代动作", any(e.get("substitutes") for e in errs), str(errs[:1]))
    check("腿举没被误伤", "腿举" not in got, str(got))
    check("要求人工复核", out["needs_review"] is True)


def test_search_respects_injury():
    print("\n[4] 查腿时，膝盖有伤的动作不该出现在结果里")
    r = pa.call_tool("search_exercises", {"part": "腿"}, LIB, ["膝盖"])
    names = [i["name"] for i in r["items"]]
    check("保加利亚不在结果里", "保加利亚分腿蹲" not in names, str(names))
    check("颈前深蹲不在结果里", "颈前深蹲" not in names, str(names))
    check("腿举还在", "腿举" in names, str(names))


# ---------------------------------------------------------------- 5. 工具容错

def test_tool_errors():
    print("\n[5] 工具自己的容错：不抛异常，给模型能看懂的错误")
    r = pa.call_tool("search_exercises", {"part": "腹肌"}, LIB, [])
    check("部位写错返回合法值列表", r.get("error") and "胸" in r.get("valid_parts", []), str(r)[:120])

    # 第一次实跑就踩了这个：「健身房」被当成器械名去库里查，一条都没有，
    # 模型只能瞎猜着再查一次。场地词要当「不限」处理。
    r = pa.call_tool("search_exercises", {"part": "胸", "equipment": "健身房"}, LIB, [])
    check("「健身房」当不限，不返回空", r.get("count", 0) > 0, str(r)[:100])

    r = pa.call_tool("search_exercises", {"part": "胸", "equipment": "龙门架"}, LIB, [])
    check("真不存在的器械要说清合法值",
          r.get("error") and r.get("valid_equipment"), str(r)[:120])

    r = pa.call_tool("get_exercise", {"name": "杠铃俯身飞鸟"}, LIB, [])
    check("查不到给候选", r.get("error") and len(r.get("suggest") or []) > 0, str(r)[:120])

    r = pa.call_tool("get_exercise", {"name": "卧推"}, LIB, [])
    check("别名能查到（卧推 → 杠铃平板卧推）", r.get("name") == "杠铃平板卧推", str(r.get("name")))

    r = pa.call_tool("没这个工具", {}, LIB, [])
    check("未知工具不抛异常", bool(r.get("error")))


def test_time_budget():
    print("\n[5b] 时长估算：5 个动作说 60 分钟做完")
    day = {"day": 1, "focus": "推日", "exercises":
           ["杠铃平板卧推", "哑铃上斜卧推", "双杠臂屈伸", "中位绳索夹胸", "仰卧臂屈伸"]}
    # 同一份计划，给 90 分钟装得下，给 30 分钟装不下 —— 判断要跟着预算走。
    tight = pa.audit([day], LIB, [], want_days=1, session_min=30)
    roomy = pa.audit([day], LIB, [], want_days=1, session_min=90)
    check("5 个动作估在 50~65 分钟（含热身和换器械）",
          50 <= tight["est_minutes"][1] <= 65, str(tight["est_minutes"]))
    check("只给 30 分钟会警告",
          any(w["type"] == "over_time" for w in tight["warnings"]), str(tight["warnings"])[:120])
    check("给 90 分钟不报警",
          not any(w["type"] == "over_time" for w in roomy["warnings"]), str(roomy["est_minutes"]))


# ---------------------------------------------------------------- 6. 有界循环

def test_max_steps():
    print("\n[6] 模型一直查个不停，步数用完必须收尾")
    forever = [resp(calls=[tool_call(f"c{i}", "search_exercises", {"part": "胸"})]) for i in range(10)]
    finish = resp(content=json.dumps({"summary": "凑合", "days": [
        {"day": 1, "focus": "推日", "exercises": ["杠铃平板卧推"]},
        {"day": 2, "focus": "拉日", "exercises": ["单手钢线下拉"]},
    ]}, ensure_ascii=False))
    llm = Scripted(forever[:pa.MAX_STEPS] + [finish])
    out = pa.plan_week(PROFILE, llm, lib=LIB)
    check("停在 MAX_STEPS", out["steps"] == pa.MAX_STEPS + 1, str(out["steps"]))
    check("标记超步数", out["stopped"] == "max_steps", out["stopped"])
    check("收尾那次没再给工具",
          llm.calls[pa.MAX_STEPS]["tools"] is None, str(llm.calls[pa.MAX_STEPS]["tools"]))
    check("仍然产出了计划", len(out["plan"]) == 2)


def test_no_json():
    print("\n[7] 模型死活不给 JSON")
    llm = Scripted([resp(content="我觉得你应该多练胸。"), resp(content="真的，多练胸。")])
    out = pa.plan_week(PROFILE, llm, lib=LIB)
    check("不给 JSON 就明说", out["audit"]["errors"][0]["type"] == "no_json", str(out["audit"]["errors"]))
    check("plan 为空而不是瞎编", out["plan"] == [])
    check("要求人工复核", out["needs_review"] is True)


# ---------------------------------------------------------------- 8. 数字不可信

def test_numbers_from_library():
    print("\n[8] 模型在动作名后面塞了自己编的数字")
    data = {"summary": "x", "days": [
        {"day": 1, "focus": "推日", "exercises": ["杠铃平板卧推 4 组 6 次", "中位绳索夹胸"]},
        {"day": 2, "focus": "拉日", "exercises": ["单手钢线下拉"]},
    ]}
    llm = Scripted([resp(content=json.dumps(data, ensure_ascii=False))])
    out = pa.plan_week({**PROFILE, "days": 2}, llm, lib=LIB)
    e = out["plan"][0]["exercises"][0]
    check("名字带后缀也能认出来", e["name"] == "杠铃平板卧推", e["name"])
    check("次数用库里的 15/12/10/8，不是模型写的 6 次", "8" in e["reps"] and "6 次" not in e["reps"], e["reps"])


# ---------------------------------------------------------------- 9. 去重与连练

def test_audit_warnings():
    print("\n[9] 同一动作重复、大肌群连练")
    days = [{"day": 1, "focus": "胸", "exercises": ["杠铃平板卧推", "中位绳索夹胸"]},
            {"day": 2, "focus": "胸", "exercises": ["杠铃平板卧推", "哑铃上斜卧推"]}]
    rep = pa.audit(days, LIB, [])
    types_ = {w["type"] for w in rep["warnings"]}
    check("抓到动作重复", "duplicate" in types_, str(types_))
    check("抓到胸连练", "back_to_back" in types_, str(types_))
    check("没排腿的会提示覆盖缺口", "coverage" in types_, str(types_))


def test_fullbody_not_flagged():
    print("\n[10] 全身训练天天练到背腿，不该报「连着练」")
    days = [
        {"day": 1, "focus": "全身 A", "exercises": ["哑铃平板卧推", "对握高位下拉", "腿举", "哑铃侧平举", "平板支撑"]},
        {"day": 2, "focus": "全身 B", "exercises": ["罗马尼亚硬拉", "单手器械划船", "臀桥", "仰卧绳索面拉", "钢线弯举"]},
    ]
    rep = pa.audit(days, LIB, [])
    check("不报 back_to_back",
          not any(w["type"] == "back_to_back" for w in rep["warnings"]), str(rep["warnings"])[:150])


def test_day_renumbered():
    print("\n[11] 模型按星期编号（1 和 4），输出要重排成 1、2")
    data = {"summary": "x", "days": [
        {"day": 1, "focus": "推日", "exercises": ["哑铃平板卧推"]},
        {"day": 4, "focus": "拉日", "exercises": ["单手器械划船"]},
    ]}
    llm = Scripted([resp(content=json.dumps(data, ensure_ascii=False))])
    out = pa.plan_week({**PROFILE, "days": 2}, llm, lib=LIB)
    check("编号是 1、2 而不是 1、4", [d["day"] for d in out["plan"]] == [1, 2],
          str([d["day"] for d in out["plan"]]))


def test_dumbbell_keeps_bodyweight():
    print("\n[12] 只有哑铃时，自重动作不该被一起过滤掉")
    r = pa.call_tool("search_exercises", {"part": "核心", "equipment": "哑铃"}, LIB, [])
    names = [i["name"] for i in r.get("items", [])]
    check("核心还能查到平板支撑", "平板支撑" in names, str(names))
    r = pa.call_tool("search_exercises", {"part": "背", "equipment": "哑铃"}, LIB, [])
    check("背确实没有哑铃动作，要说清库里有什么",
          r.get("count") == 0 and r.get("in_library"), str(r)[:150])


def main():
    test_happy_path()
    test_fake_exercise()
    test_injury_conflict()
    test_search_respects_injury()
    test_tool_errors()
    test_time_budget()
    test_max_steps()
    test_no_json()
    test_numbers_from_library()
    test_audit_warnings()
    test_fullbody_not_flagged()
    test_day_renumbered()
    test_dumbbell_keeps_bodyweight()
    print()
    if fails:
        print(f"失败 {len(fails)} 项：{fails}")
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
