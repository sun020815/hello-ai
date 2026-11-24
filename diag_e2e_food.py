"""端到端：精排分数不等于用户拿到什么。

README 已经写过这个道理（「看检索质量用 hit@1，看用户拿到什么必须用端到端」）。
这次有一批题卡在 0.77~0.94，正适合验证兜底到底兜不兜得住：

  过线的（午餐 0.97）      直接答，看答得像不像
  踩线的（晚餐 0.93 早饭 0.94）走兜底，看模型判不判「资料够」
  低分的（中午 0.77 晚上 0.78）也要看，可能判资料不足
  误放的（蛋白粉 0.96）      正文没有「几勺」，看模型会不会硬编数字

最后一条最关键：正文一只字没提「勺」，如果模型答得出勺数，那就是在编。
"""
from fastapi.testclient import TestClient

import main

main.QUERY_LOG_ON = False
c = TestClient(main.app)
H = {"x-access-code": main.ACCESS_CODE} if main.ACCESS_CODE else {}

QS = [
    ("午餐吃什么", "应命中 diet-food-05（中午一平碗米饭…）"),
    ("晚餐吃什么", "应命中 diet-food-05（晚上半碗米饭…）"),
    ("早饭吃什么", "应命中 diet-food-05（早上两个鸡蛋…）"),
    ("中午吃什么", "同上，口语版"),
    ("晚上吃什么", "同上，口语版"),
    ("蛋白粉一天喝几勺", "正文没有「勺」，答出具体勺数=编"),
    ("肌酸一天吃多少", "库外，应拒"),
    ("一周瘦几斤算正常", "应命中 lose-01"),
    ("坚果一天吃多少", "应命中 diet-food-04"),
]

for q, expect in QS:
    r = c.post("/rag", json={"question": q}, headers=H)
    d = r.json()
    srcs = d.get("sources") or []
    print("=" * 72)
    print(f"Q: {q}")
    print(f"   期望：{expect}")
    print(f"   来源：{[s.get('id') for s in srcs]}  兜底={d.get('fallback_used')}  "
          f"拒答={d.get('refused')}  top={srcs[0].get('score') if srcs else None}")
    print(f"   A: {(d.get('answer') or '').replace(chr(10), ' ')[:180]}")
