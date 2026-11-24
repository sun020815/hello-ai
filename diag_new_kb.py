"""补完减脂内容后的即时验证：那 5 条真实问法现在答不答得上，负例还拒不许。

不进测试套件，因为它要真调上游（embed + rerank + 生成），跑一次几毛钱。
放在这里是为了留下「补库前 / 补库后」的对照，面试被问起时能直接跑给人看。
"""
from fastapi.testclient import TestClient

import main

main.QUERY_LOG_ON = False
c = TestClient(main.app)

REAL = ["减脂期怎么安排有氧", "平时训练怎么做才能减脂呢", "我最近想减脂该怎么去吃",
        "肌酸一天吃多少", "你叫什么名字", "坐姿划船要不要夹肩胛骨"]
NEG = ["我膝盖前交叉韧带撕裂怎么康复", "深蹲要不要买腰带", "推荐个蛋白粉牌子"]


H = {"x-access-code": main.ACCESS_CODE} if main.ACCESS_CODE else {}


def ask(q):
    r = c.post("/rag", json={"question": q}, headers=H)
    if r.status_code != 200:
        print("  !! HTTP", r.status_code, r.text[:80])
        return False, {}
    d = r.json()
    return bool(d.get("sources")), d


print("== 应该答：真实问法日志里落空的 5 条 ==")
for q in REAL:
    ok, d = ask(q)
    topics = [s.get("topic") for s in (d.get("sources") or [])]
    tag = "答 " if ok else ("寒暄" if d.get("smalltalk") else "NO ")
    print(f"  {tag} {q}  -> {topics}")

print("\n== 应该拒：库里确实没有的 ==")
for q in NEG:
    ok, d = ask(q)
    print(f"  {'!! 答了' if ok else '拒了  '} {q}  -> {[s.get('topic') for s in (d.get('sources') or [])]}")
