"""演示页契约测试：不起服务、不联网、不花钱。

为什么需要这个：前端和后端是两个文件，后端把 `sources` 改名成 `hits`，
Python 不会报错，页面只是安静地显示「资料不足」。这类断裂靠人眼看不住，
必须把「页面依赖哪些 DOM id、调用哪些接口、读哪些字段」钉住。

跑法：uv run python test_web.py
"""
import re
import sys
from pathlib import Path

from fastapi.testclient import TestClient

import main as app

HERE = Path(__file__).parent
fails = []


def check(name, cond, extra=""):
    print(("  ok   " if cond else "  FAIL ") + name + (f"   {extra}" if extra else ""))
    if not cond:
        fails.append(name)


print("【一、页面能出来】")
r = TestClient(app.app).get("/")
check("GET / 返回 200", r.status_code == 200, r.status_code)
check("Content-Type 是 HTML", "text/html" in r.headers.get("content-type", ""))
html = r.text
check("页面有内容", len(html) > 3000, f"{len(html)} 字节")

print("\n【二、页面依赖的 DOM id 都存在】")
for dom_id in ("q", "go", "out", "mode", "fb", "examples", "cfg", "dot", "status"):
    check(f'#{dom_id}', f'id="{dom_id}"' in html)

print("\n【三、页面调的接口后端真的有】")
called = set(re.findall(r"fetch\('(/[a-z]+)", html))
routes = {getattr(rt, "path", "") for rt in app.app.routes}
check("页面至少调一个接口", bool(called), sorted(called))
for p in sorted(called):
    check(p, p in routes)

print("\n【四、/meta 的字段和类型】")
m = TestClient(app.app).get("/meta").json()
for k, t in (("kb_chunks", int), ("kb_topics", int), ("rerank_model", str),
             ("rerank_threshold", float), ("fallback_enabled", bool),
             ("uncertain_band", float)):
    check(f"{k} 是 {t.__name__}", isinstance(m.get(k), t), repr(m.get(k)))
check("语料条数 > 0", m.get("kb_chunks", 0) > 0, m.get("kb_chunks"))
check("阈值在 0~1 之间", 0 < m.get("rerank_threshold", -1) < 1, m.get("rerank_threshold"))

print("\n【五、页面读的字段，后端返回体里确实有】")
# 这一条是粗粒度双向检查：字段名两边都得出现。真正的字段级验证在 e2e_fallback.py 里。
src = (HERE / "main.py").read_text(encoding="utf-8")
for field in ("answer", "sources", "fallback_used", "tokens", "mode",
              "topic", "text", "score", "uncertain"):
    check(f"{field}: HTML + main.py 都有", field in html and field in src)

print("\n【六、uncertain 的判定语义（纯函数，不花钱）】")
# 钉住的是「边界包含等号」和「空召回不算边界」这两条：
# 前者如果不写进测试，下次有人把 < 改成 <=、或者反过来，没人会发现。
check("距门线 0.001 → True", app.near_gate([{"score": 0.951}], 0.95) is True)
check("距门线 0.003 → True", app.near_gate([{"score": 0.947}], 0.95) is True)
check("距门线 0.010 → False", app.near_gate([{"score": 0.940}], 0.95) is False)
check("正好等于门线 → True", app.near_gate([{"score": 0.95}], 0.95) is True)
check("空 ranking → False（没召回不是边界）", app.near_gate([], 0.95) is False)
check("带子可覆盖", app.near_gate([{"score": 0.958}], 0.95, band=0.01) is True)

print("\n【七、没有引外部资源（容器离线要能开）】")
for bad in ("http://cdn", "https://cdn", "unpkg.com", "jsdelivr", "googleapis"):
    check(f"不含 {bad}", bad not in html)

print()
if fails:
    print(f"失败 {len(fails)} 项：")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("全部通过")
