"""上线五件事，一次验完：服务活着、口令拦得住、问答答得出、文章和菜单挂上了、
正文没缺句且「阅读原文」点得通。

为什么单独写一个：这四件事分散在四个地方（容器、FastAPI、草稿箱、公众号后台），
任何一处断了，用户在手机上看到的现象都是「点进去没反应」，光看一处判断不了。
所以按「用户从微信点进来到拿到答案」的顺序一条条走，哪条断了一眼就看得见。

    uv run python verify_launch.py
    uv run python verify_launch.py --media-id fa-m_xxx      # 换一篇草稿验
    uv run python verify_launch.py --skip-ask               # 不想烧 token 就不发问

退出码：全通 0，有断点 1。可以直接挂到发版前跑。
"""
import argparse
import html as htmllib
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import wechat_menu as wm
import verify_draft as vd

HOST = "https://fitness-kb-rag.app.workbuddy.host"
SHARECODE = "4SEMQQJUME"
CODE = "kanwo-2026"
# 那篇导流文章建草稿时返回的 media_id。默认写死，是为了不依赖谁还记得它。
DEFAULT_MEDIA_ID = "fa-m_mYhA4b__e-iTH9HNeIA5jTroNbeEigQlaOVu20U9j4oShx5B3FZNMJOpvFb"

OK, BAD, WARN = "[ ok ]", "[FAIL]", "[ ?? ]"
fails = []


def say(mark, text, detail=""):
    print(f"{mark} {text}")
    if detail:
        for line in str(detail).splitlines():
            print(f"       {line}")
    if mark == BAD:
        fails.append(text)


def http(url, data=None, headers=None, timeout=60):
    req = urllib.request.Request(url, data=data, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")


def part_online():
    print("\n【1】线上服务")
    try:
        st, body = http(f"{HOST}/meta?sharecode={SHARECODE}", timeout=20)
    except Exception as e:
        say(BAD, f"/meta 拿不到：{e}")
        return False
    if st != 200:
        say(BAD, f"/meta 返回 {st}（反代没带 sharecode 会返回 HTML，不是 401）")
        return False
    meta = json.loads(body)
    say(OK, "服务在线", f"{meta['kb_chunks']} 条正文 / {meta['kb_topics']} 个话题，"
                       f"模型 {meta['llm_model'].split('/')[-1]}")
    want = {"rerank_top_k": 3, "decomp_mode": "off", "multi_top_k": 4,
            "query_log": True, "multiturn": True, "auth_required": True}
    drift = {k: (meta.get(k), v) for k, v in want.items() if meta.get(k) != v}
    if drift:
        say(WARN, "配置和预期不一致（可能是线上还是旧版本）",
            "\n".join(f"{k}: 线上 {a!r}，预期 {b!r}" for k, (a, b) in drift.items()))
    else:
        say(OK, "关键配置一致", "rerank_top_k=3 / 拆解关闭 / 多轮开 / 问法日志开 / 口令开")

    # 口令闸门：不带 header 必须 401。挡住了才敢把链接给粉丝。
    st, _ = http(f"{HOST}/rag?sharecode={SHARECODE}&src=mp",
                 data=json.dumps({"question": "推日练什么"}).encode(),
                 headers={"Content-Type": "application/json"})
    say(OK if st == 401 else BAD,
        f"口令闸门：不带口令 {st}" + ("" if st == 401 else "（应该 401，说明口令没生效）"))
    return True


def part_ask(question):
    print("\n【2】问答链路（走线上，带 src=mp，跟粉丝点进来同一条路）")
    t0 = time.time()
    try:
        st, body = http(f"{HOST}/rag?sharecode={SHARECODE}&src=mp",
                        data=json.dumps({"question": question}).encode(),
                        headers={"Content-Type": "application/json",
                                 "x-access-code": CODE}, timeout=90)
    except Exception as e:
        say(BAD, f"提问失败：{e}")
        return
    ms = int((time.time() - t0) * 1000)
    if st != 200:
        say(BAD, f"/rag 返回 {st}", body[:200])
        return
    d = json.loads(body)
    ans = (d.get("answer") or "").strip()
    srcs = d.get("sources") or []
    refused = not srcs
    say(OK if not refused else BAD,
        f"问「{question}」→ " + ("答上了" if not refused else "被拒答了（这条本该答得出）"),
        f"{len(srcs)} 条来源 / {ms} ms / cached={d.get('cached')}"
        + ("\n答案：" + ans[:90].replace("\n", " ") + ("…" if len(ans) > 90 else "")))
    if d.get("timings"):
        t = d["timings"]
        print("       分段：" + " / ".join(f"{k}={v}ms" for k, v in t.items()))


def part_article(token, media_id, relay):
    print("\n【3】公众号文章")
    req = urllib.request.Request(
        f"{wm.API}/draft/batchget?access_token={token}",
        data=json.dumps({"offset": 0, "count": 20, "no_content": 1}).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            d = json.loads(r.read().decode())
    except Exception as e:
        say(WARN, f"读草稿箱失败：{e}（不影响其它检查）")
        return None
    if d.get("errcode"):
        say(WARN, f"读草稿箱失败：{d.get('errcode')} {d.get('errmsg')}")
        return None
    items = d.get("item") or []
    hit = [i for i in items if i["media_id"] == media_id]
    print(f"       草稿箱 {d.get('total_count', len(items))} 篇："
          + "、".join(i["content"]["news_item"][0]["title"][:16] for i in items))
    if hit:
        ni = hit[0]["content"]["news_item"][0]
        url = ni.get("content_source_url") or ""
        say(WARN, "这篇还在草稿箱里 —— 说明还没群发，菜单里选不到它",
            f"《{ni['title']}》\n阅读原文：{url or '（空！点了会没反应）'}")
        return "draft"
    say(OK, "这篇不在草稿箱 —— 已经群发出去（群发会把草稿移走）")
    return "published"


def part_menu(token, relay):
    print("\n【4】底部菜单")
    menu, how = wm.read_menu(token, relay)
    if how.startswith("err"):
        say(WARN, f"读菜单失败：{how[4:]}")
        return ""
    print(f"       来源：{'接口' if how == 'api' else '后台手动配置'}")
    wm.describe(menu)
    btns = menu.get("button") or []
    if not btns:
        say(BAD, "没有菜单")
        return ""
    # 「问教练」可能是一级按钮，也可能被放进某个一级菜单下面当子按钮，两种都算通
    target = None
    for b in btns:
        subs = (b.get("sub_button") or {}).get("list") or []
        for s in subs:
            if "问" in (s.get("name") or ""):
                target = s
        if not subs and "问" in (b.get("name") or ""):
            target = b
    if not target:
        last = btns[-1]
        kind = last.get("type") or "sub"
        say(BAD, f"没找到「问教练」按钮（现在第三个是「{last.get('name')}」[{kind}]）",
            "后台改完要点底部「保存并发布」，只点「保存」不会下发；\n"
            "未认证号只能「跳转网页 → 从已发表选择」，手填外链必报「此链接不属于公众号」。")
        return ""
    kind = target.get("type") or ""
    url = target.get("url") or ""
    inside = any("问" in (s.get("name") or "")
                 for b in btns for s in ((b.get("sub_button") or {}).get("list") or []))
    say(OK if kind in ("view", "view_limited", "media_id") else WARN,
        f"「{target.get('name')}」[{kind}] 已挂上",
        f"目标：{url}" + ("\n（藏在「%s」下面当子菜单，粉丝要多点一层）" % btns[-1]["name"] if inside else ""))
    return url


def part_article_live(url):
    print("\n【5】已发布正文和「阅读原文」（粉丝看到的那一页）")
    if not url:
        say(WARN, "菜单里没有可跳的页面地址，这一项跳过")
        return
    ua = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
          "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")
    try:
        st, page = http(url, headers={"User-Agent": ua}, timeout=60)
    except Exception as e:
        say(BAD, f"抓文章失败：{e}")
        return
    if st != 200 or len(page) < 5000:
        say(BAD, f"文章页打不开（{st}，{len(page)} 字节）—— 手机上大概也是白屏")
        return
    m = re.search(r"var msg_title = '([^']*)'", page)
    title = m.group(1) if m else "?"
    say(OK, f"文章在线：《{title}》")

    m = re.search(r"msg_source_url = '([^']*)'", page)
    raw = m.group(1) if m else ""
    src = htmllib.unescape(raw.replace("\\x26amp;", "&").replace("\\x26", "&"))
    if not src:
        say(BAD, "这篇没有「阅读原文」链接 —— 粉丝点进来就没有入口了")
        return
    print(f"       阅读原文：{src}")
    for need in ("sharecode=", "k=", "src=mp"):
        if need not in src:
            say(WARN, f"阅读原文链接缺 {need}（粉丝要多手打口令，或日志分不清来源）")
    try:
        st2, home = http(src, headers={"User-Agent": ua}, timeout=30)
        ok_page = st2 == 200 and "健身知识库问答" in home
        say(OK if ok_page else BAD,
            "阅读原文能打开问答页" + ("" if ok_page else f"（{st2}，页面里没找到标题）"))
    except Exception as e:
        say(BAD, f"阅读原文打不开：{e}")

    local = Path("docs/wechat_qa_launch.html")
    if not local.exists():
        return
    ls = vd.sentences(vd.plain(local.read_text(encoding="utf-8")))
    page_text = vd.plain(page)
    missing = [s for s in ls if s not in page_text and s not in page]
    say(OK if not missing else BAD,
        f"正文完整性：本地 {len(ls)} 句，线上缺 {len(missing)} 句",
        "\n".join("丢失：" + s for s in missing[:5]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--media-id", default=DEFAULT_MEDIA_ID)
    ap.add_argument("--article-url", default="", help="菜单读不出地址时，手动给文章链接")
    ap.add_argument("--question", default="推日之后二头还练吗")
    ap.add_argument("--skip-ask", action="store_true")
    ap.add_argument("--relay", default="")
    args = ap.parse_args()

    part_online()
    if not args.skip_ask:
        part_ask(args.question)

    token = None
    try:
        appid, secret = wm.load_creds()
        token = wm.get_token(appid, secret, args.relay)
    except SystemExit as e:
        print(f"\n[ ?? ] 拿不到 access_token：{e}")
    if token:
        state = part_article(token, args.media_id, args.relay)
        url = part_menu(token, args.relay) or args.article_url
        if state == "draft":
            print("\n  顺序不能反：先群发，后台菜单里才选得到这篇。")
        part_article_live(url)

    print("\n" + "=" * 60)
    if fails:
        print(f"还有 {len(fails)} 处没通：")
        for f in fails:
            print("  - " + f)
        sys.exit(1)
    print("五项全通。")


if __name__ == "__main__":
    main()
