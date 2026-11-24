"""推完草稿后拉回来比对：逐句核对微信有没有吞内容。

为什么必须做这一步：微信的草稿接口对 HTML 会做一轮重写（包 section、改样式、
删不认识的标签）。正常情况文字不动，但历史上确实出现过整段丢失 —— 而且
接口返回的 errcode 是 0，不报错。等发出去了才发现正文少一段，只能删文重发，
而订阅号一天只能群发一次。

比对方式：两边都剥掉标签取纯文字，按句子切开，看本地的句子有没有全部出现在
微信返回的正文里。多出来的（微信加的）不管，少了的（被吞的）必须报出来。
"""
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

ENV = Path.home() / ".workbuddy" / "wechat-mp.env"
API = "https://api.weixin.qq.com/cgi-bin"


def creds():
    d = {}
    for line in ENV.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s.startswith("export "):
            s = s[7:]
        if s and not s.startswith("#") and "=" in s:
            k, v = s.split("=", 1)
            d[k.strip()] = v.strip().strip("'\"")
    return d["WECHAT_APPID"], d["WECHAT_SECRET"]


def get(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read().decode())


def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def plain(html: str) -> str:
    """剥标签 + 还原实体。section/span 一堆，直接去掉。"""
    t = re.sub(r"<[^>]+>", "", html)
    for a, b in (("&nbsp;", " "), ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"),
                 ("&quot;", '"'), ("&#39;", "'")):
        t = t.replace(a, b)
    return re.sub(r"[ \t\u00a0]+", " ", t)


SENT = re.compile(r"[^。？！\n]+[。？！]?")


def sentences(text: str):
    out = []
    for s in SENT.findall(text):
        s = s.strip()
        if len(s) >= 2:          # 单个标点、单字不参与比对
            out.append(s)
    return out


def main():
    media_id = sys.argv[1]
    local_html = Path(sys.argv[2]).read_text(encoding="utf-8")

    appid, secret = creds()
    tok = get(f"{API}/token?grant_type=client_credential&appid={appid}&secret={secret}")
    if not tok.get("access_token"):
        sys.exit(f"取 token 失败：{tok}")
    t = tok["access_token"]

    d = post(f"{API}/draft/get?access_token={t}", {"media_id": media_id})
    if d.get("errcode"):
        sys.exit(f"拉草稿失败：{d['errcode']} {d.get('errmsg')}")
    items = d.get("news_item") or []
    if not items:
        sys.exit("草稿里没有图文")

    item = items[0]
    remote_html = item.get("content") or ""
    print(f"草稿标题  : {item.get('title')}")
    print(f"摘要      : {item.get('digest')}")
    print(f"作者      : {item.get('author')}")
    print(f"本地 HTML : {len(local_html)} 字符 → 微信: {len(remote_html)} 字符")
    print()

    ls, rs = sentences(plain(local_html)), sentences(plain(remote_html))
    rset = set(rs)
    missing = [s for s in ls if s not in rset]

    print(f"本地句子 {len(ls)} 条，微信返回 {len(rs)} 条")
    if missing:
        print(f"\n[FAIL] 有 {len(missing)} 句在微信返回里找不到：")
        for s in missing:
            print(f"   - {s}")
        sys.exit(1)

    print("\n[OK] 逐句比对通过，没有内容丢失")
    print(f"\n正文开头：{plain(remote_html)[:60].strip()}")
    print(f"正文结尾：{plain(remote_html)[-60:].strip()}")


if __name__ == "__main__":
    main()
