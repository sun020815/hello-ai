"""从微信公众号接口取文章。

用途：把公众号里已有的文章取回本地，再交给 import_articles.py 切成 chunk 进知识库。

能力边界（实测，2026-11）：
  - 草稿箱 draft/batchget          ✅ 可读
  - 永久图文素材 material/batchget  ✅ 可读（但新版「发布」通道的文章不在这里）
  - 已发布列表 freepublish/batchget ❌ 订阅号返回 48001 api unauthorized，只有服务号有权限

用法：
  uv run python pull_wechat_articles.py                 # 列出草稿箱 + 素材库
  uv run python pull_wechat_articles.py --dump          # 连正文一起导出到 .wechat_pull/

前置条件：
  1. 本机出口 IP 已在公众号后台 IP 白名单里
  2. 关掉代理的 TUN 模式（FlClash 等）。开着 TUN 时请求会以 IPv4-mapped IPv6
     (::ffff:x.x.x.x) 发出，微信按 IPv6 比对白名单，IPv4 永远匹配不上，
     报 40164 invalid ip ... not in whitelist
"""

import argparse
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

ENV_FILE = Path.home() / ".workbuddy" / "wechat-mp.env"
OUT_DIR = Path(".wechat_pull")


def load_creds():
    """从 wechat-mp.env 读 appid / secret，不写死在代码里。"""
    if not ENV_FILE.exists():
        sys.exit(f"找不到凭证文件：{ENV_FILE}")
    creds = {}
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        k, v = line.split("=", 1)
        creds[k.strip()] = v.strip().strip("'\"")
    if not creds.get("WECHAT_APPID") or not creds.get("WECHAT_SECRET"):
        sys.exit("凭证文件里缺 WECHAT_APPID 或 WECHAT_SECRET")
    return creds["WECHAT_APPID"], creds["WECHAT_SECRET"]


def get_token(appid, secret):
    url = ("https://api.weixin.qq.com/cgi-bin/token"
           f"?grant_type=client_credential&appid={appid}&secret={secret}")
    with urllib.request.urlopen(url, timeout=30) as r:
        d = json.loads(r.read().decode())
    if not d.get("access_token"):
        code, msg = d.get("errcode"), d.get("errmsg", "")
        print(f"取 token 失败：{code} {msg}")
        if code == 40164:
            print("\n这是 IP 白名单问题，两种可能：")
            print("  1. 本机出口 IP 不在后台白名单里 —— 去 设置与开发 → 基本配置 添加")
            print("  2. 开着代理 TUN 模式 —— 退出代理再来，TUN 会让请求变成 IPv6 形式")
        sys.exit(1)
    return d["access_token"]


def api(path, token, body=None):
    """POST 列表类接口；返回 dict，不通就返回 errcode。"""
    if body is None:
        url = f"https://api.weixin.qq.com/cgi-bin/{path}?access_token={token}"
        req = urllib.request.Request(url)
    else:
        url = f"https://api.weixin.qq.com/cgi-bin/{path}?access_token={token}"
        req = urllib.request.Request(
            url,
            data=json.dumps(body, ensure_ascii=False).encode(),
            headers={"Content-Type": "application/json"},
        )
    with urllib.request.urlopen(req, timeout=40) as r:
        return json.loads(r.read().decode())


def strip_html(html: str) -> str:
    """把公众号正文 HTML 转成纯文本，尽量保住段落。"""
    t = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.S | re.I)
    t = re.sub(r"<br\s*/?>", "\n", t, flags=re.I)
    t = re.sub(r"</(p|div|section|h[1-6]|li|blockquote)>", "\n\n", t, flags=re.I)
    t = re.sub(r"<[^>]+>", "", t)
    t = (t.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<")
          .replace("&gt;", ">").replace("&quot;", '"').replace("&ldquo;", "“")
          .replace("&rdquo;", "”"))
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n[ ]+", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def collect_drafts(token):
    out, offset = [], 0
    while True:
        d = api("draft/batchget", token, {"offset": offset, "count": 20, "no_content": 0})
        if d.get("errcode"):
            print(f"  草稿箱读取失败：{d['errcode']} {d.get('errmsg')}")
            return out
        items = d.get("item", [])
        out.extend(items)
        offset += len(items)
        if not items or offset >= d.get("total_count", 0):
            break
    return out


def collect_material(token):
    out, offset = [], 0
    while True:
        d = api("material/batchget_material", token,
                {"type": "news", "offset": offset, "count": 20})
        if d.get("errcode"):
            print(f"  素材库读取失败：{d['errcode']} {d.get('errmsg')}")
            return out
        items = d.get("item", [])
        out.extend(items)
        offset += len(items)
        if not items or offset >= d.get("total_count", 0):
            break
    return out


def collect_published(token):
    """订阅号走不通，保留调用以便换成服务号时直接可用。"""
    d = api("freepublish/batchget", token, {"offset": 0, "count": 20, "no_content": 1})
    if d.get("errcode"):
        print(f"  已发布列表读取失败：{d['errcode']} {d.get('errmsg')}")
        print("  （48001 = api unauthorized，发布能力只有服务号有）")
        return []
    return d.get("item", [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", action="store_true", help="导出正文到 .wechat_pull/")
    args = ap.parse_args()

    appid, secret = load_creds()
    print(f"appid: {appid[:6]}***{appid[-4:]}")
    token = get_token(appid, secret)
    print("token 获取成功\n")

    print("[1/3] 草稿箱")
    drafts = collect_drafts(token)
    print(f"  {len(drafts)} 篇")

    print("[2/3] 永久图文素材")
    materials = collect_material(token)
    print(f"  {len(materials)} 篇")

    print("[3/3] 已发布列表")
    published = collect_published(token)
    print(f"  {len(published)} 篇\n")

    rows = []
    for src, items in (("草稿", drafts), ("素材", materials)):
        for it in items:
            news = (it.get("content") or {}).get("news_item", [])
            for k, ni in enumerate(news):
                rows.append({
                    "来源": src,
                    "索引": k,
                    "标题": ni.get("title", ""),
                    "摘要": (ni.get("digest") or "")[:60],
                    "链接": ni.get("url", ""),
                    "media_id": it.get("media_id", ""),
                    "_html": ni.get("content", "") if args.dump else "",
                })

    if not rows:
        print("没有可导出的文章。")
        return

    print(f"{'来源':4s}  {'标题'}")
    print("-" * 60)
    for r in rows:
        print(f"{r['来源']:4s}  {r['标题'][:44]}")

    if args.dump:
        OUT_DIR.mkdir(exist_ok=True)
        manifest = []
        for i, r in enumerate(rows, 1):
            text = strip_html(r.pop("_html"))
            slug = re.sub(r"[^\w\u4e00-\u9fa5]+", "-", r["标题"])[:30] or f"article{i}"
            path = OUT_DIR / f"{i:02d}-{slug}.md"
            path.write_text(f"# {r['标题']}\n\n{text}\n", encoding="utf-8")
            r["文件"] = str(path)
            r["字数"] = len(text)
            manifest.append(r)
        (OUT_DIR / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n已导出 {len(manifest)} 篇到 {OUT_DIR}/（正文 {sum(r['字数'] for r in manifest)} 字）")
        print("下一步：检查内容后，把要入库的加进 import_articles.py 的 ARTICLES 列表")


if __name__ == "__main__":
    main()
