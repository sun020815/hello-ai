"""把问答页挂到公众号底部菜单：粉丝点一下就能问。

为什么值得做：评测集里那 86 条口语题是**对着知识库正文反推**出来的，
用词必然重叠，98% 的命中率是自己骗自己。要拿到真实问法，只有一个办法 ——
让真人在公众号里问。菜单是成本最低的入口：不用改文章、不用发链接。

用法（分三步，别跳）：
    uv run python wechat_menu.py --show                    # 1. 先看现在的菜单长什么样
    uv run python wechat_menu.py --dry-run                 # 2. 只看要提交的 JSON
    uv run python wechat_menu.py --force                   # 3. 确认后再真的写

    uv run python wechat_menu.py --delete                  # 删掉菜单
    uv run python wechat_menu.py --relay http://IP:8765    # 走固定出口 IP 的中转

前置条件：出口 IP 在公众号后台的 IP 白名单里（否则报 40164）。
家用宽带是动态 IP，所以本机经常不在白名单 —— 这时用 --relay 走那台
有固定 IP 的机器。

**注意：create 会整体覆盖现有菜单。** 所以默认不写，必须显式 --force。
"""
import argparse
import base64
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ENV_FILE = Path.home() / ".workbuddy" / "wechat-mp.env"
DEFAULT_TOKEN_FILE = Path(".wx_token.tmp")
BACKUP_FILE = Path(".wechat_menu_backup.json")
API = "https://api.weixin.qq.com/cgi-bin"


def load_creds():
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


def relay_call(relay: str, method: str, url: str, body: dict | None = None) -> dict:
    """把一次微信调用交给中转机发。中转机出口 IP 固定，白名单只需填一次。"""
    token = os.environ.get("RELAY_TOKEN", "").strip()
    if not token:
        sys.exit("走 --relay 需要环境变量 RELAY_TOKEN（和中转机上配的是同一个）")
    spec = {"url": url, "method": method}
    if body is not None:
        spec["body_b64"] = base64.b64encode(
            json.dumps(body, ensure_ascii=False).encode()).decode()
        spec["headers"] = {"Content-Type": "application/json"}
    req = urllib.request.Request(
        relay.rstrip("/") + "/proxy",
        data=json.dumps(spec).encode(),
        headers={"Content-Type": "application/json", "X-Relay-Token": token},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def direct_call(method: str, url: str, body: dict | None = None) -> dict:
    data = json.dumps(body, ensure_ascii=False).encode() if body is not None else None
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=40) as r:
        return json.loads(r.read().decode())


def call(method, url, body=None, relay=""):
    return relay_call(relay, method, url, body) if relay else direct_call(method, url, body)


def get_token(appid, secret, relay=""):
    url = (f"{API}/token?grant_type=client_credential&appid={appid}&secret={secret}")
    d = call("GET", url, None, relay)
    if not d.get("access_token"):
        code = d.get("errcode")
        print(f"取 token 失败：{code} {d.get('errmsg')}")
        if code == 40164:
            print("\n这是 IP 白名单问题（本机出口 IP 不在后台白名单里）。三种解法：")
            print("  1. 把报错里那个 IP 加进 设置与开发 → 基本配置 → IP 白名单")
            print("  2. 用 --relay http://<固定IP机器>:8765 走中转（推荐，动态 IP 一次配好）")
            print("  3. 退出代理的 TUN 模式（TUN 会让请求变成 IPv6 形式，永远匹配不上）")
        sys.exit(1)
    return d["access_token"]


def get_token_wait(appid, secret, relay="", timeout=1200, interval=15):
    """等白名单生效。

    后台改完白名单到真正生效之间有一段不确定的延迟（实测几分钟到十几分钟）。
    与其让人反复手动跑脚本，不如让它自己等：失败只打一行进度，成功那一刻才说话。

    只有 40164（IP 不在白名单）值得等。换 appid、secret 错了这类问题等多久都好不了，
    立刻退出，别让人对着一个不会有结果的循环发呆。
    """
    t0 = time.time()
    n = 0
    while True:
        n += 1
        try:
            d = call("GET", f"{API}/token?grant_type=client_credential"
                            f"&appid={appid}&secret={secret}", None, relay)
        except Exception as e:                     # 断网、超时：也算「还没好」，继续等
            d = {"errcode": -1, "errmsg": f"{type(e).__name__}: {str(e)[:60]}"}
        if d.get("access_token"):
            print(f"  ✓ 白名单已生效（第 {n} 次尝试，等了 {time.time() - t0:.0f} 秒）\n")
            return d["access_token"]

        code = d.get("errcode")
        left = timeout - (time.time() - t0)
        if left <= 0:
            print(f"  × 等了 {timeout} 秒仍是 {code}：{str(d.get('errmsg'))[:100]}")
            print("\n  两个最可能的原因：")
            print("   1. 白名单里填的 IP 不是微信看到的那个 ——"
                  "浏览器里查到的「我的 IP」常常是代理的美国节点，")
            print("      必须用报错里 invalid ip 后面那个（本次是 115.63.70.186）")
            print("   2. 加完没点保存，或者要管理员扫码确认才生效")
            sys.exit(1)
        if code != 40164:
            print(f"  × 不是白名单问题，不用等了：{code} {str(d.get('errmsg'))[:100]}")
            sys.exit(1)
        print(f"  … 还没生效（{left:.0f}s 内继续试，每 {interval}s 一次）")
        time.sleep(interval)


def build_menu(url: str, name: str):
    """一个按钮就够。名字必须 ≤ 16 字节，5 个汉字是上限。"""
    if len(name.encode("utf-8")) > 16:
        sys.exit(f"按钮名「{name}」超过 16 字节（微信限制），换个短点的")
    return {"button": [{"type": "view", "name": name, "url": url}]}


def read_menu(token, relay=""):
    """读菜单，两种账号两条路。

    认证号能直接调 menu/get；**未认证订阅号没有 menu/* 权限（48001）**，
    但「后台手动配的菜单」能用 get_current_selfmenu_info 读出来。
    这个区别必须说清：API 写不进去 ≠ 配不了菜单，后台点几下一样能挂。

    返回 (菜单字典, 来源)，来源是 "api" / "selfmenu" / "err:错误码"。
    """
    d = call("GET", f"{API}/menu/get?access_token={token}", None, relay)
    if not d.get("errcode"):
        return d.get("menu") or {}, "api"
    if d.get("errcode") != 48001:
        return {}, f"err:{d.get('errcode')}"
    e = call("GET", f"{API}/get_current_selfmenu_info?access_token={token}", None, relay)
    if e.get("errcode"):
        return {}, f"err:{e.get('errcode')}"
    return e.get("selfmenu_info") or {}, "selfmenu"


def describe(menu):
    """把人看的菜单打出来。两种来源结构不同，统一成同样的样子。"""
    btns = menu.get("button") or []
    if not btns:
        print("  （没有菜单）")
        return
    for b in btns:
        subs = (b.get("sub_button") or {}).get("list") or []
        kind = b.get("type") or ("sub" if subs else "?")
        tail = f"  → {b['url']}" if b.get("url") else ""
        print(f"  - {b.get('name')}  [{kind}]{tail}")
        for s in subs:
            stail = f"  → {s['url']}" if s.get("url") else ""
            print(f"      · {s.get('name')}  [{s.get('type')}]{stail}")


def show_menu(token, relay=""):
    menu, how = read_menu(token, relay)
    if how.startswith("err"):
        print(f"读取菜单失败：{how[4:]} (读不出来不代表配不了，见下面说明)")
        return None
    print(f"当前菜单（来源：{'接口' if how == 'api' else '后台手动配置'}）：")
    describe(menu)
    if how == "selfmenu":
        print("\n  注：这个号是未认证订阅号，没有菜单写入接口权限，"
              "所以菜单是在后台手动配的 —— 加按钮也得去后台加。")
    return menu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="https://fitness-kb-rag.app.workbuddy.host")
    ap.add_argument("--sharecode", default="4SEMQQJUME")
    ap.add_argument("--code", default="kanwo-2026", help="访问口令，写进链接免输入")
    ap.add_argument("--name", default="问教练", help="按钮名（≤5 个汉字）")
    ap.add_argument("--src", default="mp", help="来源标记，日志里用来区分公众号来的")
    ap.add_argument("--relay", default="", help="中转地址，如 http://1.2.3.4:8765")
    ap.add_argument("--show", action="store_true", help="只看现有菜单，不改")
    ap.add_argument("--dry-run", action="store_true", help="只打印要提交的 JSON")
    ap.add_argument("--delete", action="store_true", help="删除菜单")
    ap.add_argument("--force", action="store_true", help="确认覆盖现有菜单")
    ap.add_argument("--wait", action="store_true",
                    help="反复重试直到 IP 白名单生效（改完白名单用这个，不用干等）")
    ap.add_argument("--wait-timeout", type=int, default=1200, help="最多等多少秒，默认 1200")
    args = ap.parse_args()

    q = {"sharecode": args.sharecode, "k": args.code, "src": args.src}
    url = args.host.rstrip("/") + "/?" + urllib.parse.urlencode(q)

    print(f"目标链接：{url}")
    print(f"  说明：k= 是口令，写在链接里粉丝不用手输；src={args.src} 进日志，")
    print(f"        用来把「粉丝真问的」和「我自己点着玩的」分开。\n")

    # dry-run 不需要 token：先让它能在完全没网/没白名单的机器上跑通，
    # 否则「想先看看要提交什么」这一步会被 40164 挡在门外。
    menu = build_menu(url, args.name)
    if args.dry_run:
        print(json.dumps(menu, ensure_ascii=False, indent=2))
        print("\n（--dry-run 不提交。确认无误后加 --force 真写）")
        return

    appid, secret = load_creds()
    token = (get_token_wait(appid, secret, args.relay, args.wait_timeout)
             if args.wait else get_token(appid, secret, args.relay))

    if args.show:
        show_menu(token, args.relay)
        return

    if args.delete:
        if not args.force:
            sys.exit("删除是破坏性操作，加 --force 确认")
        d = call("GET", f"{API}/menu/delete?access_token={token}", None, args.relay)
        print("删除结果：", d)
        return

    # 先把现有菜单和账号类型摸清楚，后面每一步的判断都靠它：
    # 能不能写菜单，取决于这个号有没有 menu/* 权限，而不是取决于网络通不通。
    current, how = read_menu(token, args.relay)
    api_writable = how == "api"

    if not args.force:
        print("将要提交的菜单：")
        print(json.dumps(menu, ensure_ascii=False, indent=2))
        print("\ncreate 会**整体覆盖**现有菜单，所以默认不写。")
        print("先跑 --show 看一眼现在的菜单，确认要覆盖后加 --force。")
        return

    if api_writable:
        # 覆盖前先备份。create 是整体覆盖，万一把别的菜单冲掉了，
        # 手里有这个 JSON 就能照着逐字填回去 —— 不然只能凭记忆重搭。
        BACKUP_FILE.write_text(json.dumps(current, ensure_ascii=False, indent=2),
                               encoding="utf-8")
        print(f"已备份现有菜单 → {BACKUP_FILE}")
        if current.get("button"):
            print("现有菜单（下面这个会被整体替换）：")
            describe(current)
        else:
            print("现有菜单是空的（没有会被冲掉的东西）")
        print()

        d = call("POST", f"{API}/menu/create?access_token={token}", menu, args.relay)
        print("创建结果：", d)
        if d.get("errcode") == 0:
            print("\n成功。微信客户端可能要等几分钟才刷新（重进公众号或杀掉重开）。")
            print(f"之后粉丝点「{args.name}」就会打开问答页，问题会记进 .qa_log.jsonl。")
            print("看重什么就问什么：uv run python analyze_queries.py --src mp")
        else:
            sys.exit(1)
        return

    # 走到这里说明这个号没有菜单写入权限（未认证订阅号，48001）。
    # 不是死路：后台手动配的菜单，粉丝看到的效果完全一样，而且不依赖 IP 白名单。
    print("这个号不能用接口写菜单（未认证订阅号，48001）。**后台手动配，效果一样。**\n")
    print("现有菜单（来自后台，加按钮时要避开的位置）：")
    describe(current)
    print()
    print("操作路径：")
    print("  mp.weixin.qq.com → 内容与互动 → 自定义菜单")
    print(f"  → 选一个位置 → 「跳转网页」→ 填下面这个地址：")
    print(f"\n    {url}\n")
    print("  注意：一级菜单上限就是 3 个，上面三个占满了。")
    print(f"  要么把某个不常用的挪成子菜单腾位置，要么把「{args.name}」挂到已有子菜单里。")
    print("\n配完在手机上杀掉微信重开，点进去问一句，再用 analyze_queries.py --src mp 验证有没有记上。")


if __name__ == "__main__":
    main()
