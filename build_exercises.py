"""动作库生成器：把「已发布语料里讲过的动作」抽成结构化数据。

为什么要有这个东西
------------------
问答是「一句话进、一段话出」，它解决不了「给我排一周计划」。
排计划需要的是**可被程序筛选的实体**：部位、器械、难度、组次、禁忌。
知识库里的 128 条正文是散文，程序没法拿它算「周三练完腿、周四还能不能练腿」。

所以这里做的是**同一批知识库的第二种视图**，不是第二份内容：
每条动作都带 `ref`，指回 kb.json 里讲它的那条 chunk（`push-02`、`legs-02`……），
`build` 时会校验 ref 必须存在。这样训练计划里出现的每个动作，
都能在问答里查到它的细节；反过来，问答里的动作也不会跟计划对不上。

几条硬规矩（都是被坑出来的）
--------------------------
1. **动作名以语料为准，不自己发明**。语料里没有的（比如「卷腹」「提踵」0 次），
   这里一条都不放。放了就是计划里出现一个学员问不到细节的动作。
2. **组次照抄，不改写**。`push-02` 写「杠铃平板卧推 4 组，15、12、10、8 递减」，
   这里就是 `sets=4 / reps=15,12,10,8`。数字被「顺一下」之后就再也对不回原文了。
3. **禁忌只写语料明确说过的**。`newbie-03` 写了膝痛换腿举、肩痛换哑铃卧推、
   腰痛换臀桥，这三条进 `contra`；没写过的部位不硬猜。
4. 这个文件是唯一入口，`exercises.json` 不要手改 —— 跟 kb.json 同理。

跑法：uv run --with numpy python build_exercises.py   （numpy 其实用不到，纯标准库也行）
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
KB_PATH = ROOT / "kb.json"
OUT_PATH = ROOT / "exercises.json"

PARTS = {"胸", "背", "肩", "腿", "臀", "手臂", "核心"}
EQUIPS = {"杠铃", "哑铃", "器械", "绳索", "史密斯", "自重", "弹力带"}
PATTERNS = {"水平推", "垂直推", "水平拉", "垂直拉", "蹲", "髋铰链", "单腿",
            "伸膝", "膝屈", "肘屈", "肘伸", "肩外展", "肩水平外展", "核心", "内收"}
LOADS = {"复合", "孤立", "辅助"}


def ex(id, name, part, pattern, equipment, level, load, sets, reps, rest, cue,
       ref, aliases=(), sub="", secondary=(), contra=(), substitutes=()):
    """一条动作。参数多，所以用具名函数而不是裸字典 —— 少一个字段会当场报错，
    而裸字典会静默生成一条缺字段的数据，等到排计划时才炸。"""
    return {
        "id": id, "name": name, "aliases": list(aliases),
        "part": part, "sub": sub, "secondary": list(secondary),
        "pattern": pattern, "equipment": equipment, "level": level, "load": load,
        "sets": sets, "reps": reps, "rest": rest,
        "cue": cue, "contra": list(contra), "substitutes": list(substitutes),
        "ref": ref,
    }


EXERCISES = [
    # ---------- 胸 ----------
    ex("bb-bench", "杠铃平板卧推", "胸", "水平推", "杠铃", 3, "复合",
       "4 组", "15 / 12 / 10 / 8 递减加重", 90,
       "肩胛后缩下沉，脚踩实，杠触胸下沿；推起时肩不要送出去",
       "push-02", aliases=["平板卧推", "卧推", "杠铃卧推"], sub="中胸",
       secondary=["三头", "前束"], contra=["肩"], substitutes=["哑铃卧推"]),
    ex("db-bench", "哑铃平板卧推", "胸", "水平推", "哑铃", 2, "复合",
       "5 组", "8 到 10 次", 90,
       "幅度比杠铃深三到五厘米，顶端往中间挤，弱侧自己推完",
       "mid-01", aliases=["哑铃卧推", "平板哑铃卧推"], sub="中胸",
       secondary=["三头", "前束"]),
    ex("db-incline", "哑铃上斜卧推", "胸", "水平推", "哑铃", 3, "复合",
       "4 组", "12 次", 80,
       "凳角 25 到 30 度，下放到哑铃与胸下沿平齐，顶端两铃不碰",
       "push-02", aliases=["上斜卧推", "上斜哑铃卧推"], sub="上胸",
       secondary=["前束", "三头"]),
    ex("dip", "双杠臂屈伸", "胸", "水平推", "自重", 3, "复合",
       "4 组", "12 次或做到力竭", 80,
       "躯干前倾 30 到 40 度压在下胸，下到肩肘持平就停",
       "push-02", aliases=["双杠", "臂屈伸"], sub="下胸",
       secondary=["三头"], contra=["肩"],
       substitutes=["绳索夹胸"]),
    ex("cable-fly", "中位绳索夹胸", "胸", "肩水平外展", "绳索", 1, "孤立",
       "4 组", "15 次", 60,
       "滑轮与肩齐平，肘微曲角度固定，只想两个手肘互相靠拢",
       "cable-02", aliases=["绳索夹胸", "夹胸"], sub="中胸"),
    ex("low-cable-fly", "低位绳索夹胸", "胸", "肩水平外展", "绳索", 1, "孤立",
       "4 组", "15 次", 60,
       "滑轮调最低，由下往上抱，手腕在锁骨高度交叉",
       "upper-04", aliases=["低位夹胸"], sub="上胸"),
    ex("db-close-press", "窄距哑铃推胸", "胸", "水平推", "哑铃", 2, "孤立",
       "3 组", "12 次", 75,
       "两铃全程并紧不分开，掌心相对，每次推起都往中间挤",
       "lower-04", aliases=["窄距推胸"], sub="中缝",
       secondary=["三头"]),

    # ---------- 背 ----------
    ex("single-cable-pulldown", "单手钢线下拉", "背", "垂直拉", "绳索", 2, "复合",
       "4 组", "10 到 12 次", 90,
       "手肘先动，想象用肘去够裤兜，顶峰收缩停一秒",
       "pull-04", aliases=["单手下拉", "钢线下拉"], sub="背阔",
       secondary=["二头"], contra=["肩"]),
    ex("neutral-pulldown", "对握高位下拉", "背", "垂直拉", "器械", 1, "复合",
       "3 组", "10 到 12 次", 75,
       "向下发力不是向后，肩胛不要往前翻",
       "back-04", aliases=["高位下拉"], sub="背阔",
       secondary=["二头"]),
    ex("single-machine-row", "单手器械划船", "背", "水平拉", "器械", 1, "复合",
       "3 组", "10 到 12 次", 75,
       "躯干固定不后仰，肘贴身体拉到腰侧",
       "pull-02", aliases=["器械划船", "单手划船"], sub="背阔",
       secondary=["二头"], contra=["腰"]),
    ex("close-elbow-row", "窄距坐姿开肘划船", "背", "水平拉", "绳索", 2, "复合",
       "3 组", "12 次", 75,
       "手肘向外打开，靠躯干后仰甩重量就变成练腰",
       "pull-05", aliases=["开肘划船", "坐姿划船"], sub="中下斜方",
       secondary=["后束"]),
    ex("reverse-fly", "蝴蝶机反向飞鸟", "背", "肩水平外展", "器械", 1, "孤立",
       "3 组", "15 次", 60,
       "后束单独练，肘微曲，肩胛不夹",
       "pull-02", aliases=["反向飞鸟", "反向蝴蝶机"], sub="后束",
       secondary=["上背"]),

    # ---------- 肩 ----------
    ex("smith-press", "史密斯实力推", "肩", "垂直推", "史密斯", 2, "复合",
       "4 组", "8 到 12 次", 90,
       "核心收紧不顶腰，下放到下巴附近推起",
       "legs-02", aliases=["史密斯推举", "实力推"], sub="前束",
       secondary=["三头"], contra=["肩"]),
    ex("bb-ohp", "站姿杠铃实力推", "肩", "垂直推", "杠铃", 3, "复合",
       "4 到 5 组", "6 到 10 次", 120,
       "唯一上大重量的主项，每组留两次余量 RPE 8，严禁力竭",
       "shoulder-04", aliases=["杠铃推举", "站姿推举"], sub="前束",
       secondary=["三头"], contra=["肩", "腰"]),
    ex("y-raise", "Y 字侧平举", "肩", "肩外展", "哑铃", 1, "孤立",
       "3 组", "15 次", 60,
       "俯身或上斜凳做，拇指朝上，往斜前方抬成 Y 字",
       "push-02", aliases=["Y 字举", "Y 抬"], sub="中束"),
    ex("db-lateral", "哑铃侧平举", "肩", "肩外展", "哑铃", 1, "孤立",
       "3 组", "15 次", 60,
       "小重量可以力竭，肩袖压力小；动作变形就减重量",
       "legs-02", aliases=["侧平举"], sub="中束"),
    ex("rev-pec-deck", "坐姿器械反飞鸟", "肩", "肩水平外展", "器械", 1, "孤立",
       "4 组", "15 次", 60,
       "后束放在第一个做，精力最充沛时给最高质量刺激",
       "shoulder-04", aliases=["器械反飞鸟"], sub="后束"),
    ex("face-pull", "仰卧绳索面拉", "肩", "肩水平外展", "绳索", 1, "孤立",
       "4 组", "12 到 15 次", 60,
       "拉到脸前，肘高于腕，想着把肩胛往后下方收",
       "shoulder-04", aliases=["面拉"], sub="后束",
       secondary=["上背"]),
    ex("db-front-raise", "哑铃对握前平举", "肩", "肩外展", "哑铃", 1, "孤立",
       "3 组", "12 到 15 次", 60,
       "对握掌心相对像端水，正手容易肩峰撞击",
       "shoulder-05", aliases=["前平举"], sub="前束"),

    # ---------- 腿 / 臀 ----------
    ex("single-leg-rdl", "单腿硬拉", "腿", "单腿", "哑铃", 3, "复合",
       "每侧 4 组", "10 到 12 次", 90,
       "髋往后推不是弯腰，支撑腿微屈，骨盆不翻",
       "legs-02", aliases=["单腿罗马尼亚硬拉"], sub="腘绳与臀",
       secondary=["臀", "核心"], contra=["腰"]),
    ex("bulgarian", "保加利亚分腿蹲", "腿", "单腿", "哑铃", 2, "复合",
       "每侧 3 组", "10 到 12 次", 90,
       "重心在前腿，后腿只负责平衡不发力；直立版偏大腿根，俯身版偏臀",
       "single-03", aliases=["保加利亚蹲", "保加利亚分腿蹲"], sub="股四与臀",
       secondary=["臀"], contra=["膝"]),
    ex("front-squat", "颈前深蹲", "腿", "蹲", "杠铃", 3, "复合",
       "3 组", "8 到 10 次", 120,
       "肘高抬、躯干直立，膝痛就换腿举",
       "legs-02", aliases=["前蹲"], sub="股四",
       secondary=["臀", "核心"], contra=["膝"], substitutes=["腿举"]),
    ex("leg-press", "腿举", "腿", "蹲", "器械", 1, "复合",
       "3 组", "12 到 15 次", 90,
       "腰贴凳面不拱起，膝盖不锁死",
       "newbie-03", aliases=["倒蹬"], sub="股四",
       secondary=["臀"]),
    ex("rdl", "罗马尼亚硬拉", "腿", "髋铰链", "杠铃", 2, "复合",
       "3 组", "10 到 12 次", 90,
       "杠铃贴腿走，膝盖只微屈，靠屈髋把重量送下去",
       "dead-01", aliases=["直腿硬拉"], sub="腘绳与臀",
       secondary=["臀", "下背"], contra=["腰"], substitutes=["臀桥"]),
    ex("leg-extension", "坐姿腿屈伸", "腿", "伸膝", "器械", 1, "孤立",
       "3 组", "20 次", 60,
       "臀部压紧凳面不翘起，每次向心都锁到最紧；孤立动作靠代谢压力",
       "iso-04", aliases=["腿屈伸"], sub="股四", contra=["膝"]),
    ex("leg-curl", "俯身腿弯举", "腿", "膝屈", "器械", 2, "孤立",
       "5 组", "内旋 6 次 + 外旋 6 次算一组", 60,
       "内旋位主打向心，外旋位主打离心控制，骨盆不反弓",
       "iso-01", aliases=["腿弯举"], sub="腘绳"),
    ex("hip-thrust", "臀桥", "臀", "髋铰链", "自重", 1, "孤立",
       "3 组", "15 次", 60,
       "仰卧屈膝脚踩地，臀发力顶髋，顶端夹紧停一秒；腰有旧伤用这个替硬拉",
       "dead-03", aliases=["臀推", "桥式"], sub="臀大肌",
       secondary=["腘绳"]),

    # ---------- 手臂 ----------
    ex("lying-triceps-ext", "仰卧臂屈伸", "手臂", "肘伸", "杠铃", 1, "孤立",
       "4 组", "12 到 15 次", 70,
       "大臂固定不动，只做肘伸，肘尖朝上不外张",
       "push-02", aliases=["仰卧臂屈伸", "碎颅式"], sub="三头",
       secondary=["三头长头"]),
    ex("cable-curl", "钢线弯举", "手臂", "肘屈", "绳索", 1, "孤立",
       "3 组", "12 到 15 次", 60,
       "大臂夹紧不晃，离心三秒放下",
       "pull-02", aliases=["绳索弯举"], sub="二头"),
    ex("overhead-triceps-ext", "颈后臂屈伸", "手臂", "肘伸", "哑铃", 1, "孤立",
       "3 组", "12 到 15 次", 60,
       "肘尖朝天不外张，下放到颈后拉到最长",
       "shoulder-04", aliases=["过顶臂屈伸"], sub="三头长头"),
    ex("bent-over-curl", "俯身弯举", "手臂", "肘屈", "哑铃", 1, "孤立",
       "3 组", "12 到 15 次", 60,
       "躯干前倾固定，靠二头把重量卷起来，不甩腰",
       "shoulder-04", aliases=["俯身二头弯举"], sub="二头"),

    # ---------- 核心 ----------
    ex("plank", "平板支撑", "核心", "核心", "自重", 1, "孤立",
       "3 组", "40 到 60 秒", 45,
       "等长收缩，臀不塌不翘，肋骨下沉",
       "term-01", aliases=["平板"], sub="核心"),
    ex("copenhagen", "哥本哈根支撑", "核心", "内收", "自重", 2, "孤立",
       "2 组", "每侧 20 秒", 45,
       "侧躺上腿放凳面、下腿夹紧凳子，身体成一条线；练髋内收和侧腹",
       "legs-05", aliases=["哥本哈根"], sub="髋内收"),
]


def main() -> int:
    kb = json.loads(KB_PATH.read_text(encoding="utf-8"))
    kb_ids = {d["id"] for d in kb}
    errs: list[str] = []
    seen_id: dict[str, int] = {}
    seen_name: dict[str, str] = {}

    for i, e in enumerate(EXERCISES):
        tag = f"[{i}] {e.get('name', '?')}"
        if e["id"] in seen_id:
            errs.append(f"{tag}: id 重复（与第 {seen_id[e['id']]} 条同 id）")
        seen_id[e["id"]] = i
        for n in [e["name"], *e["aliases"]]:
            if n in seen_name and seen_name[n] != e["id"]:
                errs.append(f"{tag}: 名称/别名「{n}」与 {seen_name[n]} 撞了")
            seen_name[n] = e["id"]
        if e["part"] not in PARTS:
            errs.append(f"{tag}: part「{e['part']}」不在 {sorted(PARTS)}")
        if e["pattern"] not in PATTERNS:
            errs.append(f"{tag}: pattern「{e['pattern']}」不在 {sorted(PATTERNS)}")
        if e["equipment"] not in EQUIPS:
            errs.append(f"{tag}: equipment「{e['equipment']}」不在 {sorted(EQUIPS)}")
        if e["load"] not in LOADS:
            errs.append(f"{tag}: load「{e['load']}」不在 {sorted(LOADS)}")
        if e["level"] not in (1, 2, 3):
            errs.append(f"{tag}: level 只能是 1/2/3")
        # 这条最重要：动作必须能追回一条真实存在的知识库正文。
        # 追不回去，就等于计划里出现一个学员问不到细节的动作。
        if e["ref"] not in kb_ids:
            errs.append(f"{tag}: ref「{e['ref']}」在 kb.json 里不存在")
    # 替代动作允许写别名（「哑铃卧推」是 db-bench 的别名，不是它的 name），
    # 所以必须等所有动作的名称都进了 seen_name 之后再校验，不能边遍历边查。
    for i, e in enumerate(EXERCISES):
        for s in e["substitutes"]:
            if s not in seen_name:
                errs.append(f"[{i}] {e['name']}: 替代动作「{s}」不在动作库里")

    if errs:
        print("动作库校验失败：", file=sys.stderr)
        for x in errs:
            print("  -", x, file=sys.stderr)
        return 1

    # ref 到 chunk 的映射由 main 运行时按需做，这里只存动作本身，
    # 避免 kb 改了标题之后动作库里留着一份过期的正文副本。
    order = ["胸", "背", "肩", "腿", "臀", "手臂", "核心"]
    items = sorted(EXERCISES, key=lambda e: (order.index(e["part"]), -e["level"], e["name"]))
    out = {
        "version": 2,
        "count": len(items),
        "parts": {p: sum(1 for e in items if e["part"] == p) for p in order},
        "items": items,
    }
    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"写好 {OUT_PATH.name}：{len(items)} 个动作")
    for p in order:
        names = [e["name"] for e in items if e["part"] == p]
        print(f"  {p}（{len(names)}）：{'、'.join(names)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
