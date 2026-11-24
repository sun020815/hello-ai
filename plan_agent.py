"""多步工具调用：模型先查动作库，再排一周计划。

这个模块要解决的不是「让模型写一份计划」——那件事一次调用就能干，
而且干得挺像。要解决的是**它写的东西不能全信**：

  - 会不会编一个库里没有的动作？（「杠铃俯身飞鸟」听起来很专业）
  - 会不会给出跟禁忌冲突的安排？（膝盖疼还安排保加利亚蹲）
  - 会不会组次数字是拍的？（侧平举 4 组 6 次）

所以这里做了三层，缺一层都挡不住：

  第一层 数据在手：动作库是结构化的（`exercises.json`），34 个动作每个都
          指回一条已发布正文的 id。模型必须先 `search_exercises` 才能拿到
          真实动作名，不能凭记忆写。
  第二层 结构上不给机会：模型只输出**动作名 + 天数分配**，组次/间歇/要点
          全部由服务端从库里填。计划里出现的每一个数字都有出处，
          「模型编数字」这条路在结构上就不存在，不靠提示词求它别编。
  第三层 代码层再校验一次：模型交答案前可以自己调 `check_plan` 自查，
          但**不管它查没查、查的结果是什么，服务端都会重新校验一遍**。
          工具是给模型看的，校验是给自己兜底的 —— 模型说「我查过了」
          不能当成证据，这跟人一样。

一个刻意的取舍：**不追求「全自动排出一份完美计划」**。
排不出来就说排不出来（`needs_review: true` + 哪一步出的问题），
不拿一份看着完整、实则动作是编的计划糊弄过去。
"""

import json
import re
import time
from difflib import SequenceMatcher
from pathlib import Path

MAX_STEPS = 6                  # 最多几次「模型说话 → 调工具」，超了强制收尾
PLAN_BUDGET = 90.0             # 整条链的时间预算（秒），超了用已有结果收尾
MAX_DAYS = 5
MIN_DAYS = 2
MAX_ACTIONS_PER_DAY = 8
SEARCH_LIMIT = 10

# 伤病词 → 动作库 contra 标签。用户说的是「膝盖」「腰」，库里写的是「膝」「腰」，
# 不统一就得靠子串匹配，但「肩袖」和「肩」是两件事，所以先做精确表，再做前缀兜底。
INJURY_MAP = {
    "膝盖": "膝", "膝关节": "膝", "半月板": "膝", "髌骨": "膝",
    "腰": "腰", "腰椎": "腰", "下背": "腰", "腰突": "腰",
    "肩": "肩", "肩膀": "肩", "肩袖": "肩", "肩峰": "肩",
}


# ---------------------------------------------------------------- 动作库

class Library:
    """动作库的查询层。做成类是为了能塞假数据进单测，不用读磁盘。"""

    def __init__(self, items):
        self.items = list(items)
        self.by_id = {e["id"]: e for e in self.items}
        self._alias = {}
        for e in self.items:
            self._alias[e["name"]] = e
            for a in e.get("aliases", []):
                self._alias.setdefault(a, e)
        self.parts = sorted({e["part"] for e in self.items})

    @classmethod
    def load(cls, path=None):
        p = Path(path) if path else Path(__file__).parent / "exercises.json"
        data = json.loads(p.read_text(encoding="utf-8"))
        return cls(data["items"])

    def resolve(self, name):
        """名称 → 动作。先精确（含别名），再包含，最后拼音级别的相似度。

        顺序不能反：先做模糊匹配的话，「哑铃卧推」会被「上斜哑铃卧推」抢走。
        """
        name = (name or "").strip()
        if not name:
            return None
        if name in self._alias:
            return self._alias[name]
        # 「杠铃平板卧推 4 组」这种带后缀的，去掉尾巴再试一次
        cut = re.split(r"[\s（(·,，/]", name)[0]
        if cut in self._alias:
            return self._alias[cut]
        hits = [e for n, e in self._alias.items() if name in n or n in name]
        if len(hits) == 1:
            return hits[0]
        if hits:
            # 多个都沾边时取最长的那个名字：名字越长描述越具体
            return max(hits, key=lambda e: len(e["name"]))
        return None

    def suggest(self, name, n=4):
        """找不到时给候选。不给候选，模型下一步只会瞎猜另一个名字。"""
        scored = []
        for alias, e in self._alias.items():
            r = SequenceMatcher(None, name, alias).ratio()
            if e["name"] in [x["name"] for x in scored]:
                continue
            scored.append({"name": e["name"], "score": round(r, 2), "_e": e})
        scored.sort(key=lambda x: -x["score"])
        return [{"name": s["name"], "score": s["score"]} for s in scored[:n]]

    def search(self, part=None, equipment=None, exclude=(), level_max=None, limit=SEARCH_LIMIT):
        out = []
        for e in self.items:
            if part and part not in ("全身", "全部") and e["part"] != part:
                # 协同肌也算：臀腿日找「臀」，单腿硬拉的主部位是腿，但它是练臀的主力。
                if part not in e.get("secondary", []):
                    continue
            if equipment and e["equipment"] != equipment:
                # 自重动作不需要器械。家里只有哑铃时，平板支撑、臀桥照样能做；
                # 把自重一起过滤掉，会出现「哑铃条件下核心一个动作都没有」的假缺口。
                if not (equipment != "自重" and e["equipment"] == "自重"):
                    continue
            if level_max and e["level"] > level_max:
                continue
            if set(exclude) & set(e.get("contra", [])):
                continue
            out.append(e)
        # 复合动作排前面：排计划的骨架是复合动作，孤立动作是补丁。
        out.sort(key=lambda e: (e["load"] != "复合", e["level"], e["name"]))
        return out[:limit]

    @staticmethod
    def brief(e, full=False):
        """给模型看的精简版：够它判断选不选，不带一大堆用不上的字段。

        默认**不带 cue**。第一次实跑时带了，10 次工具调用烧掉 1.9 万 token ——
        cue 是给人看的一句话要点，模型选动作时用不上，等它真要讲某个动作
        自然会调 get_exercise。列表接口塞满字段是最常见的 token 浪费。
        """
        d = {"name": e["name"], "part": e["part"], "sub": e.get("sub", ""),
             "equipment": e["equipment"], "load": e["load"], "level": e["level"],
             "sets": e["sets"], "reps": e["reps"], "rest": e["rest"],
             "contra": e.get("contra", [])}
        if full:
            d["cue"] = e.get("cue", "")
        return d


# ---------------------------------------------------------------- 工具

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_exercises",
            "description": ("按部位查动作库。排计划之前必须先调用，动作名只能从返回结果里选，"
                            "不许自己写库里没有的动作。返回复合动作在前。"),
            "parameters": {
                "type": "object",
                "properties": {
                    "part": {"type": "string", "enum": ["胸", "背", "肩", "腿", "臀", "手臂", "核心", "全身"],
                             "description": "要练的部位"},
                    "equipment": {"type": "string", "enum": ["杠铃", "哑铃", "器械", "绳索", "史密斯", "自重"],
                                  "description": "器械限制，不填表示不限"},
                    "exclude_injuries": {"type": "array", "items": {"type": "string"},
                                         "description": "要避开的伤病标签，如 [\"膝\",\"腰\"]"},
                    "level_max": {"type": "integer", "description": "难度上限 1-3，新手填 2"},
                },
                "required": ["part"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_exercise",
            "description": ("查单个动作的要点、禁忌和替代动作。组次系统会自动填，不用为了组次查这个；"
                            "只在需要确认「这个人能不能做」或「有没有替代动作」时用。"),
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string", "description": "动作名"}},
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_plan",
            "description": ("交最终答案之前必须调用一次。把你排好的计划草稿交上来，"
                            "检查动作是否存在、是否跟伤病冲突、部位是否连练、有没有漏掉大肌群。"),
            "parameters": {
                "type": "object",
                "properties": {
                    "days": {
                        "type": "array",
                        "description": "每天一个对象",
                        "items": {
                            "type": "object",
                            "properties": {
                                "day": {"type": "integer"},
                                "focus": {"type": "string", "description": "这一天的主题，如「推日」"},
                                "exercises": {"type": "array", "items": {"type": "string"},
                                              "description": "动作名列表，必须是动作库里的名字"},
                            },
                            "required": ["day", "focus", "exercises"],
                        },
                    }
                },
                "required": ["days"],
            },
        },
    },
]


# 场地词不是器械词。用户说的是「健身房」，模型把它当 equipment 传进来，
# 严格的枚举校验会直接拒，但它其实想表达的是「不限」。
VENUE_WORDS = {"健身房", "都有", "不限", "器械齐全", "学校健身房", "任何"}


def call_tool(name, args, lib, injuries, want_days=None, session_min=None):
    """执行一个工具。**任何错误都以结果的形式返回，不抛异常。**

    为什么：抛异常等于整条链断掉，模型没机会自己纠正；而「部位写错了」
    这种错误，它看到合法值列表就能改对。工具的错误信息是给模型看的输入，
    不是给程序员的堆栈。
    """
    if name == "search_exercises":
        part = args.get("part")
        if part not in lib.parts + ["全身", "全部"]:
            return {"error": f"没有「{part}」这个部位", "valid_parts": lib.parts}
        eq = args.get("equipment")
        if eq in VENUE_WORDS:
            eq = None                       # 「健身房」= 不限，不是「找不到器械叫健身房的动作」
        valid_eq = {e["equipment"] for e in lib.items}
        if eq and eq not in valid_eq:
            # 第一次实跑就是这么空手而归的：模型传了「健身房」，库里没有这个器械，
            # 返回 0 条，它只能瞎猜着再查一次。枚举要校验，并且告诉它合法值。
            return {"error": f"没有「{eq}」这种器械", "valid_equipment": sorted(valid_eq),
                    "hint": "不限器械就不要传这个参数"}
        inj = list(args.get("exclude_injuries") or [])
        # 用户档案里的伤病是硬约束，模型可以再补，但不能减
        for i in injuries:
            tag = INJURY_MAP.get(i, i)
            if tag not in inj:
                inj.append(tag)
        items = lib.search(part=part, equipment=eq, exclude=inj,
                           level_max=args.get("level_max"))
        if not items:
            # 空结果要说清「为什么空」。只回一句「没有」不行 —— 实跑里模型会换着
            # 器械条件重试，一路把步数和 token 烧完（C 档案 44k token、6 步）。
            # 告诉它库里这个部位原本有什么，它就能直接判断「这个条件下确实练不了」。
            elsewhere = [f"{e['name']}（{e['equipment']}）" for e in lib.search(part=part, limit=6)]
            return {"count": 0, "items": [],
                    "hint": f"「{part}」在{args.get('equipment') or '不限器械'}条件下没有可用动作",
                    "in_library": elsewhere,
                    "advice": "别重复查同一个部位。条件实在不够就直接在 summary 里说明这个部位练不了"}
        return {"count": len(items), "items": [lib.brief(e) for e in items]}

    if name == "get_exercise":
        e = lib.resolve(args.get("name"))
        if not e:
            return {"error": f"动作库里没有「{args.get('name')}」", "suggest": lib.suggest(args.get("name") or "")}
        d = lib.brief(e, full=True)
        d["aliases"] = e.get("aliases", [])
        d["substitutes"] = e.get("substitutes", [])
        d["ref"] = e["ref"]
        return d

    if name == "check_plan":
        days = args.get("days") or []
        rep = audit(days, lib, injuries, want_days, session_min)
        return rep

    return {"error": f"没有叫 {name} 的工具"}


# ---------------------------------------------------------------- 校验

def _parts_of(day, lib):
    ps = []
    for n in day.get("exercises") or []:
        e = lib.resolve(n if isinstance(n, str) else n.get("name", ""))
        if e:
            ps.append(e["part"])
    return ps


_SET_RE = re.compile(r"\d+")
# 一组做功按 30 秒算（8~15 次、每次 2~4 秒，取中间），加上组间休息就是一组的总时间。
# 但只算「组数 × (做功 + 休息)」会系统性低估：找器械、调重量、第一组热身找感觉、
# 换动作之间走过去，每个动作实打实多出两三分钟。第一版没算这块，
# 5 个动作算出来 35 分钟 —— 真实训练里 5 个动作 60 分钟是常态。
SET_WORK_SEC = 30
PER_EXERCISE_OVERHEAD_SEC = 180   # 热身组 + 调重量 + 换器械
SESSION_WARMUP_MIN = 6            # 关节热身和激活，动作库里有专门一段讲这个


def estimate_minutes(day, lib):
    """粗估一天要练多久。「每侧 3 组」要按 6 组算，否则单腿日会被低估一半。"""
    total = SESSION_WARMUP_MIN * 60.0
    for n in day.get("exercises") or []:
        e = lib.resolve(n if isinstance(n, str) else n.get("name", ""))
        if not e:
            continue
        m = _SET_RE.search(e.get("sets", ""))
        sets = int(m.group()) if m else 3
        if "每侧" in e.get("sets", ""):
            sets *= 2
        total += sets * (SET_WORK_SEC + (e.get("rest") or 60)) + PER_EXERCISE_OVERHEAD_SEC
    return round(total / 60)


def audit(days, lib, injuries=(), want_days=None, session_min=None):
    """一份计划的体检报告。工具 `check_plan` 和最终返回共用这一份逻辑。

    共用是刻意的：如果工具里的规则和代码层的规则各写一套，迟早出现
    「工具说没问题、服务端说不合法」，那时候信谁都是错的。
    """
    errors, warnings = [], []
    tags = {INJURY_MAP.get(i, i) for i in injuries}
    unknown = []
    seen = {}
    by_part = {}

    if not isinstance(days, list) or not days:
        return {"ok": False, "errors": [{"type": "empty", "detail": "计划是空的"}],
                "warnings": [], "unknown": [], "by_part": {}, "missing": lib.parts,
                "est_minutes": {}}

    if not (MIN_DAYS <= len(days) <= MAX_DAYS):
        errors.append({"type": "day_count",
                       "detail": f"训练天数 {len(days)} 不在 {MIN_DAYS}~{MAX_DAYS} 之间"})
    # 用户说一周练 4 天，模型排了 5 天，这不是「风格差异」而是没按要求做。
    # 早期不查这条，前端显示「4 天计划」实际有 5 天，看着像前端 bug。
    if want_days and len(days) != int(want_days):
        errors.append({"type": "day_mismatch",
                       "detail": f"要求 {want_days} 天，实际排了 {len(days)} 天"})

    for i, d in enumerate(days):
        exs = d.get("exercises") or []
        if not exs:
            errors.append({"type": "empty_day", "day": d.get("day", i + 1), "detail": "这天没有动作"})
            continue
        if len(exs) > MAX_ACTIONS_PER_DAY:
            warnings.append({"type": "too_many", "day": d.get("day", i + 1),
                             "detail": f"{len(exs)} 个动作偏多，一般 4 到 6 个"})
        for n in exs:
            name = n if isinstance(n, str) else n.get("name", "")
            e = lib.resolve(name)
            if not e:
                unknown.append({"day": d.get("day", i + 1), "name": name})
                continue
            if set(tags) & set(e.get("contra", [])):
                errors.append({
                    "type": "injury_conflict", "day": d.get("day", i + 1),
                    "name": e["name"], "contra": e["contra"],
                    "substitutes": e.get("substitutes", []),
                    "detail": f"{e['name']} 对{'、'.join(sorted(set(tags) & set(e['contra'])))}不友好",
                })
            seen.setdefault(e["name"], []).append(d.get("day", i + 1))
            by_part[e["part"]] = by_part.get(e["part"], 0) + 1

    # 合并成一条：动作可选范围小的条件下（比如只有哑铃），重复是必然的，
    # 逐个报会一次冒出五六条一模一样的提醒，重要的那条反而被淹掉。
    dup = [(n, ds) for n, ds in seen.items() if len(ds) > 1]
    if dup:
        detail = "；".join(f"{n} 排了 {len(ds)} 次" for n, ds in dup[:4])
        if len(dup) > 4:
            detail += f" 等 {len(dup)} 个动作"
        warnings.append({"type": "duplicate", "items": [n for n, _ in dup],
                         "detail": f"重复用了几个动作：{detail}"})

    # 相邻两天练同一部位：推日和拉日都有三头，完全避免不现实，所以只警告。
    # 全身训练（一天覆盖 4 个以上部位）必须排除在外 —— A 档案排的就是全身 A/B/C，
    # 每天都练到背和腿，那是这种分化的定义，不是安排失误。第一版没排除，
    # 两天全身训练被连报两次「背、腿连着练」，属于用错了判据。
    for a, b in zip(days, days[1:]):
        pa_, pb_ = set(_parts_of(a, lib)), set(_parts_of(b, lib))
        big = (pa_ & pb_) & {"胸", "背", "腿", "臀"}
        if big and not (len(pa_) >= 4 and len(pb_) >= 4):
            warnings.append({"type": "back_to_back", "days": [a.get("day"), b.get("day")],
                             "parts": sorted(big),
                             "detail": f"第 {a.get('day')}、{b.get('day')} 天连着练 {'、'.join(sorted(big))}"})

    missing = [p for p in ("胸", "背", "腿") if p not in by_part]
    if missing:
        warnings.append({"type": "coverage", "parts": missing,
                         "detail": f"一周里没有排 {'、'.join(missing)}"})

    # 时长估算：只有动作库是结构化的（有组数和间歇）才算得出来。
    # 不查这条，就会出现「5 个动作、60 分钟」这种纸面上成立、实际做不完的计划。
    est = {}
    if session_min:
        for i, d in enumerate(days):
            key_day = d.get("day", i + 1)
            m = estimate_minutes(d, lib)
            est[key_day] = m
            if m > session_min * 1.15:
                warnings.append({"type": "over_time", "day": key_day, "minutes": m,
                                 "detail": f"第 {key_day} 天按组数和间歇算要 {m} 分钟，"
                                           f"比说的 {session_min} 分钟多"})

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "unknown": unknown,
        "by_part": by_part,
        "missing": missing,
        "est_minutes": est,
    }


# ---------------------------------------------------------------- 输出组装

def extract_json(text):
    """从模型回复里抠出 JSON。按可靠性排序：代码块 → 首尾花括号。"""
    if not text:
        return None
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    cands = [m.group(1)] if m else []
    i, j = text.find("{"), text.rfind("}")
    if i != -1 and j > i:
        cands.append(text[i:j + 1])
    for c in cands:
        try:
            return json.loads(c)
        except json.JSONDecodeError:
            continue
    return None


PLAN_FIELDS = ("id", "name", "part", "sub", "equipment", "sets", "reps", "rest", "cue", "ref", "contra")


def normalize(data, lib):
    """模型给的「动作名」换成动作库里的完整条目。

    数字（组次/间歇/要点）全部由这里填，不采用模型写的。原因见模块开头第二层。
    """
    days, dropped = [], []
    for i, d in enumerate(data.get("days") or []):
        if not isinstance(d, dict):
            continue
        exs = []
        for n in d.get("exercises") or []:
            name = n if isinstance(n, str) else (n.get("name") if isinstance(n, dict) else "")
            e = lib.resolve(name)
            if not e:
                dropped.append({"day": i + 1, "name": name})
                continue
            item = {k: e.get(k, "") for k in PLAN_FIELDS}
            exs.append(item)
        if exs:
            # 天数编号**不用模型给的**。D 档案里模型按星期编号，给了 day 1 和 day 4，
            # 前端就显示成「第 4 天」；而它明明只排了两天。编号是展示用的序号，
            # 由服务端按顺序重排最稳，模型只需要决定「有几天、每天练什么」。
            days.append({
                "day": len(days) + 1,
                "focus": (d.get("focus") or "").strip() or "训练日",
                "note": (d.get("note") or "").strip(),
                "exercises": exs,
            })
    return days, dropped


# ---------------------------------------------------------------- 提示词

SYSTEM = """你是一名健身教练，正在给一个学员排一周训练计划。

你手上有三个工具：search_exercises（按部位查动作库）、get_exercise（查单个动作细节）、
check_plan（交答案前自检）。

硬规则：
1. 动作只能从 search_exercises 的返回结果里选。库里没有的动作一律不许写，
   哪怕它很常见、很合理。你写的动作名，服务端会逐个比对动作库。
2. 组次和间歇由系统从动作库填，你不要写数字，只写动作名。也不用调 get_exercise 去查组次。
3. 每个部位都要先查一次再排，但**只查一次**。同一个部位重复查，结果一模一样，白花时间。
   凭印象排等于编，编出来的东西我会直接剔除。
4. 交最终答案之前，必须调一次 check_plan 自查；有 error 就改，改完再查一次。
5. 学员有伤病时，避开标注了对应禁忌的动作，优先用 substitutes 里的替代动作。
6. 排不出来就说排不出来，不要拿一份看着完整的假计划糊弄。

最终只输出 JSON，不要别的文字：
{"summary":"一句话说明这套计划的逻辑","days":[{"day":1,"focus":"推日","note":"可选，这一天的重点","exercises":["动作名","动作名"]}]}
"""


def build_user_prompt(profile):
    p = profile or {}
    lines = [
        f"目标：{p.get('goal') or '增肌'}",
        f"一周能练：{p.get('days') or 3} 天，每次约 {p.get('session_min') or 60} 分钟",
        f"场地器械：{p.get('equipment') or '健身房'}",
        f"水平：{p.get('level') or '新手'}",
    ]
    inj = p.get("injuries") or []
    lines.append(f"伤病/避开的部位：{'、'.join(inj) if inj else '无'}")
    if p.get("note"):
        lines.append(f"补充：{p['note']}")
    lines.append("")
    lines.append("排之前先把要用的部位都查一遍。")
    return "\n".join(lines)


# ---------------------------------------------------------------- 主流程

def _msg_to_dict(msg):
    """把 SDK 的 message 对象转成能塞回 messages 的 dict。"""
    if isinstance(msg, dict):
        return msg
    if hasattr(msg, "model_dump"):
        return msg.model_dump(exclude_none=True)
    return {"role": "assistant", "content": getattr(msg, "content", "") or ""}


def plan_week(profile, llm, lib=None, max_steps=MAX_STEPS, budget=PLAN_BUDGET):
    """跑一条完整的「查 → 排 → 自检 → 输出」链路。

    llm(messages, tools=None) -> response，注入进来是为了单测能塞一个
    按脚本返回 tool_calls 的假模型；真实调用走 main.call_llm。
    """
    lib = lib or Library.load()
    injuries = list((profile or {}).get("injuries") or [])
    want_days = (profile or {}).get("days")
    session_min = (profile or {}).get("session_min")
    t0 = time.perf_counter()
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": build_user_prompt(profile)},
    ]
    trace, tokens, steps = [], 0, 0
    final_text = ""
    stopped = ""

    for step in range(1, max_steps + 1):
        if time.perf_counter() - t0 > budget:
            stopped = "timeout"
            break
        resp = llm(messages, tools=TOOL_SCHEMAS)
        usage = getattr(resp, "usage", None)
        tokens += getattr(usage, "total_tokens", 0) or 0
        msg = resp.choices[0].message
        calls = getattr(msg, "tool_calls", None)
        steps = step

        if not calls:
            final_text = msg.content or ""
            break

        # 一条 assistant 可能同时要调好几个工具，全部执行、全部回填。
        # 少回一条 tool 消息，下一次请求就会被上游拒绝（每个 tool_call_id 必须有回应）。
        messages.append(_msg_to_dict(msg))
        for tc in calls:
            name = tc.function.name
            raw = tc.function.arguments
            ts = time.perf_counter()
            try:
                args = json.loads(raw) if raw else {}
                if not isinstance(args, dict):
                    raise ValueError
            except (json.JSONDecodeError, ValueError):
                result = {"error": "参数不是合法 JSON，请重发这个调用"}
                args = {"_raw": (raw or "")[:120]}
            else:
                result = call_tool(name, args, lib, injuries, want_days, session_min)
            ms = (time.perf_counter() - ts) * 1000
            ok = not (isinstance(result, dict) and result.get("error"))
            trace.append({
                "step": step, "tool": name, "args": args, "ok": ok, "ms": round(ms, 2),
                "brief": _brief(result),
            })
            messages.append({
                "role": "tool", "tool_call_id": tc.id,
                "content": json.dumps(result, ensure_ascii=False)[:6000],
            })
    else:
        stopped = "max_steps"

    # 步数用完还没给答案：明确收尾，让它用已查到的东西直接出 JSON。
    # 这里不再给 tools —— 再给一次它可能又去查，永远收不了尾。
    if not final_text and stopped in ("max_steps",):
        try:
            resp = llm(messages + [{"role": "user", "content": "步数用完了，用已查到的动作直接输出 JSON。"}],
                       tools=None)
            tokens += getattr(getattr(resp, "usage", None), "total_tokens", 0) or 0
            final_text = resp.choices[0].message.content or ""
            steps += 1
        except Exception as e:            # 收尾失败不该把整条链弄崩
            stopped = f"finalize_failed: {type(e).__name__}"

    data = extract_json(final_text)
    if data is None and not stopped:
        # 模型给了一段人话没给 JSON。再要一次，只要 JSON。
        try:
            resp = llm(messages + [{"role": "user",
                                    "content": "只输出 JSON，格式：{\"summary\":\"\",\"days\":[{\"day\":1,\"focus\":\"\",\"exercises\":[\"动作名\"]}]}"}],
                       tools=None)
            tokens += getattr(getattr(resp, "usage", None), "total_tokens", 0) or 0
            final_text = resp.choices[0].message.content or ""
            steps += 1
            data = extract_json(final_text)
        except Exception as e:
            stopped = f"finalize_failed: {type(e).__name__}"

    if data is None:
        return {
            "plan": [], "summary": "",
            "audit": {"ok": False, "errors": [{"type": "no_json", "detail": "模型没有给出可解析的计划"}],
                      "warnings": [], "unknown": [], "by_part": {}, "missing": lib.parts,
                "est_minutes": {}},
            "trace": trace, "steps": steps, "tokens": tokens,
            "needs_review": True, "dropped": [],
            "stopped": stopped or "no_json",
            "elapsed_ms": round((time.perf_counter() - t0) * 1000),
        }

    days, dropped = normalize(data, lib)
    rep = audit(days, lib, injuries, want_days, session_min)
    # 模型自己写的「动作名」被剔除时，也必须算进体检报告 —— 不能只在 dropped 里悄悄记一笔。
    if dropped:
        rep["errors"].append({"type": "unknown_exercise", "items": dropped,
                              "detail": f"{len(dropped)} 个动作不在库里，已剔除"})
        rep["ok"] = False

    return {
        "plan": days,
        "summary": (data.get("summary") or "").strip(),
        "audit": rep,
        "trace": trace,
        "steps": steps,
        "tokens": tokens,
        "needs_review": (not rep["ok"]) or bool(stopped),
        "dropped": dropped,
        "stopped": stopped,
        "elapsed_ms": round((time.perf_counter() - t0) * 1000),
    }


def _brief(result):
    """trace 里每条工具结果的一句话摘要，给前端展示「它查到了什么」。"""
    if not isinstance(result, dict):
        return str(result)[:80]
    if result.get("error"):
        return "错误：" + str(result["error"])[:80]
    if "items" in result:
        if not result["items"]:
            return "空结果：" + str(result.get("hint") or "")[:80]
        names = [i["name"] for i in result["items"][:5]]
        more = "" if len(result["items"]) <= 5 else f" 等 {len(result['items'])} 个"
        return "、".join(names) + more
    if "by_part" in result:
        if result.get("errors"):
            return "查到问题：" + "；".join(e.get("detail", e["type"]) for e in result["errors"])[:100]
        ws = result.get("warnings") or []
        if ws:
            # 把提醒也显示出来：第二次实跑里模型正是读到「侧平举排了两次」这条，
            # 才自己把动作换掉的。只写「自检通过」就看不出它为什么改。
            return f"通过，{len(ws)} 条提醒：" + "；".join(
                (w.get("detail") or w.get("type")) for w in ws)[:100]
        return "自检通过，部位覆盖：" + "、".join(f"{k}{v}" for k, v in result["by_part"].items())
    if result.get("name"):
        return f"{result['name']}：{result.get('sets','')} {result.get('reps','')}"
    return json.dumps(result, ensure_ascii=False)[:80]
