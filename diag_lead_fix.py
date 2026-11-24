"""改主题句的验收：目标题要上来，相邻题不能动，负例不能松。

README 记着一条教训：改主题句会引入新的字面重叠。上次把 upper-01 写成
「上斜卧推的凳子该调多少度」，结果「器械推胸座椅调多高」被从中胸拉到上胸。
所以改完必须把相邻话题的题全跑一遍，不能只看目标题涨没涨。

这次动的是：
  lose-01      开头加「减脂一周掉几斤算正常、缺口开多大才合适」
  diet-food-04 开头加「坚果一天吃多少、蛋黄吃几个、炒菜油用多少」
两个都在「减脂」语义簇里，最可能被抢的是减脂蛋白 / 减脂碳水 / 减脂三餐 /
增肌蛋白（diet-02，前面已经出现过抢答），以及所有含「一周」「吃多少」的训练题。
"""
import main

TH = main.RERANK_THRESHOLD

TARGET = [
    ("一周瘦几斤算正常", "lose-01"),
    ("一周减几斤算正常", "lose-01"),
    ("一周掉几斤算正常", "lose-01"),
    ("七天掉几斤才算正常", "lose-01"),
    ("掉秤多少算正常", "lose-01"),
    ("坚果一天吃多少", "diet-food-04"),
    ("坚果吃多少", "diet-food-04"),
    ("坚果一天吃几颗", "diet-food-04"),
    ("一天吃多少坚果", "diet-food-04"),
    ("坚果每天吃多少才算够", "diet-food-04"),
]

# 相邻正例：不许掉分、不许换话题
NEIGHBOR = [
    ("减脂期蛋白质吃多少克", "lose-diet-01"),
    ("减脂期碳水一天吃多少克", "lose-diet-01"),
    ("减脂期脂肪吃多少克", "lose-diet-01"),
    ("减脂期蛋白吃什么", "diet-food-01"),
    ("减脂期碳水吃什么", "diet-food-03"),
    ("一天三餐怎么吃", "diet-food-05"),
    ("炒菜油一天用多少", "diet-food-04"),
    ("蛋黄一天吃几个", "diet-food-04"),
    ("菜一天吃多少", "diet-food-04"),
    ("减脂期怎么保住力量", "lose-02"),
    ("有氧一周做几次", "cardio-01"),
    ("增肌期蛋白质吃多少", "diet-02"),
    ("一周练几次合适", "param-01"),
    ("午餐吃什么", "diet-01"),
]

NEG = [
    "肌酸一天吃多少",
    "减脂期肌酸一天吃多少",
    "减脂期要不要吃肌酸",
    "我膝盖疼吃什么",
    "减脂期膝盖疼吃什么",
    "蛋白粉一天喝几勺",
    "局部减脂可能吗",
    "碳水循环怎么安排",
]


def run(title, cases, expect_gold=None):
    print("=" * 74)
    print(title)
    print("=" * 74)
    bad = 0
    for item in cases:
        q, gold = (item if isinstance(item, tuple) else (item, None))
        r = main.rank_one(q)
        t = r[0]
        ok = t["score"] >= TH
        note = ""
        if gold and t["id"] != gold:
            note = f"  !! 抢答 gold={gold}"
            bad += 1
        if gold and not ok:
            note += "  !! 该答未答"
            bad += 1
        print(f"  {t['score']:.4f} [{'过' if ok else '拒'}] {q:<24} -> {t['id']:<13} {t['topic']}{note}")
    return bad


if __name__ == "__main__":
    n = run("【目标】应该从拒变过", TARGET)
    n += run("【相邻】不许掉分 / 不许换话题", NEIGHBOR)
    n += run("【负例】不许松动", NEG)
    print(f"\n问题数：{n}")
