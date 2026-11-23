# -*- coding: utf-8 -*-
"""极简 BM25，专门用来补向量的短板。

为什么要它：向量比的是「意思像不像」，遇到「面拉」vs「硬拉」这种只差一个字的
动作名会判错——它俩在语义空间里太近了。BM25 比的是「字面命中多少」，
「面拉」这个词只出现在面拉那条资料里，一词命中就能把排序拉开。

中文怎么切词：不用 jieba，直接切字符 bigram。
「躺姿绳索面拉」→ 躺姿 / 姿绳 / 绳索 / 索面 / 面拉
好处是零依赖、不用维护词典；坏处是会切出「索面」这种无意义组合，
但这类组合在语料里出现次数少，IDF 会把它们压下去，实测不影响效果。
英文和数字整体保留（「3 到 5 克」里的 3 和 5 各算一个词）。

BM25 的三个参数：
  k1 控制「同一个词出现多次」的加成上限，1.2~2.0 之间，取 1.5
  b  控制「长文档惩罚」的力度，0 是不惩罚、1 是全额惩罚，取 0.75
  这两个是多年沿用的经验值，没在本地数据上重扫——样本只有 87 条，扫出来的
  数字没有统计意义，不如直接用通用值。
"""

import math
import re
from collections import Counter

_SPLIT = re.compile(r"[^\u4e00-\u9fa5a-zA-Z0-9]+")
_CJK = re.compile(r"[\u4e00-\u9fa5]")


def tokenize(text: str):
    """切成检索用的词。中文走字符 bigram，英文数字整体保留。"""
    tokens = []
    for seg in _SPLIT.split(text):
        if not seg:
            continue
        if _CJK.search(seg):
            if len(seg) == 1:
                tokens.append(seg)
            else:
                tokens.extend(seg[i:i + 2] for i in range(len(seg) - 1))
        else:
            tokens.append(seg.lower())
    return tokens


def tokenize_jieba(text: str):
    """用 jieba 搜索模式切词，作为 bigram 的对照组。

    cut_for_search 会同时给出粗粒度词和细粒度子词（「膝关节」→ 膝 / 关节 / 膝关节），
    对「用户问法和资料用词不完全一致」的情况更宽容。
    代价是多一个依赖，而且专业术语要靠自定义词典补（jieba 不认识「面拉」）。
    """
    import jieba
    return [w.strip().lower() for w in jieba.cut_for_search(text) if w.strip()]


class BM25:
    def __init__(self, docs, k1: float = 1.5, b: float = 0.75, tokenizer=None):
        self.tok = tokenizer or tokenize
        self.tokens = [self.tok(d) for d in docs]
        self.k1 = k1
        self.b = b
        self.n_docs = len(docs)
        self.avg_len = (sum(len(t) for t in self.tokens) / self.n_docs) if self.n_docs else 0.0

        self.tfs = [Counter(t) for t in self.tokens]

        # IDF：一个词在越多文档里出现，区分度越低，权重越小。
        # 用 log(1 + ...) 的形式，保证 IDF 永远为正 —— 经典 BM25 在词出现在
        # 超过一半文档时会给出负 IDF，反而给「命中」扣分，短小的知识库里很容易踩到。
        df = Counter()
        for toks in self.tokens:
            df.update(set(toks))
        self.idf = {
            w: math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5))
            for w, n in df.items()
        }

    def scores(self, query: str):
        """给全库每条资料打分，返回与构造时 docs 顺序一致的分数列表。"""
        q_tokens = self.tok(query)
        if not q_tokens:
            return [0.0] * self.n_docs

        out = []
        for i, tf in enumerate(self.tfs):
            doc_len = len(self.tokens[i])
            total = 0.0
            for w in q_tokens:
                f = tf.get(w)
                if not f:
                    continue
                idf = self.idf.get(w, 0.0)
                denom = f + self.k1 * (1 - self.b + self.b * doc_len / self.avg_len)
                total += idf * (f * (self.k1 + 1)) / denom
            out.append(round(total, 4))
        return out

    def best_matched_idf(self, query: str):
        """每题返回「在这条资料里命中的词里，最高的那个 IDF」。

        为什么需要它：BM25 总分高不代表命中可信。「推荐一部健身题材的电影」
        里「健身」也能命中一堆资料，但它是个到处都有的常见词，IDF 很低；
        「面拉」只在一条资料里出现，IDF 接近上限。
        **判「这个命中靠不靠得住」，看的是命中的词有多稀有，不是总分多高。**
        """
        q_tokens = self.tok(query)
        out = []
        for tf in self.tfs:
            best = 0.0
            for w in q_tokens:
                if w in tf:
                    best = max(best, self.idf.get(w, 0.0))
            out.append(round(best, 3))
        return out

    def idf_of(self, token: str):
        return self.idf.get(token, 0.0)
