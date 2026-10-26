"""中文分词 + BM25 关键词检索。

BM25 可以理解成“按关键词出现得多不多、稀不稀有来打分”的老办法，
不需要模型，结果也容易解释。
"""

from __future__ import annotations

import math
import re
import unicodedata
import warnings
from collections import Counter

warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import jieba  # noqa: E402

jieba.setLogLevel(60)

from .. import config  # noqa: E402

STOPWORDS = set(
    """
的 了 是 在 和 与 及 或 等 中 对 为 从 其 这 那 有 也 就 都 而 被 把 将 能 可 会 要
上 下 内 外 时 个 种 些 吗 呢 啊 吧 之 所 以 于 并 且 还 更 很 最 一个 一种 一些 以及
通过 基于 进行 可以 能够 具有 需要 本文 我们 他们 它们 这个 那个 这些 那些 因此
所以 但是 然而 如果 因为 由于 其中 对于 关于 使用 采用 提出 表明 显示 包括 属于
什么 哪些 哪个 哪 如何 怎么 怎样 为什么 为何 是否 多少 请问 请 介绍 一下 告诉
通常 一般 主要 相关 方面 情况 内容 时候 哪里 哪儿 哪种 怎么样 用于
the a an and or of to in on for with by as at from is are was were be been being this
that these those it its we our they their can may using based into via such which what
how why when where than then also more most not no do does did has have had but if while
within between among about over under per
""".split()
)

PUNCT = re.compile(r"^[\W_]+$", re.UNICODE)

# 用户口语里的词，原文可能写成别的说法；只在原词不在索引里时替换。
# 顺序代表优先级，优先用原文最可能出现的那个说法。
QUERY_SYNONYMS = {
    "缺点": ["挑战", "不足", "局限"],
    "优点": ["优势", "长处"],
    "区别": ["差异", "对比"],
    "原理": ["机制", "机理"],
    "流程": ["步骤", "阶段"],
    "评估": ["评价", "指标"],
    # 口语说法 → 论文里的正式说法
    "编造": ["幻觉"],
    "出处": ["可追溯", "来源"],
    "核对": ["验证", "事实一致性"],
    "找全": ["召回", "检索质量"],
    "资料": ["文档", "语料"],
    "跟不上": ["动态更新", "增量"],
    "用图": ["知识图谱"],
    "图结构": ["知识图谱"],
    "好不好": ["评估", "评价"],
    "不好": ["评估", "评价"],
    "可信": ["可信度", "可靠性", "事实一致性"],
    "配合": ["协同", "联合"],
}

# 论文标题里常用中文全称，不写英文缩写
TOPIC_ALIASES = {
    "rag": ["检索增强生成", "检索增强"],
    "llm": ["大语言模型", "大模型"],
    "plm": ["预训练语言模型"],
    "nlp": ["自然语言处理"],
    "ir": ["信息检索"],
    "odqa": ["开放领域问答", "开放域问答"],
    "fid": ["fusion-in-decoder", "fusion in decoder"],
}


def tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).lower()
    tokens: list[str] = []
    for raw in jieba.lcut_for_search(text):
        tok = raw.strip()
        if not tok or PUNCT.match(tok):
            continue
        if tok in STOPWORDS:
            continue
        if tok.isdigit():
            continue
        if len(tok) == 1:
            # 单个汉字/字母噪音较大，只保留少量技术符号
            if tok not in {"q", "k"}:
                continue
        tokens.append(tok)
    return tokens


class BM25:
    def __init__(self, tokenized_docs: list[list[str]]):
        self.docs = tokenized_docs
        self.n = len(tokenized_docs)
        self.doc_len = [len(t) for t in tokenized_docs]
        self.avgdl = sum(self.doc_len) / self.n if self.n else 0.0
        self.tf = [Counter(t) for t in tokenized_docs]
        self.df: Counter[str] = Counter()
        for tokens in tokenized_docs:
            self.df.update(set(tokens))
        self.idf = {
            term: math.log(1 + (self.n - df + 0.5) / (df + 0.5))
            for term, df in self.df.items()
        }

    def scores(self, query_tokens: list[str]) -> list[float]:
        out = [0.0] * self.n
        if not self.n or not query_tokens:
            return out
        k1, b = config.BM25_K1, config.BM25_B
        for term in set(query_tokens):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, counter in enumerate(self.tf):
                f = counter.get(term, 0)
                if not f:
                    continue
                dl = self.doc_len[i] or 1
                denom = f + k1 * (1 - b + b * dl / (self.avgdl or 1))
                out[i] += idf * f * (k1 + 1) / denom
        return out
