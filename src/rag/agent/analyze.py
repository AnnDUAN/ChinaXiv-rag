"""检索前的问题分析：判断意图、改写查询、拆出子问题。

只调用一次大模型，输出结构化的 QueryAnalysis。
分析失败时退回原问题，不阻塞后续检索和回答。
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from ..llm import (
    LLMClient,
    LLMConnectionError,
    Message,
    StructuredRunner,
)

Intent = Literal[
    "definition",
    "fact",
    "numeric",
    "comparison",
    "multi_paper",
    "follow_up",
    "other",
]

INTENT_LABELS = {
    "definition": "定义/概念",
    "fact": "事实/方法",
    "numeric": "数字/指标",
    "comparison": "对比",
    "multi_paper": "多论文归纳",
    "follow_up": "追问",
    "other": "其他",
}

ANALYZE_SYSTEM_PROMPT = """你是检索前的问题分析器，只做三件事：
1. 判断用户问题的意图；
2. 把问题改写成更适合论文检索的查询；
3. 必要时把复杂问题拆成 1～3 个子问题。

规则：
1. 不要回答问题，不要使用任何论文内容。
2. intent 只能从下面的值里选一个：
   - definition：问“是什么、定义、概念、含义、指什么”
   - fact：问原理、机制、方法、流程、原因、作用、怎么做
   - numeric：问数量、规模、参数、指标、实验结果、百分比
   - comparison：问两种方法/两篇论文/两个方案的区别、优劣、对比
   - multi_paper：问“这些论文、有哪些路线、分几类、综述、分别有哪些工作”
   - follow_up：依赖前文或上一轮的追问
   - other：以上都不明显
3. rewritten_query 要保留原文里的专有名词、模型名、系统名和英文缩写；
   补上论文里常见的同义说法，写成简短、具体的检索词，不要写成空泛的问句。
4. sub_queries 的要求：
   - comparison 和 multi_paper：给 2～3 个，分别覆盖对比对象或不同路线；
   - 其他意图：最多给 1 个，通常留空；
   - 不要重复 rewritten_query，不要写“相关研究”“综述”这种空泛查询。
5. 如果问题里有明确实体，改写时必须保留实体，不能只写主题词。
6. 只输出符合 schema 的 JSON。"""


class QueryAnalysis(BaseModel):
    """一次问题分析的结果。"""

    intent: Intent = Field(
        description="问题意图，只能取定义中列出的值。"
    )
    rewritten_query: str = Field(
        description="适合论文检索的改写查询，保留专有名词和英文缩写。"
    )
    sub_queries: list[str] = Field(
        default_factory=list,
        description=(
            "复杂问题拆出的子查询；comparison/multi_paper 给 2～3 个，"
            "其他意图最多 1 个。"
        ),
    )
    reason: str = Field(
        default="",
        description="一句话说明为什么这样分类和改写。",
    )


def _normalize(text: str) -> str:
    return re.sub(r"[\s，。！？、；：,.!?;:（）()【】\[\]]+", "", text).lower()


def _clean_query(value: str, *, max_chars: int = 120) -> str:
    query = " ".join((value or "").split()).strip()
    if len(query) > max_chars:
        return ""
    return query


def infer_intent_hint(question: str) -> Intent:
    """给模型结果做兜底，也用于分析失败时判断一个粗意图。"""
    if re.search(
        r"这些论文|哪些论文|研究现状|综述|有哪几类|分几类|"
        r"哪些不同|分别有哪些|都有哪些|哪几条路线",
        question,
    ):
        return "multi_paper"
    if re.search(r"区别|差异|对比|比较|相比|谁更好|优劣|vs\.?|VS", question):
        return "comparison"
    if re.search(
        r"多少|几组|几个|几种|几类|指标|准确率|召回率|F1|mAP|mIoU|"
        r"mDice|AUC|百分比|%|参数|规模|数量",
        question,
    ):
        return "numeric"
    if re.search(r"是什么|什么是|定义|概念|含义|指什么|指的是", question):
        return "definition"
    if re.search(
        r"怎么|如何|为什么|原理|机制|方法|流程|原因|作用|影响|"
        r"解决|实现",
        question,
    ):
        return "fact"
    return "other"


def clean_analysis(
    analysis: QueryAnalysis,
    question: str,
) -> QueryAnalysis:
    """清理模型输出：去重、限长、限制子查询数量。"""
    intent = analysis.intent
    if intent == "other":
        hint = infer_intent_hint(question)
        if hint != "other":
            intent = hint

    rewritten = _clean_query(analysis.rewritten_query)
    if not rewritten:
        rewritten = question.strip()

    max_sub = 3 if intent in {"comparison", "multi_paper"} else 1
    seen = {_normalize(rewritten), _normalize(question)}
    sub_queries: list[str] = []
    for value in analysis.sub_queries:
        query = _clean_query(value)
        if not query:
            continue
        key = _normalize(query)
        if key in seen:
            continue
        seen.add(key)
        sub_queries.append(query)
        if len(sub_queries) >= max_sub:
            break

    return analysis.model_copy(
        update={
            "intent": intent,
            "rewritten_query": rewritten,
            "sub_queries": sub_queries,
        }
    )


def build_analyze_messages(question: str) -> list[Message]:
    return [
        Message("system", ANALYZE_SYSTEM_PROMPT),
        Message(
            "user",
            f"用户问题：{question}\n\n请分析并输出 JSON。",
        ),
    ]


def analyze_question(
    question: str,
    *,
    client: LLMClient,
) -> tuple[QueryAnalysis, str | None]:
    """返回 (分析结果, 分析过程错误)。

    网络/鉴权错误向上抛；其他结构化失败退回原问题，
    保证 Agent 不会因为一次分析失败就完全跑不起来。
    """
    messages = build_analyze_messages(question)
    try:
        analysis = StructuredRunner(client).run(messages, QueryAnalysis)
    except LLMConnectionError:
        raise
    except Exception as exc:  # noqa: BLE001
        fallback = QueryAnalysis(
            intent=infer_intent_hint(question),
            rewritten_query=question.strip(),
            sub_queries=[],
            reason="分析失败，直接使用原问题检索。",
        )
        return clean_analysis(fallback, question), str(exc)

    return clean_analysis(analysis, question), None


__all__ = [
    "INTENT_LABELS",
    "Intent",
    "QueryAnalysis",
    "analyze_question",
    "build_analyze_messages",
    "clean_analysis",
    "infer_intent_hint",
]
