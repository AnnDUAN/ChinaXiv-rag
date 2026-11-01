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
    "single",
    "comparison",
    "multi_paper",
]

INTENT_LABELS = {
    "single": "单篇问题",
    "comparison": "对比",
    "multi_paper": "多论文归纳",
}

ANALYZE_SYSTEM_PROMPT = """你是检索前的问题分析器，只做三件事：
1. 判断用户问题的意图；
2. 把问题改写成更适合论文检索的查询；
3. 必要时把复杂问题拆成子问题。

规则：
1. 不要回答问题，不要使用任何论文内容。
2. intent 只能从下面的值里选一个：
   - single：单篇问题。问定义、概念、原理、机制、方法、流程、
     数字、指标、追问或其他普通问题，都归到这一类；
   - comparison：两个方法/两篇论文/两个方案的区别、优劣、对比；
   - multi_paper：问题里出现“这些论文、有哪些路线、分几类、
     综述、分别有哪些工作”，需要跨论文归纳。
3. rewritten_query 要保留原文里的专有名词、模型名、系统名和英文缩写；
   补上论文里常见的同义说法，写成简短、具体的检索词，不要写成空泛的问句。
4. sub_queries 的要求：
   - single：不要给子查询；
   - comparison：给 1～2 个，分别覆盖对比对象；
   - multi_paper：给 2～3 个，分别覆盖不同路线或不同方面；
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
            "复杂问题拆出的子查询；single 不给，comparison 给 1～2 个，"
            "multi_paper 给 2～3 个。"
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
    return "single"


def clean_analysis(
    analysis: QueryAnalysis,
    question: str,
) -> QueryAnalysis:
    """清理模型输出：去重、限长、限制子查询数量。"""
    intent = analysis.intent
    rewritten = _clean_query(analysis.rewritten_query)
    if not rewritten:
        rewritten = question.strip()

    max_sub = {
        "single": 0,
        "comparison": 2,
        "multi_paper": 3,
    }[intent]
    seen = {_normalize(rewritten), _normalize(question)}
    sub_queries: list[str] = []
    if max_sub > 0:
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
