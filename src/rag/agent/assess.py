"""判断当前证据够不够，以及下一轮该查什么。

这里只让模型做一个很窄的判断，不让它自由规划整个任务。
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from ..llm import (
    LLMClient,
    LLMConnectionError,
    Message,
    StructuredRunner,
)

ASSESS_SYSTEM_PROMPT = """你是检索流程的检查员，只做一件事：
判断下面这些原文片段，是否足以回答用户的问题。

规则：
1. 不要回答问题，不要使用片段之外的知识。
2. 如果现有片段已经能支撑一个明确答案，把 sufficient 设为 true。
3. 如果不够，用 missing 说明缺什么。
4. 如果换一个检索角度可能找到，把 next_query 写成一句简短、完整、
   和已有查询不同的问题；如果换角度也没用，next_query 留空。
5. 不要为了让流程继续而随便编造 next_query。"""


class EvidenceAssessment(BaseModel):
    """一次证据充分性判断。"""

    sufficient: bool = Field(
        description="现有原文片段是否足以回答用户问题。"
    )
    missing: str = Field(
        default="",
        description="如果证据不够，具体缺哪方面信息；够则留空。",
    )
    next_query: str = Field(
        default="",
        description=(
            "如果还需要检索，给出一个和已有查询不同的简短问题；"
            "不需要或无法再查时留空。"
        ),
    )
    reason: str = Field(
        default="",
        description="做出这个判断的简短原因。",
    )


def _normalize_query(text: str) -> str:
    return re.sub(r"[\s，。！？、；：,.!?;:（）()【】\[\]]+", "", text).lower()


def clean_next_query(
    value: str,
    used_queries: list[str],
    *,
    max_chars: int = 120,
) -> str:
    """去掉空值、过长值和已经查过的词。"""
    query = " ".join((value or "").split()).strip()
    if not query:
        return ""
    if len(query) > max_chars:
        return ""
    normalized = _normalize_query(query)
    if not normalized:
        return ""
    if any(normalized == _normalize_query(old) for old in used_queries):
        return ""
    return query


def build_assess_messages(
    question: str,
    results: list[dict[str, Any]],
    queries: list[str],
    *,
    max_evidence: int = 12,
    max_chars_per_chunk: int = 1200,
) -> list[Message]:
    evidence: list[str] = []
    for index, result in enumerate(results[:max_evidence], 1):
        section = " > ".join(result.get("section_path") or [])
        text = (result.get("text") or "")[:max_chars_per_chunk]
        evidence.append(
            f"[{index}] 论文：《{result.get('paper_title', '')}》"
            f" | 章节：{section}\n{text}"
        )

    used = "\n".join(f"- {query}" for query in queries[-5:]) or "（还没有查过）"
    user = (
        f"用户问题：{question}\n\n"
        f"已经用过的检索词：\n{used}\n\n"
        "当前找到的片段：\n\n"
        + ("\n\n".join(evidence) if evidence else "（没有找到片段）")
        + "\n\n请判断这些片段是否足够回答问题。"
    )
    return [
        Message("system", ASSESS_SYSTEM_PROMPT),
        Message("user", user),
    ]


def assess_evidence(
    question: str,
    results: list[dict[str, Any]],
    queries: list[str],
    *,
    client: LLMClient,
    max_evidence: int = 12,
    max_chars_per_chunk: int = 1200,
) -> tuple[EvidenceAssessment, str | None]:
    """返回 (判断结果, 判断过程错误)。

    没有检索结果时不去调用模型，直接判定“不足”。
    模型判断失败时也不会无限重试，而是让 Agent 用现有证据收尾。
    """
    if not results:
        return (
            EvidenceAssessment(
                sufficient=False,
                missing="没有检索到相关片段",
                next_query="",
                reason="没有召回结果",
            ),
            None,
        )

    messages = build_assess_messages(
        question,
        results,
        queries,
        max_evidence=max_evidence,
        max_chars_per_chunk=max_chars_per_chunk,
    )
    try:
        assessment = StructuredRunner(client).run(
            messages, EvidenceAssessment
        )
    except LLMConnectionError:
        # 网络/鉴权问题应该让任务失败，不能假装证据足够。
        raise
    except Exception as exc:  # noqa: BLE001
        return (
            EvidenceAssessment(
                sufficient=True,
                missing="",
                next_query="",
                reason="证据判断失败，使用现有片段继续回答",
            ),
            str(exc),
        )

    if assessment.sufficient:
        assessment.next_query = ""
    else:
        assessment.next_query = clean_next_query(
            assessment.next_query, queries
        )
    return assessment, None
