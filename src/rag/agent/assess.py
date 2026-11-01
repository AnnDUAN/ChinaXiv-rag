"""判断当前证据够不够，以及下一步是回答还是联网。

只让模型做一个很窄的判断，不回答问题。
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..llm import (
    LLMClient,
    LLMConnectionError,
    Message,
    StructuredRunner,
)

NextAction = Literal["answer", "web_search", "give_up"]

ASSESS_SYSTEM_PROMPT = """你是检索流程的证据检查员，只做一件事：
判断下面的证据是否足以回答用户问题。

规则：
1. 不要回答问题，不要使用证据之外的知识。
2. 如果现有证据已经能支撑一个明确答案，sufficient 设为 true，
   next_action 设为 answer。
3. 如果证据不够，而且还能联网，next_action 设为 web_search，
   并在 web_query 里写一个适合搜索引擎的简短查询。
4. web_query 要保留实体和专有名词，不要照抄论文检索词，
   不要写“相关研究”“综述”这种空泛查询。
5. 如果已经联网过，或者当前不能联网，next_action 设为 give_up，
   不要反复要求联网。
6. 不要为了让流程继续而编造 web_query。"""


class EvidenceAssessment(BaseModel):
    """一次证据充分性判断。"""

    sufficient: bool = Field(
        description="现有证据是否足以回答用户问题。"
    )
    missing: str = Field(
        default="",
        description="如果证据不够，具体缺哪方面信息；够则留空。",
    )
    next_action: NextAction = Field(
        default="answer",
        description="下一步动作：answer、web_search 或 give_up。",
    )
    web_query: str = Field(
        default="",
        description=(
            "需要联网时使用的搜索查询；不联网或已经联网过时留空。"
        ),
    )
    reason: str = Field(
        default="",
        description="做出这个判断的简短原因。",
    )


def clean_web_query(value: str, *, max_chars: int = 160) -> str:
    query = " ".join((value or "").split()).strip()
    if not query or len(query) > max_chars:
        return ""
    return query


def build_assess_messages(
    question: str,
    results: list[dict[str, Any]],
    queries: list[str],
    *,
    web_used: bool,
    web_available: bool,
    max_evidence: int = 12,
    max_chars_per_chunk: int = 1200,
) -> list[Message]:
    evidence: list[str] = []
    for index, result in enumerate(results[:max_evidence], 1):
        source_type = result.get("source_type") or "local"
        if source_type == "web":
            evidence.append(
                f"[{index}] 网页：《{result.get('title', '')}》"
                f" | 来源：{result.get('url', '')}"
                f" | 发布时间：{result.get('published_at') or '未知'}\n"
                f"{str(result.get('text') or '')[:max_chars_per_chunk]}"
            )
        else:
            section = " > ".join(result.get("section_path") or [])
            evidence.append(
                f"[{index}] 本地论文：《{result.get('paper_title', '')}》"
                f" | 章节：{section}\n"
                f"{str(result.get('text') or '')[:max_chars_per_chunk]}"
            )

    used = "\n".join(f"- {query}" for query in queries[-5:]) or "（还没有查过）"
    user = (
        f"用户问题：{question}\n\n"
        f"已经用过的本地检索词：\n{used}\n\n"
        f"是否已经联网搜索过：{'是' if web_used else '否'}\n"
        f"当前是否允许联网：{'是' if web_available else '否'}\n\n"
        "当前找到的证据：\n\n"
        + ("\n\n".join(evidence) if evidence else "（没有找到片段）")
        + "\n\n请判断这些证据是否足够回答问题。"
    )
    return [
        Message("system", ASSESS_SYSTEM_PROMPT),
        Message("user", user),
    ]


def _no_evidence_assessment(
    question: str,
    *,
    web_used: bool,
    web_available: bool,
) -> EvidenceAssessment:
    if web_available and not web_used:
        return EvidenceAssessment(
            sufficient=False,
            missing="本地没有检索到相关片段",
            next_action="web_search",
            web_query=clean_web_query(question) or question.strip(),
            reason="本地没有检索结果，尝试联网搜索。",
        )
    return EvidenceAssessment(
        sufficient=False,
        missing="没有检索到相关片段",
        next_action="give_up",
        reason="没有可用证据，也无法继续联网。",
    )


def assess_evidence(
    question: str,
    results: list[dict[str, Any]],
    queries: list[str],
    *,
    client: LLMClient,
    web_used: bool,
    web_available: bool,
    max_evidence: int = 12,
    max_chars_per_chunk: int = 1200,
) -> tuple[EvidenceAssessment, str | None]:
    """返回 (判断结果, 判断过程错误)。"""
    if not results:
        return (
            _no_evidence_assessment(
                question,
                web_used=web_used,
                web_available=web_available,
            ),
            None,
        )

    messages = build_assess_messages(
        question,
        results,
        queries,
        web_used=web_used,
        web_available=web_available,
        max_evidence=max_evidence,
        max_chars_per_chunk=max_chars_per_chunk,
    )
    try:
        assessment = StructuredRunner(client).run(
            messages, EvidenceAssessment
        )
    except LLMConnectionError:
        raise
    except Exception as exc:  # noqa: BLE001
        return (
            EvidenceAssessment(
                sufficient=True,
                missing="",
                next_action="answer",
                reason="证据判断失败，使用现有片段继续回答。",
            ),
            str(exc),
        )

    assessment = _clean_assessment(
        assessment,
        web_used=web_used,
        web_available=web_available,
    )
    return assessment, None


def _clean_assessment(
    assessment: EvidenceAssessment,
    *,
    web_used: bool,
    web_available: bool,
) -> EvidenceAssessment:
    if assessment.sufficient:
        return assessment.model_copy(
            update={"next_action": "answer", "web_query": ""}
        )

    web_query = clean_web_query(assessment.web_query)
    if assessment.next_action == "web_search":
        if web_used or not web_available or not web_query:
            return assessment.model_copy(
                update={"next_action": "give_up", "web_query": ""}
            )
    else:
        web_query = ""

    return assessment.model_copy(update={"web_query": web_query})


__all__ = [
    "EvidenceAssessment",
    "NextAction",
    "assess_evidence",
    "build_assess_messages",
    "clean_web_query",
]
