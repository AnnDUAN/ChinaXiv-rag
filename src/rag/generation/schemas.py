"""问答模块的结构化输出模型。

模型只负责“写答案”和“指出用了哪段原文”，
编号是否存在、引文是否真的来自原文，由 Python 再校验一遍。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AnswerCitation(BaseModel):
    """一条出处。"""

    source_id: int = Field(
        ge=1,
        description="引用的片段编号，必须对应上下文里的 [1]、[2]……",
    )
    quote: str = Field(
        description=(
            "支持该结论的原文片段，必须逐字复制自对应片段，"
            "不要改写、概括或补全。"
        )
    )


class StructuredAnswer(BaseModel):
    """大模型回答的结构化结果。"""

    answer: str = Field(
        description=(
            "给用户看的中文回答。先给结论，再给必要解释；"
            "引用出处时在句末写 [编号]。"
        )
    )
    citations: list[AnswerCitation] = Field(
        default_factory=list,
        description="回答实际使用的原文出处列表。",
    )
    insufficient_evidence: bool = Field(
        default=False,
        description="原文片段是否不足以回答这个问题。",
    )
    confidence: Literal["high", "medium", "low"] = Field(
        default="medium",
        description="对回答是否有原文支撑的把握程度。",
    )
