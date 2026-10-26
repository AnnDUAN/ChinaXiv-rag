"""从对话文本中抽取候选长期记忆。

模型只负责语义抽取；主体、来源、日期夹紧和重要度范围由代码处理。
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Sequence

from pydantic import BaseModel, Field

from ..llm import (
    LLMClient,
    Message,
    StructuredRunner,
    build_client,
    load_llm_config,
)
from .models import MemoryCandidate, MemoryKind

_EXTRACT_SYS = """\
你在从一段对话里抽取「值得长期记住的陈述」。

只抽取关于用户或用户所认可世界的稳定信息，不抽取提问、寒暄和一次性追问。
例如：
  「我以后都希望用中文回答」 → preference
  「我在做 agentic_rag 项目」 → fact
  「我上周搬到了杭州」 → event
  「这个方案先按 PostgreSQL 做」 → task

规则：
1. 每条 content 写成一句独立陈述，脱离对话也能看懂；
2. 同一件事只输出一条；
3. 用户纠正旧说法时，kind 用 correction；
4. 用户明确说的内容 is_speculation=false；你推断的填 true；
5. importance 1-10：长期身份/硬约束 9-10，稳定偏好 7-8，一般事实 5-6，
   一次性小事 3-4，寒暄 1-2；
6. confidence：用户明确说 0.9，带保留 0.6-0.7，推断 0.4；
7. valid_from 只在用户明确说了日期时填写，否则填 null；
8. 只输出 JSON。
"""


class _ExtractedItem(BaseModel):
    kind: MemoryKind = Field(description="记忆类别")
    content: str = Field(description="一句独立陈述，不超过 80 字")
    importance: int = Field(ge=1, le=10, description="重要度 1-10")
    confidence: float = Field(ge=0.0, le=1.0, description="置信度 0-1")
    is_speculation: bool = Field(
        description="true=模型推断；false=用户明确说或确认"
    )
    valid_from: date | None = Field(
        default=None,
        description="这件事从哪天开始成立；不确定填 null",
    )


class _ExtractionOut(BaseModel):
    items: list[_ExtractedItem] = Field(default_factory=list)


def _default_runner(client: LLMClient | None = None) -> StructuredRunner:
    if client is None:
        client = build_client(load_llm_config())
    return StructuredRunner(client)


def extract_candidates(
    text: str,
    *,
    subject: str = "user:self",
    source: str = "conversation",
    session_id: str | None = None,
    today: date | None = None,
    runner: StructuredRunner | None = None,
    client: LLMClient | None = None,
) -> list[MemoryCandidate]:
    """把一段对话文本抽成候选记忆。"""
    text = (text or "").strip()
    if not text:
        return []

    today = today or datetime.now(timezone.utc).date()
    r = runner or _default_runner(client)
    out = r.run(
        [
            Message(
                "system",
                _EXTRACT_SYS + f"\n当前参考日期：{today.isoformat()}。",
            ),
            Message("user", text),
        ],
        _ExtractionOut,
    )
    assert isinstance(out, _ExtractionOut)

    candidates: list[MemoryCandidate] = []
    for item in out.items:
        content = (item.content or "").strip()
        if not content:
            continue
        valid_from = item.valid_from or today
        if valid_from > today:
            valid_from = today
        confidence = item.confidence
        if item.is_speculation:
            confidence = min(confidence, 0.6)
        candidates.append(
            MemoryCandidate(
                kind=item.kind,
                content=content,
                subject=subject,
                importance=max(1, min(10, item.importance)),
                confidence=max(0.0, min(1.0, confidence)),
                is_speculation=item.is_speculation,
                valid_from=valid_from,
                source=source,
                session_id=session_id,
            )
        )
    return candidates
