"""Agent 一次运行的可持久化状态。

状态会整体存进 PostgreSQL 的 JSONB 字段里，
这样程序中断后可以按 run_id 接着跑。
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class AgentStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentState(BaseModel):
    """一次 Agent 任务的全部进度。"""

    model_config = ConfigDict(extra="ignore")

    version: int = 2
    question: str
    status: AgentStatus = AgentStatus.RUNNING

    # analyze：分析意图和改写查询；retrieve：批量执行查询；
    # assess：判断证据够不够；web_search：联网搜索；
    # answer：生成回答；done：完成。
    phase: Literal[
        "analyze", "retrieve", "assess", "web_search", "answer", "done"
    ] = "analyze"
    step_count: int = 0
    max_steps: int = Field(default=8, ge=1)

    # 这里沿用原名字，实际表示“最多执行几次检索查询”。
    max_search_rounds: int = Field(default=4, ge=1)
    # 实际执行过的查询数量（改写查询 + 子查询）。
    search_rounds: int = Field(default=0, ge=0)
    top_k: int = Field(default=5, ge=1)
    mode: str = "hybrid"
    rerank: bool = False

    # 问题分析结果：intent、rewritten_query、sub_queries、reason。
    analysis: dict[str, Any] | None = None
    # 最近一次证据判断结果（只存判断摘要，不存网页正文）。
    assessment: dict[str, Any] | None = None

    # 已经执行过的本地查询；pending_queries 是分析后等待执行的查询。
    queries: list[str] = Field(default_factory=list)
    pending_queries: list[str] = Field(default_factory=list)

    # 联网只允许一次；网页正文放在 Runner 内存里，不进入状态。
    web_used: bool = False
    pending_web_query: str | None = None

    # 去重后的证据片段；不同查询的结果都会放在这里。
    results: list[dict[str, Any]] = Field(default_factory=list)
    # 每次查询的摘要，不存完整正文。
    searches: list[dict[str, Any]] = Field(default_factory=list)

    answer: dict[str, Any] | None = None
    error: str | None = None

    @property
    def result_count(self) -> int:
        return len(self.results)
