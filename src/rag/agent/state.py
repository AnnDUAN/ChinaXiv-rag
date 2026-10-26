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

    version: int = 1
    question: str
    status: AgentStatus = AgentStatus.RUNNING

    # retrieve：还要检索；assess：判断证据；answer：生成回答；done：完成
    phase: Literal["retrieve", "assess", "answer", "done"] = "retrieve"
    step_count: int = 0
    max_steps: int = Field(default=8, ge=1)

    search_rounds: int = 0
    max_search_rounds: int = Field(default=3, ge=1)
    top_k: int = Field(default=5, ge=1)
    mode: str = "hybrid"
    rerank: bool = False

    # 已经用过的检索词，避免模型用同一个问题反复查。
    queries: list[str] = Field(default_factory=list)
    pending_query: str | None = None

    # 去重后的证据片段；不同轮次的结果都会放在这里。
    results: list[dict[str, Any]] = Field(default_factory=list)
    # 每轮检索的摘要，不存完整正文。
    searches: list[dict[str, Any]] = Field(default_factory=list)
    # 每次“证据够不够”的判断结果。
    assessments: list[dict[str, Any]] = Field(default_factory=list)

    answer: dict[str, Any] | None = None
    error: str | None = None

    @property
    def result_count(self) -> int:
        return len(self.results)
