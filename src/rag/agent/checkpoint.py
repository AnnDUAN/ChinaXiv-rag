"""Agent 运行的 checkpoint 存储。

第一版只做两件事：
1. 把每次运行的完整状态存进 agent_runs；
2. 每个步骤存一条 agent_steps，方便中断后继续和事后查看。

没有引入后台队列，也没有做分布式锁；本地单进程使用足够了。
"""

from __future__ import annotations

from typing import Any, Protocol

from psycopg.types.json import Jsonb

from ..db.engine import connect
from .state import AgentState


class CheckpointStore(Protocol):
    """Runner 依赖的最小存储接口，测试时可以换成内存实现。"""

    def create_run(self, state: AgentState) -> int: ...

    def save(self, run_id: int, state: AgentState) -> None: ...

    def load(self, run_id: int) -> AgentState | None: ...

    def start_step(
        self,
        run_id: int,
        step_no: int,
        node: str,
        input_payload: dict[str, Any],
    ) -> int: ...

    def finish_step(
        self,
        step_id: int,
        status: str,
        *,
        output: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None: ...

    def mark_interrupted(self, run_id: int) -> int: ...

    def list_runs(self, limit: int = 20) -> list[dict[str, Any]]: ...

    def trace(self, run_id: int) -> dict[str, Any] | None: ...


class PostgresCheckpointStore:
    """用 PostgreSQL 保存 Agent 状态。"""

    def create_run(self, state: AgentState) -> int:
        with connect() as conn:
            row = conn.execute(
                """
                INSERT INTO agent_runs
                    (question, status, phase, step_count, state, error)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING run_id
                """,
                (
                    state.question,
                    state.status.value,
                    state.phase,
                    state.step_count,
                    Jsonb(state.model_dump(mode="json")),
                    state.error,
                ),
            ).fetchone()
        return int(row["run_id"])

    def save(self, run_id: int, state: AgentState) -> None:
        with connect() as conn:
            conn.execute(
                """
                UPDATE agent_runs
                   SET question = %s,
                       status = %s,
                       phase = %s,
                       step_count = %s,
                       state = %s,
                       error = %s,
                       updated_at = now()
                 WHERE run_id = %s
                """,
                (
                    state.question,
                    state.status.value,
                    state.phase,
                    state.step_count,
                    Jsonb(state.model_dump(mode="json")),
                    state.error,
                    run_id,
                ),
            )

    def load(self, run_id: int) -> AgentState | None:
        with connect() as conn:
            row = conn.execute(
                "SELECT state FROM agent_runs WHERE run_id = %s",
                (run_id,),
            ).fetchone()
        if not row:
            return None
        return AgentState.model_validate(row["state"])

    def start_step(
        self,
        run_id: int,
        step_no: int,
        node: str,
        input_payload: dict[str, Any],
    ) -> int:
        with connect() as conn:
            row = conn.execute(
                """
                INSERT INTO agent_steps
                    (run_id, step_no, node, status, input_payload)
                VALUES (%s, %s, %s, 'running', %s)
                ON CONFLICT (run_id, step_no) DO UPDATE SET
                    node = EXCLUDED.node,
                    status = 'running',
                    input_payload = EXCLUDED.input_payload,
                    output_payload = '{}'::jsonb,
                    error = NULL,
                    started_at = now(),
                    ended_at = NULL
                RETURNING id
                """,
                (
                    run_id,
                    step_no,
                    node,
                    Jsonb(input_payload),
                ),
            ).fetchone()
        return int(row["id"])

    def finish_step(
        self,
        step_id: int,
        status: str,
        *,
        output: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        with connect() as conn:
            conn.execute(
                """
                UPDATE agent_steps
                   SET status = %s,
                       output_payload = %s,
                       error = %s,
                       ended_at = now()
                 WHERE id = %s
                """,
                (status, Jsonb(output or {}), error, step_id),
            )

    def mark_interrupted(self, run_id: int) -> int:
        """把上次进程留下的 running 步骤标成中断，方便重新执行。"""
        with connect() as conn:
            rows = conn.execute(
                """
                UPDATE agent_steps
                   SET status = 'interrupted',
                       ended_at = now()
                 WHERE run_id = %s AND status = 'running'
                 RETURNING id
                """,
                (run_id,),
            ).fetchall()
        return len(rows)

    def list_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        with connect() as conn:
            rows = conn.execute(
                """
                SELECT run_id, question, status, phase, step_count,
                       error, created_at, updated_at
                FROM agent_runs
                ORDER BY run_id DESC
                LIMIT %s
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def trace(self, run_id: int) -> dict[str, Any] | None:
        with connect() as conn:
            run = conn.execute(
                """
                SELECT run_id, question, status, phase, step_count,
                       state, error, created_at, updated_at
                FROM agent_runs
                WHERE run_id = %s
                """,
                (run_id,),
            ).fetchone()
            if not run:
                return None
            steps = conn.execute(
                """
                SELECT step_no, node, status, input_payload,
                       output_payload, error, started_at, ended_at
                FROM agent_steps
                WHERE run_id = %s
                ORDER BY step_no
                """,
                (run_id,),
            ).fetchall()
        return {"run": dict(run), "steps": [dict(s) for s in steps]}
