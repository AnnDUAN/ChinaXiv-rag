"""一个很小、可控的检索 Agent。

流程只有四种状态：
retrieve（检索）→ assess（判断够不够）→ answer（回答）→ done（结束）

它不会自由调用一堆工具，也不会无限循环：
- 最多补查 max_search_rounds 轮；
- 最多执行 max_steps 个步骤；
- 模型想重复同一个检索词时直接停止补查；
- 补查没有带回新片段时也停止。
"""

from __future__ import annotations

import re
from typing import Any, Callable

from .. import config
from ..generation.answer import NO_EVIDENCE_TEXT, answer_from_search
from ..llm import LLMClient, LLMConfig, build_client
from ..retrieval.search import search_chunks
from .assess import assess_evidence
from .checkpoint import CheckpointStore, PostgresCheckpointStore
from .state import AgentState, AgentStatus


class AgentError(RuntimeError):
    """Agent 运行过程中的可预期错误。"""


def _normalize_query(text: str) -> str:
    return re.sub(r"[\s，。！？、；：,.!?;:（）()【】\[\]]+", "", text).lower()


def _short(text: str, limit: int = 500) -> str:
    text = text or ""
    return text if len(text) <= limit else text[:limit] + "…"


class AgentRunner:
    """执行一次 Agent 任务，并把每一步写入 checkpoint。"""

    def __init__(
        self,
        *,
        client: LLMClient | None = None,
        llm_config: LLMConfig | None = None,
        store: CheckpointStore | None = None,
        search_fn: Callable[..., dict[str, Any]] | None = None,
        answer_fn: Callable[..., dict[str, Any]] | None = None,
        max_steps: int = 8,
        max_search_rounds: int = 3,
        top_k: int = config.DEFAULT_TOP_K,
        mode: str = "hybrid",
        rerank: bool = False,
        max_evidence: int = 12,
        log_to_db: bool = True,
    ) -> None:
        self.client = client
        self.llm_config = llm_config
        self.store = store or PostgresCheckpointStore()
        self.search_fn = search_fn or search_chunks
        self.answer_fn = answer_fn or answer_from_search
        self.max_steps = max(1, max_steps)
        self.max_search_rounds = max(1, max_search_rounds)
        self.top_k = max(1, top_k)
        self.mode = mode
        self.rerank = rerank
        self.max_evidence = max(1, max_evidence)
        self.log_to_db = log_to_db

    # ------------------------------------------------------------------
    # 入口
    # ------------------------------------------------------------------
    def run(
        self,
        question: str | None = None,
        *,
        resume_run_id: int | None = None,
        max_steps: int | None = None,
        max_search_rounds: int | None = None,
    ) -> dict[str, Any]:
        resumed = False

        if resume_run_id is not None:
            state = self.store.load(resume_run_id)
            if state is None:
                raise AgentError(f"找不到 Agent 运行 #{resume_run_id}")
            if state.status is AgentStatus.COMPLETED:
                return self._result(resume_run_id, state, resumed=True)

            # 续跑也需要大模型；先确认配置可用，避免把状态改成 running
            # 之后才在节点里失败。
            self._get_client()

            # 上一次进程可能停在某个步骤执行到一半的位置。
            self.store.mark_interrupted(resume_run_id)
            state.status = AgentStatus.RUNNING
            state.error = None
            if max_steps is not None:
                state.max_steps = max(1, max_steps)
            if max_search_rounds is not None:
                state.max_search_rounds = max(1, max_search_rounds)
            self.store.save(resume_run_id, state)
            run_id = resume_run_id
            resumed = True
        else:
            if not question or not question.strip():
                raise AgentError("问题不能为空。")
            # 先确认大模型能正常创建，避免先建一条空任务。
            self._get_client()
            state = AgentState(
                question=question.strip(),
                max_steps=self.max_steps,
                max_search_rounds=self.max_search_rounds,
                top_k=self.top_k,
                mode=self.mode,
                rerank=self.rerank,
            )
            run_id = self.store.create_run(state)

        try:
            self._loop(run_id, state)
        except Exception as exc:  # noqa: BLE001
            # 保留当前 phase，方便 resume 时从同一个节点重试。
            state.status = AgentStatus.FAILED
            state.error = str(exc)
            try:
                self.store.save(run_id, state)
            except Exception:
                pass

        return self._result(run_id, state, resumed=resumed)

    def _loop(self, run_id: int, state: AgentState) -> None:
        while state.status is AgentStatus.RUNNING:
            if state.phase == "done":
                state.status = AgentStatus.COMPLETED
                self.store.save(run_id, state)
                return
            if state.step_count >= state.max_steps:
                raise AgentError(
                    f"达到最大步数 {state.max_steps}，已停止。"
                    f"可以运行 `rag agent resume {run_id} --max-steps "
                    f"{state.max_steps + 2}` 继续。"
                )

            if state.phase == "retrieve":
                self._retrieve(run_id, state)
            elif state.phase == "assess":
                self._assess(run_id, state)
            elif state.phase == "answer":
                self._answer(run_id, state)
            else:
                raise AgentError(f"未知的 Agent 阶段：{state.phase}")

    # ------------------------------------------------------------------
    # 三个节点
    # ------------------------------------------------------------------
    def _retrieve(self, run_id: int, state: AgentState) -> None:
        if state.search_rounds >= state.max_search_rounds:
            state.phase = "answer"
            self.store.save(run_id, state)
            return

        query = (state.pending_query or state.question).strip()
        if not query:
            raise AgentError("检索词为空。")
        if any(
            _normalize_query(query) == _normalize_query(old)
            for old in state.queries
        ):
            state.pending_query = None
            state.phase = "assess"
            self.store.save(run_id, state)
            return

        step_no = state.step_count + 1
        step_id = self.store.start_step(
            run_id,
            step_no,
            "retrieve",
            {
                "query": query,
                "round": state.search_rounds + 1,
                "top_k": state.top_k,
                "mode": state.mode,
                "rerank": state.rerank,
            },
        )
        try:
            payload = self.search_fn(
                query,
                top_k=state.top_k,
                mode=state.mode,
                rerank=state.rerank,
            )
        except Exception as exc:  # noqa: BLE001
            self.store.finish_step(step_id, "failed", error=str(exc))
            raise

        if payload.get("error"):
            message = str(payload["error"])
            self.store.finish_step(step_id, "failed", error=message)
            raise AgentError(message)

        seen = {
            str(item.get("chunk_id"))
            for item in state.results
            if item.get("chunk_id")
        }
        added = 0
        merged = list(state.results)
        for item in payload.get("results") or []:
            chunk_id = str(item.get("chunk_id") or "")
            if not chunk_id or chunk_id in seen:
                continue
            copied = dict(item)
            copied["agent_query"] = query
            copied["agent_round"] = state.search_rounds + 1
            merged.append(copied)
            seen.add(chunk_id)
            added += 1

        state.results = merged
        state.search_rounds += 1
        state.queries.append(query)
        state.pending_query = None
        summary = {
            "query": query,
            "round": state.search_rounds,
            "mode": payload.get("mode"),
            "result_count": len(payload.get("results") or []),
            "new_count": added,
            "vector_error": payload.get("vector_error"),
            "rerank_applied": bool(payload.get("rerank_applied")),
        }
        state.searches.append(summary)
        state.phase = "assess"
        state.step_count = step_no
        state.error = None

        self.store.finish_step(step_id, "success", output=summary)
        self.store.save(run_id, state)

    def _assess(self, run_id: int, state: AgentState) -> None:
        if not state.results:
            state.phase = "answer"
            self.store.save(run_id, state)
            return

        step_no = state.step_count + 1
        step_id = self.store.start_step(
            run_id,
            step_no,
            "assess",
            {
                "evidence_count": len(state.results),
                "search_rounds": state.search_rounds,
            },
        )
        try:
            assessment, assessment_error = assess_evidence(
                state.question,
                state.results,
                state.queries,
                client=self._get_client(),
                max_evidence=self.max_evidence,
            )
        except Exception as exc:  # noqa: BLE001
            self.store.finish_step(step_id, "failed", error=str(exc))
            raise

        record = assessment.model_dump()
        record["round"] = state.search_rounds
        if assessment_error:
            record["error"] = assessment_error
        state.assessments.append(record)

        last_new = (
            int(state.searches[-1].get("new_count") or 0)
            if state.searches
            else 0
        )
        if assessment.sufficient:
            state.phase = "answer"
            state.pending_query = None
        elif (
            assessment.next_query
            and state.search_rounds < state.max_search_rounds
            and last_new > 0
        ):
            state.phase = "retrieve"
            state.pending_query = assessment.next_query
        else:
            # 没给出新检索词、补查没有新片段、或者轮数用完了，直接收尾。
            state.phase = "answer"
            state.pending_query = None

        state.step_count = step_no
        state.error = None
        output = {
            "sufficient": assessment.sufficient,
            "missing": assessment.missing,
            "next_query": assessment.next_query,
            "reason": assessment.reason,
        }
        if assessment_error:
            output["error"] = _short(assessment_error, 300)

        self.store.finish_step(step_id, "success", output=output)
        self.store.save(run_id, state)

    def _answer(self, run_id: int, state: AgentState) -> None:
        step_no = state.step_count + 1
        step_id = self.store.start_step(
            run_id,
            step_no,
            "answer",
            {
                "evidence_count": len(state.results),
                "search_rounds": state.search_rounds,
            },
        )
        try:
            evidence = []
            for index, item in enumerate(
                state.results[: self.max_evidence], 1
            ):
                copied = dict(item)
                copied["rank"] = index
                evidence.append(copied)

            search = {
                "query": state.question,
                "mode": state.mode,
                "results": evidence,
                "candidate_count": len(evidence),
                "agent": True,
                "agent_rounds": state.search_rounds,
                "agent_queries": list(state.queries),
            }

            if not evidence:
                payload = {
                    "query": state.question,
                    "answer": NO_EVIDENCE_TEXT,
                    "mode": state.mode,
                    "sources": [],
                    "search": search,
                    "structured": None,
                    "citation_issues": [],
                    "structured_error": None,
                }
            else:
                payload = self.answer_fn(
                    state.question,
                    search,
                    structured=True,
                    config_value=self.llm_config,
                    client=self._get_client(),
                    log_to_db=self.log_to_db,
                )
                if payload.get("error"):
                    raise AgentError(str(payload["error"]))
        except Exception as exc:  # noqa: BLE001
            self.store.finish_step(step_id, "failed", error=str(exc))
            raise

        state.answer = payload
        state.status = AgentStatus.COMPLETED
        state.phase = "done"
        state.pending_query = None
        state.step_count = step_no
        state.error = None

        structured = payload.get("structured") or {}
        output = {
            "answer": _short(str(payload.get("answer") or ""), 500),
            "source_count": len(payload.get("sources") or []),
            "citation_count": len(structured.get("citations") or []),
            "citation_issues": payload.get("citation_issues") or [],
            "structured_error": payload.get("structured_error"),
        }
        self.store.finish_step(step_id, "success", output=output)
        self.store.save(run_id, state)

    # ------------------------------------------------------------------
    def _get_client(self) -> LLMClient:
        if self.client is None:
            self.client = build_client(self.llm_config)
        return self.client

    @staticmethod
    def _result(
        run_id: int,
        state: AgentState,
        *,
        resumed: bool,
    ) -> dict[str, Any]:
        return {
            "run_id": run_id,
            "question": state.question,
            "status": state.status.value,
            "phase": state.phase,
            "step_count": state.step_count,
            "search_rounds": state.search_rounds,
            "queries": list(state.queries),
            "assessments": list(state.assessments),
            "answer": state.answer,
            "error": state.error,
            "resumed": resumed,
        }
