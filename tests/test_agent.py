"""Agent 流程和 checkpoint 的离线测试。"""

from __future__ import annotations

import json
import unittest
from typing import Any

from rag.agent import AgentRunner
from rag.agent.assess import clean_next_query
from rag.llm import Capabilities, MockClient
from rag.generation.answer import NO_EVIDENCE_TEXT

RESULT_A = {
    "rank": 1,
    "score": 0.9,
    "chunk_id": "chunk-a",
    "paper_id": "paper-a",
    "paper_title": "论文 A",
    "section_path": ["方法"],
    "page_start": 1,
    "page_end": 1,
    "text": "这是第一轮找到的证据。",
}

RESULT_B = {
    "rank": 1,
    "score": 0.8,
    "chunk_id": "chunk-b",
    "paper_id": "paper-b",
    "paper_title": "论文 B",
    "section_path": ["实验"],
    "page_start": 2,
    "page_end": 2,
    "text": "这是第二轮找到的证据。",
}


class MemoryCheckpointStore:
    """测试用内存 checkpoint，行为尽量贴近 PostgreSQL 版本。"""

    def __init__(self) -> None:
        self.runs: dict[int, dict[str, Any]] = {}
        self.steps: dict[int, dict[str, Any]] = {}
        self._next_run = 1
        self._next_step = 1

    def create_run(self, state) -> int:
        run_id = self._next_run
        self._next_run += 1
        self.runs[run_id] = {
            "run_id": run_id,
            "question": state.question,
            "status": state.status.value,
            "phase": state.phase,
            "step_count": state.step_count,
            "state": state.model_copy(deep=True),
            "error": state.error,
        }
        return run_id

    def save(self, run_id: int, state) -> None:
        row = self.runs[run_id]
        row.update(
            {
                "question": state.question,
                "status": state.status.value,
                "phase": state.phase,
                "step_count": state.step_count,
                "state": state.model_copy(deep=True),
                "error": state.error,
            }
        )

    def load(self, run_id: int):
        row = self.runs.get(run_id)
        if row is None:
            return None
        return row["state"].model_copy(deep=True)

    def start_step(
        self,
        run_id: int,
        step_no: int,
        node: str,
        input_payload: dict[str, Any],
    ) -> int:
        for step_id, step in self.steps.items():
            if step["run_id"] == run_id and step["step_no"] == step_no:
                step.update(
                    {
                        "node": node,
                        "status": "running",
                        "input_payload": dict(input_payload),
                        "output_payload": {},
                        "error": None,
                    }
                )
                return step_id
        step_id = self._next_step
        self._next_step += 1
        self.steps[step_id] = {
            "step_id": step_id,
            "run_id": run_id,
            "step_no": step_no,
            "node": node,
            "status": "running",
            "input_payload": dict(input_payload),
            "output_payload": {},
            "error": None,
        }
        return step_id

    def finish_step(
        self,
        step_id: int,
        status: str,
        *,
        output: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        self.steps[step_id].update(
            {
                "status": status,
                "output_payload": dict(output or {}),
                "error": error,
            }
        )

    def mark_interrupted(self, run_id: int) -> int:
        count = 0
        for step in self.steps.values():
            if step["run_id"] == run_id and step["status"] == "running":
                step["status"] = "interrupted"
                count += 1
        return count

    def test_steps(self, run_id: int) -> list[dict[str, Any]]:
        return sorted(
            (
                step
                for step in self.steps.values()
                if step["run_id"] == run_id
            ),
            key=lambda x: x["step_no"],
        )

    def list_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        return list(self.runs.values())[-limit:]

    def trace(self, run_id: int) -> dict[str, Any] | None:
        run = self.runs.get(run_id)
        if run is None:
            return None
        return {
            "run": {
                **run,
                "state": run["state"].model_dump(mode="json"),
            },
            "steps": self.test_steps(run_id),
        }


def assessment(
    sufficient: bool,
    *,
    next_query: str = "",
    missing: str = "",
) -> str:
    return json.dumps(
        {
            "sufficient": sufficient,
            "next_query": next_query,
            "missing": missing,
            "reason": "测试判断",
        },
        ensure_ascii=False,
    )


def answer_json(source_id: int, quote: str) -> str:
    return json.dumps(
        {
            "answer": f"测试答案 [{source_id}]",
            "citations": [{"source_id": source_id, "quote": quote}],
            "insufficient_evidence": False,
            "confidence": "high",
        },
        ensure_ascii=False,
    )


def make_client(responses: list[str]) -> MockClient:
    return MockClient(
        caps=Capabilities(json_schema=False, json_object=True),
        responses=responses,
    )


def make_search():
    calls: list[str] = []

    def search(query: str, **kwargs: Any) -> dict[str, Any]:
        calls.append(query)
        if query == "空问题":
            results: list[dict[str, Any]] = []
        elif "补充" in query:
            results = [dict(RESULT_B)]
        else:
            results = [dict(RESULT_A)]
        return {
            "query": query,
            "mode": "hybrid",
            "results": results,
            "rerank_applied": False,
            "vector_error": None,
        }

    return search, calls


class AgentFlowTest(unittest.TestCase):
    def test_sufficient_after_first_search(self) -> None:
        store = MemoryCheckpointStore()
        search, calls = make_search()
        client = make_client(
            [assessment(True), answer_json(1, RESULT_A["text"])]
        )
        runner = AgentRunner(
            client=client,
            store=store,
            search_fn=search,
            log_to_db=False,
        )

        result = runner.run("初始问题")

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["search_rounds"], 1)
        self.assertEqual(result["queries"], ["初始问题"])
        self.assertEqual(result["answer"]["citation_issues"], [])
        self.assertEqual(calls, ["初始问题"])
        self.assertEqual(
            [s["node"] for s in store.test_steps(result["run_id"])],
            ["retrieve", "assess", "answer"],
        )
        self.assertEqual(len(client.calls), 2)

    def test_insufficient_triggers_second_search(self) -> None:
        store = MemoryCheckpointStore()
        search, calls = make_search()
        client = make_client(
            [
                assessment(False, next_query="补充问题", missing="缺少实验"),
                assessment(True),
                answer_json(2, RESULT_B["text"]),
            ]
        )
        runner = AgentRunner(
            client=client,
            store=store,
            search_fn=search,
            log_to_db=False,
        )

        result = runner.run("初始问题")

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["search_rounds"], 2)
        self.assertEqual(result["queries"], ["初始问题", "补充问题"])
        self.assertEqual(len(result["answer"]["search"]["results"]), 2)
        self.assertEqual(result["answer"]["citation_issues"], [])
        self.assertEqual(calls, ["初始问题", "补充问题"])
        self.assertEqual(result["step_count"], 5)
        self.assertEqual(
            [s["node"] for s in store.test_steps(result["run_id"])],
            ["retrieve", "assess", "retrieve", "assess", "answer"],
        )

    def test_repeated_query_does_not_loop(self) -> None:
        store = MemoryCheckpointStore()
        search, calls = make_search()
        client = make_client(
            [assessment(False, next_query="初始问题"), answer_json(1, RESULT_A["text"])]
        )
        runner = AgentRunner(
            client=client,
            store=store,
            search_fn=search,
            log_to_db=False,
        )

        result = runner.run("初始问题")

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["search_rounds"], 1)
        self.assertEqual(calls, ["初始问题"])
        self.assertEqual(result["assessments"][0]["next_query"], "")

    def test_no_results_returns_no_evidence_answer(self) -> None:
        store = MemoryCheckpointStore()
        search, calls = make_search()
        client = make_client([])
        runner = AgentRunner(
            client=client,
            store=store,
            search_fn=search,
            log_to_db=False,
        )

        result = runner.run("空问题")

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["answer"]["answer"], NO_EVIDENCE_TEXT)
        self.assertEqual(calls, ["空问题"])
        self.assertEqual(len(client.calls), 0)
        self.assertEqual(
            [s["node"] for s in store.test_steps(result["run_id"])],
            ["retrieve", "answer"],
        )

    def test_resume_from_checkpoint(self) -> None:
        store = MemoryCheckpointStore()
        search, calls = make_search()

        first_client = make_client(
            [assessment(False, next_query="补充问题", missing="缺少实验")]
        )
        first_runner = AgentRunner(
            client=first_client,
            store=store,
            search_fn=search,
            max_steps=2,
            log_to_db=False,
        )
        first = first_runner.run("初始问题")
        self.assertEqual(first["status"], "failed")
        self.assertIn("最大步数", first["error"])
        self.assertEqual(first["phase"], "retrieve")

        second_client = make_client(
            [assessment(True), answer_json(2, RESULT_B["text"])]
        )
        second_runner = AgentRunner(
            client=second_client,
            store=store,
            search_fn=search,
            log_to_db=False,
        )
        second = second_runner.run(
            resume_run_id=first["run_id"], max_steps=8
        )

        self.assertTrue(second["resumed"])
        self.assertEqual(second["run_id"], first["run_id"])
        self.assertEqual(second["status"], "completed")
        self.assertEqual(second["queries"], ["初始问题", "补充问题"])
        self.assertEqual(calls, ["初始问题", "补充问题"])
        self.assertEqual(second["answer"]["citation_issues"], [])

    def test_trace_contains_steps(self) -> None:
        store = MemoryCheckpointStore()
        search, _ = make_search()
        client = make_client(
            [assessment(True), answer_json(1, RESULT_A["text"])]
        )
        runner = AgentRunner(
            client=client,
            store=store,
            search_fn=search,
            log_to_db=False,
        )
        result = runner.run("初始问题")
        trace = store.trace(result["run_id"])

        self.assertIsNotNone(trace)
        self.assertEqual(len(trace["steps"]), 3)
        self.assertEqual(trace["steps"][0]["node"], "retrieve")


class QueryCleanTest(unittest.TestCase):
    def test_duplicate_and_empty_queries_are_removed(self) -> None:
        self.assertEqual(clean_next_query("", ["初始问题"]), "")
        self.assertEqual(
            clean_next_query("初始，问题", ["初始问题"]), ""
        )
        self.assertEqual(
            clean_next_query("换一个角度", ["初始问题"]), "换一个角度"
        )
        self.assertEqual(
            clean_next_query("长" * 121, ["初始问题"]), ""
        )


if __name__ == "__main__":
    unittest.main()
