"""Agent 新流程和 checkpoint 的离线测试。"""

from __future__ import annotations

import json
import unittest
from typing import Any

from rag.agent import (
    AgentRunner,
    QueryAnalysis,
    analyze_question,
    clean_analysis,
    infer_intent_hint,
)
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
    "text": "这是子查询找到的证据。",
}

RESULT_C = {
    "rank": 1,
    "score": 0.7,
    "chunk_id": "chunk-c",
    "paper_id": "paper-c",
    "paper_title": "论文 C",
    "section_path": ["讨论"],
    "page_start": 3,
    "page_end": 3,
    "text": "这是另一个子查询找到的证据。",
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


def analysis_json(
    intent: str,
    rewritten_query: str,
    *,
    sub_queries: list[str] | None = None,
) -> str:
    return json.dumps(
        {
            "intent": intent,
            "rewritten_query": rewritten_query,
            "sub_queries": sub_queries or [],
            "reason": "测试分析",
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


def make_search(mapping: dict[str, list[dict[str, Any]]] | None = None):
    calls: list[str] = []
    mapping = mapping or {}

    def search(query: str, **kwargs: Any) -> dict[str, Any]:
        calls.append(query)
        if query == "空问题":
            results: list[dict[str, Any]] = []
        else:
            results = [
                dict(item)
                for item in mapping.get(query, [RESULT_A])
            ]
        return {
            "query": query,
            "mode": "hybrid",
            "results": results,
            "rerank_applied": False,
            "vector_error": None,
        }

    return search, calls


class AgentFlowTest(unittest.TestCase):
    def test_analysis_rewrites_and_searches_once(self) -> None:
        store = MemoryCheckpointStore()
        search, calls = make_search()
        client = make_client(
            [
                analysis_json(
                    "definition",
                    "双沙漏 细腰 智能生态网络",
                ),
                answer_json(1, RESULT_A["text"]),
            ]
        )
        runner = AgentRunner(
            client=client,
            store=store,
            search_fn=search,
            log_to_db=False,
        )

        result = runner.run("双沙漏模型是什么？")

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["search_rounds"], 1)
        self.assertEqual(result["queries"], ["双沙漏 细腰 智能生态网络"])
        self.assertEqual(result["analysis"]["intent"], "definition")
        self.assertEqual(calls, ["双沙漏 细腰 智能生态网络"])
        self.assertEqual(result["step_count"], 3)
        self.assertEqual(
            [s["node"] for s in store.test_steps(result["run_id"])],
            ["analyze", "retrieve", "answer"],
        )
        self.assertEqual(len(client.calls), 2)

    def test_subqueries_are_all_searched_in_one_step(self) -> None:
        store = MemoryCheckpointStore()
        search, calls = make_search(
            {
                "幻觉治理": [dict(RESULT_A)],
                "元认知门控": [dict(RESULT_B)],
                "系统性证伪": [dict(RESULT_C)],
            }
        )
        client = make_client(
            [
                analysis_json(
                    "multi_paper",
                    "幻觉治理",
                    sub_queries=["元认知门控", "系统性证伪"],
                ),
                answer_json(2, RESULT_B["text"]),
            ]
        )
        runner = AgentRunner(
            client=client,
            store=store,
            search_fn=search,
            log_to_db=False,
        )

        result = runner.run("这些论文里，幻觉治理有哪些路线？")

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["search_rounds"], 3)
        self.assertEqual(
            result["queries"],
            ["幻觉治理", "元认知门控", "系统性证伪"],
        )
        self.assertEqual(calls, result["queries"])
        self.assertEqual(len(result["searches"]), 3)
        self.assertEqual(
            [s["node"] for s in store.test_steps(result["run_id"])],
            ["analyze", "retrieve", "answer"],
        )
        self.assertEqual(len(result["answer"]["search"]["results"]), 3)

    def test_duplicate_subquery_is_removed(self) -> None:
        store = MemoryCheckpointStore()
        search, calls = make_search()
        client = make_client(
            [
                analysis_json(
                    "fact",
                    "初始问题",
                    sub_queries=["初始问题", "初始，问题", "补充角度"],
                ),
                answer_json(1, RESULT_A["text"]),
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
        self.assertEqual(result["queries"], ["初始问题", "补充角度"])
        self.assertEqual(calls, ["初始问题", "补充角度"])

    def test_no_results_returns_no_evidence_answer(self) -> None:
        store = MemoryCheckpointStore()
        search, calls = make_search()
        client = make_client(
            [analysis_json("fact", "空问题")]
        )
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
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(
            [s["node"] for s in store.test_steps(result["run_id"])],
            ["analyze", "retrieve", "answer"],
        )

    def test_resume_from_checkpoint(self) -> None:
        store = MemoryCheckpointStore()
        search, calls = make_search()

        first_client = make_client(
            [analysis_json("fact", "改写后的问题")]
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
        self.assertEqual(first["phase"], "answer")
        self.assertEqual(first["queries"], ["改写后的问题"])

        second_client = make_client(
            [answer_json(1, RESULT_A["text"])]
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
        self.assertEqual(second["queries"], ["改写后的问题"])
        self.assertEqual(calls, ["改写后的问题"])
        self.assertEqual(second["answer"]["citation_issues"], [])
        self.assertEqual(
            [s["node"] for s in store.test_steps(second["run_id"])],
            ["analyze", "retrieve", "answer"],
        )

    def test_trace_contains_three_steps(self) -> None:
        store = MemoryCheckpointStore()
        search, _ = make_search()
        client = make_client(
            [
                analysis_json("fact", "改写后的问题"),
                answer_json(1, RESULT_A["text"]),
            ]
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
        self.assertEqual(
            [s["node"] for s in trace["steps"]],
            ["analyze", "retrieve", "answer"],
        )
        self.assertEqual(len(trace["steps"]), 3)


class AnalysisTest(unittest.TestCase):
    def test_infer_intent_hint(self) -> None:
        self.assertEqual(
            infer_intent_hint("这些论文里有哪些不同路线？"),
            "multi_paper",
        )
        self.assertEqual(
            infer_intent_hint("两种方法有什么区别？"),
            "comparison",
        )
        self.assertEqual(
            infer_intent_hint("实验一共做了多少组？"),
            "numeric",
        )
        self.assertEqual(
            infer_intent_hint("双沙漏模型是什么？"),
            "definition",
        )
        self.assertEqual(
            infer_intent_hint("这个方法怎么实现？"),
            "fact",
        )
        self.assertEqual(infer_intent_hint("你好"), "other")

    def test_clean_analysis_limits_and_deduplicates(self) -> None:
        fact = QueryAnalysis(
            intent="fact",
            rewritten_query="A",
            sub_queries=["B", "C", "A"],
        )
        cleaned = clean_analysis(fact, "原问题")
        self.assertEqual(cleaned.rewritten_query, "A")
        self.assertEqual(cleaned.sub_queries, ["B"])

        multi = QueryAnalysis(
            intent="multi_paper",
            rewritten_query="A",
            sub_queries=["B", "C", "D", "E", "B"],
        )
        cleaned = clean_analysis(multi, "原问题")
        self.assertEqual(cleaned.sub_queries, ["B", "C", "D"])

    def test_analysis_failure_falls_back_to_original(self) -> None:
        client = MockClient(
            caps=Capabilities(json_schema=False, json_object=True),
            responses=["不是 JSON"] * 3,
        )
        analysis, error = analyze_question(
            "这些论文里有哪些不同路线？",
            client=client,
        )
        self.assertIsNotNone(error)
        self.assertEqual(analysis.intent, "multi_paper")
        self.assertEqual(analysis.rewritten_query, "这些论文里有哪些不同路线？")
        self.assertEqual(analysis.sub_queries, [])


if __name__ == "__main__":
    unittest.main()
