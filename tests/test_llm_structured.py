"""LLM 结构化输出的离线测试，不需要网络或 API key。"""

from __future__ import annotations

import unittest

from pydantic import BaseModel

from rag.generation.answer import validate_citations
from rag.generation.schemas import AnswerCitation, StructuredAnswer
from rag.llm.base import Capabilities, LLMMethodUnsupported, Message
from rag.llm.mock import MockClient
from rag.llm.structured import StructuredRunner


class Demo(BaseModel):
    answer: str
    n: int


class StructuredRunnerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.messages = [Message("user", "测试")]

    def test_json_object_and_fenced_json(self) -> None:
        client = MockClient(
            caps=Capabilities(json_schema=False, json_object=True),
            responses=['```json\n{"answer":"ok","n":1}\n```'],
        )
        result = StructuredRunner(client).run(self.messages, Demo)
        self.assertEqual(result.answer, "ok")
        self.assertEqual(result.n, 1)

    def test_repair_after_invalid_json(self) -> None:
        client = MockClient(
            caps=Capabilities(json_schema=False, json_object=True),
            responses=["not json", '{"answer":"fixed","n":2}'],
        )
        result = StructuredRunner(client).run(self.messages, Demo)
        self.assertEqual(result.answer, "fixed")
        self.assertEqual(len(client.calls), 2)

    def test_unsupported_schema_falls_back(self) -> None:
        class UnsupportedClient(MockClient):
            def chat(self, messages, **kwargs):
                response_format = kwargs.get("response_format") or {}
                if response_format.get("type") == "json_schema":
                    raise LLMMethodUnsupported(
                        "response_format json_schema not supported"
                    )
                return super().chat(messages, **kwargs)

        client = UnsupportedClient(
            caps=Capabilities(json_schema=True, json_object=True),
            responses=['{"answer":"fallback","n":3}'],
        )
        result = StructuredRunner(client).run(self.messages, Demo)
        self.assertEqual(result.answer, "fallback")


class CitationValidationTest(unittest.TestCase):
    def test_valid_quote(self) -> None:
        answer = StructuredAnswer(
            answer="结论 [1]",
            citations=[
                AnswerCitation(source_id=1, quote="第一段原文")
            ],
        )
        valid, issues = validate_citations(
            answer, [{"text": "这是第一段原文内容。"}]
        )
        self.assertEqual(len(valid), 1)
        self.assertEqual(issues, [])

    def test_invalid_source_and_marker(self) -> None:
        answer = StructuredAnswer(
            answer="结论 [2]",
            citations=[AnswerCitation(source_id=9, quote="不存在")],
        )
        valid, issues = validate_citations(
            answer, [{"text": "这是第一段原文内容。"}]
        )
        self.assertEqual(valid, [])
        self.assertTrue(any("不存在" in issue for issue in issues))
        self.assertTrue(any("[2]" in issue for issue in issues))


if __name__ == "__main__":
    unittest.main()
