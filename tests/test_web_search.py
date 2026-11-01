"""联网搜索的参数组装和结果清洗测试。"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from rag.generation.answer import build_messages
from rag.web.mcp_search import (
    _build_arguments,
    _extract_items,
    _normalize_results,
    _redact,
)


class WebSearchArgumentTest(unittest.TestCase):
    def test_arguments_follow_tool_schema(self) -> None:
        tool = SimpleNamespace(
            inputSchema={
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer"},
                    "search_depth": {"type": "string"},
                    "include_raw_content": {"type": "boolean"},
                    "unsupported": {"type": "string"},
                }
            }
        )
        args = _build_arguments(tool, "最新消息", 3)
        self.assertEqual(args["query"], "最新消息")
        self.assertEqual(args["max_results"], 3)
        self.assertEqual(args["search_depth"], "basic")
        self.assertTrue(args["include_raw_content"])
        self.assertNotIn("unsupported", args)

    def test_extract_from_structured_content(self) -> None:
        result = SimpleNamespace(
            structuredContent={
                "results": [
                    {
                        "title": "标题",
                        "url": "https://example.com",
                        "content": "正文",
                        "score": 0.8,
                    }
                ]
            },
            content=[],
        )
        items = _extract_items(result)
        self.assertEqual(items[0]["title"], "标题")
        self.assertEqual(items[0]["content"], "正文")

    def test_normalize_and_deduplicate(self) -> None:
        items = [
            {
                "title": "A",
                "url": "https://example.com/a",
                "content": "正文 A",
            },
            {
                "title": "A 重复",
                "url": "https://example.com/a",
                "content": "重复正文",
            },
            {
                "title": "B",
                "url": "https://example.com/b",
                "content": "正文 B",
            },
        ]
        results = _normalize_results(items, limit=5)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["source_type"], "web")
        self.assertEqual(results[0]["url"], "https://example.com/a")

    def test_api_key_is_redacted(self) -> None:
        text = _redact(
            "https://mcp.tavily.com/mcp/?tavilyApiKey=secret&x=1"
        )
        self.assertNotIn("secret", text)
        self.assertIn("tavilyApiKey=***", text)

    def test_web_evidence_is_sent_as_web_source(self) -> None:
        messages = build_messages(
            "问题",
            [
                {
                    "source_type": "web",
                    "title": "网页标题",
                    "url": "https://example.com",
                    "text": "网页正文",
                    "published_at": "2026-10-31",
                    "rank": 1,
                }
            ],
            structured=False,
        )
        user = messages[1].content
        self.assertIn("网页", user)
        self.assertIn("https://example.com", user)
        self.assertIn("网页正文", user)


if __name__ == "__main__":
    unittest.main()
