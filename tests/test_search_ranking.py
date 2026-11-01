"""检索排序辅助逻辑的离线测试。"""

from __future__ import annotations

import unittest

from rag.indexing.keyword_index import bm25_text, index_signature
from rag.retrieval.search import (
    _is_numeric_query,
    _numeric_density,
    _score_boost,
    _select_diverse,
)


class BM25TextFieldTest(unittest.TestCase):
    def test_title_is_not_dumped_into_every_chunk(self) -> None:
        chunk = {
            "paper_title": "UNIQUE TITLE",
            "section_path": ["方法"],
            "text": "正文内容",
        }
        text = bm25_text(chunk)
        self.assertNotIn("UNIQUE TITLE", text)
        self.assertIn("方法", text)
        self.assertIn("正文内容", text)

    def test_signature_changes_when_text_changes(self) -> None:
        first = [{"chunk_id": "c1", "text": "旧正文"}]
        second = [{"chunk_id": "c1", "text": "新正文"}]
        self.assertNotEqual(index_signature(first), index_signature(second))


class ScoreBoostTest(unittest.TestCase):
    def test_numeric_query_detection(self) -> None:
        self.assertTrue(_is_numeric_query("实验一共做了多少组？"))
        self.assertTrue(_is_numeric_query("准确率是多少？"))
        self.assertFalse(_is_numeric_query("这个方法怎么实现？"))

    def test_table_and_numeric_boost(self) -> None:
        table = {"type": "table", "text": "表 1 数据"}
        body = {"type": "body", "text": "准确率 0.92，样本 1234 条"}
        plain = {"type": "body", "text": "这是一段普通正文。"}

        self.assertGreater(_score_boost(table, True), 1.0)
        self.assertGreater(_score_boost(body, True), 1.0)
        self.assertEqual(_score_boost(plain, False), 1.0)
        self.assertGreater(
            _score_boost(table, True), _score_boost(table, False)
        )

    def test_numeric_density(self) -> None:
        self.assertTrue(_numeric_density("准确率 0.92，样本 12345 条，提升 10.5%"))
        self.assertFalse(_numeric_density("这是一段普通正文。"))


class DiversityTest(unittest.TestCase):
    def test_same_paper_is_capped(self) -> None:
        docs = [
            {
                "paper_id": "p1",
                "section_path": [f"章节{i}"],
                "text": f"第 {i} 段",
            }
            for i in range(5)
        ]
        tokens = [[f"t{i}"] for i in range(5)]
        selected = _select_diverse(
            docs, list(range(5)), tokens, limit=5
        )
        self.assertEqual(selected, [0, 1])


if __name__ == "__main__":
    unittest.main()
