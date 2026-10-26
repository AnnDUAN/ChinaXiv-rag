"""长期记忆的离线单元测试。"""

from __future__ import annotations

import json
import unittest
from datetime import date

from rag.llm.base import Capabilities
from rag.llm.mock import MockClient
from rag.memory.extract import extract_candidates
from rag.memory.gate import GateConfig, GateDecision, decide
from rag.memory.models import MemoryCandidate, MemoryItem, MemoryKind, MemoryStatus
from rag.memory.recall import RecalledMemory, format_memories, pack


def item(
    *,
    content: str = "用户偏好简洁回答",
    kind: MemoryKind = MemoryKind.PREFERENCE,
    embedding: list[float] | None = None,
    importance: int = 8,
) -> MemoryItem:
    return MemoryItem(
        id=1,
        namespace="default",
        subject="user:self",
        kind=kind,
        content=content,
        importance=importance,
        confidence=0.9,
        status=MemoryStatus.ACTIVE,
        is_speculation=False,
        fingerprint="x",
        t_valid_from=date.today(),
        embedding=embedding or [1.0, 0.0],
    )


class GateTest(unittest.TestCase):
    def test_same_embedding_is_duplicate(self) -> None:
        candidate = MemoryCandidate(
            kind=MemoryKind.PREFERENCE,
            content="用户希望回答简短",
            embedding=[1.0, 0.0],
            importance=8,
        )
        result = decide(candidate, [item()])
        self.assertEqual(result.decision, GateDecision.DUPLICATE)

    def test_low_importance_is_rejected(self) -> None:
        candidate = MemoryCandidate(
            kind=MemoryKind.EVENT,
            content="今天天气不错",
            importance=2,
        )
        result = decide(candidate, [])
        self.assertEqual(result.decision, GateDecision.REJECT)

    def test_similar_same_kind_is_update(self) -> None:
        candidate = MemoryCandidate(
            kind=MemoryKind.PREFERENCE,
            content="用户偏好非常简洁的回答",
            embedding=[0.88, 0.48],
            importance=8,
        )
        result = decide(
            candidate,
            [item(embedding=[1.0, 0.0])],
            config=GateConfig(update_cosine=0.80),
        )
        self.assertEqual(result.decision, GateDecision.UPDATE)

    def test_different_kind_can_be_new(self) -> None:
        candidate = MemoryCandidate(
            kind=MemoryKind.EVENT,
            content="用户上周去了杭州",
            embedding=[0.7, 0.7],
            importance=6,
        )
        result = decide(candidate, [item(kind=MemoryKind.PREFERENCE)])
        self.assertEqual(result.decision, GateDecision.NEW)


class PackTest(unittest.TestCase):
    def test_pack_respects_budget(self) -> None:
        rows = [
            RecalledMemory(item=item(content="第一条"), score=0.9, relevance=0.9),
            RecalledMemory(item=item(content="第二条"), score=0.8, relevance=0.8),
            RecalledMemory(item=item(content="第三条"), score=0.7, relevance=0.7),
        ]
        packed = pack(rows, budget_chars=6)
        self.assertEqual([r.item.content for r in packed], ["第一条", "第二条"])

    def test_format(self) -> None:
        text = format_memories(
            [RecalledMemory(item=item(), score=0.8, relevance=0.8)]
        )
        self.assertIn("【长期记忆】", text)
        self.assertIn("用户偏好简洁回答", text)


class ExtractTest(unittest.TestCase):
    def test_extract_from_mock_llm(self) -> None:
        payload = {
            "items": [
                {
                    "kind": "preference",
                    "content": "用户偏好简洁的中文回答",
                    "importance": 8,
                    "confidence": 0.9,
                    "is_speculation": False,
                    "valid_from": None,
                }
            ]
        }
        client = MockClient(
            caps=Capabilities(json_schema=False, json_object=True),
            responses=[json.dumps(payload, ensure_ascii=False)],
        )
        candidates = extract_candidates(
            "用户：以后请用简洁的中文回答。",
            client=client,
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].kind, MemoryKind.PREFERENCE)
        self.assertEqual(candidates[0].importance, 8)


if __name__ == "__main__":
    unittest.main()
