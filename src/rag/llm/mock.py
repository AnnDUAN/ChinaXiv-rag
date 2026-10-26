"""离线测试用的假客户端。"""

from __future__ import annotations

from typing import Any, Iterator

from .base import Capabilities, Message


class MockClient:
    """按顺序返回预设文本，方便测试结构化输出的降级和修复。"""

    def __init__(
        self,
        *,
        name: str = "mock",
        caps: Capabilities | None = None,
        responses: list[str] | None = None,
        default: str = "[mock] 未配置回复",
    ) -> None:
        self.name = name
        self.caps = caps or Capabilities(
            json_schema=True, json_object=True, streaming=True
        )
        self._responses = list(responses or [])
        self._default = default
        self.calls: list[list[Message]] = []
        self.kwargs: list[dict[str, Any]] = []

    def chat(self, messages: list[Message], **kwargs: Any) -> str:
        self.calls.append(list(messages))
        self.kwargs.append(dict(kwargs))
        if self._responses:
            return self._responses.pop(0)
        return self._default

    def stream(self, messages: list[Message], **kwargs: Any) -> Iterator[str]:
        yield self.chat(messages, **kwargs)
