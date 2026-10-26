"""LLM 传输层的最小接口。

这一层只关心“把消息发出去，把文本拿回来”，
不负责提示词、检索或业务校验。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Literal, Protocol, TypeVar

from pydantic import BaseModel

TModel = TypeVar("TModel", bound=BaseModel)


@dataclass(frozen=True)
class Capabilities:
    """模型/接口支持哪些调用方式。

    这里不做自动探测：一次网络抖动就可能被误记成“不支持”，
    之后再也走不到这个模式。能力由配置显式声明。
    """

    json_schema: bool = False
    json_object: bool = False
    tool_calling: bool = False
    streaming: bool = True
    max_context: int = 32_000


@dataclass(frozen=True)
class Message:
    """一条对话消息。"""

    role: Literal["system", "user", "assistant", "tool"]
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


class LLMClient(Protocol):
    """纯传输层接口。"""

    name: str
    caps: Capabilities

    def chat(self, messages: list[Message], **kwargs: Any) -> str: ...

    def stream(self, messages: list[Message], **kwargs: Any) -> Iterator[str]: ...


class LLMError(Exception):
    """本项目所有 LLM 错误的基类。"""


class LLMConnectionError(LLMError):
    """网络、超时或限流。"""


class LLMMethodUnsupported(LLMError):
    """当前接口不支持这种调用方式，例如 response_format。"""
