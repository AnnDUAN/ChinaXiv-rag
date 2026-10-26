"""OpenAI 兼容客户端。

直接用 OpenAI SDK 的 `OpenAI` 类，不经过 LangChain。
DeepSeek、通义千问兼容模式、OpenAI、本地 vLLM/Ollama 都可以接。
"""

from __future__ import annotations

from typing import Any, Iterator

from openai import (
    APIConnectionError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)

from .base import (
    Capabilities,
    LLMConnectionError,
    LLMError,
    Message,
)

_TRANSIENT = (APIConnectionError, APITimeoutError, RateLimitError)


def _content_text(content: Any) -> str:
    """OpenAI 的 content 可能是字符串，也可能是分段列表。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text") or part.get("content")
                if text:
                    parts.append(str(text))
            else:
                text = getattr(part, "text", None)
                if text:
                    parts.append(str(text))
        return "".join(parts)
    return str(content)


class OpenAICompatClient:
    """实现 LLMClient 协议的 OpenAI 兼容客户端。"""

    def __init__(
        self,
        *,
        name: str = "openai_compat",
        base_url: str,
        api_key: str,
        model: str,
        caps: Capabilities | None = None,
        default_temperature: float = 0.2,
        timeout: float = 120.0,
    ) -> None:
        if not base_url:
            raise ValueError("base_url 为空")
        if not api_key:
            raise ValueError("api_key 为空")
        if not model:
            raise ValueError("model 为空")

        self.name = name
        self.caps = caps or Capabilities()
        self._model = model
        self._default_temperature = default_temperature
        clean_base = base_url.rstrip("/")
        # 兼容旧配置：如果填到了 /chat/completions，SDK 自己会再拼一次。
        suffix = "/chat/completions"
        if clean_base.endswith(suffix):
            clean_base = clean_base[: -len(suffix)].rstrip("/")
        self._client = OpenAI(
            base_url=clean_base,
            api_key=api_key,
            timeout=timeout,
        )

    def chat(self, messages: list[Message], **kwargs: Any) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[m.to_dict() for m in messages],
                **self._params(kwargs),
            )
        except _TRANSIENT as exc:
            raise LLMConnectionError(f"{self.name}: {exc}") from exc
        except Exception as exc:
            raise LLMError(f"{self.name}: {exc}") from exc

        choices = response.choices or []
        if not choices:
            raise LLMError(f"{self.name}: 接口没有返回 choices")
        return _content_text(choices[0].message.content)

    def stream(self, messages: list[Message], **kwargs: Any) -> Iterator[str]:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[m.to_dict() for m in messages],
                stream=True,
                **self._params(kwargs),
            )
            for chunk in response:
                choices = chunk.choices or []
                if not choices:
                    continue
                text = _content_text(choices[0].delta.content)
                if text:
                    yield text
        except _TRANSIENT as exc:
            raise LLMConnectionError(f"{self.name}: {exc}") from exc
        except Exception as exc:
            raise LLMError(f"{self.name}: {exc}") from exc

    def _params(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        params = dict(kwargs)
        params.setdefault("temperature", self._default_temperature)
        return params
