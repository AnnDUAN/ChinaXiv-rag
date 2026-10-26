"""结构化输出：按能力逐级降级，并在格式错误时尝试修复。

调用方只需要给一个 Pydantic 模型：

    runner = StructuredRunner(client)
    result = runner.run(messages, MySchema)

尝试顺序：
1. json_schema：把 Pydantic 生成的 JSON Schema 直接交给接口；
2. json_object：要求接口只返回 JSON 对象，并在提示词里附上字段说明；
3. free_text：普通文本返回，再从 ```json 或大括号里提取 JSON。

如果模型返回的 JSON 结构不对，会用错误信息再请求一次；
如果接口明确表示不支持某种 response_format，就记住并换下一种方式。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Sequence, TypeVar

from pydantic import BaseModel, ValidationError

from .base import (
    LLMClient,
    LLMConnectionError,
    LLMMethodUnsupported,
    Message,
)

TModel = TypeVar("TModel", bound=BaseModel)
DEFAULT_METHODS = ("json_schema", "json_object", "free_text")

# 同一个进程里，某个客户端一旦明确说不支持某种方式，就不再重复尝试。
_UNSUPPORTED: dict[str, set[str]] = {}


@dataclass(frozen=True)
class StructuredAttempt:
    method: str
    detail: str


class StructuredError(RuntimeError):
    """所有结构化方式都失败了。"""

    def __init__(self, attempts: Sequence[StructuredAttempt]) -> None:
        self.attempts = list(attempts)
        lines = "\n".join(
            f"  - [{a.method}] {a.detail[:150]}" for a in self.attempts
        )
        super().__init__(
            f"结构化输出失败，共 {len(self.attempts)} 次尝试：\n{lines}"
        )


class StructuredRunner:
    """在任意 LLMClient 上提供结构化输出。"""

    def __init__(
        self,
        client: LLMClient,
        *,
        methods: Sequence[str] = DEFAULT_METHODS,
        max_repair: int = 1,
        error_truncate: int = 600,
    ) -> None:
        self._client = client
        self._methods = tuple(methods)
        self._max_repair = max_repair
        self._error_truncate = error_truncate

    def run(self, messages: list[Message], schema: type[TModel]) -> TModel:
        attempts: list[StructuredAttempt] = []
        unsupported = _UNSUPPORTED.setdefault(self._client.name, set())

        for method in self._methods:
            if method in unsupported or not self._declared(method):
                continue

            call = getattr(self, f"_call_{method}")
            for round_no in range(self._max_repair + 1):
                detail = attempts[-1].detail if round_no > 0 else None
                try:
                    return call(messages, schema, detail)
                except LLMConnectionError:
                    # 网络问题换结构化方式也没用，直接交给上层。
                    raise
                except Exception as exc:  # noqa: BLE001
                    attempts.append(
                        StructuredAttempt(
                            method, self._truncate(str(exc))
                        )
                    )
                    if self._looks_like_unsupported(exc):
                        unsupported.add(method)
                        break
                    if not self._is_repairable(exc):
                        break

        raise StructuredError(attempts)

    # ------------------------------------------------------------------
    # 三种调用方式
    # ------------------------------------------------------------------
    def _call_json_schema(
        self,
        messages: list[Message],
        schema: type[TModel],
        detail: str | None,
    ) -> TModel:
        msgs = self._with_hint(messages, detail)
        raw = self._client.chat(
            msgs,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "schema": schema.model_json_schema(),
                    "strict": True,
                },
            },
        )
        return schema.model_validate_json(raw)

    def _call_json_object(
        self,
        messages: list[Message],
        schema: type[TModel],
        detail: str | None,
    ) -> TModel:
        msgs = self._with_spec(messages, schema, detail)
        raw = self._client.chat(
            msgs,
            response_format={"type": "json_object"},
        )
        return self._parse_loose(raw, schema)

    def _call_free_text(
        self,
        messages: list[Message],
        schema: type[TModel],
        detail: str | None,
    ) -> TModel:
        msgs = self._with_spec(messages, schema, detail)
        raw = self._client.chat(msgs)
        return self._parse_loose(raw, schema)

    # ------------------------------------------------------------------
    # 提示词和解析
    # ------------------------------------------------------------------
    @staticmethod
    def _with_hint(
        messages: list[Message], detail: str | None
    ) -> list[Message]:
        if not detail:
            return list(messages)
        return [
            Message(
                "system",
                "你上一次的输出无法被解析为符合要求的 JSON，"
                f"错误如下：\n{detail}\n请修正后重新输出。",
            ),
            *messages,
        ]

    @staticmethod
    def _with_spec(
        messages: list[Message],
        schema: type[TModel],
        detail: str | None,
    ) -> list[Message]:
        spec = [
            "你必须只输出一个合法的 JSON 对象。",
            "不要 markdown 代码块，不要任何解释性文字。",
            f"JSON 的字段必须且只能是：{list(schema.model_fields)}",
            "字段定义：",
            json.dumps(
                schema.model_json_schema(), ensure_ascii=False, indent=2
            ),
            "不要编造材料里没有的内容；材料不足时按 schema 如实填写，"
            "例如把 insufficient_evidence 设为 true、citations 设为空数组。",
        ]
        if detail:
            spec += [
                "",
                "你上一次输出有问题，错误如下：",
                detail,
                "请修正后重新输出。",
            ]
        return [Message("system", "\n".join(spec)), *messages]

    @staticmethod
    def _parse_loose(raw: str, schema: type[TModel]) -> TModel:
        text = (raw or "").strip()
        fence = re.search(r"```(?:json)?\s*(.+?)\s*```", text, re.S)
        if fence:
            text = fence.group(1).strip()

        # 优先直接解析；失败再截取第一个 { 到最后一个 }。
        try:
            return schema.model_validate_json(text)
        except ValidationError:
            pass

        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            text = text[start : end + 1]
        return schema.model_validate_json(text)

    # ------------------------------------------------------------------
    # 能力与错误判断
    # ------------------------------------------------------------------
    def _declared(self, method: str) -> bool:
        caps = self._client.caps
        if method == "json_schema":
            return caps.json_schema
        if method == "json_object":
            return caps.json_object
        return True

    @staticmethod
    def _looks_like_unsupported(exc: Exception) -> bool:
        if isinstance(exc, LLMMethodUnsupported):
            return True
        msg = str(exc).lower()
        mentions_param = (
            "response_format" in msg
            or "json_schema" in msg
            or "json_object" in msg
        )
        if not mentions_param:
            return False
        return any(
            key in msg
            for key in (
                "not supported",
                "unsupported",
                "unknown",
                "unrecognized",
                "invalid parameter",
            )
        )

    @staticmethod
    def _is_repairable(exc: Exception) -> bool:
        return isinstance(exc, (ValidationError, json.JSONDecodeError))

    def _truncate(self, text: str) -> str:
        if len(text) <= self._error_truncate:
            return text
        return (
            f"{text[:self._error_truncate]}"
            f"...(已截断，原长 {len(text)} 字)"
        )
