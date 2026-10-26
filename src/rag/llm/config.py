"""读取 LLM 配置并创建客户端。

配置优先级：
1. 环境变量 RAG_LLM_*；
2. config/llm.json。

兼容 OpenAI 的服务都可以用：DeepSeek、通义千问兼容模式、
OpenAI、本地 vLLM / Ollama 等。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from .. import config
from .base import Capabilities
from .openai_client import OpenAICompatClient


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    api_key: str
    model: str
    timeout: float = 120.0
    temperature: float = 0.2
    capabilities: Capabilities = Capabilities(
        json_schema=True,
        json_object=True,
    )

    def build(self) -> OpenAICompatClient:
        return OpenAICompatClient(
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model,
            caps=self.capabilities,
            default_temperature=self.temperature,
            timeout=self.timeout,
        )


def _as_bool(value: object, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off", ""}
    return bool(value)


def _env_bool(name: str, default: bool) -> bool:
    return _as_bool(os.environ.get(name), default)


def load_llm_config(path: Path | None = None) -> LLMConfig:
    """读取配置；环境变量优先于 config/llm.json。"""
    path = path or config.LLM_CONFIG
    data: dict = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise RuntimeError(f"配置文件格式不对：{path} ({exc})") from exc
    elif not (
        os.environ.get("RAG_LLM_BASE_URL")
        and os.environ.get("RAG_LLM_MODEL")
    ):
        raise RuntimeError(
            "还没有配置大模型。请编辑 config/llm.json，"
            "或设置 RAG_LLM_BASE_URL / RAG_LLM_API_KEY / RAG_LLM_MODEL。"
        )

    base_url = (
        os.environ.get("RAG_LLM_BASE_URL") or str(data.get("base_url", ""))
    ).strip()
    api_key = (
        os.environ.get("RAG_LLM_API_KEY") or str(data.get("api_key", ""))
    ).strip()
    model = (
        os.environ.get("RAG_LLM_MODEL") or str(data.get("model", ""))
    ).strip()

    timeout = float(
        os.environ.get("RAG_LLM_TIMEOUT") or data.get("timeout") or 120
    )
    temperature = float(
        os.environ.get("RAG_LLM_TEMPERATURE")
        or data.get("temperature")
        or 0.2
    )

    caps_data = data.get("capabilities") or {}
    capabilities = Capabilities(
        json_schema=_env_bool(
            "RAG_LLM_JSON_SCHEMA",
            _as_bool(caps_data.get("json_schema"), True),
        ),
        json_object=_env_bool(
            "RAG_LLM_JSON_OBJECT",
            _as_bool(caps_data.get("json_object"), True),
        ),
        tool_calling=_as_bool(
            caps_data.get("tool_calling"), False
        ),
        streaming=_as_bool(caps_data.get("streaming"), True),
        max_context=int(caps_data.get("max_context", 32_000)),
    )

    if not base_url or not model:
        raise RuntimeError(
            "还没有配置完整的大模型信息。请编辑 config/llm.json，"
            "填写 base_url、api_key、model；"
            "也可以设置 RAG_LLM_BASE_URL / RAG_LLM_API_KEY / RAG_LLM_MODEL。"
        )
    if not api_key:
        raise RuntimeError(
            "还没有配置 API key。请在 config/llm.json 填写 api_key；"
            "本地服务如果不校验 key，可以填 EMPTY。"
        )

    return LLMConfig(
        base_url=base_url.rstrip("/"),
        api_key=api_key,
        model=model,
        timeout=timeout,
        temperature=temperature,
        capabilities=capabilities,
    )


def build_client(config_value: LLMConfig | None = None) -> OpenAICompatClient:
    return (config_value or load_llm_config()).build()
