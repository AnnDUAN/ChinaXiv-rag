"""LLM 客户端和结构化输出。

这里不依赖 LangChain：
- base.py：消息、能力声明、错误类型
- openai_client.py：直接用 OpenAI SDK 发送请求
- structured.py：JSON Schema → JSON 模式 → 普通文本解析，逐级降级
- config.py：从 config/llm.json 或环境变量创建客户端
- mock.py：离线测试用的假客户端
"""

from .base import (
    Capabilities,
    LLMClient,
    LLMConnectionError,
    LLMError,
    LLMMethodUnsupported,
    Message,
)
from .config import LLMConfig, build_client, load_llm_config
from .mock import MockClient
from .openai_client import OpenAICompatClient
from .structured import StructuredError, StructuredRunner

__all__ = [
    "Capabilities",
    "LLMClient",
    "LLMConnectionError",
    "LLMError",
    "LLMMethodUnsupported",
    "Message",
    "LLMConfig",
    "build_client",
    "load_llm_config",
    "MockClient",
    "OpenAICompatClient",
    "StructuredError",
    "StructuredRunner",
]
