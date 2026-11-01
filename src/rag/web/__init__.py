"""联网搜索：通过外部 MCP 服务搜索网页。"""

from .mcp_search import (
    MCPWebSearch,
    WebSearchConfig,
    WebSearchError,
    load_web_search_config,
    web_search_available,
)

__all__ = [
    "MCPWebSearch",
    "WebSearchConfig",
    "WebSearchError",
    "load_web_search_config",
    "web_search_available",
]
