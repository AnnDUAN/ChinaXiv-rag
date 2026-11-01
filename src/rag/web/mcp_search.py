"""通过 MCP 调用外部联网搜索，默认适配 Tavily 远程 MCP。

网页结果只返回给本次运行使用，不写入本地论文索引或数据库。
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from .. import config


class WebSearchError(RuntimeError):
    """联网搜索不可用或调用失败。"""


@dataclass(frozen=True)
class WebSearchConfig:
    url: str
    tool: str = ""
    timeout: float = 20.0
    max_results: int = 5


def _read_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise WebSearchError(f"联网搜索配置格式不对：{path}") from exc


def load_web_search_config(path: Path | None = None) -> WebSearchConfig:
    """读取 MCP 配置；环境变量优先于 config/web_search.json。"""
    data = _read_file(path or config.WEB_SEARCH_CONFIG)
    url = (os.environ.get("RAG_WEB_SEARCH_URL") or data.get("url") or "").strip()
    api_key = (
        os.environ.get("TAVILY_API_KEY") or data.get("api_key") or ""
    ).strip()

    if not url and api_key:
        url = (
            "https://mcp.tavily.com/mcp/?tavilyApiKey="
            + quote(api_key, safe="")
        )
    if not url:
        raise WebSearchError(
            "还没有配置联网搜索。请复制 config/web_search.example.json "
            "为 config/web_search.json，或设置 RAG_WEB_SEARCH_URL / "
            "TAVILY_API_KEY。"
        )

    tool = (
        os.environ.get("RAG_WEB_SEARCH_TOOL")
        or data.get("tool")
        or "tavily-search"
    ).strip()
    timeout = float(
        os.environ.get("RAG_WEB_SEARCH_TIMEOUT")
        or data.get("timeout")
        or 20
    )
    max_results = int(
        os.environ.get("RAG_WEB_SEARCH_MAX_RESULTS")
        or data.get("max_results")
        or 5
    )
    return WebSearchConfig(
        url=url,
        tool=tool,
        timeout=max(1.0, timeout),
        max_results=max(1, max_results),
    )


def web_search_available() -> bool:
    """只检查配置是否存在，不发起网络请求。"""
    try:
        load_web_search_config()
    except WebSearchError:
        return False
    return True


def _properties(tool: Any) -> dict[str, Any]:
    schema = getattr(tool, "inputSchema", None) or {}
    if not isinstance(schema, dict):
        return {}
    properties = schema.get("properties") or {}
    return properties if isinstance(properties, dict) else {}


def _build_arguments(
    tool: Any,
    query: str,
    max_results: int,
) -> dict[str, Any]:
    """根据 MCP 工具的输入 schema 组装参数，避免发送不支持的字段。"""
    properties = _properties(tool)
    args: dict[str, Any] = {"query": query}

    for name in (
        "max_results",
        "maxResults",
        "limit",
        "top_k",
        "num_results",
    ):
        if name in properties:
            args[name] = max_results
            break
    if "search_depth" in properties:
        args["search_depth"] = "basic"
    if "include_raw_content" in properties:
        args["include_raw_content"] = True
    return args


def _normalize_result(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        text = str(item or "").strip()
        if not text:
            return None
        return {
            "source_type": "web",
            "title": "网页搜索结果",
            "url": "",
            "text": text,
            "score": 0.0,
            "published_at": None,
            "retrieved_at": _now(),
        }

    url = str(item.get("url") or item.get("link") or "").strip()
    title = str(title_value(item) or url or "网页搜索结果").strip()
    text = str(
        item.get("content")
        or item.get("raw_content")
        or item.get("snippet")
        or item.get("text")
        or ""
    ).strip()
    if not text and not url:
        return None
    try:
        score = float(item.get("score") or 0.0)
    except (TypeError, ValueError):
        score = 0.0
    return {
        "source_type": "web",
        "title": title,
        "url": url,
        "text": text[:2000],
        "score": score,
        "published_at": (
            item.get("published_date")
            or item.get("published_at")
            or item.get("date")
        ),
        "retrieved_at": _now(),
    }


def title_value(item: dict[str, Any]) -> str:
    return str(item.get("title") or item.get("name") or "")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _redact(text: str) -> str:
    """错误信息里不要带出 Tavily API Key。"""
    return re.sub(
        r"(tavilyApiKey=)[^&\s]+",
        r"\1***",
        text,
        flags=re.I,
    )


def _extract_items(result: Any) -> list[Any]:
    structured = getattr(result, "structuredContent", None)
    if structured is not None:
        if isinstance(structured, list):
            return structured
        if isinstance(structured, dict):
            for key in ("results", "data", "items"):
                value = structured.get(key)
                if isinstance(value, list):
                    return value

    text_parts: list[str] = []
    for block in getattr(result, "content", []) or []:
        text = getattr(block, "text", None)
        if text:
            text_parts.append(str(text))
    text = "\n".join(text_parts).strip()
    if not text:
        return []

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return [{"content": text}]

    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        for key in ("results", "data", "items"):
            value = parsed.get(key)
            if isinstance(value, list):
                return value
        return [parsed]
    return [{"content": str(parsed)}]


def _normalize_results(items: list[Any], limit: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for item in items:
        normalized = _normalize_result(item)
        if normalized is None:
            continue
        url = normalized.get("url") or ""
        if url and url in seen_urls:
            continue
        if url:
            seen_urls.add(url)
        results.append(normalized)
        if len(results) >= limit:
            break
    return results


class MCPWebSearch:
    """一个很小的同步 MCP 客户端，专门用于搜索。"""

    def __init__(self, config_value: WebSearchConfig | None = None) -> None:
        self.config = config_value or load_web_search_config()

    @classmethod
    def from_config(cls) -> "MCPWebSearch":
        return cls()

    def search(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """同步调用；返回标准化后的网页结果。"""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._search_async(query, top_k))
        raise WebSearchError(
            "当前进程已有事件循环，不能同步调用 MCP 搜索。"
        )

    async def _search_async(
        self,
        query: str,
        top_k: int | None,
    ) -> list[dict[str, Any]]:
        limit = max(1, min(top_k or self.config.max_results, 10))
        try:
            return await asyncio.wait_for(
                self._run(query, limit),
                timeout=self.config.timeout + 5,
            )
        except asyncio.TimeoutError as exc:
            raise WebSearchError("联网搜索超时。") from exc
        except WebSearchError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise WebSearchError(
                f"联网搜索失败：{_redact(str(exc))}"
            ) from exc

    async def _run(self, query: str, limit: int) -> list[dict[str, Any]]:
        async with streamable_http_client(self.config.url) as (
            read_stream,
            write_stream,
        ):
            async with ClientSession(
                read_stream,
                write_stream,
                read_timeout_seconds=self.config.timeout,
            ) as session:
                await session.initialize()
                tools = await session.list_tools()
                tool = self._pick_tool(tools.tools)
                result = await session.call_tool(
                    tool.name,
                    _build_arguments(tool, query, limit),
                    read_timeout_seconds=self.config.timeout,
                )
                return _normalize_results(
                    _extract_items(result), limit
                )

    def _pick_tool(self, tools: list[Any]) -> Any:
        names = [str(getattr(tool, "name", "")) for tool in tools]
        if self.config.tool:
            for tool in tools:
                if getattr(tool, "name", "") == self.config.tool:
                    return tool
        for tool in tools:
            name = str(getattr(tool, "name", "")).lower()
            if "search" in name:
                return tool
        raise WebSearchError(
            "MCP 服务里没有找到搜索工具。可用工具："
            + (", ".join(names) if names else "无")
        )


__all__ = [
    "MCPWebSearch",
    "WebSearchConfig",
    "WebSearchError",
    "load_web_search_config",
    "web_search_available",
]
