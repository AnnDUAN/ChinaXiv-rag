"""检索 + 带出处的回答生成。

传输层在 `rag.llm` 里：
- 直接使用 OpenAI SDK，不经过 LangChain；
- 结构化输出优先走 JSON Schema，不支持时自动降级到 JSON 模式和文本解析；
- 回答里的引用编号和引文会再由 Python 做一次机械校验。
"""

from __future__ import annotations

import re
import time
from typing import Any

from ..llm import (
    LLMClient,
    LLMConfig,
    LLMConnectionError,
    Message,
    StructuredError,
    StructuredRunner,
    build_client,
    load_llm_config,
)
from ..retrieval.search import search_chunks
from .schemas import AnswerCitation, StructuredAnswer

SYSTEM_PROMPT = """你是一个严谨的中文学术论文问答助手。
请严格遵守：
1. 只能根据提供的原文片段回答，不要使用片段以外的知识。
2. 每个结论后面用 [编号] 标注来源，例如 [1]、[2]。
3. 如果片段不足以回答，直接说“当前语料中没有足够信息”，不要猜测或编造。
4. 用中文回答，先给结论，再给必要解释，控制在 300 字左右。
5. 不要复述整段原文，不要输出与问题无关的内容。"""

STRUCTURED_SYSTEM_PROMPT = SYSTEM_PROMPT + """
6. 你必须输出一个 JSON 对象，包含 answer、citations、insufficient_evidence、confidence。
7. citations 里每条都要有 source_id 和 quote；quote 必须逐字复制自对应原文片段，不要改写。
8. answer 里的 [编号] 只能使用上下文里真实存在的编号。"""

NO_EVIDENCE_TEXT = "当前语料中没有足够信息回答这个问题。"


def _pages(result: dict[str, Any]) -> str:
    start = result.get("page_start")
    end = result.get("page_end")
    if start == end:
        return f"第 {start} 页"
    return f"第 {start}–{end} 页"


def _sources(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sources = []
    for result in results:
        sources.append(
            {
                "rank": result.get("rank"),
                "chunk_id": result.get("chunk_id", ""),
                "title": result.get("paper_title", ""),
                "pages": _pages(result),
                "section": " > ".join(result.get("section_path") or []),
            }
        )
    return sources


def build_messages(
    query: str,
    results: list[dict[str, Any]],
    structured: bool = False,
    memory_text: str | None = None,
) -> list[Message]:
    """把检索结果和长期记忆拼成发给大模型的 messages。"""
    blocks: list[str] = []
    for rank, result in enumerate(results, 1):
        section = " > ".join(result.get("section_path") or [])
        blocks.append(
            f"[{rank}] 论文：《{result.get('paper_title', '')}》"
            f" | {_pages(result)} | 章节：{section}\n"
            f"原文：{result.get('text', '')}"
        )
    memory = f"{memory_text.strip()}\n\n" if memory_text else ""
    user = (
        memory
        + f"问题：{query}\n\n"
        "原文片段：\n\n"
        + "\n\n".join(blocks)
        + "\n\n请根据以上片段回答，并在结论后标注 [编号]。"
    )
    system = STRUCTURED_SYSTEM_PROMPT if structured else SYSTEM_PROMPT
    return [
        Message("system", system),
        Message("user", user),
    ]


def prepare_answer(
    query: str,
    top_k: int = 5,
    mode: str = "hybrid",
    rerank: bool = False,
    structured: bool = False,
    memory_text: str | None = None,
) -> dict[str, Any]:
    """只做检索和提示词拼装，不调用大模型。"""
    search = search_chunks(
        query, top_k=top_k, mode=mode, rerank=rerank
    )
    results = search.get("results", [])
    return {
        "query": query,
        "search": search,
        "sources": _sources(results),
        "messages": (
            build_messages(
                query,
                results,
                structured=structured,
                memory_text=memory_text,
            )
            if results
            else []
        ),
    }


def _squash(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def validate_citations(
    answer: StructuredAnswer,
    results: list[dict[str, Any]],
) -> tuple[list[AnswerCitation], list[str]]:
    """检查引用编号是否存在、引文是否真的来自原文。"""
    issues: list[str] = []
    valid: list[AnswerCitation] = []

    for citation in answer.citations:
        index = citation.source_id - 1
        if index < 0 or index >= len(results):
            issues.append(f"引用编号 [{citation.source_id}] 不存在")
            continue
        source_text = _squash(results[index].get("text", ""))
        quote = _squash(citation.quote)
        if not quote:
            issues.append(f"[{citation.source_id}] 的 quote 为空")
            continue
        if quote not in source_text:
            issues.append(
                f"[{citation.source_id}] 的 quote 不是原文子串"
            )
            continue
        valid.append(citation)

    markers = {
        int(m)
        for m in re.findall(r"\[(\d{1,3})\]", answer.answer or "")
    }
    for marker in sorted(markers - set(range(1, len(results) + 1))):
        issues.append(f"回答里出现了不存在的引用编号 [{marker}]")
    if not answer.citations and not answer.insufficient_evidence:
        issues.append("回答没有给出任何出处，也没有说明证据不足")

    return valid, issues


def call_llm(
    config_value: LLMConfig,
    messages: list[Message],
) -> str:
    """普通文本调用，保留给调试或简单场景。"""
    return build_client(config_value).chat(messages)


def _log_answer(
    payload: dict[str, Any],
    started: float,
    *,
    log_to_db: bool,
) -> dict[str, Any]:
    """把本次问答写入数据库；数据库不可用不影响回答。"""
    if not log_to_db:
        return payload
    try:
        from ..db.engine import connect
        from ..db.repo import log_query

        search = payload.get("search") or {}
        with connect() as conn:
            query_id = log_query(
                conn,
                question=payload["query"],
                mode=str(payload.get("mode") or ""),
                rerank=bool(search.get("rerank_applied")),
                structured=bool(payload.get("structured")),
                answer=payload.get("answer", ""),
                structured_answer=payload.get("structured"),
                citation_issues=payload.get("citation_issues") or [],
                latency_ms=int((time.perf_counter() - started) * 1000),
                results=search.get("results") or [],
            )
        payload["db_query_id"] = query_id
    except Exception as exc:  # noqa: BLE001
        payload["db_error"] = str(exc)
    return payload


def _generate_from_prepared(
    *,
    query: str,
    search: dict[str, Any],
    results: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    messages: list[Message],
    structured: bool,
    llm_client: LLMClient,
    log_to_db: bool,
    memory_text: str | None,
    started: float,
) -> dict[str, Any]:
    """给定已经检索好的片段，只负责调用大模型并校验引用。"""
    if structured:
        try:
            parsed = StructuredRunner(llm_client).run(
                messages, StructuredAnswer
            )
        except LLMConnectionError:
            raise
        except StructuredError as exc:
            # 结构化失败不一定代表模型不会回答，退回普通文本。
            plain_messages = build_messages(
                query,
                results,
                structured=False,
                memory_text=memory_text,
            )
            raw = llm_client.chat(plain_messages)
            payload = {
                "query": query,
                "answer": raw,
                "mode": search.get("mode"),
                "sources": sources,
                "search": search,
                "structured": None,
                "citation_issues": [],
                "structured_error": str(exc),
            }
            return _log_answer(payload, started, log_to_db=log_to_db)

        valid_citations, issues = validate_citations(parsed, results)
        answer_text = parsed.answer.strip()
        if parsed.insufficient_evidence or not answer_text:
            answer_text = answer_text or NO_EVIDENCE_TEXT
        payload = {
            "query": query,
            "answer": answer_text,
            "mode": search.get("mode"),
            "sources": sources,
            "search": search,
            "structured": {
                **parsed.model_dump(),
                "citations": [c.model_dump() for c in valid_citations],
            },
            "citation_issues": issues,
            "structured_error": None,
        }
        return _log_answer(payload, started, log_to_db=log_to_db)

    raw = llm_client.chat(messages)
    payload = {
        "query": query,
        "answer": raw,
        "mode": search.get("mode"),
        "sources": sources,
        "search": search,
        "structured": None,
        "citation_issues": [],
        "structured_error": None,
    }
    return _log_answer(payload, started, log_to_db=log_to_db)


def answer_from_search(
    query: str,
    search: dict[str, Any],
    *,
    structured: bool = True,
    config_value: LLMConfig | None = None,
    client: LLMClient | None = None,
    log_to_db: bool = True,
    memory_text: str | None = None,
) -> dict[str, Any]:
    """基于已经检索好的 search 结果生成答案。

    Agent 会先自己完成多轮检索，最后再调用这里生成一次答案，
    避免把“检索”和“生成”绑死在一个函数里。
    """
    started = time.perf_counter()
    results = list(search.get("results") or [])
    sources = _sources(results)
    base = {
        "query": query,
        "search": search,
        "sources": sources,
    }
    if not results:
        return {**base, "error": "没有检索到相关段落，无法回答。"}

    messages = build_messages(
        query,
        results,
        structured=structured,
        memory_text=memory_text,
    )
    llm_client = client or build_client(config_value or load_llm_config())
    return _generate_from_prepared(
        query=query,
        search=search,
        results=results,
        sources=sources,
        messages=messages,
        structured=structured,
        llm_client=llm_client,
        log_to_db=log_to_db,
        memory_text=memory_text,
        started=started,
    )


def answer_question(
    query: str,
    top_k: int = 5,
    mode: str = "hybrid",
    rerank: bool = False,
    structured: bool = True,
    config_value: LLMConfig | None = None,
    client: LLMClient | None = None,
    log_to_db: bool = True,
    memory_text: str | None = None,
) -> dict[str, Any]:
    """检索 + 调用大模型，返回答案和出处。

    structured=True 时优先使用结构化输出；如果所有结构化方式都失败，
    会退回普通文本回答，并在 `structured_error` 里留下原因。
    """
    started = time.perf_counter()
    prepared = prepare_answer(
        query,
        top_k=top_k,
        mode=mode,
        rerank=rerank,
        structured=structured,
        memory_text=memory_text,
    )
    search = prepared["search"]
    if search.get("error"):
        return {**prepared, "error": search["error"]}
    if not prepared["messages"]:
        return {**prepared, "error": "没有检索到相关段落，无法回答。"}

    if client is None:
        llm_config = config_value or load_llm_config()
        llm_client = build_client(llm_config)
    else:
        llm_client = client
    results = search.get("results", [])
    return _generate_from_prepared(
        query=query,
        search=search,
        results=results,
        sources=prepared["sources"],
        messages=prepared["messages"],
        structured=structured,
        llm_client=llm_client,
        log_to_db=log_to_db,
        memory_text=memory_text,
        started=started,
    )
