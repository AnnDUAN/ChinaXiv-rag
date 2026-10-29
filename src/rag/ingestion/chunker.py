"""把解析结果切成适合检索的小块。

原则：
- 不跨章节切；
- 尽量按句子边界切；
- 每块 400～600 字，块之间留一点重叠；
- 表格单独成块，保留 HTML 合并单元格，超长表格重复表头；
- 公式跟相邻正文放一起，不生成只有公式的块；
- 每块都记录论文、章节、页码、前后块，方便以后追溯。
"""

from __future__ import annotations

import html as html_module
import json
import re
from pathlib import Path
from typing import Any

from .. import config
from ..common.utils import hard_split, split_sentences

FORMULA_RE = re.compile(r"(\$\$.*?\$\$|\$[^$\n]+\$)", re.S)


def _protect_formulas(text: str, formulas: dict[str, str]) -> str:
    """把公式换成短占位符，避免句号/长度切分把公式拦腰截断。"""

    def repl(match: re.Match[str]) -> str:
        key = f"[[F{len(formulas)}]]"
        formulas[key] = match.group(1)
        return key

    return FORMULA_RE.sub(repl, text)


def _restore_formulas(text: str, formulas: dict[str, str]) -> str:
    for key, value in formulas.items():
        text = text.replace(key, value)
    return text


def _units_for_paragraphs(
    paragraphs: list[dict[str, Any]],
    section: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    units: list[dict[str, Any]] = []
    formulas: dict[str, str] = {}
    for p_idx, para in enumerate(paragraphs):
        text = (para.get("text") or "").strip()
        if not text:
            continue
        text = _protect_formulas(text, formulas)
        sentence_parts: list[str] = []
        for sent in split_sentences(text):
            sentence_parts.extend(hard_split(sent, config.CHUNK_MAX))
        for s_idx, sent in enumerate(sentence_parts):
            units.append(
                {
                    "text": sent,
                    "page_start": para.get("page_start", section.get("page_start", 1)),
                    "page_end": para.get("page_end", section.get("page_end", 1)),
                    "new_para": s_idx == 0 and p_idx > 0,
                }
            )
    return units, formulas


def _join_units(units: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for i, unit in enumerate(units):
        if i > 0 and unit.get("new_para"):
            parts.append("\n")
        parts.append(unit["text"])
    return "".join(parts).strip()


def _group_len(group: list[dict[str, Any]]) -> int:
    total = sum(len(unit["text"]) for unit in group)
    # _join_units 会在段落之间插入换行，这里也要算上
    total += sum(1 for unit in group[1:] if unit.get("new_para"))
    return total


def _primary_chunks(units: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """贪心装句子；上限是硬限制，超不过 600 字。"""
    chunks: list[list[dict[str, Any]]] = []
    cur: list[dict[str, Any]] = []
    cur_len = 0

    for unit in units:
        u_len = len(unit["text"])
        sep = 1 if cur and unit.get("new_para") else 0
        # 不管当前块有多短，只要再加一句会超过硬上限，就先切。
        # 这样偶尔会产生短块，下面再把特别短的并回前一块。
        if cur and (
            cur_len + sep + u_len > config.CHUNK_MAX
            or (
                cur_len >= config.CHUNK_TARGET
                and cur_len + sep + u_len > config.CHUNK_TARGET + 100
            )
        ):
            chunks.append(cur)
            cur = []
            cur_len = 0
            sep = 0
        cur.append(unit)
        cur_len += sep + u_len

    if cur:
        # 最后一块太短，优先并回上一块
        if (
            chunks
            and cur_len < config.CHUNK_MIN
            and _group_len(chunks[-1]) + cur_len <= config.CHUNK_MAX
        ):
            chunks[-1].extend(cur)
        else:
            chunks.append(cur)

    # 同一章节内，特别短的块并到前一块；装不下就保留
    merged: list[list[dict[str, Any]]] = []
    for group in chunks:
        if merged and _group_len(group) < 100:
            if _group_len(merged[-1]) + _group_len(group) <= config.CHUNK_MAX:
                merged[-1].extend(group)
                continue
        merged.append(group)
    return merged


def _text_units(
    text: str,
    page: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """把一段文字拆成句子单元，供摘要等单独内容复用。"""
    units: list[dict[str, Any]] = []
    formulas: dict[str, str] = {}
    text = _protect_formulas(text, formulas)
    for sent in split_sentences(text):
        for part in hard_split(sent, config.CHUNK_MAX):
            units.append(
                {
                    "text": part,
                    "page_start": page,
                    "page_end": page,
                    "new_para": False,
                }
            )
    return units, formulas


def _overlap_prefix(prev_text: str, overlap: int) -> str:
    """取上一块末尾最多 overlap 个字符。

    优先按完整句子取；如果最后一句本身就超过 overlap，
    只取最后 overlap 个字符，避免把整句搬过来把下一块撑大。
    """
    if overlap <= 0 or not prev_text:
        return ""
    formulas: dict[str, str] = {}
    prev_text = _protect_formulas(prev_text, formulas)
    sentences = split_sentences(prev_text)
    picked: list[str] = []
    total = 0
    for sent in reversed(sentences):
        if total + len(sent) > overlap:
            break
        picked.insert(0, sent)
        total += len(sent)
        if total >= overlap:
            break
    prefix = "".join(picked).strip()
    if not prefix:
        prefix = prev_text[-overlap:].strip()
    return _restore_formulas(prefix[:overlap].strip(), formulas)


def _split_table_html(html: str, max_len: int) -> list[str]:
    """表格太长时按行拆开，每块重复第一行表头。"""
    if len(html) <= max_len:
        return [html]
    rows = re.findall(r"<tr\b[^>]*>.*?</tr>", html, re.S | re.I)
    if len(rows) <= 1:
        return [html]

    header = rows[0]
    body = rows[1:]
    chunks: list[str] = []
    current = [header]

    for row in body:
        candidate = "<table>" + "".join(current + [row]) + "</table>"
        if len(candidate) > max_len and len(current) > 1:
            chunks.append("<table>" + "".join(current) + "</table>")
            current = [header, row]
        else:
            current.append(row)

    if len(current) > 1 or not chunks:
        chunks.append("<table>" + "".join(current) + "</table>")
    return chunks


def _html_table_text(html: str) -> str:
    """把 HTML 表格压成纯文本，只给 BM25/向量索引用，不替换原文。"""
    text = re.sub(r"<br\s*/?>", "\n", html, flags=re.I)
    text = re.sub(r"</t[dh]>", " | ", text, flags=re.I)
    text = re.sub(r"</tr>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    text = html_module.unescape(text)
    lines = []
    for line in text.splitlines():
        line = re.sub(r"\s+", " ", line).strip(" |")
        if line:
            lines.append(line)
    return "\n".join(lines)


def _table_chunks(para: dict[str, Any]) -> list[tuple[str, str]]:
    """返回 (保留 HTML 的块文本, 给索引用的纯文本)。"""
    caption = (para.get("caption") or "").strip()
    html = (para.get("html") or "").strip()
    if not html:
        text = (para.get("text") or "").strip()
        idx = text.find("<table")
        if idx < 0:
            return []
        caption = caption or text[:idx].strip(" ：:")
        html = text[idx:].strip()
    if not html:
        return []

    prefix = f"{caption}\n\n" if caption else ""
    body_limit = max(400, config.TABLE_CHUNK_MAX - len(prefix))
    parts = _split_table_html(html, body_limit)

    result: list[tuple[str, str]] = []
    for i, part in enumerate(parts):
        if i == 0:
            cap = caption
        else:
            cap = f"{caption}（续表）" if caption else "（续表）"
        text = f"{cap}\n\n{part}" if cap else part
        index_text = f"{cap}\n{_html_table_text(part)}" if cap else _html_table_text(part)
        result.append((text, index_text))
    return result


def chunk_parsed(parsed: dict[str, Any]) -> dict[str, Any]:
    meta = parsed["meta"]
    title = meta.get("title") or parsed.get("paper_id", "")
    paper_id = parsed["paper_id"]
    chunks: list[dict[str, Any]] = []

    def add_chunk(
        text: str,
        section_path: list[str],
        chunk_type: str,
        page_start: int,
        page_end: int,
        index_text: str | None = None,
    ) -> None:
        text = text.strip()
        if len(text) < 20:
            return
        # 正文里偶尔会残留纯公式/表格碎片，中文字符太少就不建索引
        if chunk_type == "body" and len(re.findall(r"[一-鿿]", text)) < 8:
            return
        if index_text is None:
            index_text = text
        chunks.append(
            {
                "chunk_id": "",
                "paper_id": paper_id,
                "type": chunk_type,
                "section_path": section_path,
                "page_start": page_start,
                "page_end": page_end,
                "text": text,
                "index_text": f"《{title}》 > {' > '.join(section_path)} > {index_text}",
                "char_count": len(text),
                "prev_id": "",
                "next_id": "",
            }
        )

    # 摘要单独成块；太长时按句子切开，避免一个块接近 1000 字
    if meta.get("abstract"):
        units, formulas = _text_units(meta["abstract"], 1)
        for group in _primary_chunks(units):
            add_chunk(
                _restore_formulas(_join_units(group), formulas),
                ["摘要"],
                "abstract",
                1,
                1,
            )
    if meta.get("keywords"):
        add_chunk("；".join(meta["keywords"]), ["关键词"], "keywords", 1, 1)

    # 正文按章节切；表格单独成块，公式跟前后正文放一起
    for section in parsed.get("sections", []):
        if section.get("type") != "body":
            continue
        section_path = section.get("section_path") or [section.get("title", "")]

        def flush_text(paragraphs: list[dict[str, Any]]) -> None:
            units, formulas = _units_for_paragraphs(paragraphs, section)
            if not units:
                return
            primary = _primary_chunks(units)
            section_chunks: list[dict[str, Any]] = []
            for group in primary:
                text = _restore_formulas(_join_units(group), formulas)
                if not text:
                    continue
                section_chunks.append(
                    {
                        "text": text,
                        "page_start": min(u["page_start"] for u in group),
                        "page_end": max(u["page_end"] for u in group),
                    }
                )

            # 块间重叠（只在同一段连续正文内）
            for i, item in enumerate(section_chunks):
                if i > 0:
                    # 先算这一块还剩多少空间，重叠不能把总长度顶过 600
                    room = max(0, config.CHUNK_MAX - len(item["text"]))
                    prefix = _overlap_prefix(
                        section_chunks[i - 1]["text"],
                        min(config.CHUNK_OVERLAP, room),
                    )
                    if prefix and not item["text"].startswith(prefix):
                        item["text"] = prefix + item["text"]
                add_chunk(
                    item["text"],
                    section_path,
                    "body",
                    item["page_start"],
                    item["page_end"],
                )

        text_run: list[dict[str, Any]] = []
        pending_equations: list[str] = []
        for para in section.get("paragraphs", []):
            kind = para.get("kind") or "text"
            if kind == "table":
                flush_text(text_run)
                text_run = []
                pending_equations = []
                for table_text, table_index in _table_chunks(para):
                    add_chunk(
                        table_text,
                        section_path,
                        "table",
                        para.get("page_start", section.get("page_start", 1)),
                        para.get("page_end", section.get("page_end", 1)),
                        index_text=table_index,
                    )
                continue

            if kind == "equation":
                latex = (para.get("latex") or para.get("text") or "").strip()
                if not latex:
                    continue
                wrapped = f"$${latex}$$"
                if text_run:
                    prev = text_run[-1]
                    prev["text"] = (prev.get("text") or "").rstrip() + "\n" + wrapped
                    prev["page_end"] = max(
                        prev.get("page_end", para.get("page_end", 1)),
                        para.get("page_end", 1),
                    )
                else:
                    pending_equations.append(wrapped)
                continue

            text = (para.get("text") or "").strip()
            if not text:
                continue
            if pending_equations:
                text = "\n".join(pending_equations) + "\n" + text
                pending_equations = []
            text_run.append(
                {
                    "text": text,
                    "page_start": para.get(
                        "page_start", section.get("page_start", 1)
                    ),
                    "page_end": para.get(
                        "page_end", section.get("page_end", 1)
                    ),
                }
            )

        flush_text(text_run)
        # 只有公式、后面没有正文说明时，不生成纯公式块。

    # 编号 + 前后指针
    for i, chunk in enumerate(chunks, 1):
        chunk["chunk_id"] = f"{paper_id}-c{i:04d}"
    for i, chunk in enumerate(chunks):
        if i > 0:
            chunk["prev_id"] = chunks[i - 1]["chunk_id"]
        if i + 1 < len(chunks):
            chunk["next_id"] = chunks[i + 1]["chunk_id"]

    return {
        "paper_id": paper_id,
        "meta": meta,
        "chunk_count": len(chunks),
        "stats": {
            "count": len(chunks),
            "avg_chars": round(sum(c["char_count"] for c in chunks) / len(chunks), 1) if chunks else 0,
            "min_chars": min((c["char_count"] for c in chunks), default=0),
            "max_chars": max((c["char_count"] for c in chunks), default=0),
            "by_type": {
                t: sum(1 for c in chunks if c["type"] == t)
                for t in sorted({c["type"] for c in chunks})
            },
        },
        "chunks": chunks,
    }


def chunk_file(parsed_path: str | Path, out_path: str | Path | None = None) -> dict[str, Any]:
    parsed_path = Path(parsed_path)
    parsed = json.loads(parsed_path.read_text(encoding="utf-8"))
    result = chunk_parsed(parsed)
    if out_path is None:
        out_path = config.CHUNKS_DIR / f"{parsed['paper_id']}.json"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
