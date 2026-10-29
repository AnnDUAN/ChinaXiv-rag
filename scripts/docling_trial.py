"""用 Docling 的 JSON 输出做一次切片试验。

这个脚本不是正式入库流程，只把 Docling 解析结果切成和现有 RAG
接近的块，方便对比阅读顺序、表格结构和切片质量。

试验环境（Python 3.10，CPU 版 Docling）：
    uv venv --python 3.10 .venv-docling
    uv pip install --python .venv-docling/Scripts/python.exe docling

用法：
    .venv-docling/Scripts/python.exe scripts/docling_trial.py
    .venv-docling/Scripts/python.exe scripts/docling_trial.py --ids 202610.00009
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag.common.utils import clean_text, hard_split, split_sentences  # noqa: E402

TARGET = 450
MAX_LEN = 600
OVERLAP = 70


def _resolve(doc: dict[str, Any], ref: str) -> tuple[str, dict[str, Any]]:
    parts = ref.lstrip("#/").split("/")
    if len(parts) != 2:
        raise ValueError(f"无法解析引用：{ref}")
    kind, index = parts
    return kind, doc[kind][int(index)]


def _walk(doc: dict[str, Any], ref: str) -> Iterable[tuple[str, dict[str, Any]]]:
    kind, item = _resolve(doc, ref)
    if kind == "groups":
        for child in item.get("children", []):
            yield from _walk(doc, child["$ref"])
    else:
        yield kind, item


def _body_items(doc: dict[str, Any]) -> Iterable[tuple[str, dict[str, Any]]]:
    for child in doc["body"]["children"]:
        yield from _walk(doc, child["$ref"])


def _pages(item: dict[str, Any]) -> tuple[int, int]:
    pages = [
        int(p["page_no"])
        for p in item.get("prov", [])
        if isinstance(p, dict) and p.get("page_no")
    ]
    return (min(pages), max(pages)) if pages else (0, 0)


def _heading_level(text: str, stack: list[tuple[int, str]]) -> int:
    m = re.match(r"^(\d+(?:\.\d+)*)[\s　]+", text)
    if m:
        return len(m.group(1).split("."))
    # Docling 目前把所有标题都给成 level 1；像“（1）结果变量”
    # 这种子标题按当前层级往下放一层，避免把章节路径清空。
    if re.match(r"^[（(]\d+[）)]", text) and stack:
        if re.match(r"^[（(]\d+[）)]", stack[-1][1]):
            return stack[-1][0]
        return len(stack) + 1
    return 1


def _subtract_heading(stack: list[tuple[int, str]], text: str) -> list[str]:
    level = _heading_level(text, stack)
    while stack and stack[-1][0] >= level:
        stack.pop()
    stack.append((level, text))
    return [title for _, title in stack]


def _chunk_text(text: str) -> list[str]:
    units: list[str] = []
    for paragraph in text.split("\n"):
        paragraph = clean_text(paragraph)
        if not paragraph:
            continue
        for sentence in split_sentences(paragraph):
            units.extend(hard_split(sentence, MAX_LEN))

    groups: list[list[str]] = []
    current: list[str] = []
    current_len = 0
    for unit in units:
        if current and current_len + len(unit) > MAX_LEN:
            groups.append(current)
            current, current_len = [], 0
        current.append(unit)
        current_len += len(unit)
        if current_len >= TARGET:
            groups.append(current)
            current, current_len = [], 0
    if current:
        groups.append(current)

    chunks: list[str] = []
    for i, group in enumerate(groups):
        text = "".join(group).strip()
        if i > 0 and chunks:
            prefix = chunks[-1][-OVERLAP:].strip()
            room = MAX_LEN - len(text)
            if prefix and room > 0:
                text = (prefix[:room] + text).strip()
        if text:
            chunks.append(text)
    return chunks


def _table_markdown(table: dict[str, Any]) -> tuple[str, list[str], list[str]]:
    data = table.get("data", {})
    rows = int(data.get("num_rows") or 0)
    cols = int(data.get("num_cols") or 0)
    cells = data.get("table_cells") or []
    if not rows or not cols:
        return "", [], []

    grid = [["" for _ in range(cols)] for _ in range(rows)]
    for cell in cells:
        r = int(cell.get("start_row_offset_idx") or 0)
        c = int(cell.get("start_col_offset_idx") or 0)
        if not (0 <= r < rows and 0 <= c < cols):
            continue
        text = clean_text(cell.get("text") or "").replace("|", "\\|")
        text = re.sub(r"\s+", " ", text)
        grid[r][c] = text

    lines = ["| " + " | ".join(row) + " |" for row in grid]
    header = [lines[0], "| " + " | ".join(["---"] * cols) + " |"] if lines else []
    body = lines[1:] if len(lines) > 1 else []
    return "\n".join(header + body), header, body


def _table_chunks(
    table: dict[str, Any],
    doc: dict[str, Any],
    section_path: list[str],
) -> list[str]:
    captions = []
    for ref in table.get("captions", []):
        _, item = _resolve(doc, ref["$ref"])
        text = clean_text(item.get("text") or "")
        if text:
            captions.append(text)
    caption = " ".join(captions)
    markdown, header, body = _table_markdown(table)
    if not markdown:
        return []

    prefix = f"{caption}\n\n" if caption else ""
    if len(prefix + markdown) <= MAX_LEN:
        return [prefix + markdown]

    chunks: list[str] = []
    current: list[str] = []
    for row in body:
        candidate = "\n".join(header + current + [row])
        if current and len(prefix + candidate) > MAX_LEN:
            suffix = "（续表）" if chunks else ""
            chunks.append(prefix + suffix + "\n\n" + "\n".join(header + current))
            current = [row]
        else:
            current.append(row)
    if current:
        suffix = "（续表）" if chunks else ""
        chunks.append(prefix + suffix + "\n\n" + "\n".join(header + current))
    return chunks


def _load_title(paper_id: str, doc: dict[str, Any]) -> str:
    old = ROOT / "data" / "chunks" / f"{paper_id}.json"
    if old.exists():
        try:
            data = json.loads(old.read_text(encoding="utf-8"))
            title = (data.get("meta") or {}).get("title")
            if title:
                return str(title)
        except Exception:
            pass
    for text in doc.get("texts", []):
        if text.get("label") == "section_header" and text.get("text"):
            return clean_text(text["text"])
    return paper_id


def build_chunks(doc: dict[str, Any], paper_id: str) -> dict[str, Any]:
    title = _load_title(paper_id, doc)
    stack: list[tuple[int, str]] = []
    section_path: list[str] = []
    paragraph_buffer: list[str] = []
    buffer_pages: list[tuple[int, int]] = []
    chunks: list[dict[str, Any]] = []
    counts = {"text": 0, "table": 0}
    body_started = False

    def add_text_chunk(text: str, page_start: int, page_end: int) -> None:
        path = list(section_path)
        index_text = f"《{title}》 > {' > '.join(path)} > {text}"
        chunks.append(
            {
                "chunk_id": f"{paper_id}-d{len(chunks) + 1:04d}",
                "paper_id": paper_id,
                "type": "text",
                "section_path": path,
                "page_start": page_start,
                "page_end": page_end,
                "text": text,
                "index_text": index_text,
                "char_count": len(text),
            }
        )
        counts["text"] += 1

    def flush_text() -> None:
        nonlocal paragraph_buffer, buffer_pages
        if not paragraph_buffer:
            return
        text = "\n".join(paragraph_buffer).strip()
        paragraph_buffer = []
        pages = [(start, end) for start, end in buffer_pages if start]
        buffer_pages = []
        page_start = min((p[0] for p in pages), default=1)
        page_end = max((p[1] for p in pages), default=1)
        if not text:
            return
        for part in _chunk_text(text):
            add_text_chunk(part, page_start, page_end)

    for kind, item in _body_items(doc):
        label = item.get("label")
        if label in {"page_header", "page_footer"}:
            continue

        if kind == "texts" and label == "section_header":
            flush_text()
            heading = clean_text(item.get("text") or "")
            if not heading:
                continue
            if not body_started:
                if re.match(r"^\d+(?:\.\d+)*[\s　]+", heading) or re.match(
                    r"^(摘\s*要|Abstract|关键词|Key\s*words?)",
                    heading,
                    re.I,
                ):
                    body_started = True
                else:
                    continue
            section_path = _subtract_heading(stack, heading)
            continue

        if kind == "texts" and label in {"text", "list_item"}:
            if not body_started:
                continue
            text = clean_text(item.get("text") or "")
            if text:
                paragraph_buffer.append(text)
                page_start, page_end = _pages(item)
                if page_start:
                    buffer_pages.append((page_start, page_end))
            continue

        if kind == "tables":
            flush_text()
            page_start, page_end = _pages(item)
            for table_text in _table_chunks(item, doc, section_path):
                path = list(section_path)
                chunks.append(
                    {
                        "chunk_id": f"{paper_id}-d{len(chunks) + 1:04d}",
                        "paper_id": paper_id,
                        "type": "table",
                        "section_path": path,
                        "page_start": page_start,
                        "page_end": page_end,
                        "text": table_text,
                        "index_text": f"《{title}》 > {' > '.join(path)} > {table_text}",
                        "char_count": len(table_text),
                    }
                )
                counts["table"] += 1

    flush_text()

    old_count = 0
    old_path = ROOT / "data" / "chunks" / f"{paper_id}.json"
    if old_path.exists():
        try:
            old = json.loads(old_path.read_text(encoding="utf-8"))
            old_count = int(old.get("chunk_count") or len(old.get("chunks", [])))
        except Exception:
            pass

    return {
        "paper_id": paper_id,
        "title": title,
        "docling_pages": len(doc.get("pages", {})),
        "docling_tables": len(doc.get("tables", [])),
        "docling_texts": len(doc.get("texts", [])),
        "old_chunks": old_count,
        "chunk_count": len(chunks),
        "counts": counts,
        "chunks": chunks,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json-dir",
        default=str(ROOT / "data" / "docling_probe"),
        help="Docling JSON 所在目录",
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "data" / "docling_probe" / "chunks"),
        help="切片输出目录",
    )
    parser.add_argument("--ids", nargs="*", help="只处理这些论文 ID")
    args = parser.parse_args()

    json_dir = Path(args.json_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(json_dir.glob("*.json"))
    if args.ids:
        wanted = set(args.ids)
        paths = [p for p in paths if p.stem in wanted]

    summaries = []
    for path in paths:
        if path.parent == out_dir:
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        result = build_chunks(doc, path.stem)
        (out_dir / f"{path.stem}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        summaries.append(result)
        print(
            f"{path.stem}: 页 {result['docling_pages']} | "
            f"表 {result['docling_tables']} | "
            f"旧切片 {result['old_chunks']} -> "
            f"新切片 {result['chunk_count']} "
            f"(文本 {result['counts']['text']}, 表格 {result['counts']['table']})"
        )

    if summaries:
        print("\n--- 表格切片示例 ---")
        for result in summaries:
            for chunk in result["chunks"]:
                if chunk["type"] == "table":
                    print(f"\n[{result['paper_id']}] {chunk['section_path']}")
                    print(chunk["text"][:1200])
                    break
            else:
                continue
            break

        print("\n--- 文本切片示例 ---")
        for result in summaries:
            for chunk in result["chunks"]:
                if chunk["type"] == "text" and chunk["char_count"] >= 200:
                    print(f"\n[{result['paper_id']}] {chunk['section_path']}")
                    print(chunk["text"][:800])
                    break
            else:
                continue
            break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
