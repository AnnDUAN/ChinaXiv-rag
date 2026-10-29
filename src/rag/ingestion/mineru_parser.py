"""把 MinerU 的结构化结果整理成项目内部使用的解析格式。

MinerU 的 middle JSON 已经按阅读顺序给出了每个块的类型、页码、坐标和内容。
这里不重新猜版面，只做三件事：

1. 去掉页眉、页脚、水印、脚注、预印本说明、分类号等脏数据；
2. 整理标题、作者、单位、摘要、关键词；
3. 保留正文、表格和公式，供后面的切片步骤使用。

图片只记录数量，不进入正文，也不参与检索。
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import jieba

from .. import config
from ..common.utils import file_sha256, normalize_spaces

NOISE_TYPES = {
    "header",
    "footer",
    "page_number",
    "aside_text",
    "page_footnote",
}
TEXT_TYPES = {"text", "ref_text"}
HEADING_TYPES = {"paragraph_title"}
TABLE_TYPES = {"table"}
EQUATION_TYPES = {"equation"}
IMAGE_TYPES = {"image", "chart"}

AFFILIATION_WORDS = (
    "大学",
    "学院",
    "研究所",
    "研究院",
    "实验室",
    "研究中心",
    "学校",
    "公司",
    "科学院",
    "医院",
    "协会",
    "学会",
    "图书馆",
    "出版社",
)

CLASSIFICATION_RE = re.compile(
    r"^[\[【(（［]?\s*(中图分类号|分类号|文献标识码|文献标志码|文章编号|DOI|"
    r"收稿日期|修回日期|基金项目|作者简介|通信作者|通信联系人|预印本说明)"
    r"\s*[\]】)）］]?\s*[：:]?",
    re.I,
)
PREPRINT_RE = re.compile(r"^(预印本|preprint)", re.I)
ABSTRACT_RE = re.compile(r"^(?:摘\s*要|提\s*要)\s*[：:]?$", re.I)
ABSTRACT_INLINE_RE = re.compile(
    r"^[\[【(（［]\s*(?:摘\s*要|提\s*要)\s*[\]】)）］]\s*(.+)$",
    re.S,
)
ABSTRACT_COLON_RE = re.compile(r"^(?:摘\s*要|提\s*要)\s*[：:]\s*(.+)$", re.S)
ABSTRACT_PLAIN_RE = re.compile(r"^(?:摘\s*要|提\s*要)\s+(.+)$", re.S)
ABSTRACT_EN_RE = re.compile(r"^abstract\s*[：:]?$", re.I)
KEYWORDS_RE = re.compile(
    r"^[\[【(（［]?\s*(关键词|关键字|关\s*键\s*词|key\s*words?)\s*"
    r"[\]】)）］]?\s*[：:]?\s*",
    re.I,
)
REFERENCE_RE = re.compile(
    r"^[\[【]?\s*参\s*考\s*文\s*献\s*[\]】]?\s*(?:[（(]\s*references?\s*[）)])?\s*[：:]?$",
    re.I,
)
REFERENCE_EN_RE = re.compile(r"^references?\s*[：:]?$", re.I)
BODY_HEADING_RE = re.compile(r"^\d{1,2}(?:\.\d{1,2}){0,3}[.、]?\s+\S")
LIST_HEADING_RE = re.compile(r"^[（(]\d{1,2}[）)]\s*\S")
LIST_NUMBER_RE = re.compile(r"^\d{1,2}[.、]\s+\S")
CAPTION_RE = re.compile(r"^(表|图)\s*\d+(?:[-.]\d+)*\s*\S")
END_MATTER_RE = re.compile(
    r"^(作者贡献(声明)?|利益冲突(声明)?|数据可用性(声明)?)\s*[：:]?$"
)
ABS_TITLE_SKIP_RE = re.compile(
    r"^(摘\s*要|提\s*要|关\s*键\s*词|关键词|关键字|abstract|key\s*words?|目录|引言)\b",
    re.I,
)


@dataclass
class Block:
    page: int
    order: int
    type: str
    text: str
    bbox: list[float]
    level: int | None = None
    end_page: int = 0

    @property
    def width(self) -> float:
        if len(self.bbox) != 4:
            return 0.0
        return max(0.0, self.bbox[2] - self.bbox[0])

    @property
    def center_x(self) -> float:
        if len(self.bbox) != 4:
            return 0.5
        return (self.bbox[0] + self.bbox[2]) / 2


def _content_text(item: dict[str, Any]) -> str:
    """MinerU 的 content 可能是字符串，也可能是嵌套列表。"""
    parts: list[str] = []
    for piece in item.get("content", []):
        if isinstance(piece, str):
            parts.append(piece)
            continue
        if not isinstance(piece, dict):
            parts.append(str(piece))
            continue
        value = piece.get("content", "")
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, list):
            for sub in value:
                if isinstance(sub, dict):
                    parts.append(str(sub.get("content", "")))
                else:
                    parts.append(str(sub))
        else:
            parts.append(str(value))
    return "".join(parts)


def _clean_value(text: str, keep_newlines: bool = False) -> str:
    # 只做 Unicode 规范化，不做 NFKC，避免把中文的“？”、“（）”改成半角。
    text = unicodedata.normalize("NFC", text or "")
    text = text.replace("​", "").replace("﻿", "")
    text = text.replace("\xa0", " ").replace("　", " ")
    # 去掉 MinerU 偶尔带出的私用区字符，保留正常文字和标点。
    text = "".join(
        ch for ch in text if not (0xE000 <= ord(ch) <= 0xF8FF)
    )
    if keep_newlines:
        lines = [normalize_spaces(line) for line in text.splitlines()]
        return "\n".join(line for line in lines if line).strip()
    return normalize_spaces(text)


def _flatten_blocks(raw: dict[str, Any]) -> list[Block]:
    blocks: list[Block] = []
    order = 0
    for page_no, page in enumerate(raw.get("pages", []), 1):
        page_idx = int(page.get("page_idx", page_no - 1)) + 1
        for item in page.get("blocks", []):
            order += 1
            block_type = str(item.get("type") or "text")
            text = _content_text(item)
            if block_type in TABLE_TYPES:
                text = _clean_value(text, keep_newlines=True)
            else:
                text = _clean_value(text)
            if not text:
                continue
            bbox = item.get("bbox") or []
            if not isinstance(bbox, list) or len(bbox) != 4:
                bbox = []
            level = item.get("level")
            blocks.append(
                Block(
                    page=page_idx,
                    order=order,
                    type=block_type,
                    text=text,
                    bbox=[float(x) for x in bbox],
                    level=int(level) if isinstance(level, int) else None,
                    end_page=page_idx,
                )
            )
    return blocks


def _table_has_caption(text: str) -> bool:
    idx = text.find("<table")
    if idx < 0:
        return False
    return bool(text[:idx].strip(" ：:"))


def _merge_continuation_tables(blocks: list[Block]) -> list[Block]:
    """MinerU 有时会把一张跨页表拆成多个 table 块。

    后一块没有表题、且前面紧挨着的有效块也是表格时，把它的行接回上一张表。
    """
    merged: list[Block] = []
    for block in blocks:
        if block.type in TABLE_TYPES and not _table_has_caption(block.text):
            previous = None
            for item in reversed(merged):
                if item.type not in NOISE_TYPES:
                    previous = item
                    break
            if previous is not None and previous.type in TABLE_TYPES:
                rows = re.findall(
                    r"<tr\b[^>]*>.*?</tr>", block.text, re.S | re.I
                )
                close = previous.text.rfind("</tbody>")
                if close < 0:
                    close = previous.text.rfind("</table>")
                if rows and close >= 0:
                    previous.text = (
                        previous.text[:close]
                        + "".join(rows)
                        + previous.text[close:]
                    )
                    previous.end_page = max(previous.end_page, block.end_page)
                    continue
        merged.append(block)
    return merged


def _is_noise(text: str) -> bool:
    return bool(
        CLASSIFICATION_RE.match(text)
        or PREPRINT_RE.match(text)
        or re.match(r"^This\s+version\s+posted", text, re.I)
        or re.match(r"^ChinaXiv:", text, re.I)
    )


def _looks_like_person(text: str) -> bool:
    value = re.sub(r"^作者\s*[：:]\s*", "", text)
    value = re.sub(r"[（(].*$", "", value).strip()
    value = re.sub(r"[\d¹²³⁴⁵⁶⁷⁸⁹⁰]+", "", value).strip(
        " *†‡∗,，、;；"
    )
    if re.fullmatch(r"[一-鿿]{2,4}", value):
        return True
    if re.fullmatch(r"[A-Za-z][A-Za-z .'\-]{1,59}", value):
        return True
    return False


def _looks_like_name(text: str) -> bool:
    if re.fullmatch(r"[一-鿿]{2,6}", text):
        return True
    if re.fullmatch(
        r"[A-Za-z][A-Za-z.'\-]*(?:\s+[A-Za-z][A-Za-z.'\-]*){1,3}",
        text,
    ):
        return True
    return False


def _parse_author_names(text: str) -> list[str]:
    text = normalize_spaces(text)
    text = re.sub(r"^作者\s*[：:]\s*", "", text)
    text = re.sub(r"[（(][^）)]*[）)]", "", text)
    # “卢玉鑫 1,2”“张淏1”这类上标编号直接贴在人名后面
    text = re.sub(
        r"(?<=[一-鿿])\s*\d+[A-Za-z]?(?:[,，]\d+[A-Za-z]?)*[*†‡∗]*",
        "",
        text,
    )
    text = re.sub(r"(?<=[一-鿿])\s*[¹²³⁴⁵⁶⁷⁸⁹⁰]+", "", text)
    text = re.sub(
        r"(?<=[A-Za-z])\d+[A-Za-z]?(?:[,，]\d+[A-Za-z]?)*"
        r"[*†‡∗¹²³⁴⁵⁶⁷⁸⁹⁰]*",
        "",
        text,
    )
    text = re.sub(r"[*†‡∗]+$", "", text).strip(" ,，、;；.)）]")

    # 英文作者用逗号分隔，名和姓之间的空格不能当分隔符。
    if re.search(r"[A-Za-z]", text) and not re.search(r"[一-鿿]", text):
        text = re.sub(r"[*†‡∗]", "", text)
        text = re.sub(r"\b\d+\b", "", text)
        parts = [
            normalize_spaces(item).strip(" ,，、;；.")
            for item in re.split(r"[，,、;；]+", text)
            if item.strip()
        ]
        return [item for item in parts if _looks_like_name(item)]

    raw = [
        item.strip()
        for item in re.split(r"[，,、;；\s]+", text)
        if item.strip()
    ]
    # “张 三”被空格拆成两个单字时，先合回去。
    merged_raw: list[str] = []
    i = 0
    while i < len(raw):
        if (
            re.fullmatch(r"[一-鿿]", raw[i])
            and i + 1 < len(raw)
            and re.fullmatch(r"[一-鿿]", raw[i + 1])
        ):
            merged_raw.append(raw[i] + raw[i + 1])
            i += 2
        else:
            merged_raw.append(raw[i])
            i += 1

    names: list[str] = []
    for item in merged_raw:
        item = item.strip(" *†‡∗,，.;；")
        if re.fullmatch(r"[\d\W_]+", item):
            continue
        if _looks_like_name(item):
            names.append(item)
    return names


def _clean_affiliation(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = normalize_spaces(text)
    text = re.sub(r"^[\d¹²³⁴⁵⁶⁷⁸⁹⁰]+\s*[.、]?\s*", "", text)
    if text.startswith(("（", "(")) and text.endswith(("）", ")")):
        text = text[1:-1]
    # 同一行里可能有多个并列单位：“（A）（B）” -> “A；B”
    text = text.replace("）（", "；").replace(")（", "；")
    text = text.replace("）(", "；").replace(")(", "；")
    text = re.sub(r"[）)]\s*\d{1,2}\s*[（(]", "；", text)
    text = re.sub(r"^[（(]|[）)]$", "", text)
    # “（1. A大学；2. B学院）”去掉外层括号后，还要再清一次编号。
    text = re.sub(r"^[\d¹²³⁴⁵⁶⁷⁸⁹⁰]+\s*[.、]?\s*", "", text)
    text = re.sub(r"([；;])\s*\d{1,2}\s*[.、]\s*", r"\1", text)
    # 同一行里的第二个单位：“…… 北京 100101 2 中国科学院大学……”
    text = re.sub(r"\s+\d{1,2}\s+(?=[一-鿿]{2,})", "；", text)
    # “a.航海学院”这种学院编号
    text = re.sub(r"(?<![A-Za-z])[a-z]\.(?=[一-鿿])", "", text)
    # 单位后面经常跟着邮箱/电话，检索时不需要。
    text = re.split(
        r"(?:E-?mail|邮箱|联系电话|电话|ORCID)\s*[：:]?",
        text,
        maxsplit=1,
        flags=re.I,
    )[0]
    return normalize_spaces(text).strip("；;，, ")


def _find_index(
    blocks: list[Block],
    start: int,
    predicate: Any,
) -> int | None:
    for i in range(start, len(blocks)):
        if predicate(blocks[i]):
            return i
    return None


def _abstract_heading_parts(block: Block) -> tuple[bool, str]:
    """返回 (是否摘要标题, 标题块里自带的摘要正文)。"""
    if block.type not in HEADING_TYPES | TEXT_TYPES:
        return False, ""
    text = block.text.strip()
    if ABSTRACT_RE.match(text) or ABSTRACT_EN_RE.match(text):
        return True, ""
    match = ABSTRACT_INLINE_RE.match(text)
    if match:
        return True, match.group(1).strip()
    match = ABSTRACT_COLON_RE.match(text)
    if match:
        return True, match.group(1).strip()
    match = ABSTRACT_PLAIN_RE.match(text)
    if match:
        return True, match.group(1).strip()
    return False, ""


def _is_abstract_heading(block: Block) -> bool:
    return _abstract_heading_parts(block)[0]


def _is_keywords_line(block: Block) -> bool:
    return block.type in TEXT_TYPES | HEADING_TYPES and bool(
        KEYWORDS_RE.match(block.text)
    )


def _is_body_heading(block: Block) -> bool:
    if block.type not in HEADING_TYPES:
        return False
    text = block.text
    if ABSTRACT_RE.match(text) or ABSTRACT_EN_RE.match(text):
        return False
    return bool(
        BODY_HEADING_RE.match(text)
        or LIST_HEADING_RE.match(text)
    )


def _is_reference_heading(block: Block) -> bool:
    if block.type not in HEADING_TYPES | TEXT_TYPES:
        return False
    return bool(
        REFERENCE_RE.match(block.text)
        or REFERENCE_EN_RE.match(block.text)
    )


def _front_matter(blocks: list[Block]) -> dict[str, Any]:
    abstract_idx = _find_index(blocks, 0, _is_abstract_heading)

    title = ""
    title_idx: int | None = None
    for i, block in enumerate(blocks):
        if abstract_idx is not None and i >= abstract_idx:
            break
        if block.type == "doc_title" and block.text:
            title = block.text
            title_idx = i
            break

    # 如果 MinerU 没有标出 doc_title，优先取摘要之前第一个像标题的段落。
    if not title and abstract_idx is not None:
        candidates = [
            (i, b)
            for i, b in enumerate(blocks[:abstract_idx])
            if b.type in TEXT_TYPES | HEADING_TYPES
            and not _is_noise(b.text)
            and not any(word in b.text for word in AFFILIATION_WORDS)
            and not _looks_like_person(b.text)
            and not ABS_TITLE_SKIP_RE.match(b.text)
        ]
        candidates.sort(key=lambda item: (item[1].type != "paragraph_title", item[0]))
        if candidates:
            title_idx, title_block = candidates[0]
            title = title_block.text

    title, inline_authors = _split_title_authors(title)
    authors: list[str] = []
    affiliations: list[str] = []
    for name in _parse_author_names(inline_authors):
        if name not in authors:
            authors.append(name)
    if abstract_idx is not None:
        author_end = abstract_idx
    else:
        # 没有识别到摘要时，只在前置区域找作者，避免把正文整段吞进来。
        start = (title_idx + 1) if title_idx is not None else 0
        author_end = min(len(blocks), start + 20)
        for i in range(start, author_end):
            if blocks[i].type in HEADING_TYPES:
                author_end = i
                break

    for block in blocks[(title_idx + 1 if title_idx is not None else 0):author_end]:
        if block.type in NOISE_TYPES or block.type in IMAGE_TYPES:
            continue
        text = block.text
        if not text or _is_noise(text):
            continue
        if CLASSIFICATION_RE.match(text):
            continue

        has_affiliation = any(word in text for word in AFFILIATION_WORDS)
        # 作者已经找到后，后面的“1 南京师范大学……”一般就是单位行，
        # 不要再从里面拆出“南京”这种假作者。
        if has_affiliation and authors:
            cleaned = _clean_affiliation(text)
            if cleaned and cleaned not in affiliations:
                affiliations.append(cleaned)
            continue

        prefix = re.split(r"[（(]", text, 1)[0]
        prefix = re.sub(r"^作者\s*[：:]\s*", "", prefix).strip()
        names = _parse_author_names(prefix)

        # “李宇轩重庆邮电大学……”这种名字和单位粘在一起、又没有括号。
        # 用 jieba 先把开头的人名切出来，比按固定长度硬猜稳。
        if has_affiliation and not re.search(r"[（(]", text):
            tokens = [t for t in jieba.lcut(prefix) if t.strip()]
            next_is_affiliation = (
                len(tokens) > 1
                and any(word in tokens[1] for word in AFFILIATION_WORDS)
            )
            if (
                tokens
                and not next_is_affiliation
                and re.fullmatch(r"[一-鿿]{2,3}", tokens[0])
            ):
                name = tokens[0]
                if (
                    len(name) == 2
                    and len(tokens) > 1
                    and re.fullmatch(r"[一-鿿]", tokens[1])
                ):
                    name += tokens[1]
                names = [name]

        if not names:
            if has_affiliation:
                cleaned = _clean_affiliation(text)
                if cleaned and cleaned not in affiliations:
                    affiliations.append(cleaned)
            continue

        for name in names:
            if name not in authors:
                authors.append(name)

        if has_affiliation:
            inside_parts = re.findall(r"[（(]([^）)]+)[）)]", text)
            for inside in inside_parts:
                if any(word in inside for word in AFFILIATION_WORDS):
                    cleaned = _clean_affiliation(inside)
                    if cleaned and cleaned not in affiliations:
                        affiliations.append(cleaned)
            if not inside_parts and prefix.startswith(names[0]):
                rest = prefix[len(names[0]):]
                cleaned = _clean_affiliation(rest)
                if cleaned and cleaned not in affiliations:
                    affiliations.append(cleaned)

    # 有些期刊把作者信息放在首页脚注的“第一作者/通信作者”里。
    if not authors:
        for block in blocks[: min(len(blocks), 30)]:
            for match in re.finditer(
                r"(?:第一作者|通信作者|作者简介|作者)\s*[：:]?\s*"
                r"([一-鿿]{2,4})(?=[（(，,。；;]|$)",
                block.text,
            ):
                name = match.group(1)
                if name not in authors:
                    authors.append(name)

    abstract = ""
    keywords: list[str] = []
    keywords_idx: int | None = None
    if abstract_idx is not None:
        abstract_start = abstract_idx + 1
        abstract_stop = len(blocks)
        for i in range(abstract_start, len(blocks)):
            block = blocks[i]
            if _is_keywords_line(block):
                keywords_idx = i
                abstract_stop = i
                break
            if block.type == "doc_title" or (
                block.type in HEADING_TYPES and ABSTRACT_EN_RE.match(block.text)
            ):
                abstract_stop = i
                break
        abstract_lines = []
        _, inline_abstract = _abstract_heading_parts(blocks[abstract_idx])
        if inline_abstract:
            abstract_lines.append(inline_abstract)
        for block in blocks[abstract_start:abstract_stop]:
            if block.type not in TEXT_TYPES:
                continue
            if _is_noise(block.text):
                continue
            abstract_lines.append(block.text)
        abstract = "\n".join(abstract_lines).strip()

    if abstract_idx is not None:
        body_candidate = _find_index(blocks, abstract_idx + 1, _is_body_heading)
        keyword_stop = body_candidate if body_candidate is not None else len(blocks)
        # 有些论文先放英文 Abstract，再放中文关键词，所以单独再找一遍。
        if keywords_idx is None:
            for i in range(abstract_idx + 1, keyword_stop):
                if _is_keywords_line(blocks[i]):
                    keywords_idx = i
                    break
        if keywords_idx is not None and keywords_idx < keyword_stop:
            rest = KEYWORDS_RE.sub("", blocks[keywords_idx].text, count=1).strip()
            if not rest:
                for block in blocks[keywords_idx + 1:keyword_stop]:
                    if block.type in TEXT_TYPES and not _is_noise(block.text):
                        rest = block.text.strip()
                        break
            keywords = [
                item.strip()
                for item in re.split(r"[；;，,、]+", rest)
                if item.strip()
            ]
            # “可重构芯片 演进历程 新型构架芯片”这种用空格分隔的
            if len(keywords) == 1 and " " in keywords[0]:
                spaced = [item for item in keywords[0].split() if item]
                if len(spaced) > 1 and all(len(item) <= 10 for item in spaced):
                    keywords = spaced
        else:
            keywords_idx = None

    return {
        "title": title,
        "title_idx": title_idx,
        "authors": authors,
        "affiliations": affiliations,
        "abstract": abstract,
        "abstract_idx": abstract_idx,
        "keywords": keywords,
        "keywords_idx": keywords_idx,
    }


def _body_start(blocks: list[Block], front: dict[str, Any]) -> int:
    start_after = (front.get("keywords_idx") or front.get("abstract_idx") or -1) + 1
    idx = _find_index(blocks, start_after, _is_body_heading)
    if idx is not None:
        return idx

    # 没有编号标题时，取摘要/关键词之后第一个“像章节”的标题。
    for i in range(start_after, len(blocks)):
        block = blocks[i]
        if block.type != "paragraph_title":
            continue
        if ABSTRACT_EN_RE.match(block.text):
            continue
        if _is_keywords_line(block):
            continue
        return i
    return start_after


def _reference_start(blocks: list[Block], body_start: int) -> int | None:
    idx = _find_index(blocks, body_start, _is_reference_heading)
    if idx is not None:
        return idx
    for i in range(body_start, len(blocks)):
        block = blocks[i]
        if block.type != "ref_text":
            continue
        if re.match(r"^[\[［]\d{1,3}[\]］]", block.text):
            return i
        # 英文期刊常用 APA 格式，没有 [1]，但会连续出现很多 ref_text。
        window = blocks[i:i + 20]
        if sum(1 for item in window if item.type == "ref_text") >= 3:
            return i
    return None


def _heading_level(block: Block, stack: list[tuple[int, str]]) -> int:
    text = block.text
    m = re.match(r"^(\d{1,2}(?:\.\d{1,2}){0,3})[.、]?\s+", text)
    if m:
        return len(m.group(1).split("."))
    if LIST_HEADING_RE.match(text) and stack:
        # “（1）（2）（3）”通常是同一级的小标题，不要越挂越深。
        if LIST_HEADING_RE.match(stack[-1][1]):
            return stack[-1][0]
        return min(stack[-1][0] + 1, 4)
    if block.level and block.level >= 1:
        return min(max(int(block.level), 1), 4)
    return 1


def _table_parts(text: str) -> tuple[str, str]:
    idx = text.find("<table")
    if idx < 0:
        return "", text.strip()
    caption = normalize_spaces(text[:idx]).strip(" ：:")
    return caption, text[idx:].strip()


def _normalize_caption(text: str) -> str:
    text = normalize_spaces(text).strip(" ：:")
    text = re.sub(
        r"^((?:表|图)\s*\d+(?:[-.]\d+)*)\s*",
        r"\1 ",
        text,
    )
    return text


def _split_title_authors(text: str) -> tuple[str, str]:
    """有些论文把标题和作者粘在同一个 doc_title 块里。"""
    tokens = list(jieba.cut(text))
    position = 0
    for index, token in enumerate(tokens):
        start = position
        position += len(token)
        if not re.search(r"[\d¹²³⁴⁵⁶⁷⁸⁹⁰]", token):
            continue
        # 第一个数字前面的那个 2～3 字中文词，通常就是第一作者。
        previous = index - 1
        while previous >= 0 and not re.fullmatch(
            r"[一-鿿]{2,3}", tokens[previous]
        ):
            previous -= 1
        if previous < 0:
            continue
        name_start = sum(len(item) for item in tokens[:previous])
        rest = text[name_start:]
        groups = re.findall(
            r"[一-鿿]{2,4}\s*[\d¹²³⁴⁵⁶⁷⁸⁹⁰]+",
            rest,
        )
        if len(groups) >= 2 and name_start >= 4:
            return text[:name_start].strip(), rest.strip()
        break
    return text, ""


def _caption_from_equation(latex: str) -> str:
    """MinerU 偶尔把表题识别成公式，例如 \\text {表7 ...}。"""
    match = re.match(
        r"^\\text\s*\{\s*((?:表|图)\s*\d+(?:[-.]\d+)*[^{}]*)\s*\}$",
        latex.strip(),
    )
    if not match:
        return ""
    return _normalize_caption(match.group(1))


def _clean_body_text(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        line = _clean_value(line)
        if not line or _is_noise(line) or CLASSIFICATION_RE.match(line):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _next_effective_type(
    blocks: list[Block],
    index: int,
    body_end: int,
) -> str:
    for block in blocks[index + 1:body_end]:
        if block.type not in NOISE_TYPES:
            return block.type
    return ""


def _build_sections(
    blocks: list[Block],
    body_start: int,
    body_end: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    sections: list[dict[str, Any]] = []
    stack: list[tuple[int, str]] = []
    current: dict[str, Any] | None = None
    counts = {
        "paragraphs": 0,
        "tables": 0,
        "equations": 0,
        "images_ignored": 0,
        "duplicates": 0,
    }
    seen_text: set[str] = set()
    pending_caption = ""
    pending_text_caption: Block | None = None

    def append_text(block: Block, text: str) -> None:
        nonlocal current
        if current is None:
            return
        text = _clean_body_text(text)
        if not text:
            return
        key = re.sub(r"\s+", "", text)
        if key in seen_text:
            counts["duplicates"] += 1
            return
        seen_text.add(key)
        current["paragraphs"].append(
            {
                "kind": "text",
                "text": text,
                "page_start": block.page,
                "page_end": block.end_page or block.page,
            }
        )
        current["page_start"] = min(current["page_start"], block.page)
        current["page_end"] = max(
            current["page_end"], block.end_page or block.page
        )
        counts["paragraphs"] += 1

    def flush_pending_text() -> None:
        nonlocal pending_caption, pending_text_caption
        if pending_text_caption is None:
            return
        block = pending_text_caption
        pending_text_caption = None
        pending_caption = ""
        append_text(block, block.text)

    for index in range(body_start, body_end):
        block = blocks[index]
        if block.type in NOISE_TYPES:
            continue
        if block.type in IMAGE_TYPES:
            counts["images_ignored"] += 1
            pending_caption = ""
            pending_text_caption = None
            continue
        if block.type == "doc_title":
            continue
        if pending_text_caption is not None and block.type not in TABLE_TYPES:
            flush_pending_text()

        if block.type in HEADING_TYPES:
            text = block.text
            if _is_reference_heading(block) or END_MATTER_RE.match(text):
                break
            if not text:
                continue
            # 有时 MinerU 会把“表10 ……”标成标题，后面紧跟真正的表格。
            # 这里先存起来，等表格块出现时当表题用，不单独建章节。
            if CAPTION_RE.match(text):
                pending_caption = _normalize_caption(text)
                pending_text_caption = None
                continue
            # 二级标题下面又出现“1. ……”时，通常是正文里的编号列表，
            # 不是新章节；把它当正文，避免切出一堆很小的章节。
            if stack and stack[-1][0] >= 2 and LIST_NUMBER_RE.match(text):
                append_text(block, text)
                continue
            pending_caption = ""
            level = _heading_level(block, stack)
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, text))
            path = [title for _, title in stack]
            current = {
                "section_id": f"s{len(sections) + 1}",
                "level": level,
                "title": text,
                "section_path": path,
                "type": "body",
                "page_start": block.page,
                "page_end": block.page,
                "headings": [text],
                "paragraphs": [],
                "text": "",
            }
            sections.append(current)
            seen_text = set()
            continue

        if block.type in TABLE_TYPES:
            if current is None:
                continue
            caption, html = _table_parts(block.text)
            if not caption and pending_caption:
                caption = pending_caption
            elif not caption and pending_text_caption is not None:
                caption = _normalize_caption(pending_text_caption.text)
            pending_caption = ""
            pending_text_caption = None
            if not html:
                continue
            current["paragraphs"].append(
                {
                    "kind": "table",
                    "text": (f"{caption}\n{html}" if caption else html),
                    "html": html,
                    "caption": caption,
                    "page_start": block.page,
                    "page_end": block.end_page or block.page,
                }
            )
            current["page_start"] = min(current["page_start"], block.page)
            current["page_end"] = max(
                current["page_end"], block.end_page or block.page
            )
            counts["tables"] += 1
            counts["paragraphs"] += 1
            continue

        if block.type in EQUATION_TYPES:
            if current is None:
                continue
            latex = block.text.strip()
            if not latex:
                continue
            caption = _caption_from_equation(latex)
            if caption:
                next_type = _next_effective_type(blocks, index, body_end)
                pending_caption = (
                    caption
                    if caption.startswith("表") and next_type in TABLE_TYPES
                    else ""
                )
                pending_text_caption = None
                continue
            pending_caption = ""
            current["paragraphs"].append(
                {
                    "kind": "equation",
                    "text": latex,
                    "latex": latex,
                    "page_start": block.page,
                    "page_end": block.end_page or block.page,
                }
            )
            current["page_start"] = min(current["page_start"], block.page)
            current["page_end"] = max(
                current["page_end"], block.end_page or block.page
            )
            counts["equations"] += 1
            counts["paragraphs"] += 1
            continue

        if block.type not in TEXT_TYPES:
            continue
        if current is None:
            continue
        pending_caption = ""

        text = _clean_body_text(block.text)
        if not text:
            continue
        # 表题有时被当成普通正文，例如“表 9 地区异质性分析”。
        # 只有下一块确实是表格时才当表题；否则仍按正文保留。
        if CAPTION_RE.match(text) and len(text) <= 120:
            next_type = _next_effective_type(blocks, index, body_end)
            if text.startswith("表") and next_type in TABLE_TYPES:
                pending_caption = _normalize_caption(text)
                pending_text_caption = block
                continue
            # 图片不参与检索，图题直接丢掉；普通正文里的“图 1 显示……”保留。
            if text.startswith("图") and next_type in IMAGE_TYPES:
                continue

        append_text(block, text)

    flush_pending_text()

    for section in sections:
        section["text"] = "\n\n".join(
            p["text"] for p in section["paragraphs"] if p.get("text")
        )
    return sections, counts


def _column_check(blocks: list[Block]) -> tuple[int, bool]:
    """用 MinerU 的坐标粗略检查双栏阅读顺序。

    MinerU 已经排好阅读顺序；这里只做“是否可疑”的提示，不重排。
    """
    narrow = [
        b
        for b in blocks
        if b.type in TEXT_TYPES | HEADING_TYPES
        and 0.05 < b.width < 0.55
    ]
    left = [b for b in narrow if b.center_x < 0.47]
    right = [b for b in narrow if b.center_x > 0.53]
    if len(left) < 3 or len(right) < 3:
        return 1, False

    # 左右两栏的块如果大量交错，说明阅读顺序可能有问题。
    sequence = [
        0 if b.center_x < 0.47 else 1
        for b in narrow
        if b.center_x < 0.47 or b.center_x > 0.53
    ]
    inversions = 0
    for i, value in enumerate(sequence):
        if value == 1 and 0 in sequence[i + 1:]:
            inversions += 1
    suspect = inversions > max(2, int(len(narrow) * 0.15))
    return 2, suspect


def _page_reports(blocks: list[Block], raw_pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_page: dict[int, list[Block]] = {}
    for block in blocks:
        by_page.setdefault(block.page, []).append(block)

    reports: list[dict[str, Any]] = []
    for page_no in range(1, len(raw_pages) + 1):
        page_blocks = by_page.get(page_no, [])
        kept = [
            b
            for b in page_blocks
            if b.type not in NOISE_TYPES and b.type not in IMAGE_TYPES
        ]
        columns, suspect = _column_check(page_blocks)
        reports.append(
            {
                "page": page_no,
                "raw_blocks": len(raw_pages[page_no - 1].get("blocks", [])),
                "kept_blocks": len(kept),
                "kept_chars": sum(len(b.text) for b in kept),
                "tables": sum(1 for b in page_blocks if b.type in TABLE_TYPES),
                "equations": sum(1 for b in page_blocks if b.type in EQUATION_TYPES),
                "images": sum(1 for b in page_blocks if b.type in IMAGE_TYPES),
                "columns": columns,
                "reading_order_suspect": suspect,
            }
        )
    return reports


def parse_mineru_json(
    json_path: str | Path,
    paper_id: str | None = None,
    overrides: dict[str, Any] | None = None,
    out_path: str | Path | None = None,
) -> dict[str, Any]:
    json_path = Path(json_path)
    paper_id = paper_id or json_path.stem
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    raw_pages = raw.get("pages", [])
    if not raw_pages:
        raise ValueError(f"MinerU 结果没有页面：{json_path}")

    blocks = _merge_continuation_tables(_flatten_blocks(raw))
    front = _front_matter(blocks)
    title = front["title"] or paper_id
    body_start = _body_start(blocks, front)
    ref_start = _reference_start(blocks, body_start)
    body_end = ref_start if ref_start is not None else len(blocks)
    sections, counts = _build_sections(blocks, body_start, body_end)
    pages = _page_reports(blocks, raw_pages)

    front_end = front.get("abstract_idx")
    front_blocks = blocks[: front_end] if front_end is not None else []
    removed = {
        "页眉页脚": sum(
            1
            for b in blocks
            if b.type in {"header", "footer", "page_number", "aside_text"}
        ),
        "脚注": sum(1 for b in blocks if b.type == "page_footnote"),
        "图片": counts["images_ignored"],
        "参考文献": (
            sum(1 for b in blocks[ref_start:] if b.type == "ref_text")
            if ref_start is not None
            else 0
        ),
        "重复段落": counts["duplicates"],
        "元数据噪声": sum(
            1
            for b in front_blocks
            if _is_noise(b.text) or CLASSIFICATION_RE.match(b.text)
        ),
    }

    suspicious_pages = [
        p["page"]
        for p in pages
        if (
            p["kept_chars"] < 80
            and p["raw_blocks"] > 0
            and p["tables"] == 0
            and p["images"] == 0
        )
        or p["reading_order_suspect"]
    ]
    warnings: list[str] = []
    if not front["title"]:
        warnings.append("没有识别到标题，暂时用论文编号代替。")
    if not front["authors"]:
        warnings.append("没有识别到作者。")
    if not front["affiliations"]:
        warnings.append("没有识别到作者单位。")
    if not front["abstract"]:
        warnings.append("没有识别到中文摘要。")
    if not front["keywords"]:
        warnings.append("没有识别到中文关键词。")
    if ref_start is None:
        warnings.append("没有识别到参考文献标题，可能把参考文献混进正文。")
    if len(sections) < 3:
        warnings.append("识别到的正文章节少于 3 个，请检查标题格式。")
    if suspicious_pages:
        warnings.append(f"这些页面文字很少或双栏顺序可疑，建议抽查：{suspicious_pages}")
    if counts["images_ignored"]:
        warnings.append(f"忽略了 {counts['images_ignored']} 个图片/图表块。")

    meta = {
        "paper_id": paper_id,
        "title": title,
        "authors": front["authors"],
        "affiliations": front["affiliations"],
        "abstract": front["abstract"],
        "keywords": front["keywords"],
        "parser": "mineru-standard",
        "source": "",
        "source_url": "",
        "category": "",
    }
    pdf_path = config.PDF_DIR / f"{paper_id}.pdf"
    if pdf_path.exists():
        meta["pdf"] = str(pdf_path.relative_to(config.ROOT))
        meta["pdf_sha256"] = file_sha256(pdf_path)
    if overrides:
        for key, value in overrides.items():
            if value not in (None, "", [], {}):
                meta[key] = value
    meta.setdefault("title", paper_id)

    quality = {
        "page_count": len(raw_pages),
        "raw_chars": sum(len(b.text) for b in blocks),
        "body_chars": sum(len(s["text"]) for s in sections),
        "paragraph_count": counts["paragraphs"],
        "section_count": len(sections),
        "body_section_count": len(sections),
        "table_count": counts["tables"],
        "equation_count": counts["equations"],
        "image_count": counts["images_ignored"],
        "reference_found": ref_start is not None,
        "reference_start_page": blocks[ref_start].page if ref_start is not None else None,
        "body_start_page": blocks[body_start].page if body_start < len(blocks) else None,
        "removed": removed,
        "suspicious_pages": suspicious_pages,
        "warnings": warnings,
    }
    parsed = {
        "paper_id": paper_id,
        "meta": meta,
        "pages": pages,
        "sections": sections,
        "quality": quality,
    }

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(parsed, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return parsed


def clean_markdown(parsed: dict[str, Any]) -> str:
    """生成一份方便人工检查的 Markdown，不参与入库。"""
    meta = parsed.get("meta", {})
    lines = [f"# {meta.get('title') or parsed.get('paper_id', '')}", ""]
    if meta.get("authors"):
        lines.extend(["作者：" + "、".join(meta["authors"]), ""])
    if meta.get("affiliations"):
        lines.extend(["单位：" + "；".join(meta["affiliations"]), ""])
    if meta.get("abstract"):
        lines.extend(["## 摘要", "", meta["abstract"], ""])
    if meta.get("keywords"):
        lines.extend(["关键词：" + "；".join(meta["keywords"]), ""])

    for section in parsed.get("sections", []):
        level = min(max(int(section.get("level", 2)), 2), 5)
        lines.extend(["#" * level + " " + section.get("title", ""), ""])
        for para in section.get("paragraphs", []):
            kind = para.get("kind")
            if kind == "table":
                if para.get("caption"):
                    lines.extend([para["caption"], ""])
                lines.extend([para.get("html", ""), ""])
            elif kind == "equation":
                lines.extend(["$$", para.get("latex", para.get("text", "")), "$$", ""])
            else:
                lines.extend([para.get("text", ""), ""])
    return "\n".join(lines).rstrip() + "\n"


def iter_mineru_json(directory: str | Path) -> Iterable[Path]:
    return sorted(Path(directory).glob("*.json"))


__all__ = [
    "clean_markdown",
    "iter_mineru_json",
    "parse_mineru_json",
]
