"""PDF → 分页原文 + 章节结构 + 元数据 + 质量报告。

思路：
1. 先按行取出文字和位置（坐标），这样才能处理双栏。
2. 去掉页眉、页脚、脚注、公式/图注这类小字。
3. 按“左栏→右栏”的顺序读，跨栏/跨页的句子会接回去。
4. 用编号（1、2.1、3.2.1）和常见标题词识别章节。
5. 原始分页文字完整保留，方便出问题时回查。
"""

from __future__ import annotations
 
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pymupdf

from .. import config
from ..common.utils import (
    clean_text,
    file_sha256,
    join_lines,
    normalize_for_match,
    normalize_spaces,
)

CJK = re.compile(r"[一-鿿]")
NUMBER_HEADING = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,2})[ 　]+(\S.{0,48})$")
REF_ITEM = re.compile(r"^[\[［]\d{1,3}[\]］]")
LIST_ITEM = re.compile(r"^[（(]\d{1,2}[）)]|^[①-⑳]")
FOOTNOTE_PREFIX = re.compile(
    r"^(收稿日期|修回日期|基金项目|作者简介|通信作者|通信联系人|引用格式|DOI|"
    r"中图分类号|文献标识码|文章编号)[：:]",
    re.I,
)

REFERENCE_HEADING = re.compile(
    r"^\s*[\[【]?\s*参\s*考\s*文\s*献\s*[\]】]?\s*"
    r"(?:[（(]\s*references?\s*[）)])?\s*[：:]?\s*$",
    re.I,
)


def _margin_noise_reason(line: Line, page_width: float) -> str | None:
    """判断页边文字属于哪类噪声；不是噪声则返回 None。"""
    text = normalize_spaces(line.text)

    # 竖排文字只要落在左右页边，基本就是水印/侧边标注。
    # 不依赖“离边缘多少百分比”这种偶然条件，202609.00318 的
    # ChinaXiv:...v1 就是一条旋转 90° 的长行。
    if line.is_vertical and (
        line.cx < 0.15 * page_width or line.cx > 0.85 * page_width
    ):
        return "页边竖排文字"

    if re.fullmatch(r"\d{1,3}", text) and line.x1 < 0.12 * page_width:
        return "页边标注/行号"
    if line.x0 > 0.92 * page_width or line.x1 < 0.08 * page_width:
        return "页边标注/行号"
    if re.match(r"^(作者简介|通信联系人)[：:]", text):
        return "页边标注/行号"
    return None


def _is_margin_noise(line: Line, page_width: float) -> bool:
    return _margin_noise_reason(line, page_width) is not None


@dataclass
class Line:
    page: int
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    size: float
    bold: bool = False
    # PyMuPDF 给出的文字方向；横排是 (1, 0)，竖排通常是 (0, -1)
    direction: tuple[float, float] = (1.0, 0.0)

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def is_vertical(self) -> bool:
        """方向接近竖直，通常是页边水印或侧边标注。"""
        return abs(self.direction[0]) < 0.5


@dataclass
class Block:
    kind: str  # heading / paragraph
    lines: list[Line] = field(default_factory=list)
    text: str = ""
    level: int = 0
    title: str = ""
    section_type: str = "body"

    @property
    def page_start(self) -> int:
        return min(l.page for l in self.lines)

    @property
    def page_end(self) -> int:
        return max(l.page for l in self.lines)


def _is_cjk(text: str) -> bool:
    return bool(CJK.search(text))


def _fingerprint(text: str) -> str:
    """把数字和标点抹掉，用来识别每页重复的页眉页脚。"""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\d+", "#", text)
    text = re.sub(r"[\W_]+", "", text, flags=re.UNICODE)
    return text


def extract_raw_lines(doc: pymupdf.Document) -> tuple[dict[int, list[Line]], list[str]]:
    raw_pages: dict[int, list[Line]] = {}
    raw_texts: list[str] = []
    for pno, page in enumerate(doc):
        lines: list[Line] = []
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                text = "".join(s.get("text", "") for s in spans).strip()
                if not text:
                    continue
                x0, y0, x1, y1 = line["bbox"]
                size = max((s.get("size", 0) for s in spans), default=0)
                bold = any(
                    (s.get("flags", 0) & 16) or "Bold" in s.get("font", "")
                    for s in spans
                )
                direction = line.get("dir", (1.0, 0.0))
                lines.append(
                    Line(
                        pno + 1,
                        text,
                        x0,
                        y0,
                        x1,
                        y1,
                        size,
                        bold,
                        (float(direction[0]), float(direction[1])),
                    )
                )
        raw_pages[pno + 1] = lines
        raw_texts.append(page.get_text("text"))
    return raw_pages, raw_texts


def _body_font_size(
    raw_pages: dict[int, list[Line]], ref_start: tuple[int, float] | None
) -> float:
    """按字符数加权找正文字号；参考文献字小且量很大，先排除。"""
    weights: Counter[float] = Counter()
    for pno, lines in raw_pages.items():
        for line in lines:
            if ref_start and (pno > ref_start[0] or (pno == ref_start[0] and line.y0 >= ref_start[1] - 2)):
                continue
            weights[round(line.size, 1)] += len(line.text)
    if not weights:
        for lines in raw_pages.values():
            for line in lines:
                weights[round(line.size, 1)] += len(line.text)
    return weights.most_common(1)[0][0] if weights else 10.0


def _find_reference_start(raw_pages: dict[int, list[Line]]) -> tuple[int, float] | None:
    for pno in sorted(raw_pages):
        for line in sorted(raw_pages[pno], key=lambda l: (l.y0, l.x0)):
            if REFERENCE_HEADING.match(normalize_spaces(line.text)):
                return pno, line.y0
    return None


def _is_after_reference(line: Line, ref_start: tuple[int, float] | None) -> bool:
    if ref_start is None:
        return False
    ref_page, ref_y = ref_start
    return line.page > ref_page or (line.page == ref_page and line.y0 >= ref_y - 2)


def _is_front_matter_line(
    line: Line, cutoff: tuple[int, float] | None
) -> bool:
    """判断这一行是否在第一个正文标题之前（标题/作者/摘要/关键词区域）。"""
    if cutoff is None:
        # 找不到正文标题时，至少把第一页当作前置内容
        return line.page == 1
    page, y = cutoff
    return line.page < page or (line.page == page and line.y0 < y - 2)


def _running_pattern(text: str) -> bool:
    text = normalize_spaces(text)
    patterns = [
        r"^第\s*\d+\s*(卷|期)",
        r"^Vol\.?\s*\d+",
        r"^\d{4}\s*年(\s*\d+\s*月)?$",
        r"^October\s+\d{4}$",
        r"^https?://",
        r"^微\s*电\s*子\s*学\s*与\s*计\s*算\s*机",
        r"^MICROELECTRONICS",
        r"^.{2,12}[，,]\s*等\s*[：:]",
    ]
    return any(re.search(p, text, re.I) for p in patterns)


def _looks_like_page_number(
    line: Line, body_size: float, page_height: float
) -> bool:
    """页码：在页边、纯数字，而且字号不能明显大于正文。

    第 3 页顶部的“2”虽然也是纯数字，但字号 14.1 比正文 12 大，
    它是章节标题编号，不是页码，所以这里要放行。
    """
    text = normalize_spaces(line.text)
    if not re.fullmatch(r"[-—–]?\s*\d{1,4}\s*[-—–]?", text):
        return False
    if line.size > body_size + 0.5:
        return False
    in_margin = (
        line.y0 < config.HEADER_RATIO * page_height
        or line.y0 > config.FOOTER_RATIO * page_height
    )
    return in_margin


def _record_removed(
    removed: Counter,
    debug: list[dict[str, Any]] | None,
    reason: str,
    line: Line,
) -> None:
    """统计删除数量；调试模式下同时记下这一行为什么被删。"""
    removed[reason] += 1
    if debug is None:
        return
    debug.append(
        {
            "page": line.page,
            "reason": reason,
            "text": line.text,
            "bbox": [
                round(line.x0, 1),
                round(line.y0, 1),
                round(line.x1, 1),
                round(line.y1, 1),
            ],
            "size": round(line.size, 1),
            "direction": [
                round(line.direction[0], 3),
                round(line.direction[1], 3),
            ],
        }
    )


def filter_lines(
    raw_pages: dict[int, list[Line]],
    page_width: float,
    page_height: float,
    body_size: float,
    ref_start: tuple[int, float] | None,
    front_cutoff: tuple[int, float] | None = None,
    debug: list[dict[str, Any]] | None = None,
) -> tuple[dict[int, list[Line]], Counter]:
    """去掉页眉页脚、脚注和小字号辅助文字。

    debug 传入一个空列表时，会逐行记录“哪一行因为什么被删”，
    只给检查命令用，不影响正常解析。
    """
    removed: Counter = Counter()

    # 先统计“在页边距里重复出现”的文本，跨页重复的才删
    marginal_pages: dict[str, set[int]] = defaultdict(set)
    for pno, lines in raw_pages.items():
        for line in lines:
            if line.y0 < config.HEADER_RATIO * page_height or line.y0 > 0.90 * page_height:
                fp = _fingerprint(line.text)
                if fp:
                    marginal_pages[fp].add(pno)

    kept: dict[int, list[Line]] = {}
    for pno, lines in raw_pages.items():
        out: list[Line] = []
        for line in lines:
            text = normalize_spaces(line.text)
            in_top = line.y0 < config.HEADER_RATIO * page_height
            in_bottom = line.y0 > config.FOOTER_RATIO * page_height
            marginal = line.y0 < config.HEADER_RATIO * page_height or line.y0 > 0.90 * page_height
            fp = _fingerprint(text)

            # 页码要先判断，避免被后面的“页眉页脚”规则抢先，
            # 也避免把字号较大的章节编号“2”误当页码。
            if _looks_like_page_number(line, body_size, page_height):
                _record_removed(removed, debug, "页码", line)
                continue
            if in_bottom:
                _record_removed(removed, debug, "页眉页脚", line)
                continue
            if in_top and line.size <= body_size + 0.5 and _running_pattern(text):
                _record_removed(removed, debug, "页眉页脚", line)
                continue
            if (
                marginal
                and fp
                and len(marginal_pages.get(fp, ())) >= 3
                and len(fp) >= 2
                and line.size <= body_size + 0.5
            ):
                _record_removed(removed, debug, "页眉页脚", line)
                continue
            if FOOTNOTE_PREFIX.match(text) or text.startswith("http"):
                _record_removed(removed, debug, "脚注", line)
                continue
            margin_reason = _margin_noise_reason(line, page_width)
            if margin_reason:
                _record_removed(removed, debug, margin_reason, line)
                continue
            # 第 2 页以后，页面顶部有时是上一页摘要/正文的续行，
            # 字号会比中文正文小一点（例如英文摘要 10.4 vs 正文 12）。
            # 只对“接近正文大小的小字”放行，真正很小的公式/图注仍会删。
            top_continuation = (
                line.page > 1
                and line.y0 < config.HEADER_RATIO * page_height
                and line.size >= body_size - 2.0
            )
            # 首页标题/作者/摘要/关键词区域经常使用比正文小的字号，
            # 不能按正文标准当公式/表格删掉；但 8 号以下的真正小字
            # （上标、脚注标记等）仍然删除。
            front_matter_text = (
                _is_front_matter_line(line, front_cutoff)
                and line.size >= 8.0
            )
            if (
                line.size < body_size - config.SMALL_FONT_DELTA
                and not front_matter_text
                and not top_continuation
                and not _is_after_reference(line, ref_start)
            ):
                _record_removed(removed, debug, "小字号(公式/表格/图注)", line)
                continue
            out.append(line)
        kept[pno] = out
    return kept, removed


def _merge_same_line_fragments(lines: list[Line], page_width: float) -> list[Line]:
    """把同一行里被拆开的小块（如正文和上标引用）合并。"""
    if not lines:
        return []
    lines = sorted(lines, key=lambda l: (round(l.cy / 3), l.x0))
    split = page_width / 2
    groups: list[list[Line]] = []
    for line in lines:
        placed = False
        for group in reversed(groups[-6:]):
            last = group[-1]
            overlap = min(last.y1, line.y1) - max(last.y0, line.y0)
            min_h = min(last.height, line.height) or 1
            gap = line.x0 - last.x1
            same_side = (last.cx < split) == (line.cx < split)
            becomes_full = (max(last.x1, line.x1) - min(group[0].x0, line.x0)) > 0.55 * page_width
            # 章节编号和标题之间通常有 12～15 的空隙，比普通词间距大。
            # 只有“上一段是纯编号”时才放宽到 20，避免把普通正文误合并。
            is_heading_number = (
                re.fullmatch(
                    r"\d{1,2}(?:\.\d{1,2}){0,2}",
                    normalize_spaces(last.text),
                )
                is not None
            )
            max_gap = 20 if is_heading_number else 12
            if overlap > 0.5 * min_h and -2 <= gap <= max_gap and (same_side or becomes_full):
                group.append(line)
                placed = True
                break
        if not placed:
            groups.append([line])

    merged: list[Line] = []
    for group in groups:
        group.sort(key=lambda l: l.x0)
        text = group[0].text
        for nxt in group[1:]:
            if (
                re.fullmatch(
                    r"\d{1,2}(?:\.\d{1,2}){0,2}",
                    normalize_spaces(text),
                )
                and not nxt.text[:1].isdigit()
            ):
                # “3” + “方法与体系结构设计”要变成“3 方法与…”，
                # 否则编号识别规则要求中间有空格，会认不出来。
                text = text.rstrip() + " " + nxt.text.lstrip()
            else:
                text = join_lines(text, nxt.text)
        merged.append(
            Line(
                page=group[0].page,
                text=text,
                x0=min(l.x0 for l in group),
                y0=min(l.y0 for l in group),
                x1=max(l.x1 for l in group),
                y1=max(l.y1 for l in group),
                size=max(l.size for l in group),
                bold=any(l.bold for l in group),
                direction=group[0].direction,
            )
        )
    return merged


def _vertical_regions(lines: list[Line], gap_threshold: float = 45.0) -> list[list[Line]]:
    """按横向空白带把页面切成上下几块（表格/整页图会把版面隔开）。"""
    if not lines:
        return []
    intervals: list[list[float]] = []
    for y0, y1 in sorted((l.y0, l.y1) for l in lines):
        if not intervals or y0 - intervals[-1][1] > gap_threshold:
            intervals.append([y0, y1])
        else:
            intervals[-1][1] = max(intervals[-1][1], y1)

    regions: list[list[Line]] = [[] for _ in intervals]
    for line in lines:
        idx = 0
        while idx < len(intervals) and intervals[idx][1] < line.cy:
            idx += 1
        idx = min(idx, len(intervals) - 1)
        regions[idx].append(line)
    return [r for r in regions if r]


def _order_region(lines: list[Line], page_width: float) -> list[Line]:
    """一块内部：先处理横跨两栏的行，再按左栏、右栏读。"""
    split = page_width / 2
    full_ids = {id(l) for l in lines if l.width > 0.55 * page_width}
    full = [l for l in lines if id(l) in full_ids]
    left = [l for l in lines if id(l) not in full_ids and l.cx < split]
    right = [l for l in lines if id(l) not in full_ids and l.cx >= split]
    full.sort(key=lambda l: (l.y0, l.x0))
    left.sort(key=lambda l: (l.y0, l.x0))
    right.sort(key=lambda l: (l.y0, l.x0))

    band_of: dict[int, int] = {}
    full_cys = [f.cy for f in full]
    for line in left + right:
        idx = 0
        while idx < len(full_cys) and full_cys[idx] < line.cy:
            idx += 1
        band_of[id(line)] = idx

    ordered: list[Line] = []
    for band in range(len(full) + 1):
        ordered.extend(l for l in left if band_of.get(id(l)) == band)
        ordered.extend(l for l in right if band_of.get(id(l)) == band)
        if band < len(full):
            ordered.append(full[band])
    return ordered


def _page_order(lines: list[Line], page_width: float) -> list[Line]:
    """一页的阅读顺序：先按大块空白分区，每区再左栏→右栏。"""
    ordered: list[Line] = []
    for region in _vertical_regions(lines):
        ordered.extend(_order_region(region, page_width))
    return ordered


def detect_heading(line: Line) -> tuple[int, str, str] | None:
    """返回 (层级, 标题, 类型) 或 None。"""
    text = normalize_spaces(line.text)
    if len(text) > 52:
        return None

    if REFERENCE_HEADING.match(text):
        return 1, "参考文献", "references"
    if re.match(r"^(致\s*谢|作者简介|附\s*录)[：:]?$", text):
        return 1, text.rstrip("：:"), "backmatter"
    if re.match(r"^(引\s*言|绪\s*论|相关工作|研究方法|实验(与结果)?|结果与分析|"
                r"结\s*论|结束语|总结与展望)$", text):
        return 1, text, "body"

    m = NUMBER_HEADING.match(text)
    if m:
        number, title = m.group(1), m.group(2).strip()
        # 排除“365 Copilot ...”这类正文；编号最多两位
        if len(number.split(".")[0]) <= 2 and title and not title.endswith(("。", "，", "；", ",", ";")):
            # 标题一般较短，且不以“的、了”等正文语气收尾
            if len(title) <= 40:
                return number.count(".") + 1, text, "body"
    return None


def _find_first_body_heading(
    raw_pages: dict[int, list[Line]],
    page_width: float,
    body_size: float,
) -> tuple[int, float] | None:
    """找到第一个正文标题，用来划分“前置内容”和“正文”。

    返回 (页码, 标题顶部 y 坐标)；找不到则返回 None。
    前置内容指标题/作者/单位/摘要/关键词，可能跨到第 2 页。
    """
    for pno in sorted(raw_pages):
        lines = _merge_same_line_fragments(raw_pages[pno], page_width)
        ordered = _page_order(lines, page_width)
        for line in ordered:
            heading = detect_heading(line)
            if not heading or heading[2] != "body":
                continue
            # 英文摘要里的“Web 4.0 is ...”也可能被误认成编号标题。
            # 真正的正文标题字号不会比正文小。
            if line.size >= body_size - 0.5:
                return pno, line.y0
    return None


def _ends_sentence(text: str) -> bool:
    return text.rstrip().endswith(("。", "！", "？", "!", "?", "；", ";"))


def assemble_blocks(
    ordered: list[Line],
    page_width: float,
    page_height: float,
    front_cutoff: tuple[int, float] | None = None,
) -> list[Block]:
    """把行拼成段落，并把章节标题单独拆出来。"""
    blocks: list[Block] = []
    current: list[Line] = []

    # 每页每栏的左边距，用来判断“首行缩进”
    mins: dict[tuple[int, str], float] = {}
    for line in ordered:
        if line.width > 0.55 * page_width:
            continue
        side = "L" if line.cx < page_width / 2 else "R"
        key = (line.page, side)
        mins[key] = min(mins.get(key, line.x0), line.x0)

    def flush() -> None:
        nonlocal current
        if not current:
            return
        text = current[0].text
        for nxt in current[1:]:
            text = join_lines(text, nxt.text)
        blocks.append(Block("paragraph", list(current), clean_text(text)))
        current = []

    prev: Line | None = None
    for line in ordered:
        # 前置内容里也有“4.0 is”这类会被误认成编号标题的文字，
        # 在第一个正文标题之前不做标题识别。
        heading = (
            None
            if _is_front_matter_line(line, front_cutoff)
            else detect_heading(line)
        )
        if heading:
            flush()
            level, title, sec_type = heading
            blocks.append(
                Block(
                    "heading",
                    [line],
                    clean_text(line.text),
                    level=level,
                    title=title,
                    section_type=sec_type,
                )
            )
            prev = None
            continue

        side = "L" if line.cx < page_width / 2 else "R"
        left_min = mins.get((line.page, side), line.x0)
        indent = line.x0 - left_min > 10

        new_para = False
        if prev is None:
            new_para = True
        else:
            same_place = prev.page == line.page and (
                (prev.cx < page_width / 2) == (line.cx < page_width / 2)
            )
            gap = line.y0 - prev.y1 if same_place else 99
            if indent and (_ends_sentence(prev.text) or (same_place and gap > 8)):
                new_para = True
            elif _ends_sentence(prev.text) and same_place and gap > 6:
                new_para = True
            elif same_place and gap > 12:
                new_para = True
            elif LIST_ITEM.match(normalize_spaces(line.text)) or REF_ITEM.match(normalize_spaces(line.text)):
                new_para = True
            elif prev.text.rstrip().endswith(("：", ":")) and line.text[:1] in "（(①":
                new_para = True

        if new_para:
            flush()
        current.append(line)
        prev = line

    flush()
    return blocks


def _line_texts(lines: Iterable[Line]) -> str:
    return "\n".join(clean_text(l.text) for l in lines)


def _looks_like_title_continuation(text: str) -> bool:
    """标题的副标题，常见写法是“——……”或“--……”。"""
    return bool(re.match(r"^[—\-–－]{1,3}\s*\S", normalize_spaces(text)))


_AUTHOR_STOP_WORDS = (
    "摘要",
    "关键词",
    "中图分类号",
    "文献标识码",
    "文章编号",
    "Abstract",
    "Key words",
    "DOI",
    "收稿日期",
    "基金项目",
    "作者简介",
    "通信作者",
    "预印本",
    "投稿前",
    "同行评审",
)


def _clean_author_text(text: str) -> str:
    text = normalize_spaces(text)
    text = re.sub(r"^作者\s*[：:]\s*", "", text)
    # 作者和单位写在同一行时，先把括号里的单位去掉。
    text = re.sub(r"[（(][^）)]*[）)]", "", text)
    return text.strip(" ,，、;；")


def _author_score(text: str) -> int:
    """给作者候选行打分，避免把副标题或单位行当成作者。"""
    raw = normalize_spaces(text)
    if not raw or len(raw) > 100:
        return -999
    if any(word in raw for word in _AUTHOR_STOP_WORDS):
        return -999
    if _looks_like_title_continuation(raw):
        return -999
    if re.search(r"^[\[【]", raw) or raw.endswith("。"):
        return -999
    if re.search(r"[：:]", raw) and not re.match(r"^作者\s*[：:]", raw):
        return -999

    cleaned = _clean_author_text(raw)
    if not cleaned or not _is_cjk(cleaned):
        return -999
    if any(word in cleaned for word in ("大学", "学院", "研究所", "实验室")):
        return -999

    score = 0
    # 只有 2～4 个汉字的一行，通常就是独立成一行的作者名
    if re.fullmatch(r"[一-鿿]{2,4}", cleaned):
        score += 5
    # 常见作者行：2～4 个汉字的人名开头，后面跟空格、逗号或上标
    if re.match(r"^[一-鿿]{2,4}(?=[\s，,、;；*†‡\d])", cleaned):
        score += 3
    if re.search(r"[（(]", raw):
        score += 1
    if re.search(r"\d", cleaned):
        score += 1
    if re.search(r"[,，、;；\s]", cleaned):
        score += 1
    if 2 <= len(cleaned) <= 50:
        score += 1
    if len(cleaned) > 60 and not re.search(r"[,，、;；\s]", cleaned):
        score -= 2
    return score


def _extract_metadata(
    raw_first_page: list[Line],
    paper_id: str,
    doc: pymupdf.Document,
    page_width: float,
) -> dict[str, Any]:
    raw_lines = sorted(raw_first_page, key=lambda l: (l.y0, l.x0))
    lines = [l for l in raw_lines if not _is_margin_noise(l, page_width)]
    # 首页的标签和内容经常被 PDF 拆成同一行的多个片段
    # （例如“关键词”+“智能生态网络；……”），先合并再提取。
    front_lines = sorted(
        _merge_same_line_fragments(raw_first_page, page_width),
        key=lambda l: (l.y0, l.x0),
    )
    front_lines = [
        l for l in front_lines if not _is_margin_noise(l, page_width)
    ]
    meta: dict[str, Any] = {}

    # 标题：首页最大字号里的中文行
    chinese_big = [l for l in lines if l.size >= 14 and _is_cjk(l.text)]
    if chinese_big:
        max_size = max(l.size for l in chinese_big)
        title_lines = [l for l in chinese_big if abs(l.size - max_size) < 0.3]
        title_lines.sort(key=lambda l: l.y0)
        title = title_lines[0].text
        for extra in title_lines[1:]:
            title = join_lines(title, extra.text)
        meta["title"] = normalize_spaces(title)
        title_y = title_lines[-1].y1

        # 副标题常比主标题小半号，不参与最大字号筛选，单独接上。
        for line in lines:
            if line.y0 <= title_y:
                continue
            if line.y0 - title_y > 24:
                break
            if (
                _looks_like_title_continuation(line.text)
                and line.size >= max_size - 4
                and len(CJK.findall(line.text)) >= 4
            ):
                meta["title"] = normalize_spaces(
                    join_lines(meta["title"], line.text)
                )
                title_y = line.y1
                break
    else:
        meta["title"] = ""
        title_y = 0

    # 摘要开始的位置是作者区域的硬边界，避免把摘要正文当作者。
    abstract_y = next(
        (
            l.y0
            for l in front_lines
            if re.match(
                r"^(摘\s*要|Abstract)\s*[：:]?",
                normalize_spaces(l.text),
                re.I,
            )
        ),
        title_y + 200,
    )

    # 作者：标题和摘要之间、字号 9～13，按“像人名”的程度挑最像的一行。
    # 有些 PDF 会把同一行的几位作者拆到左右两栏，这里按纵向重叠合并。
    author_candidates = [
        l
        for l in front_lines
        if title_y < l.y0 < abstract_y
        and 9 <= l.size <= 13.5
        and _is_cjk(l.text)
        and len(l.text) < 100
    ]
    author_bands: list[list[Line]] = []
    for line in sorted(author_candidates, key=lambda l: (l.y0, l.x0)):
        if author_bands:
            last = author_bands[-1][-1]
            overlap = min(last.y1, line.y1) - max(last.y0, line.y0)
            min_height = min(last.height, line.height) or 1
            if overlap > 0.5 * min_height:
                author_bands[-1].append(line)
                continue
        author_bands.append([line])

    scored: list[tuple[int, str, Line]] = []
    for band in author_bands:
        band.sort(key=lambda l: l.x0)
        text = band[0].text
        for nxt in band[1:]:
            text = join_lines(text, nxt.text)
        scored.append((_author_score(text), text, band[-1]))
    scored.sort(key=lambda item: (-item[0], item[2].y0))

    if scored and scored[0][0] >= 5:
        _, author_text, author_last = scored[0]
        author_text = _clean_author_text(author_text)
        # 去掉名字后面的上标编号：纪馨玉1,2,3,4 -> 纪馨玉
        author_text = re.sub(
            r"(?<=[一-鿿])\s*\d+(?:[,，]\d+)*[*†‡]*", "", author_text
        )
        author_text = re.sub(r"[*†‡​]+$", "", author_text).strip(" ,，、;；")
        raw_names = [
            a.strip()
            for a in re.split(r"[，,、;；\s]+", author_text)
            if a.strip()
            and a.strip() not in {"等", "et al", "et al."}
            and len(a.strip()) <= 20
        ]
        names: list[str] = []
        for name in raw_names:
            if re.fullmatch(r"[\d\W_]+", name):
                continue
            # PDF 有时把“刘徽”拆成“刘 徽”两个单字，接回去
            if (
                re.fullmatch(r"[一-鿿]", name)
                and names
                and re.fullmatch(r"[一-鿿]", names[-1])
            ):
                names[-1] += name
                continue
            names.append(name)
        meta["authors"] = [
            re.sub(r"(?<=[一-鿿])\s+(?=[一-鿿])", "", n).strip()
            for n in names
        ]
        author_y = author_last.y1
    else:
        meta["authors"] = []
        author_y = title_y
    affiliation_parts: list[str] = []
    for l in front_lines:
        text = normalize_spaces(l.text)
        if not (author_y <= l.y0 < abstract_y):
            continue
        if re.match(r"^(通讯作者|通信作者|收稿日期|基金项目)", text):
            continue
        if any(
            k in text
            for k in ("大学", "学院", "研究所", "公司", "重点实验室")
        ):
            affiliation_parts.append(text.strip("（）() "))

    affiliation_text = ""
    for part in affiliation_parts:
        affiliation_text = (
            join_lines(affiliation_text, part) if affiliation_text else part
        )
    affiliations = [
        a.strip()
        for a in re.split(r"[；;]", affiliation_text)
        if a.strip()
    ]
    meta["affiliations"] = affiliations[:5]

    # DOI：可能在页边竖排，所以从原始行里找
    for line in raw_lines:
        m = re.search(r"DOI[：:]\s*(\S+)", normalize_spaces(line.text), re.I)
        if m:
            meta["doi"] = m.group(1).rstrip("．。")
            break
    meta.setdefault("doi", "")

    # 期刊、年份、卷期、页码：从引用格式行里找
    journal = year = volume = issue = pages = ""
    for line in lines:
        text = unicodedata.normalize("NFKC", line.text)
        m = re.search(
            r"\[J\]\.\s*([^,]+),\s*(\d{4}),\s*(\d+)\s*\((\d+)\)\s*:\s*([0-9\-–]+)",
            text,
        )
        if m:
            journal, year, volume, issue, pages = (g.strip() for g in m.groups())
            break
    if not journal:
        for line in lines:
            text = normalize_spaces(line.text)
            if re.search(r"微\s*电\s*子\s*学\s*与\s*计\s*算\s*机", text):
                journal = re.sub(r"\s+", "", text.split("http")[0])
                break
    if not year:
        m = re.search(r"(20\d{2})", meta.get("doi", ""))
        year = m.group(1) if m else ""
    meta.update(
        {
            "journal": journal,
            "year": int(year) if year.isdigit() else None,
            "volume": volume,
            "issue": issue,
            "pages": pages,
            "language": "zh" if _is_cjk(meta.get("title", "")) else "en",
        }
    )

    # 摘要：标签可能带冒号、没有冒号，或者和内容被拆成同一行的两个片段
    abstract = ""
    abstract_parts: list[str] = []
    in_abstract = False
    abstract_size = 0.0
    for line in front_lines:
        text = normalize_spaces(line.text)
        if not in_abstract:
            m = re.match(r"^摘\s*要\s*[：:]?\s*(.*)$", text)
            if m:
                in_abstract = True
                abstract_size = line.size
                content = m.group(1).strip()
                if content:
                    abstract_parts.append(content)
            continue
        if re.match(
            r"^(关键词|关键字|中图分类号|文献标识码|文章编号|Abstract|Key\s*words?|DOI)",
            text,
            re.I,
        ):
            break
        # 字号突然变大，通常是进入了英文标题
        if line.size > abstract_size + 2.0:
            break
        abstract_parts.append(text)

    if abstract_parts:
        abstract = abstract_parts[0]
        for part in abstract_parts[1:]:
            abstract = join_lines(abstract, part)
    meta["abstract"] = clean_text(abstract)

    # 关键词：PDF 经常把“关键词”和内容拆成同一行的两个片段，
    # 内容还可能跨行，所以先合并同一行碎片，再收集连续行。
    keyword_source = front_lines
    keyword_parts: list[str] = []
    in_keywords = False
    keyword_size = 0.0
    for line in keyword_source:
        text = normalize_spaces(line.text)
        if not in_keywords:
            m = re.match(r"^(关键词|关键字)\s*[：:]?\s*(.*)$", text)
            if m:
                in_keywords = True
                keyword_size = line.size
                content = m.group(2).strip()
                if content:
                    keyword_parts.append(content)
            continue

        # 遇到这些标记，说明关键词已经结束
        if re.match(
            r"^(中图分类号|文献标识码|文章编号|Abstract|Key\s*words?|DOI|"
            r"收稿日期|修回日期|基金项目|作者简介|通信作者)",
            text,
            re.I,
        ):
            break
        # 字号突然变大，通常是进入了英文标题
        if line.size > keyword_size + 1.0:
            break
        # 纯英文的作者/单位行，且没有关键词分隔符，也停
        if not CJK.search(text) and not re.search(r"[；;，,]", text):
            break
        keyword_parts.append(text)

    kw_text = ""
    for part in keyword_parts:
        kw_text = join_lines(kw_text, part) if kw_text else part
    keywords = [
        k.strip()
        for k in re.split(r"[；;，,]", kw_text)
        if k.strip()
    ]
    meta["keywords"] = keywords

    # 英文摘要（有就存，没有不影响）
    en_abstract_lines: list[str] = []
    in_en = False
    for line in front_lines:
        text = normalize_spaces(line.text)
        m = re.match(r"^Abstract\s*[：:]?\s*(.*)$", text, re.I)
        if m:
            in_en = True
            content = m.group(1).strip()
            en_abstract_lines = [content] if content else []
            continue
        if in_en:
            if re.match(r"^(Key\s*words?|关键词|中图分类号)", text, re.I):
                break
            en_abstract_lines.append(text)
    if en_abstract_lines and _is_cjk(meta["abstract"]):
        en = en_abstract_lines[0]
        for part in en_abstract_lines[1:]:
            en = join_lines(en, part)
        meta["abstract_en"] = clean_text(en)

    meta["paper_id"] = paper_id
    meta["page_count"] = doc.page_count
    return meta


def _section_type_for_title(title: str) -> str:
    if re.search(r"参考文献", title):
        return "references"
    if re.search(r"致谢|作者简介|附录", title):
        return "backmatter"
    return "body"


def build_sections(blocks: list[Block], meta: dict[str, Any]) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    stack: dict[int, str] = {}
    current: dict[str, Any] | None = None
    sec_no = 0

    for block in blocks:
        if block.kind == "heading":
            level = block.level
            title = block.title
            stack[level] = title
            for deeper in [k for k in stack if k > level]:
                stack.pop(deeper, None)
            path = [stack[k] for k in sorted(stack) if k <= level]
            sec_no += 1
            sec_type = block.section_type
            if sec_type == "body":
                sec_type = _section_type_for_title(title)
            current = {
                "section_id": f"s{sec_no}",
                "level": level,
                "title": title,
                "section_path": path,
                "type": sec_type,
                "page_start": block.page_start,
                "page_end": block.page_end,
                "headings": [title],
                "paragraphs": [],
                "text": "",
            }
            sections.append(current)
            continue

        if current is None:
            continue  # 首页标题、作者、摘要等前置内容，上面单独处理
        current["paragraphs"].append(
            {
                "text": block.text,
                "page_start": block.page_start,
                "page_end": block.page_end,
            }
        )
        current["page_start"] = min(current["page_start"], block.page_start)
        current["page_end"] = max(current["page_end"], block.page_end)

    # 把摘要/关键词补到最前面，方便阅读
    front: list[dict[str, Any]] = []
    if meta.get("abstract"):
        front.append(
            {
                "section_id": "s0",
                "level": 1,
                "title": "摘要",
                "section_path": ["摘要"],
                "type": "abstract",
                "page_start": 1,
                "page_end": 1,
                "headings": ["摘要"],
                "paragraphs": [{"text": meta["abstract"], "page_start": 1, "page_end": 1}],
                "text": meta["abstract"],
            }
        )
    if meta.get("keywords"):
        kw_text = "；".join(meta["keywords"])
        front.append(
            {
                "section_id": "s0k",
                "level": 1,
                "title": "关键词",
                "section_path": ["关键词"],
                "type": "keywords",
                "page_start": 1,
                "page_end": 1,
                "headings": ["关键词"],
                "paragraphs": [{"text": kw_text, "page_start": 1, "page_end": 1}],
                "text": kw_text,
            }
        )

    for sec in sections:
        sec["text"] = "\n\n".join(p["text"] for p in sec["paragraphs"])
    return front + sections


def parse_pdf(
    pdf_path: str | Path,
    paper_id: str | None = None,
    overrides: dict[str, Any] | None = None,
    out_path: str | Path | None = None,
) -> dict[str, Any]:
    pdf_path = Path(pdf_path)
    paper_id = paper_id or pdf_path.stem
    doc = pymupdf.open(pdf_path)
    if doc.page_count == 0:
        raise ValueError(f"PDF 没有页面：{pdf_path}")

    first = doc[0]
    page_width, page_height = first.rect.width, first.rect.height

    raw_pages, raw_texts = extract_raw_lines(doc)
    ref_start = _find_reference_start(raw_pages)
    body_size = _body_font_size(raw_pages, ref_start)
    front_cutoff = _find_first_body_heading(raw_pages, page_width, body_size)
    debug: list[dict[str, Any]] = []
    kept, removed = filter_lines(
        raw_pages,
        page_width,
        page_height,
        body_size,
        ref_start,
        front_cutoff=front_cutoff,
        debug=debug,
    )

    ordered_all: list[Line] = []
    page_reports: list[dict[str, Any]] = []
    for pno in range(1, doc.page_count + 1):
        lines = _merge_same_line_fragments(kept.get(pno, []), page_width)
        ordered = _page_order(lines, page_width)
        ordered_all.extend(ordered)
        page_reports.append(
            {
                "page": pno,
                "raw_chars": len(raw_texts[pno - 1]),
                "body_chars": sum(len(l.text) for l in ordered),
                "raw_lines": len(raw_pages.get(pno, [])),
                "kept_lines": len(ordered),
                "small_font_dropped": sum(
                    1
                    for item in debug
                    if item["page"] == pno
                    and item["reason"] == "小字号(公式/表格/图注)"
                ),
                "is_empty": len(ordered) == 0,
                "text": _line_texts(ordered),
            }
        )

    blocks = assemble_blocks(
        ordered_all, page_width, page_height, front_cutoff=front_cutoff
    )
    meta = _extract_metadata(raw_pages.get(1, []), paper_id, doc, page_width)
    meta["pdf"] = str(pdf_path.relative_to(config.ROOT)) if pdf_path.is_relative_to(config.ROOT) else str(pdf_path)
    meta["pdf_sha256"] = file_sha256(pdf_path)
    meta["body_font_size"] = body_size
    meta["source"] = ""
    meta["source_url"] = ""
    meta["category"] = ""
    if overrides:
        for key, value in overrides.items():
            if value not in (None, "", [], {}):
                meta[key] = value
    meta.setdefault("title", paper_id)

    sections = build_sections(blocks, meta)

    ref_page = ref_start[0] if ref_start else None
    suspicious = [
        p["page"]
        for p in page_reports
        if p["body_chars"] < 80
        and p["raw_chars"] > 150
        and not (ref_page and p["page"] >= ref_page)
    ]
    warnings: list[str] = []
    if suspicious:
        warnings.append(f"第 {suspicious} 页正文文字很少，可能是扫描页、整页图表或排版异常。")
    if ref_start is None:
        warnings.append("没有识别到“参考文献”，请检查原文是否有该标题。")
    if sum(1 for s in sections if s["type"] == "body") < 3:
        warnings.append("识别到的正文章节少于 3 个，标题格式可能比较特殊。")
    if removed.get("小字号(公式/表格/图注)", 0) > len(ordered_all) * 0.2:
        warnings.append("较多的公式/表格/图注被排除，第一版不对这些内容建索引。")

    parsed = {
        "paper_id": paper_id,
        "meta": meta,
        "pages": page_reports,
        "sections": sections,
        "quality": {
            "page_count": doc.page_count,
            "raw_chars": sum(p["raw_chars"] for p in page_reports),
            "body_chars": sum(p["body_chars"] for p in page_reports),
            "paragraph_count": sum(len(s["paragraphs"]) for s in sections),
            "section_count": len(sections),
            "body_section_count": sum(1 for s in sections if s["type"] == "body"),
            "reference_found": ref_start is not None,
            "reference_start": list(ref_start) if ref_start else None,
            "front_matter_end": list(front_cutoff) if front_cutoff else None,
            "removed": dict(removed),
            "suspicious_pages": suspicious,
            "warnings": warnings,
        },
    }

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")
    return parsed


def _line_info(line: Line) -> dict[str, Any]:
    return {
        "text": line.text,
        "bbox": [
            round(line.x0, 1),
            round(line.y0, 1),
            round(line.x1, 1),
            round(line.y1, 1),
        ],
        "size": round(line.size, 1),
        "direction": [
            round(line.direction[0], 3),
            round(line.direction[1], 3),
        ],
        "vertical": line.is_vertical,
    }


def inspect_page(
    pdf_path: str | Path,
    page_no: int = 1,
    paper_id: str | None = None,
) -> dict[str, Any]:
    """只检查一页：原始行、删除原因、保留后的正文。

    不写 parsed/chunks/index，专门给调规则时用。
    """
    pdf_path = Path(pdf_path)
    paper_id = paper_id or pdf_path.stem
    doc = pymupdf.open(pdf_path)
    try:
        if not 1 <= page_no <= doc.page_count:
            raise ValueError(f"页码必须在 1～{doc.page_count} 之间。")
        page = doc[page_no - 1]
        page_width, page_height = page.rect.width, page.rect.height

        raw_pages, raw_texts = extract_raw_lines(doc)
        ref_start = _find_reference_start(raw_pages)
        body_size = _body_font_size(raw_pages, ref_start)
        front_cutoff = _find_first_body_heading(raw_pages, page_width, body_size)
        debug: list[dict[str, Any]] = []
        kept, removed = filter_lines(
            raw_pages,
            page_width,
            page_height,
            body_size,
            ref_start,
            front_cutoff=front_cutoff,
            debug=debug,
        )

        raw_lines = sorted(
            raw_pages.get(page_no, []), key=lambda line: (line.y0, line.x0)
        )
        merged = _merge_same_line_fragments(
            kept.get(page_no, []), page_width
        )
        ordered = _page_order(merged, page_width)
        page_removed = [
            item for item in debug if item["page"] == page_no
        ]
        page_counts = Counter(item["reason"] for item in page_removed)

        return {
            "paper_id": paper_id,
            "pdf": str(pdf_path),
            "page": page_no,
            "page_count": doc.page_count,
            "page_size": [round(page_width, 1), round(page_height, 1)],
            "body_font_size": round(body_size, 1),
            "reference_start": list(ref_start) if ref_start else None,
            "front_matter_end": list(front_cutoff) if front_cutoff else None,
            "raw_text": raw_texts[page_no - 1],
            "raw_lines": [_line_info(line) for line in raw_lines],
            "kept_lines": [_line_info(line) for line in ordered],
            "removed_lines": page_removed,
            "page_removed_counts": dict(page_counts),
            "removed_counts": dict(removed),
            "text": _line_texts(ordered),
        }
    finally:
        doc.close()
