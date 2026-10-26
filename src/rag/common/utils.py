"""通用文本小工具。"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path


def normalize_spaces(text: str) -> str:
    """把全角空格、连续空白压成一个普通空格。"""
    text = text.replace("　", " ").replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


def normalize_for_match(text: str) -> str:
    """用于比较/检索的归一化：全角转半角、转小写。"""
    return unicodedata.normalize("NFKC", text).lower()


def clean_text(text: str) -> str:
    """输出给人看的正文：保留段落内空格，去掉行首尾空白。"""
    text = text.replace("　", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def join_lines(left: str, right: str) -> str:
    """把同一段的相邻两行接起来，处理英文断词和空格。"""
    left = left.rstrip()
    right = right.lstrip()
    if not left:
        return right
    if not right:
        return left

    # 英文单词被连字符拆开：Langu- / age -> Language
    if left.endswith("-") and right[:1].isascii() and right[:1].islower():
        return left[:-1] + right

    a, b = left[-1], right[0]
    if a.isascii() and a.isalnum() and b.isascii() and b.isalnum():
        return left + " " + right
    if a in ",;:." and b.isascii() and b.isalnum():
        return left + " " + right
    return left + right


SENTENCE_END = re.compile(r"(?<=[。！？!?；;])")


def split_sentences(text: str) -> list[str]:
    """按句号、问号、感叹号、分号切句子，保留标点。"""
    text = text.strip()
    if not text:
        return []
    parts = SENTENCE_END.split(text)
    result = [p.strip() for p in parts if p.strip()]
    return result or [text]


def hard_split(text: str, max_len: int) -> list[str]:
    """超长句子兜底：先按逗号切，再按长度硬切。"""
    if len(text) <= max_len:
        return [text]
    pieces = re.split(r"(?<=[，,、：:])", text)
    out: list[str] = []
    buf = ""
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        if len(buf) + len(piece) <= max_len:
            buf += piece
        else:
            if buf:
                out.append(buf)
            while len(piece) > max_len:
                out.append(piece[:max_len])
                piece = piece[max_len:]
            buf = piece
    if buf:
        out.append(buf)
    return out or [text]


def file_sha256(path: Path, n: int = 16) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:n]


def safe_filename(name: str) -> str:
    name = re.sub(r'[\\/:*?"<>|]+', "_", name).strip()
    return name or "paper"
