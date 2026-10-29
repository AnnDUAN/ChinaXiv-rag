"""把 Docling 和 MinerU 的解析结果放到一起，方便人工对比。

输出目录：
    data/compare/<paper_id>/docling.md
    data/compare/<paper_id>/mineru.md
    data/compare/summary.md

MinerU 的 Markdown 默认会把图片写成 base64；这里只做一步清理，
换成“图片占位”，不改正文和表格。

试验环境（Python 3.12）：
    uv venv --python 3.12 .venv-mineru
    uv pip install --python .venv-mineru/Scripts/python.exe "mineru>=4.0,<5"

生成 MinerU 结果：
    .venv-mineru/Scripts/mineru-kit parse data/pdfs/论文.pdf \
        -o data/mineru_probe_standard_full \
        --tier standard --ocr-mode txt --format markdown
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DOCLING_DIR = ROOT / "data" / "docling_probe" / "clean"
MINERU_DIR = ROOT / "data" / "mineru_probe_standard_full"
OUT_DIR = ROOT / "data" / "compare_standard"


def _clean_images(text: str) -> str:
    return re.sub(
        r"!\[[^\]]*\]\(data:image/[^)]+\)",
        "![图片](图片占位)",
        text,
    )


def _table_counts(text: str) -> dict[str, int]:
    return {
        "html": text.count("<table"),
        "markdown": len(
            re.findall(r"^\|\s*---", text, re.MULTILINE)
        ),
        "images": text.count("![图片]") + text.count("<img"),
    }


def _first_table(text: str) -> str:
    html = re.search(r"<table.*?</table>", text, re.S)
    if html:
        return html.group(0)
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if re.match(r"^\|\s*---", line):
            start = max(0, i - 1)
            end = i
            while end < len(lines) and lines[end].startswith("|"):
                end += 1
            return "\n".join(lines[start:end])
    return ""


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    mineru_files = sorted(MINERU_DIR.glob("*.md"))
    for mineru_path in mineru_files:
        paper_id = mineru_path.stem
        mineru_text = _clean_images(
            mineru_path.read_text(encoding="utf-8")
        )
        paper_dir = OUT_DIR / paper_id
        paper_dir.mkdir(parents=True, exist_ok=True)

        mineru_out = paper_dir / "mineru.md"
        mineru_out.write_text(mineru_text, encoding="utf-8")

        docling_path = DOCLING_DIR / f"{paper_id}.md"
        docling_text = ""
        if docling_path.exists():
            docling_text = docling_path.read_text(encoding="utf-8")
            shutil.copyfile(docling_path, paper_dir / "docling.md")

        rows.append(
            {
                "paper_id": paper_id,
                "docling": _table_counts(docling_text),
                "mineru": _table_counts(mineru_text),
                "docling_chars": len(docling_text),
                "mineru_chars": len(mineru_text),
                "docling_first_table": _first_table(docling_text),
                "mineru_first_table": _first_table(mineru_text),
            }
        )

    summary = [
        "# Docling vs MinerU（standard 档）解析对比",
        "",
        "| 论文 | Docling 表格 | MinerU 表格 | Docling 字符 | MinerU 字符 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        d = row["docling"]
        m = row["mineru"]
        summary.append(
            f"| `{row['paper_id']}` | "
            f"{d['html'] + d['markdown']} | "
            f"{m['html'] + m['markdown']} | "
            f"{row['docling_chars']} | {row['mineru_chars']} |"
        )

    summary.extend(
        [
            "",
            "每个论文目录下：",
            "",
            "- `docling.md`：Docling 整理后的 Markdown；",
            "- `mineru.md`：MinerU 的 Markdown，图片已换成占位符。",
            "",
            "表格数量包含 HTML 表格和 Markdown 管表。",
        ]
    )
    (OUT_DIR / "summary.md").write_text(
        "\n".join(summary) + "\n", encoding="utf-8"
    )

    # 单独留一份复杂合并单元格表格的对比，方便快速检查
    target = next(
        (r for r in rows if r["paper_id"] == "202602.00063"), None
    )
    if target:
        parts = [
            "# 202602.00063 第一张表对比",
            "",
            "## Docling",
            "",
            "```markdown",
            target["docling_first_table"][:4000],
            "```",
            "",
            "## MinerU",
            "",
            "```html",
            target["mineru_first_table"][:4000],
            "```",
        ]
        (OUT_DIR / "table_compare_202602.00063.md").write_text(
            "\n".join(parts) + "\n", encoding="utf-8"
        )

    for row in rows:
        print(
            f"{row['paper_id']}: "
            f"Docling 表格 {row['docling']['html'] + row['docling']['markdown']} | "
            f"MinerU 表格 {row['mineru']['html'] + row['mineru']['markdown']}"
        )
    print(f"\n输出目录：{OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
