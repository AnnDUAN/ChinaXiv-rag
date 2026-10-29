"""把 MinerU 的 middle JSON 整理成可检查的解析结果。

用法：
    PYTHONPATH=src python scripts/mineru_clean.py
    PYTHONPATH=src python scripts/mineru_clean.py --ids 202610.00009

输出：
    data/mineru_clean/<paper_id>.json   项目内部解析格式
    data/mineru_clean/<paper_id>.md     人工检查用的清理版 Markdown
    data/mineru_clean/summary.md        每篇的清理统计
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag.ingestion.mineru_parser import (  # noqa: E402
    clean_markdown,
    parse_mineru_json,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json-dir",
        default=str(ROOT / "data" / "mineru_json_probe"),
        help="MinerU middle JSON 所在目录",
    )
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "data" / "mineru_clean"),
        help="整理结果输出目录",
    )
    parser.add_argument("--ids", nargs="*", help="只处理这些论文 ID")
    args = parser.parse_args()

    json_dir = Path(args.json_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(json_dir.glob("*.json"))
    if args.ids:
        wanted = set(args.ids)
        paths = [p for p in paths if p.stem in wanted]
    if not paths:
        print(f"没有找到 MinerU JSON：{json_dir}")
        return 1

    rows = []
    for path in paths:
        parsed = parse_mineru_json(path)
        paper_id = parsed["paper_id"]
        (out_dir / f"{paper_id}.json").write_text(
            json.dumps(parsed, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (out_dir / f"{paper_id}.md").write_text(
            clean_markdown(parsed),
            encoding="utf-8",
        )
        meta = parsed["meta"]
        quality = parsed["quality"]
        rows.append(
            {
                "paper_id": paper_id,
                "title": meta.get("title", ""),
                "authors": meta.get("authors", []),
                "affiliations": meta.get("affiliations", []),
                "abstract_chars": len(meta.get("abstract", "")),
                "keywords": meta.get("keywords", []),
                "sections": quality["section_count"],
                "tables": quality["table_count"],
                "equations": quality["equation_count"],
                "images": quality["image_count"],
                "removed": quality["removed"],
                "warnings": quality["warnings"],
            }
        )
        print(
            f"{paper_id}: {meta.get('title')}\n"
            f"  作者：{meta.get('authors')}\n"
            f"  单位：{meta.get('affiliations')}\n"
            f"  摘要 {len(meta.get('abstract', ''))} 字 | "
            f"关键词 {len(meta.get('keywords', []))} 个 | "
            f"章节 {quality['section_count']} | 表 {quality['table_count']} | "
            f"公式 {quality['equation_count']} | 图 {quality['image_count']}"
        )
        for warning in quality["warnings"]:
            print(f"  ⚠ {warning}")

    lines = [
        "# MinerU 解析与清理结果",
        "",
        "| 论文 | 标题 | 作者 | 章节 | 表 | 公式 | 图 |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['paper_id']}` | {row['title']} | "
            f"{'、'.join(row['authors'])} | {row['sections']} | "
            f"{row['tables']} | {row['equations']} | {row['images']} |"
        )
    lines.extend(
        [
            "",
            "同名 `.json` 是项目内部解析格式，`.md` 是人工检查版。",
            "表格保留 HTML 合并单元格，公式保留 LaTeX；图片只计数，不进入正文。",
        ]
    )
    (out_dir / "summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"\n输出目录：{out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
