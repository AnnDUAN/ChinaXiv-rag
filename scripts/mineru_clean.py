"""把 MinerU 的 middle JSON 整理成可检查的解析结果。

用法：
    PYTHONPATH=src python scripts/mineru_clean.py
    PYTHONPATH=src python scripts/mineru_clean.py --ids 202610.00009

输出：
    data/mineru_clean/<paper_id>.json   项目内部解析格式
    data/mineru_clean/<paper_id>.md     人工检查用的清理版 Markdown
    data/mineru_clean/summary.md        每篇的清理统计和需要人工检查的问题
    data/mineru_clean/failures.md       解析失败列表（如果有）
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


def _cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json-dir",
        default=str(ROOT / "data" / "mineru_json"),
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
    failures: list[tuple[str, str]] = []
    for n, path in enumerate(paths, 1):
        try:
            parsed = parse_mineru_json(path)
        except Exception as exc:  # noqa: BLE001
            failures.append((path.stem, str(exc)))
            print(f"[{n}/{len(paths)}] {path.stem}: 失败 {exc}")
            continue
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
            f"[{n}/{len(paths)}] {paper_id}: "
            f"章节 {quality['section_count']} | 表 {quality['table_count']} | "
            f"公式 {quality['equation_count']} | 图 {quality['image_count']} | "
            f"警告 {len(quality['warnings'])}"
        )

    total_removed: dict[str, int] = {}
    for row in rows:
        for key, value in row["removed"].items():
            total_removed[key] = total_removed.get(key, 0) + int(value)

    total = max(len(rows), 1)
    meta_ok = {
        "作者": sum(1 for row in rows if row["authors"]),
        "单位": sum(1 for row in rows if row["affiliations"]),
        "摘要": sum(1 for row in rows if row["abstract_chars"]),
        "关键词": sum(1 for row in rows if row["keywords"]),
    }
    lines = [
        "# MinerU 解析与清理结果",
        "",
        f"共处理 **{len(rows)}** 篇，失败 **{len(failures)}** 篇。",
        "",
        "元数据完整度："
        + " | ".join(
            f"{name} {count}/{total}" for name, count in meta_ok.items()
        ),
        "",
        "| 论文 | 标题 | 作者 | 章节 | 表 | 公式 | 图 | 警告 |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['paper_id']}` | {_cell(row['title'])} | "
            f"{_cell('、'.join(row['authors']) or '—')} | "
            f"{row['sections']} | {row['tables']} | {row['equations']} | "
            f"{row['images']} | {len(row['warnings'])} |"
        )
    lines.extend(
        [
            "",
            "## 清理总量",
            "",
            "| 类型 | 数量 |",
            "|---|---:|",
        ]
    )
    for key, value in sorted(total_removed.items()):
        lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            "同名 `.json` 是项目内部解析格式，`.md` 是人工检查版。",
            "表格保留 HTML 合并单元格，公式保留 LaTeX；图片只计数，不进入正文。",
            "",
            "## 需要人工检查的论文",
            "",
        ]
    )
    issue_count = 0
    for row in rows:
        # 图片本来就不入库，不把“忽略了 N 个图片”当成清洗问题。
        reasons = [
            warning
            for warning in row["warnings"]
            if not warning.startswith("忽略了 ")
        ]
        if not row["authors"]:
            reasons.append("没有作者")
        if not row["affiliations"]:
            reasons.append("没有单位")
        if not row["abstract_chars"]:
            reasons.append("没有摘要")
        reasons = list(dict.fromkeys(reasons))
        if not reasons:
            continue
        issue_count += 1
        lines.append(f"### `{row['paper_id']}` {row['title']}")
        lines.append("")
        for reason in reasons:
            lines.append(f"- {reason}")
        lines.append("")
    if not issue_count:
        lines.append("没有发现需要人工检查的论文。")

    (out_dir / "summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    failures_path = out_dir / "failures.md"
    if failures:
        failures_path.write_text(
            "# MinerU 解析失败\n\n"
            + "\n".join(f"- `{paper_id}`：{error}" for paper_id, error in failures)
            + "\n",
            encoding="utf-8",
        )
    else:
        failures_path.unlink(missing_ok=True)

    print(
        f"\n完成：成功 {len(rows)} 篇，失败 {len(failures)} 篇，"
        f"需人工检查 {issue_count} 篇。"
    )
    print(f"输出目录：{out_dir}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
