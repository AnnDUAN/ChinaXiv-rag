# data 目录说明

本目录用于存放运行时数据。

## 当前保留

- `pdfs/`：原始论文 PDF，共 106 篇；
- `mineru_json/`：MinerU middle JSON，共 103 篇；
- `mineru_clean/`：清洗后的 Markdown 和 parsed JSON，共 103 篇；`summary.md` 是汇总；
- `mineru_skipped.md`：跳过解析的超大 PDF 清单；
- `models/`：本地 embedding / rerank 模型缓存；
- `pgdata/`：PostgreSQL 数据目录；
- `catalog.json`：ChinaXiv 论文列表；
- `ignored.json`：不参与入库的论文列表。

## 需要重新生成

- `parsed/`：正式切片前的项目解析结果；
- `chunks/`：RAG 切片；
- `index/`：jieba/BM25 和向量索引。

这三个目录已经清空，等确认 MinerU 清洗结果后，由 `rag build` 重新生成。

## 常用命令

```powershell
# 下载 PDF
.\rag fetch --catalog --pages 1 --download

# 用 MinerU 生成 middle JSON（单独装在 .venv-mineru）
.\venv-mineru\Scripts\mineru-kit.exe parse data\pdfs `
  -o data\mineru_json --format middle_json --tier standard --ocr-mode txt

# 清洗 MinerU 结果，生成可检查的 Markdown
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe scripts\mineru_clean.py

# 入库：切片 + 关键词索引 + 数据库同步
.\rag build --force --embed
```
