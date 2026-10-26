# data 目录说明

本目录用于存放运行时数据，包括：

- `pdfs/`：原始论文 PDF
- `parsed/`：PDF 解析结果
- `chunks/`：切片结果
- `index/`：jieba/BM25 和向量索引
- `models/`：本地 embedding / rerank 模型缓存
- `pgdata/`：PostgreSQL 数据目录

这些文件体积较大，且包含论文原文，因此没有上传到 GitHub。

按仓库根目录 README 的命令重新生成即可：

```powershell
.\rag fetch --catalog --pages 1 --download
.\rag build
.\rag embed
```
