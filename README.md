# ChinaXiv-rag

> GitHub：https://github.com/AnnDUAN/ChinaXiv-rag
>
> 仓库不包含 `data/` 下的 PDF、解析结果、切片、索引和本地模型；按下方命令重新生成即可。

一个“本地中文论文检索”小系统：先用 **MinerU standard** 把 PDF 解析成结构化结果，清理正文脏数据后再切成能追溯原文的块，用 **jieba 分词 + BM25** 做关键词检索，再叠加本地向量做混合检索。需要更准时可以再加一层**本地重排序**，先多召回一些候选，再逐条判断相关度。没有向量索引时会自动退回 BM25。

数据默认存到 **PostgreSQL + pgvector**；数据库没启动时会自动退回 JSON 文件，不会影响检索。第一阶段的完整设计和评测说明见 `第一阶段结束文档.md`。

**检索是核心；带出处的回答生成是可选功能**，需要配置一个 OpenAI 兼容接口。问答模块直接用 OpenAI SDK，不依赖 LangChain，并支持结构化输出和引用校验。不配置模型也能正常检索，这样排查问题时能分清是解析、切片还是检索的锅。

## 当前语料

来源：**ChinaXiv（中国科学院科技论文预发布平台）** 的“计算机科学”分类（分类编号 243）。

- 只保留**标题是中文**的论文：标题里至少 4 个汉字；
- 纯英文标题的论文会在入库前被过滤掉；
- 论文列表保存在 `data/catalog.json`，按 `paper_id` 去重；
- 当前保留了 **106 篇有效 PDF**；
- 旧的字号解析结果、切片、索引和数据库数据已全部清空；
- 已生成 **103 篇** MinerU middle JSON 和清洗后的 Markdown：
  - 先跑了 40 篇 standard；
  - 剩余 63 篇用 basic 快速补跑，优先保证正文和元数据；
- 3 篇超大 PDF 按要求跳过，规则和名单见 `data/mineru_skipped.md`；
- 全量切片、向量索引和数据库还没有重建，等清洗结果确认后再做。

个别预印本版式比较特殊（超长、标题没有编号、正文里有大量公式），解析时会给出警告。以 `rag stats` 的输出和 `data/parsed/*.json` 里的 `quality` 为准。

## 当前评测

目前有三套测试集：

- `tests/questions_40.json`：43 篇语料时期建立的 21 个直接问法问题；
- `tests/questions_40_hard.json`：同期的 21 个口语化问题，包含跨论文问题，更能拉开差距；
- `tests/questions_new60.json`：新增语料上的 20 个覆盖检查题，主要确认新论文能正常检索到，不是难度基准。

每道题都指定了 `expect_paper_ids`，避免在别的论文里碰到同名词也算通过。

```powershell
# 不加重排序：看 Hit@5
.\rag eval --questions tests/questions_40_hard.json --mode hybrid

# 加重排序：看 Recall@5
.\rag eval --questions tests/questions_40_hard.json --mode hybrid --rerank

# 新增语料覆盖检查
.\rag eval --questions tests/questions_new60.json --mode hybrid
```

### 指标怎么算

- **Hit@5**：前 5 条里有没有至少一条来自目标论文、且包含答案关键词；
- **Recall@5**：题目标注的目标论文中，有多少篇在前 5 条里出现过。多论文问题比如“这些论文里有哪些不同路线”，会看 5 条结果覆盖了几篇目标论文；
- 单论文题目的 Recall@5 只有 0% 或 100%，所以多论文题更能体现这个指标。

### 旧解析器结果（仅留档）

下面的数字来自旧的字号解析器和旧切片，**不是 MinerU 重建后的当前结果**。等全量 106 篇重新入库后需要重跑一遍，再更新这里。

| 测试集 | 不重排 Hit@5 | 不重排 Recall@5 | 重排 Hit@5 | 重排 Recall@5 |
|---|---:|---:|---:|---:|
| `questions_40.json`（直接问法） | 21/21（100%） | 90.8% | — | — |
| `questions_40_hard.json`（口语化） | 19/21（90.5%） | 86.5% | — | — |
| `questions_new60.json`（新增语料覆盖） | 20/20（100%） | 100% | — | — |

旧结果里，RRF 改动后口语化这套从 18/21 提到 19/21，Recall@5 从 81.7% 提到 86.5%（不加重排）。重排序仍然保留原排序前 2 名做保底，候选池 30 个；CPU 上约 1 分钟/题，因此默认关闭。

旧的 `tests/questions.json` 和 `tests/questions_hard.json` 仍然保留，但它们对应的是 16 篇论文时期，结果已经被上面这组扩库评测取代。

## 怎么用

在项目目录下打开命令行。**PowerShell 要写成 `.\rag`；cmd.exe 里可以直接写 `rag`。**

MinerU 单独装在 `.venv-mineru`（Python 3.12），主项目环境不需要安装 MinerU。第一次使用先建好这个环境：

```powershell
uv venv --python 3.12 .venv-mineru
uv pip install --python .venv-mineru\Scripts\python.exe "mineru>=4.0,<5"
```

```powershell
# 1. 下载论文（先抓 1 页列表，再下载所有还没有 PDF 的中文论文）
.\rag fetch --catalog --pages 1 --download

# 2. 用 MinerU standard 生成 middle JSON（单独装在 .venv-mineru）
.\venv-mineru\Scripts\mineru-kit.exe parse data\pdfs `
  -o data\mineru_json --format middle_json --tier standard --ocr-mode txt

# 3. 入库：清理 MinerU 结果 + 切片 + 用 jieba 建立关键词索引
.\rag build

# 3.1 只试几篇（不跑全量）
.\rag build --ids 202602.00063 202610.00016 `
  --mineru-dir data\mineru_json --force

# 4. 建立本地向量索引（默认混合检索需要；不建也能用，会自动退回 BM25）
.\rag embed

# 4.1 第一次使用数据库：启动容器、建表、导入现有数据
docker compose up -d
.\rag db init
.\rag db import

# 5. 检索一个问题，默认是混合检索
.\rag search "大语言模型的幻觉怎么治理？"

# 5.1 需要更准时，再加本地重排序（第一次会下载约 570MB 模型）
.\rag search "大语言模型的幻觉怎么治理？" --rerank

# 6. 连续提问
.\rag repl --rerank

# 7. 可选：让大模型基于检索结果回答并标出处（先配置 config/llm.json）
.\rag ask "大语言模型怎么辅助心理构念发现？"
```

其他常用命令：

```powershell
# 只看前 3 条，并显示完整切片
.\rag search "检索增强生成有哪些挑战？" -k 3 --full

# 用重排序提高排序质量
.\rag search "检索增强生成有哪些挑战？" --rerank

# 单独重建 jieba 关键词索引
.\rag index

# 只重新切片并重建关键词索引（不重新读 MinerU JSON）
.\rag chunk

# 查看已经入库了什么
.\rag stats

# 检查旧字号解析器在某一页的细节（只做诊断，不参与现在的入库流程）
.\rag inspect 202609.00318 --page 1
.\rag inspect 202609.00318 --page 1 --raw
```

`inspect` 是旧字号解析器的诊断工具，现在正式入库走 MinerU，所以调 MinerU 清洗规则时优先看 `data/mineru_clean/*.md` 和 `parsed/*.json` 里的 `quality`。旧命令 `--raw` 仍然可以看原始行，`--all-removed` 可以看全部被删除的行。

```powershell
# 可选：建立向量索引，用 --mode hybrid / vector 检索
.\rag embed
.\rag search "怎么让回答有出处？" --mode vector

# 向量索引 + 重排序一起用
.\rag search "怎么让回答有出处？" --mode hybrid --rerank

# 重新读取 MinerU JSON + 重建全部索引
.\rag build --force --embed
```

如果 `rag` 命令不能直接用，就在 Git Bash 里运行 `bash rag.sh search "你的问题"`，或者用完整命令：

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m rag search "你的问题"
```

## 一条检索结果怎么看

```text
[1] 得分 2.607 | 词命中 100% | 综合 较高 | 第 9 页
《论文标题》 > 4 方法 > 4.1 数据构建

……原文段落……
```

- **得分**：BM25 关键词得分，只能在同一批结果里比较，不是百分比。
- **词命中**：问题里的关键词有多少被这个切片覆盖。适合判断“字面上有没有说到”。
- **综合**：根据词命中和章节匹配给出的把握程度。较低时会提示“可能没有直接答案”。
- **页码 / 章节**：方便回到原文核对。
- **切片 ID**（JSON 里）：`202610.00065-c0016` 这种，以后加生成时用来标出处。

如果用了 `--mode vector` 或 `--mode hybrid`，结果里会多一列“语义”，表示向量相似度；顶部还会显示“向量后端：pgvector”或“npz”，方便确认当前走的是数据库还是本地备份索引。

## 自动下载论文（ChinaXiv）

ChinaXiv 的检索页需要先拿到一个会话 cookie 和一个 `pageId`，然后才能按分类筛选和翻页。下载器已经自动处理这些步骤，**PDF 不需要登录**。

下载规则：

- 默认分类是“计算机科学”（编号 243）；
- 只保留中文标题，纯英文标题会被过滤；
- 论文列表保存在 `data/catalog.json`；
- 每次抓取后按 `paper_id` 和已有列表合并，重复的论文不会新增；
- PDF 统一保存为 `data/pdfs/<paper_id>.pdf`；
- 下载前会检查文件是否存在且开头是 `%PDF`，已经下好的直接跳过；
- `data/ignored.json` 里的论文编号会在抓列表时自动跳过；
- `--force` 可以强制重新下载。

```powershell
# 只抓 1 页列表并合并去重，不下载（1 页约 20 条，过滤英文后约 10 篇）
.\rag fetch --catalog --pages 1

# 抓 3 页，并按完整列表下载所有还没有 PDF 的论文
.\rag fetch --catalog --pages 3 --download

# 本次最多下载 10 篇缺失论文（已有的仍然跳过）
.\rag fetch --catalog --pages 3 --download --limit 10

# 下载单篇；如果本地已有有效 PDF，会跳过
.\rag fetch --paper 202610.00065

# 强制重新下载单篇
.\rag fetch --paper 202610.00065 --force

# 直接下载任意 PDF 直链
.\rag fetch --url https://.../xxx.pdf --name xxx
```

下载完成后运行：

```powershell
.\rag build
```

就会把新论文解析、切片并建立 jieba 关键词索引。需要向量索引时再加 `--embed`。批量下载默认每篇间隔 1.5 秒，避免请求太快。

## 数据长什么样

```text
data/
  pdfs/                放原始 PDF
  mineru_json/         MinerU standard 的 middle JSON（正式入库读这里）
  mineru_clean/        清理后的 Markdown 和 parsed JSON，方便人工检查
  parsed/              项目内部解析结果（章节、表格、公式、元数据、质量报告）
  chunks/              切片结果（关键词检索用的就是这里）
  index/               关键词索引 jieba_bm25.json；可选的向量索引 vectors.npz
  models/              本地向量模型缓存（约 90MB，和论文无关，可保留）
  catalog.json         从 ChinaXiv 抓到的论文列表
config/
  papers.json          每篇论文的来源、链接、分类等人工补充信息（可为空）
```

`parsed/*.json` 是排查问题的关键，里面保留了：

- `pages`：每页的块数、保留字符数、表格/公式/图片数量，以及双栏顺序是否可疑；
- `sections`：章节标题、层级、起止页，以及 `text` / `table` / `equation` 三种段落；
- `meta`：标题、作者、单位、摘要、关键词；
- `quality`：页数、正文字符数、章节数、表格数、公式数、删掉了多少页眉页脚/脚注/参考文献/图片，以及警告。

`chunks/*.json` 里每个切片都带：

```json
{
  "chunk_id": "202610.00065-c0016",
  "type": "body",
  "section_path": ["3 方法", "3.2 实验设置"],
  "page_start": 6,
  "page_end": 6,
  "text": "给人看的原文……",
  "index_text": "《论文标题》 > 3 方法 > 3.2 实验设置 > 原文……",
  "char_count": 545,
  "prev_id": "202610.00065-c0015",
  "next_id": "202610.00065-c0017"
}
```

`text` 和 `index_text` 分开存：前者保持原文干净，后者在开头拼上论文标题和章节路径，检索时更容易命中。

## 切片规则

- 不跨章节切；
- 按句子边界切，目标 450 字左右，最短 200 字，正文硬上限 600 字；
- 同一章节内相邻块重叠最多 70 字，不会把上一块的长句整句搬过来；
- 摘要、关键词单独成块；摘要超过 600 字也会按句子拆开；
- 参考文献不建索引，防止提问时召回一堆引用；
- 表格单独成块，保留 MinerU 的 HTML 合并单元格；超长表格按行拆开并重复表头，块里标“（续表）”；
- 表格给检索用的 `index_text` 会去掉 HTML 标签，只留单元格文字；
- 公式和前后正文放在一起，不生成只有公式的块；切分时会先把公式保护起来，不会从公式中间截断；
- 图片块直接忽略，不建索引，也不做图片理解。

参数都在 `src/rag/config.py`，只改切片参数时运行 `rag chunk` 即可（不重新读 MinerU JSON）；清洗规则或解析规则改了才需要 `rag build --force`。

## 检索规则（默认：混合检索）

默认 `hybrid` 会同时跑两路：

1. **关键词路**：jieba 分词 + BM25，适合精确术语、模型名和数字；
2. **向量路**：本地 `bge-small-zh-v1.5` 把问题和切片变成向量，适合换一种说法；
3. 两路各自排名后用 **RRF（Reciprocal Rank Fusion）** 合并，只看名次，不强行换算 BM25 分和余弦相似度；
4. RRF 权重当前是 BM25 0.5、向量 0.5，`k=60`，后面再用评测集调；
5. 用同义词表补一下口语说法，比如“缺点 → 挑战”“编造 → 幻觉”；
6. 去掉同一章节里相邻或高度重复的块，返回前 5 条；启用重排序时，RRF 的候选会交给重排序模型重新排队。

关键词索引由 `rag build` 自动建立，切片变了可以单独运行 `rag index`；向量索引由 `rag embed` 建立。没有向量索引时，`hybrid` 会自动退回 BM25。

### 可选：本地重排序

加上 `--rerank` 后，流程会变成：

```text
BM25 / 向量召回
→ 取前 30 个候选（去重）
→ 本地 bge-reranker-v2-m3-int8 逐条打分
→ 重新排序
→ 保留原排序前 2 名做保底
→ 取前 5
```

- 重排序模型会把“问题 + 候选段落”放在一起读，比 BM25 和向量更擅长判断“这段是不是真的在回答这个问题”；
- 它只能重新排已经召回的内容，**不能把没召回的段落变出来**；
- 默认关闭，所以不加 `--rerank` 时行为和以前一样；
- 第一次运行会下载约 570MB 模型到 `data/models`，之后完全本地；
- CPU 上每条问题大约多花十几秒，适合偶尔提高精度时用。

```powershell
.\rag search "怎样让大模型帮忙提出心理学里的新概念？" --rerank
.\rag eval --questions tests/questions_40_hard.json --mode hybrid --rerank
```

## 向量索引（混合模式使用）

```powershell
.\rag embed
.\rag search "你的问题"              # 默认 hybrid
.\rag search "你的问题" --mode bm25  # 只走关键词
.\rag search "你的问题" --mode vector
.\rag search "你的问题" --rerank     # 混合召回后再精排
```

向量模型是 `BAAI/bge-small-zh-v1.5`（约 90MB，CPU 可跑），第一次运行会下载，之后完全离线。切片变化后需要重新运行 `rag embed`；如果数据库可用，会同时写入 pgvector，检索优先走数据库，本地 `vectors.npz` 作为备份。向量完全不可用时会退回 BM25。

## 数据库（PostgreSQL + pgvector）

项目使用一个独立的 PostgreSQL 容器，端口是 **55433**，不会碰你之前 `finagent-pg` 占用的 55432。

```powershell
# 第一次：启动数据库容器
docker compose up -d

# 建表
.\rag db init

# 把现有的 parsed/chunks/npz 导入数据库
.\rag db import

# 查看数据库统计
.\rag db stats
```

默认连接地址：

```text
postgresql://rag:rag@127.0.0.1:55433/agentic_rag
```

可以用环境变量覆盖：

```powershell
$env:RAG_DATABASE_URL = "postgresql://用户名:密码@主机:端口/数据库名"
```

数据库里现在有：

- `papers`：论文元数据、质量和来源；
- `sections`：章节树；
- `chunks`：切片正文、页码、前后块、jieba 词；
- `embeddings`：512 维 pgvector 向量；
- `queries` / `query_results`：每次问答和它的检索结果，方便回溯和评测；
- `agent_runs` / `agent_steps`：Agent 多步运行的状态、最终答案和每一步记录。

几个行为约定：

- `rag search` 优先从数据库读切片；数据库没启动时会自动退回 JSON，不影响检索；
- `rag build`、`rag chunk`、`rag embed` 在写完 JSON 后会尝试同步数据库；数据库没开时只提示，不阻断流程；
- PDF 和本地模型文件仍然放在 `data/`，数据库不保存大文件；
- `data/pgdata/` 是数据库数据目录，已经加入 `.gitignore`；
- 向量检索现在优先走 **pgvector**，数据库不可用时才退回本地 `vectors.npz`；
- 当前 3753 个切片仍然不大，pgvector 先用精确扫描（`PGVECTOR_EXACT = True`），保证结果和本地 numpy 点积一致；数据量继续变大后可以改成 HNSW 近似检索；
- PostgreSQL 没有真正的 BM25；当前 BM25 仍然是 Python 自己实现的，数据库里只保存 jieba 分词结果。以后需要把打分也放进 SQL，可以再加；
- 数据库只是数据源，BM25 和向量索引仍然可以单独重建；
- 扩库后重新跑过现有三套测试集，结果见上面的“当前评测”。

## 带出处的结构化问答（可选）

检索只能把原文段落找出来；如果想让模型基于这些段落写一段回答，可以配置一个 **OpenAI 兼容接口**。

仓库只提供 `config/llm.example.json`，先复制为 `config/llm.json` 再填写：

```json
{
  "base_url": "https://api.deepseek.com/v1",
  "api_key": "你的 key",
  "model": "deepseek-chat",
  "timeout": 120,
  "temperature": 0.2,
  "capabilities": {
    "json_schema": true,
    "json_object": true,
    "tool_calling": false,
    "streaming": true,
    "max_context": 32000
  }
}
```

`base_url` 填到 `/v1` 这一层即可。DeepSeek、通义千问兼容模式、OpenAI、本地 vLLM/Ollama 都可以用；如果本地服务不校验 key，`api_key` 可以填 `EMPTY`。

接口能力说明：

- `json_schema`：接口是否支持 OpenAI 的 `response_format={"type":"json_schema"}`；
- `json_object`：接口是否支持只返回 JSON 对象；
- 不确定就都填 `true`，程序会自动尝试，不支持就换下一种方式；
- 两个都填 `false` 也能用，会退回到“普通文本里提取 JSON”。

也可以用环境变量覆盖配置：

```powershell
$env:RAG_LLM_BASE_URL = "https://api.deepseek.com/v1"
$env:RAG_LLM_API_KEY = "你的 key"
$env:RAG_LLM_MODEL = "deepseek-chat"
$env:RAG_LLM_JSON_SCHEMA = "false"
$env:RAG_LLM_JSON_OBJECT = "true"
```

然后运行：

```powershell
.\rag ask "大语言模型怎么辅助心理构念发现？"
```

### 结构化输出

这里没有用 LangChain 的封装客户端，而是直接用 OpenAI SDK 的 `OpenAI` 类。问答结果默认走结构化输出：

1. 优先让接口按 JSON Schema 返回；
2. 不支持时改用 JSON 模式；
3. 再不行就从普通文本里提取 JSON；
4. JSON 格式不对时，会把错误信息发回去，让模型修正一次；
5. 所有结构化方式都失败时，自动退回普通文本回答。

结构化结果大概长这样：

```json
{
  "answer": "MPBI 的维度按测量学地位分层 [1]。",
  "citations": [
    {
      "source_id": 1,
      "quote": "MPBI 的维度体系按测量学地位分层，而不是按心理学分支平铺罗列。"
    }
  ],
  "insufficient_evidence": false,
  "confidence": "high"
}
```

程序还会做二次校验：

- `source_id` 是否真的存在；
- `quote` 是否逐字出现在对应原文里；
- 答案里的 `[编号]` 是否越界。

校验不通过的引用不会被放进有效引用列表，但会在 `citation_issues` 里显示，不会静默忽略。

常用命令：

```powershell
# 默认：结构化输出
.\rag ask "大语言模型怎么辅助心理构念发现？"

# 查看完整结构化 JSON、引用和校验问题
.\rag ask "大语言模型怎么辅助心理构念发现？" --json

# 不要结构化，直接返回普通文本
.\rag ask "大语言模型怎么辅助心理构念发现？" --plain

# 只查看检索上下文和提示词，不调用模型
.\rag ask "大语言模型怎么辅助心理构念发现？" --prompt-only
```

离线测试结构化输出（不需要 API key）：

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

注意：配置模型后，问题会和前 5 条论文片段一起发送到你设置的接口。

## Agent 多步检索（实验功能）

普通 `rag ask` 是一条固定流程：检索一次，然后回答。
`rag agent ask` 会在中间多问一句“现在的证据够不够”，不够时最多再补查几次。

第一版流程：

```text
检索 → 判断证据
       ├─ 够 → 回答
       ├─ 不够，且还有新角度 → 补查（最多再查 2 次）
       └─ 不够，但没新角度 → 用现有证据回答或明确说不知道
```

它不会自由调用网页、代码等工具，也不会无限循环：

- 最多 3 轮检索（`--max-search-rounds`）；
- 最多 8 个步骤（`--max-steps`，包含最后的回答）；
- 模型想重复同一个检索词时会停止补查；
- 补查没有带回新片段时会停止补查。

运行前先更新数据库表结构：

```powershell
.\rag db init
```

常用命令：

```powershell
# 复杂问题：证据不够时自动补查
.\rag agent ask "对比论文里两种方法分别有什么优缺点？" --max-steps 8 --max-search-rounds 3

# 查看最近的 Agent 运行
.\rag agent list

# 查看某次运行的每一步：查了什么、判断了什么
.\rag agent trace 1

# 上次中途断了，从存档继续
.\rag agent resume 1 --max-steps 10
```

Agent 每次运行都会写两张表：

- `agent_runs`：保存完整进度和最终答案；
- `agent_steps`：保存每一步的输入、输出和状态。

所以程序中断后可以从上次完成的步骤继续，也能事后看出它为什么又查了一轮。
如果只是普通单篇论文问题，继续用 `rag ask` 就够了，不需要 Agent。

## 手动加一篇新论文

1. 把 PDF 放进 `data/pdfs/`，文件名用论文编号，例如 `202610.00065.pdf`；
2. 如果需要补充来源信息，在 `config/papers.json` 里用文件名（不含 `.pdf`）作为键加一段；
3. 用 MinerU 生成这篇论文的 middle JSON，放到 `data/mineru_json/202610.00065.json`；
4. 运行 `.\rag build`。

```json
{
  "my_paper": {
    "source": "ChinaXiv",
    "source_url": "https://chinaxiv.org/abs/202610.00065",
    "category": "计算机科学 / 自然语言理解与机器翻译"
  }
}
```

## 已知限制（第一版故意没做）

- 默认只做检索；回答生成是可选功能，模型只看到前 5 条片段，可能答不全或答错，使用前要核对出处；
- 结构化输出能检查“引用编号是否存在、引文是否来自原文”，但不能保证模型的理解和结论一定正确；
- MinerU standard 仍然可能把复杂表格的行列识别错，尤其是跨页表和多层表头；清洗规则会尽量合并续表，但不能保证每个单元格都对；
- 图片块不参与检索，图里的信息不会进入切片；
- 公式保留 LaTeX，和前后正文一起切；嵌入模型对纯公式不敏感，所以不要指望“只靠公式”检索到；
- 部分预印本版式比较特殊：标题可能跨行、章节没有编号、某一页几乎全是图，解析结果会出现在 `quality.warnings` 里；
- ChinaXiv 的“计算机科学”分类里有少量论文同时挂在其他学科下，主题可能偏交叉；
- 默认混合检索；没有向量索引时会自动退回 BM25，不依赖在线服务；
- Agent 多步检索是实验功能，默认不启用；它只做“补检索”，不能浏览网页或执行代码；
- Agent 的复杂问题效果还没有用真实大模型做系统评测，目前只做了离线流程测试和本地假模型冒烟测试。

## 目录结构

```text
agentic_rag/
  docker-compose.yml        PostgreSQL + pgvector 容器
  第一阶段结束文档.md        RAG v1 设计与评测总结
  Agent编排与Checkpoint设计.md  Agent 多步流程和断点续跑设计
  rag.cmd / rag.sh          命令行包装
  config/papers.json        论文补充信息
  config/llm.json           大模型接口配置（可选）
  data/                     数据（PDF、JSON、模型、pgdata）
  src/rag/
    cli.py                  命令行入口
    config.py               路径和可调参数
    common/
      utils.py              通用文本小工具
    db/                     数据库
      engine.py             连接管理
      repo.py               SQL 访问
      migrate.py            建表、导入旧 JSON
      schema.sql            表结构
    ingestion/              数据入库
      downloader.py         ChinaXiv 列表抓取 / PDF 下载
      mineru_parser.py      MinerU JSON → 清理后的章节、表格、公式、元数据
      pdf_parser.py         旧字号解析器，只保留给 inspect 诊断
      chunker.py            章节 → 检索切片（表格/公式感知）
    indexing/               建立索引
      bm25.py               分词 + BM25
      keyword_index.py      jieba 分词 + BM25 索引
      embedder.py           本地中文向量模型
      vector_index.py       向量索引的建立与读取
    retrieval/              检索
      search.py             关键词/向量/混合召回、去重、排序
      reranker.py           本地重排序模型
    llm/                    LLM 传输和结构化输出
      base.py               消息、能力声明、错误类型
      openai_client.py      直接使用 OpenAI SDK
      structured.py         JSON Schema → JSON → 文本，逐级降级
      config.py             读取 config/llm.json
      mock.py               离线测试用假客户端
    generation/             生成
      answer.py             检索与生成分离、带出处的结构化问答
      schemas.py            结构化回答的字段定义
    agent/                  多步检索 Agent
      state.py              运行状态
      assess.py             判断证据够不够、下一轮查什么
      runner.py             检索 → 判断 → 回答，最多补查几轮
      checkpoint.py         把运行状态和每一步存进 PostgreSQL
  tests/questions.json      旧版 20 个基础验收问题（16 篇语料时期）
  tests/questions_hard.json 旧版 20 个口语化难题（16 篇语料时期）
  tests/questions_40.json   43 篇时期的 21 个直接问法问题
  tests/questions_40_hard.json 43 篇时期的 21 个口语化问题
  tests/questions_new60.json   新增语料的 20 个覆盖检查问题
  tests/test_llm_structured.py  结构化输出和引用校验的离线测试
  tests/test_agent.py           Agent 流程、补查、停止和断点续跑的离线测试
```

## 下一步可以做什么

按优先级建议：

1. 配置 `config/llm.json`，用 `rag ask` 检查回答是否真的引用了原文；
2. 用真实模型跑一小批 Agent 复杂问题，和普通 `rag ask` 对比答案质量、步数和耗时；
3. 抽查新增论文的解析质量，尤其是警告多的几篇；
4. 根据评测结果调整 Agent 的最大检索轮数和补查策略；
5. 做一个简单的网页界面。
