"""项目路径与可调参数，集中放在这里，方便以后改。"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
PDF_DIR = DATA_DIR / "pdfs"
# MinerU 输出的 middle JSON 放在这里；正式 build 只读这些 JSON。
MINERU_JSON_DIR = DATA_DIR / "mineru_json"
PARSED_DIR = DATA_DIR / "parsed"
CHUNKS_DIR = DATA_DIR / "chunks"
INDEX_DIR = DATA_DIR / "index"
CONFIG_DIR = ROOT / "config"
PAPERS_CONFIG = CONFIG_DIR / "papers.json"
LLM_CONFIG = CONFIG_DIR / "llm.json"
WEB_SEARCH_CONFIG = CONFIG_DIR / "web_search.json"
DB_SCHEMA = Path(__file__).resolve().parent / "db" / "schema.sql"

# 默认连接 docker-compose.yml 里的独立数据库；
# 可以用 RAG_DATABASE_URL 覆盖，不会碰到其他项目的数据库。
DATABASE_URL = os.environ.get(
    "RAG_DATABASE_URL",
    "postgresql://rag:rag@127.0.0.1:55433/agentic_rag",
)

# 切片参数（单位：字符）
CHUNK_TARGET = 450
CHUNK_MAX = 600
CHUNK_MIN = 200
CHUNK_OVERLAP = 70
# 表格带 HTML 标签，字符数比正文膨胀，单独给一个上限。
TABLE_CHUNK_MAX = 1200

# 检索参数
BM25_K1 = 1.5
BM25_B = 0.75
RECALL_N = 20
DEFAULT_TOP_K = 5
# 最终结果里同一篇论文最多保留几块，避免标题很匹配的论文刷屏。
MAX_CHUNKS_PER_PAPER = 2

# 向量检索参数
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"
EMBED_DIM = 512
EMBED_BATCH_SIZE = 16
VECTOR_MIN_COSINE = 0.20
# 当前语料很小，pgvector 先用精确扫描，保证和原来的 npz 结果一致；
# 数据量大了再把这个改成 False，走 HNSW 近似检索。
PGVECTOR_EXACT = True
# 混合检索用 RRF（Reciprocal Rank Fusion）合并两路排名。
# 只使用名次，不使用原始分数，避免 BM25 和余弦相似度量纲不同。
# 权重先各占一半，后面可以用评测集重新调。
RRF_K = 60
RRF_BM25_WEIGHT = 0.5
RRF_VECTOR_WEIGHT = 0.5

# 强关键词保底：BM25 前几名且问题词覆盖够高时，额外加一点 BM25 名次分，
# 避免正确答案被 RRF 融合稀释。
KEYWORD_ANCHOR_TOP_N = 10
KEYWORD_ANCHOR_COVERAGE = 0.30
KEYWORD_ANCHOR_WEIGHT = 0.35

# 论文标题单独算一个 BM25 字段，给正文分数加一点标题分。
# 这样只提升整篇论文，不改变同一篇论文内部章节的相对顺序。
TITLE_BM25_WEIGHT = 0.35

# 表格和数字密集切片的轻微加权。
TABLE_SCORE_BOOST = 1.03
NUMERIC_SCORE_BOOST = 1.06
NUMERIC_MIN_DIGITS = 6
NUMERIC_DIGIT_RATIO = 0.02

# 重排序参数
# 本地交叉编码器：把“问题 + 候选段落”放在一起判断相关性。
# int8 版本约 570MB，第一次运行会下载到 data/models。
RERANK_MODEL = "BAAI/bge-reranker-v2-m3-int8"
# 先多召回一些候选，再重排；语料扩到 40 多篇后，
# 20 个候选可能漏掉目标，先用 30 个。
RERANK_CANDIDATES = 30
# 保留原排序的前几名作为“保底”，避免重排把本来正确的小节顶掉。
RERANK_ANCHORS = 2
RERANK_BATCH_SIZE = 16
