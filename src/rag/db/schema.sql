-- agentic_rag 的数据库结构。
-- 只存元数据、正文、切片、向量和问答记录；
-- PDF 文件和本地模型文件仍然放在 data/ 下面。

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS schema_version (
    version      integer PRIMARY KEY,
    applied_at   timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS papers (
    paper_id      text PRIMARY KEY,
    title         text NOT NULL,
    authors       jsonb NOT NULL DEFAULT '[]'::jsonb,
    affiliations  jsonb NOT NULL DEFAULT '[]'::jsonb,
    abstract      text,
    keywords      jsonb NOT NULL DEFAULT '[]'::jsonb,
    doi           text,
    journal       text,
    year          integer,
    source        text,
    source_url    text,
    pdf_path      text,
    status        text NOT NULL DEFAULT 'ready',
    quality       jsonb NOT NULL DEFAULT '{}'::jsonb,
    meta          jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at    timestamptz NOT NULL DEFAULT now(),
    updated_at    timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS sections (
    section_id     bigserial PRIMARY KEY,
    paper_id       text NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
    section_index  integer NOT NULL,
    section_path   jsonb NOT NULL,
    title          text,
    level          integer,
    page_start     integer,
    page_end       integer,
    section_type   text,
    UNIQUE (paper_id, section_index)
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id      text PRIMARY KEY,
    paper_id      text NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
    section_id    bigint REFERENCES sections(section_id) ON DELETE SET NULL,
    chunk_index   integer NOT NULL,
    chunk_type    text NOT NULL,
    section_path  jsonb NOT NULL DEFAULT '[]'::jsonb,
    page_start    integer,
    page_end      integer,
    text          text NOT NULL,
    index_text    text NOT NULL,
    char_count    integer NOT NULL,
    tokens        jsonb NOT NULL DEFAULT '[]'::jsonb,
    prev_id       text,
    next_id       text,
    created_at    timestamptz NOT NULL DEFAULT now(),
    UNIQUE (paper_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS chunks_paper_idx
    ON chunks (paper_id);
CREATE INDEX IF NOT EXISTS chunks_section_idx
    ON chunks (section_id);
CREATE INDEX IF NOT EXISTS chunks_tokens_gin
    ON chunks USING gin (tokens jsonb_path_ops);

CREATE TABLE IF NOT EXISTS embeddings (
    chunk_id     text PRIMARY KEY REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    model        text NOT NULL,
    dim          integer NOT NULL,
    vector       vector(512) NOT NULL,
    created_at   timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS embeddings_vector_idx
    ON embeddings USING hnsw (vector vector_cosine_ops);

-- 每次问答的记录，方便以后做评测、回溯和记忆模块。
CREATE TABLE IF NOT EXISTS queries (
    query_id          bigserial PRIMARY KEY,
    question          text NOT NULL,
    mode              text,
    rerank            boolean NOT NULL DEFAULT false,
    structured        boolean NOT NULL DEFAULT true,
    answer            text,
    structured_answer jsonb,
    citation_issues   jsonb NOT NULL DEFAULT '[]'::jsonb,
    latency_ms        integer,
    created_at        timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS query_results (
    query_id      bigint NOT NULL REFERENCES queries(query_id) ON DELETE CASCADE,
    rank          integer NOT NULL,
    chunk_id      text REFERENCES chunks(chunk_id) ON DELETE SET NULL,
    score         double precision,
    bm25_score    double precision,
    vector_score  double precision,
    rerank_score  double precision,
    PRIMARY KEY (query_id, rank)
);

CREATE INDEX IF NOT EXISTS queries_created_idx
    ON queries (created_at DESC);

-- 长期记忆
CREATE TABLE IF NOT EXISTS memories (
    id              bigserial PRIMARY KEY,
    namespace       text NOT NULL DEFAULT 'default',
    subject         text NOT NULL DEFAULT 'user:self',
    kind            text NOT NULL,
    content         text NOT NULL,
    structured      jsonb NOT NULL DEFAULT '{}'::jsonb,
    importance      smallint NOT NULL DEFAULT 5
                    CHECK (importance BETWEEN 1 AND 10),
    confidence      real NOT NULL DEFAULT 0.5
                    CHECK (confidence BETWEEN 0 AND 1),
    status          text NOT NULL DEFAULT 'active',
    is_speculation  boolean NOT NULL DEFAULT false,
    fingerprint     text NOT NULL,
    source          text,
    session_id      text,
    t_valid_from    date NOT NULL DEFAULT CURRENT_DATE,
    t_valid_to      date,
    created_at      timestamptz NOT NULL DEFAULT now(),
    updated_at      timestamptz NOT NULL DEFAULT now(),
    last_used_at    timestamptz,
    use_count       integer NOT NULL DEFAULT 0,
    superseded_by   bigint REFERENCES memories(id) ON DELETE SET NULL,
    embedding       vector(512),
    UNIQUE (namespace, fingerprint)
);

CREATE INDEX IF NOT EXISTS memories_namespace_status_idx
    ON memories (namespace, status);
CREATE INDEX IF NOT EXISTS memories_subject_idx
    ON memories (namespace, subject);
CREATE INDEX IF NOT EXISTS memories_vector_idx
    ON memories USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS memory_events (
    id          bigserial PRIMARY KEY,
    memory_id   bigint,
    action      text NOT NULL,
    detail      jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS memory_events_memory_idx
    ON memory_events (memory_id, created_at DESC);

-- Agent 多步运行的 checkpoint。
-- state 保存完整进度，steps 保存每一步的摘要，便于断点续跑和排查。
CREATE TABLE IF NOT EXISTS agent_runs (
    run_id      bigserial PRIMARY KEY,
    question    text NOT NULL,
    status      text NOT NULL DEFAULT 'running',
    phase       text NOT NULL DEFAULT 'retrieve',
    step_count  integer NOT NULL DEFAULT 0,
    state       jsonb NOT NULL DEFAULT '{}'::jsonb,
    error       text,
    created_at  timestamptz NOT NULL DEFAULT now(),
    updated_at  timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS agent_runs_status_idx
    ON agent_runs (status, updated_at DESC);

CREATE TABLE IF NOT EXISTS agent_steps (
    id              bigserial PRIMARY KEY,
    run_id          bigint NOT NULL
                    REFERENCES agent_runs(run_id) ON DELETE CASCADE,
    step_no         integer NOT NULL,
    node            text NOT NULL,
    status          text NOT NULL DEFAULT 'running',
    input_payload   jsonb NOT NULL DEFAULT '{}'::jsonb,
    output_payload  jsonb NOT NULL DEFAULT '{}'::jsonb,
    error           text,
    started_at      timestamptz NOT NULL DEFAULT now(),
    ended_at        timestamptz,
    UNIQUE (run_id, step_no)
);

CREATE INDEX IF NOT EXISTS agent_steps_run_idx
    ON agent_steps (run_id, step_no);

INSERT INTO schema_version (version)
VALUES (1)
ON CONFLICT (version) DO NOTHING;

INSERT INTO schema_version (version)
VALUES (2)
ON CONFLICT (version) DO NOTHING;

INSERT INTO schema_version (version)
VALUES (3)
ON CONFLICT (version) DO NOTHING;
