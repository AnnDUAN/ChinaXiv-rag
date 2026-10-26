"""从本地 chunks 里做关键词召回，返回最相关的原文段落。"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .. import config
from ..indexing.bm25 import BM25, QUERY_SYNONYMS, TOPIC_ALIASES, tokenize


def _topic_overlap(query_terms: set[str], title: str) -> float:
    """问题词有多少落在论文标题（含中英文别名）里。"""
    if not query_terms:
        return 0.0
    title_tokens = set(tokenize(title))
    hit = 0
    for term in query_terms:
        if term in title_tokens or any(alias in title for alias in TOPIC_ALIASES.get(term, [])):
            hit += 1
    return hit / len(query_terms)


def _is_definitional(query: str) -> bool:
    """判断是不是“X 是什么”这类整体定义问题。"""
    if not re.search(r"是什么|什么是|指什么|定义|概念|含义", query):
        return False
    # 问具体方面时不抢摘要的位置
    if re.search(
        r"核心思想|思想|原理|区别|差异|挑战|评估|应用|场景|流程|组成|模块|阶段|"
        r"特点|方法|优点|缺点|不足|局限|问题|风险|如何|怎么|为什么|哪些",
        query,
    ):
        return False
    return True


_DB_LOAD_FAILED = False


def load_chunks(chunks_dir: Path | None = None) -> list[dict[str, Any]]:
    """优先从 PostgreSQL 读切片；数据库不可用时退回 JSON。"""
    global _DB_LOAD_FAILED
    if chunks_dir is None and not _DB_LOAD_FAILED:
        try:
            from ..db.repo import load_chunks as load_db_chunks

            docs = load_db_chunks()
            if docs:
                return docs
        except Exception:
            # 数据库没启动或表还没建时，不让检索整个失败。
            _DB_LOAD_FAILED = True

    chunks_dir = chunks_dir or config.CHUNKS_DIR
    docs: list[dict[str, Any]] = []
    for path in sorted(Path(chunks_dir).glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for chunk in data.get("chunks", []):
            chunk = dict(chunk)
            chunk["paper_title"] = data.get("meta", {}).get("title", chunk.get("paper_id", ""))
            docs.append(chunk)
    return docs


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _phrase_bonus(query_tokens: list[str], text: str) -> float:
    """问题里相邻的两个词，如果原文里也连着出现，说明更像答案。"""
    compact = re.sub(r"\s+", "", text.lower())
    hits = 0
    for a, b in zip(query_tokens, query_tokens[1:]):
        if len(a) >= 2 and len(b) >= 2 and (a + b) in compact:
            hits += 1
    return 1 + 0.12 * min(hits, 3)


def _snippet(text: str, query_tokens: list[str], width: int = 220) -> str:
    lowered = text.lower()
    best_pos = -1
    for tok in sorted(set(query_tokens), key=len, reverse=True)[:8]:
        pos = lowered.find(tok.lower())
        if pos >= 0 and (best_pos < 0 or pos < best_pos):
            best_pos = pos
    if best_pos < 0:
        snippet = text[:width]
    else:
        start = max(0, best_pos - 80)
        snippet = text[start : start + width]
        if start > 0:
            snippet = "…" + snippet
    if len(text) > len(snippet) and not snippet.endswith("…"):
        snippet = snippet.rstrip() + "…"
    return snippet.replace("\n", " ").strip()


EXACT_MODEL_TERMS = re.compile(
    r"\b(?:FiD|REALM|REPLUG|GTR|GraphRAG|BERT|GPT|DPR|ColBERT|FAISS|ScaNN|"
    r"Self-RAG|RAG-RL|Toolformer|ReAct)\b",
    re.I,
)


def _semantic_conf(cosine: float) -> float:
    """把余弦相似度粗略映射成 0～1 的把握程度。"""
    return max(0.0, min(1.0, (cosine - 0.30) / 0.40))


def _select_diverse(
    docs: list[dict[str, Any]],
    ranked: list[int],
    doc_tokens: list[list[str]],
    limit: int,
) -> list[int]:
    """按顺序选结果，跳过同一章节里重复或相邻的切片。"""
    selected: list[int] = []
    selected_tokens: list[set[str]] = []
    for idx in ranked:
        doc = docs[idx]
        redundant = False
        for pos, chosen_idx in enumerate(selected):
            chosen = docs[chosen_idx]
            if doc.get("paper_id") != chosen.get("paper_id"):
                continue
            if doc.get("section_path") != chosen.get("section_path"):
                continue
            # 相邻块有重叠，选分数高的就够了
            if doc.get("chunk_id") in (chosen.get("prev_id"), chosen.get("next_id")):
                redundant = True
                break
            if _jaccard(set(doc_tokens[idx]), selected_tokens[pos]) > 0.55:
                redundant = True
                break
            a = doc.get("text", "")
            b = chosen.get("text", "")
            if a and b and (a in b or b in a):
                redundant = True
                break
        if redundant:
            continue
        selected.append(idx)
        selected_tokens.append(set(doc_tokens[idx]))
        if len(selected) >= limit:
            break
    return selected


def _context_factor(
    doc: dict[str, Any],
    q_set: set[str],
    query_tokens: list[str],
    content_terms: set[str],
    definitional: bool,
    mode: str,
) -> float:
    """向量/混合模式下用的轻量规则加成。

    向量模式尽量尊重余弦相似度的排序，规则只做很小的修正；
    混合模式里 BM25 已经带过一轮较强的章节/摘要加权，这里也只是轻加。
    """
    factor = 1.0
    path_tokens = set(tokenize(" ".join(doc.get("section_path", []))))
    # 只算有区分度的词，避免“RAG”这种每块都有的词把章节权重拉满
    path_hits = (q_set & content_terms) & path_tokens
    if mode == "hybrid" and path_hits:
        factor *= 1 + 0.05 * min(len(path_hits), 2)
    title_overlap = _topic_overlap(q_set, doc.get("paper_title", ""))
    if title_overlap:
        factor *= 1.01
    if doc.get("type") == "abstract":
        if definitional and title_overlap >= 0.5:
            factor *= 1.25 if mode == "vector" else 1.15
    phrase_hits = _phrase_bonus(query_tokens, doc.get("text", "")) - 1
    factor *= 1 + phrase_hits * (0.25 if mode == "vector" else 0.35)
    if len(re.findall(r"[（(]\d{1,2}[）)]", doc.get("text", ""))) >= 4:
        factor *= 0.98
    return factor


def search_chunks(
    query: str,
    top_k: int = config.DEFAULT_TOP_K,
    recall_n: int = config.RECALL_N,
    docs: list[dict[str, Any]] | None = None,
    mode: str = "hybrid",
    rerank: bool = False,
) -> dict[str, Any]:
    """mode: hybrid（默认，BM25+向量）/ bm25 / vector / auto。

    rerank=True 时，先多取一些候选，再用本地重排序模型重新排队。
    没有向量索引时，hybrid 会自动退回 bm25。
    """
    docs = docs if docs is not None else load_chunks()
    # 关键词块只是标签列表，不作为答案段落参与检索
    docs = [d for d in docs if d.get("type") != "keywords"]
    query_tokens = tokenize(query)
    if not docs:
        return {"query": query, "query_tokens": query_tokens, "results": [], "error": "还没有入库任何切片，先运行 build。"}
    if not query_tokens:
        return {"query": query, "query_tokens": [], "results": [], "error": "问题里没有可用于检索的词。"}

    # ---------- 1. BM25 打分（默认就是 jieba 分词） ----------
    # 优先读 data/index/jieba_bm25.json 里已经切好的词；
    # 没有索引或索引过期时，现场用 jieba 分词。
    doc_tokens: list[list[str]] | None = None
    try:
        from ..indexing.keyword_index import load_keyword_index

        keyword_index = load_keyword_index()
    except Exception:
        keyword_index = None
    if keyword_index and set(keyword_index.get("chunk_ids", [])) == {
        d.get("chunk_id", "") for d in docs
    }:
        row_of = {
            chunk_id: row
            for row, chunk_id in enumerate(keyword_index["chunk_ids"])
        }
        doc_tokens = [
            keyword_index["tokens"][row_of[d["chunk_id"]]] for d in docs
        ]
    if doc_tokens is None:
        doc_tokens = [
            tokenize(" ".join(d.get("section_path", [])) + " " + d.get("text", ""))
            for d in docs
        ]
    bm25 = BM25(doc_tokens)
    # 口语词在原文里不存在时，换成原文更可能用的说法
    expanded_tokens: list[str] = []
    for term in query_tokens:
        if term not in bm25.idf and term in QUERY_SYNONYMS:
            chosen_tokens: list[str] | None = None
            for synonym in QUERY_SYNONYMS[term]:
                pieces = tokenize(synonym)
                useful = [p for p in pieces if bm25.idf.get(p, 0) >= 0.05]
                if useful:
                    chosen_tokens = useful
                    break
            if chosen_tokens is None:
                chosen_tokens = tokenize(QUERY_SYNONYMS[term][0])
            expanded_tokens.extend(chosen_tokens or [term])
        else:
            expanded_tokens.append(term)
    expanded_tokens = list(dict.fromkeys(expanded_tokens))
    query_for_score = expanded_tokens or query_tokens

    base_scores = bm25.scores(query_for_score)
    q_set = set(query_for_score)
    definitional = _is_definitional(query)
    idf_max = 8.0
    term_idf = {t: bm25.idf.get(t, idf_max) for t in q_set}
    # 只拿“有区分度”的词计算命中度；几乎所有块都有的词不算
    effective = {t: v for t, v in term_idf.items() if v >= 0.05} or term_idf
    idf_sum = sum(effective.values()) or 1.0

    bm25_scores: list[float] = []
    coverage_by_doc: list[float] = []
    matched_by_doc: list[list[str]] = []
    for i, doc in enumerate(docs):
        score = base_scores[i]
        matched = q_set & set(doc_tokens[i])
        matched_by_doc.append(sorted(matched))
        coverage = sum(effective.get(t, 0.0) for t in matched) / idf_sum
        coverage_by_doc.append(coverage)
        if score > 0:
            # 覆盖的问题词越多，越可能是答案所在的段落
            score *= 0.8 + 0.4 * coverage
            path_hits = q_set & set(tokenize(" ".join(doc.get("section_path", []))))
            if path_hits:
                # 章节标题里直接出现了问题词，权重更高（命中越多越高）
                score *= 1 + 0.25 * len(path_hits)
            # “（1）（2）……”很多的块通常是论文贡献/提纲列表，不是具体答案
            if len(re.findall(r"[（(]\d{1,2}[）)]", doc.get("text", ""))) >= 4:
                score *= 0.9
            score *= _phrase_bonus(query_for_score, doc.get("text", ""))
            title_overlap = _topic_overlap(q_set, doc.get("paper_title", ""))
            if title_overlap:
                score *= 1.02
            if doc.get("type") == "abstract":
                # 只有“问题问的就是这篇论文的主题”时，才把摘要顶上来
                score *= 1.8 if (definitional and title_overlap >= 0.5) else 1.08
        bm25_scores.append(score)

    bm25_ranked = sorted(
        [i for i in range(len(docs)) if bm25_scores[i] > 0],
        key=lambda i: bm25_scores[i],
        reverse=True,
    )

    # ---------- 2. 向量打分（可选） ----------
    requested = mode if mode in {"auto", "bm25", "vector", "hybrid"} else "auto"
    vector_scores: list[float] | None = None
    vector_meta: dict[str, Any] | None = None
    vector_error: str | None = None

    if requested != "bm25":
        # 先尝试数据库里的 pgvector；不可用时再退回本地 npz。
        # 这样数据库没开、或者向量还没同步时，检索仍然能用。
        query_vector = None
        pg_status: dict[str, Any] | None = None
        pg_error: str | None = None

        def get_query_vector():
            nonlocal query_vector, vector_error
            if query_vector is None:
                try:
                    from ..indexing.embedder import LocalEmbedder

                    query_vector = LocalEmbedder.get().embed_query(query)
                except Exception as exc:
                    vector_error = f"本地向量模型加载失败：{exc}"
                    return None
            return query_vector

        try:
            from ..db.repo import embedding_status

            pg_status = embedding_status()
        except Exception as exc:  # 数据库没开、表还没建等
            pg_error = f"pgvector 不可用：{exc}"

        if (
            pg_status
            and pg_status.get("count") == len(docs)
            and pg_status.get("model") == config.EMBED_MODEL
            and pg_status.get("dim") == config.EMBED_DIM
        ):
            query_vector = get_query_vector()
            if query_vector is not None:
                try:
                    from ..db.repo import vector_scores as pg_vector_scores

                    pg_scores = pg_vector_scores(
                        query_vector,
                        limit=len(docs),
                        min_cosine=config.VECTOR_MIN_COSINE,
                        exact=config.PGVECTOR_EXACT,
                    )
                    vector_scores = [
                        float(
                            pg_scores.get(doc.get("chunk_id", ""), -1.0)
                        )
                        for doc in docs
                    ]
                    vector_meta = {
                        "model": pg_status.get("model"),
                        "dim": pg_status.get("dim"),
                        "count": pg_status.get("count"),
                        "built_at": pg_status.get("built_at"),
                        "backend": "pgvector",
                    }
                except Exception as exc:  # noqa: BLE001
                    vector_error = f"pgvector 检索失败：{exc}"

        # pgvector 没成功时，使用原来的本地向量文件。
        if vector_scores is None:
            try:
                from ..indexing.vector_index import load_vector_index

                loaded = load_vector_index()
            except Exception as exc:  # 索引文件损坏等
                loaded = None
                vector_error = f"读取向量索引失败：{exc}"
            else:
                if loaded is None:
                    if vector_error is None:
                        vector_error = "还没有向量索引，先运行 rag embed"
                elif loaded["meta"].get("model") != config.EMBED_MODEL:
                    vector_error = "向量索引的模型和当前配置不一致，请运行 rag embed 重建"
                elif set(loaded["chunk_ids"]) != {
                    d.get("chunk_id", "") for d in docs
                }:
                    vector_error = "向量索引和当前切片不一致，请运行 rag embed 重建"
                else:
                    query_vector = get_query_vector()
                    if query_vector is not None:
                        row_of = {
                            chunk_id: row
                            for row, chunk_id in enumerate(
                                loaded["chunk_ids"]
                            )
                        }
                        vectors = loaded["vectors"]
                        vector_scores = [
                            float(query_vector @ vectors[row_of[doc["chunk_id"]]])
                            if doc.get("chunk_id") in row_of
                            else -1.0
                            for doc in docs
                        ]
                        vector_meta = {
                            **loaded["meta"],
                            "backend": "npz",
                        }
                        vector_error = None

        if vector_scores is None and vector_error is None:
            vector_error = pg_error or "没有可用的向量索引"

    effective_mode = requested
    if requested == "auto":
        effective_mode = "hybrid" if vector_scores is not None else "bm25"
    elif requested in ("vector", "hybrid") and vector_scores is None:
        effective_mode = "bm25"

    # ---------- 3. 合并排序 ----------
    scores = [0.0] * len(docs)
    conf_scores = list(coverage_by_doc)

    # 重排序需要比最终 top_k 更大的候选池。
    # 多取一倍，给去重留出余量，最终再交给重排序模型。
    rank_limit = max(recall_n, top_k * 4)
    if rerank:
        rank_limit = max(rank_limit, config.RERANK_CANDIDATES * 2)

    if effective_mode == "bm25":
        ranked = bm25_ranked[:rank_limit]
        scores = list(bm25_scores)

    elif effective_mode == "vector":
        assert vector_scores is not None
        ranked = sorted(
            [i for i in range(len(docs)) if vector_scores[i] >= config.VECTOR_MIN_COSINE],
            key=lambda i: vector_scores[i],
            reverse=True,
        )[:rank_limit]
        for i in ranked:
            scores[i] = vector_scores[i] * _context_factor(
                docs[i], q_set, query_for_score, set(effective), definitional, "vector"
            )
        ranked.sort(key=lambda i: scores[i], reverse=True)
        conf_scores = [_semantic_conf(v) for v in vector_scores]

    else:  # hybrid：两路分数归一化后加权
        assert vector_scores is not None
        max_bm25 = max(bm25_scores) if bm25_scores else 0.0
        max_vec = max(vector_scores) if vector_scores else 0.0
        # 具体模型名（FiD、REPLUG 等）出现时，更相信关键词精确匹配；
        # 纯自然语言问法，更相信向量语义。
        weight = (
            config.HYBRID_EXACT_WEIGHT
            if EXACT_MODEL_TERMS.search(query)
            else config.HYBRID_BM25_WEIGHT
        )
        candidates: list[int] = []
        for i in range(len(docs)):
            bm25_norm = bm25_scores[i] / max_bm25 if max_bm25 > 0 else 0.0
            vec_norm = max(0.0, vector_scores[i]) / max_vec if max_vec > 0 else 0.0
            if vector_scores[i] < config.VECTOR_MIN_COSINE:
                vec_norm = 0.0
            mixed = weight * bm25_norm + (1 - weight) * vec_norm
            if mixed <= 0:
                continue
            mixed *= 100 * _context_factor(
                docs[i], q_set, query_for_score, set(effective), definitional, "hybrid"
            )
            scores[i] = mixed
            candidates.append(i)
        ranked = sorted(candidates, key=lambda i: scores[i], reverse=True)
        conf_scores = [
            max(coverage_by_doc[i], _semantic_conf(vector_scores[i]))
            for i in range(len(docs))
        ]

    # ---------- 4. 重排序（可选） ----------
    # 先取出比最终结果更多的候选，再让重排序模型逐条判断。
    # 重排序只能改变顺序，不能把没召回的内容变出来。
    rerank_error: str | None = None
    rerank_scores: dict[int, float] = {}
    rank_before_rerank: dict[int, int] = {}
    rerank_anchor_indices: list[int] = []
    rerank_candidate_count = 0

    if rerank:
        candidate_indices = _select_diverse(
            docs, ranked, doc_tokens, config.RERANK_CANDIDATES
        )
        rerank_candidate_count = len(candidate_indices)
        if candidate_indices:
            try:
                from .reranker import LocalReranker, build_rerank_text

                reranker = LocalReranker.get()
                texts = [
                    build_rerank_text(docs[idx]) for idx in candidate_indices
                ]
                values = reranker.score(query, texts)
                before = {idx: pos for pos, idx in enumerate(ranked, 1)}
                rank_before_rerank = {
                    idx: before.get(idx, 0) for idx in candidate_indices
                }
                rerank_scores = dict(zip(candidate_indices, values))
                candidate_indices.sort(
                    key=lambda idx: rerank_scores[idx], reverse=True
                )
            except Exception as exc:
                rerank_error = f"重排序失败，已使用原排序：{exc}"
                rerank_scores.clear()
                rank_before_rerank.clear()
                selected = _select_diverse(
                    docs, ranked, doc_tokens, top_k
                )
            else:
                # 保留原排序的前几名作为保底：
                # 重排擅长把“第 6～20 名”的正确段落提上来，
                # 但偶尔会把原本正确的小节压下去，所以不能完全丢掉原排序。
                anchors = _select_diverse(
                    docs, ranked, doc_tokens, config.RERANK_ANCHORS
                )
                rerank_anchor_indices = anchors
                combined: list[int] = []
                for idx in anchors + candidate_indices:
                    if idx not in combined:
                        combined.append(idx)
                selected = _select_diverse(
                    docs, combined, doc_tokens, top_k
                )
        else:
            selected = _select_diverse(docs, ranked, doc_tokens, top_k)
    else:
        selected = _select_diverse(docs, ranked, doc_tokens, top_k)

    # ---------- 5. 输出 ----------
    results = []
    for rank, idx in enumerate(selected, 1):
        doc = docs[idx]
        confidence_score = conf_scores[idx]
        if confidence_score >= 0.65:
            confidence = "较高"
        elif confidence_score >= 0.4:
            confidence = "一般"
        else:
            confidence = "较低"
        results.append(
            {
                "rank": rank,
                "score": round(scores[idx], 4),
                "coverage": round(coverage_by_doc[idx], 2),
                "confidence": confidence,
                "matched_terms": matched_by_doc[idx],
                "bm25_score": round(bm25_scores[idx], 4),
                "vector_score": round(vector_scores[idx], 4) if vector_scores is not None else None,
                "rerank_score": (
                    round(rerank_scores[idx], 4)
                    if idx in rerank_scores
                    else None
                ),
                "rank_before_rerank": rank_before_rerank.get(idx),
                "rerank_anchor": idx in rerank_anchor_indices,
                "chunk_id": doc.get("chunk_id", ""),
                "paper_id": doc.get("paper_id", ""),
                "paper_title": doc.get("paper_title", ""),
                "type": doc.get("type", ""),
                "section_path": doc.get("section_path", []),
                "page_start": doc.get("page_start"),
                "page_end": doc.get("page_end"),
                "snippet": _snippet(doc.get("text", ""), query_tokens),
                "text": doc.get("text", ""),
            }
        )

    payload = {
        "query": query,
        "query_tokens": query_tokens,
        "mode": effective_mode,
        "requested_mode": requested,
        "vector_model": vector_meta.get("model") if vector_meta else None,
        "vector_backend": vector_meta.get("backend") if vector_meta else None,
        "vector_error": vector_error,
        "candidate_count": len(ranked),
        "rerank_requested": rerank,
        "rerank_applied": bool(rerank_scores),
        "rerank_model": config.RERANK_MODEL if rerank_scores else None,
        "rerank_candidate_count": rerank_candidate_count,
        "rerank_error": rerank_error,
        "results": results,
    }
    if expanded_tokens != query_tokens:
        payload["expanded_tokens"] = expanded_tokens
    return payload
