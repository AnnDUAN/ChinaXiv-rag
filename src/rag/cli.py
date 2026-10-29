"""命令行入口：build / search / repl / stats / eval / fetch。"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from . import config
from .generation.answer import answer_question, prepare_answer
from .indexing.keyword_index import (
    KEYWORD_FILE,
    build_keyword_index,
    index_status as keyword_status,
)
from .indexing.vector_index import build_vector_index, index_status
from .ingestion.chunker import chunk_file, chunk_parsed
from .ingestion.downloader import (
    download_catalog,
    download_chinaxiv_papers,
    download_pdf,
    fetch_chinaxiv_catalog,
    merge_catalog_file,
)
from .ingestion.mineru_parser import parse_mineru_json
from .ingestion.pdf_parser import inspect_page
from .retrieval.search import load_chunks, search_chunks

MODE_NAMES = {
    "bm25": "关键词 BM25（jieba 分词）",
    "vector": "向量",
    "hybrid": "混合（BM25 + 向量）",
    "auto": "自动（有向量索引才混合）",
}


def _utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _load_paper_overrides() -> dict:
    if not config.PAPERS_CONFIG.exists():
        return {}
    try:
        return json.loads(config.PAPERS_CONFIG.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"配置文件格式不对：{config.PAPERS_CONFIG} ({exc})")
        return {}


def cmd_build(args: argparse.Namespace) -> int:
    config.PARSED_DIR.mkdir(parents=True, exist_ok=True)
    config.CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    overrides_all = _load_paper_overrides()

    if args.pdf:
        pdfs = [Path(args.pdf)]
    elif args.ids:
        pdfs = [config.PDF_DIR / f"{paper_id}.pdf" for paper_id in args.ids]
    else:
        pdfs = sorted(config.PDF_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"没有找到 PDF。把 PDF 放进：{config.PDF_DIR}")
        return 1

    mineru_dir = Path(args.mineru_dir)
    built = 0
    for n, pdf in enumerate(pdfs, 1):
        paper_id = pdf.stem
        parsed_path = config.PARSED_DIR / f"{paper_id}.json"
        chunks_path = config.CHUNKS_DIR / f"{paper_id}.json"
        if not args.force and parsed_path.exists() and chunks_path.exists():
            print(f"[{n}/{len(pdfs)}] 跳过（已入库）：{pdf.name}  加 --force 可重做")
            continue
        if not pdf.exists():
            print(f"[{n}/{len(pdfs)}] 跳过：找不到 {pdf}")
            continue
        mineru_json = mineru_dir / f"{paper_id}.json"
        if not mineru_json.exists():
            print(
                f"[{n}/{len(pdfs)}] 跳过：缺少 MinerU 解析结果 "
                f"{mineru_json}。先用 mineru-kit parse --format middle_json 生成。"
            )
            continue

        print(f"[{n}/{len(pdfs)}] 解析：{pdf.name}（MinerU JSON）")
        try:
            parsed = parse_mineru_json(
                mineru_json,
                paper_id=paper_id,
                overrides=overrides_all.get(paper_id, {}),
                out_path=parsed_path,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"    ⚠ MinerU 结果解析失败：{exc}")
            parsed_path.unlink(missing_ok=True)
            continue
        body_text = "".join(
            section.get("text", "")
            for section in parsed.get("sections", [])
            if section.get("type") == "body"
        )
        cjk_ratio = len(re.findall(r"[一-鿿]", body_text)) / max(len(body_text), 1)
        if cjk_ratio < 0.10:
            print(f"    标题：{parsed['meta'].get('title')}")
            print(
                f"    ⚠ 正文中文字符比例只有 {cjk_ratio:.1%}，"
                "可能是英文正文论文，跳过入库。"
            )
            parsed_path.unlink(missing_ok=True)
            continue
        result = chunk_parsed(parsed)
        chunks_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

        meta = parsed["meta"]
        q = parsed["quality"]
        print(f"    标题：{meta.get('title')}")
        print(
            f"    页数：{q['page_count']} | 正文字符：{q['body_chars']} | "
            f"章节：{q['body_section_count']} | 表：{q.get('table_count', 0)} | "
            f"公式：{q.get('equation_count', 0)} | "
            f"参考文献：{'有' if q['reference_found'] else '无'}"
        )
        stats = result["stats"]
        print(
            f"    切片：{stats['count']} 块 | 平均 {stats['avg_chars']} 字 | "
            f"范围 {stats['min_chars']}～{stats['max_chars']} 字"
        )
        for warning in q["warnings"]:
            print(f"    ⚠ {warning}")
        removed = q.get("removed", {})
        if removed:
            print(
                "    清理："
                f"页眉页脚 {removed.get('页眉页脚', 0)} | "
                f"脚注 {removed.get('脚注', 0)} | "
                f"参考文献 {removed.get('参考文献', 0)} | "
                f"图片 {removed.get('图片', 0)} | "
                f"重复段落 {removed.get('重复段落', 0)}"
            )
        built += 1

    print(f"\n完成：本次处理 {built} 篇。数据在 {config.DATA_DIR}")

    # 这里明确从刚写好的 chunks JSON 读取，避免读到数据库里的旧版本。
    docs = [
        d
        for d in load_chunks(config.CHUNKS_DIR)
        if d.get("type") != "keywords"
    ]
    if docs:
        kw = build_keyword_index(docs, force=True)
        print(f"关键词索引：jieba 分词，{kw['count']} 块 -> {KEYWORD_FILE}")

    if args.embed:
        code = _build_vectors(force=args.force)
        if code != 0:
            return code
    _sync_database()
    print(
        "提示：默认检索是混合模式；运行 rag embed 建立向量索引，"
        "没有索引时会自动退回 BM25。"
    )
    return 0


def cmd_index(args: argparse.Namespace) -> int:
    docs = [d for d in load_chunks() if d.get("type") != "keywords"]
    if not docs:
        print("还没有切片，先运行：rag build")
        return 1
    print(f"用 jieba 分词建立 BM25 索引：{len(docs)} 块……")
    try:
        index = build_keyword_index(docs, force=args.force)
    except Exception as exc:
        print(f"建立关键词索引失败：{exc}")
        return 1
    if index.get("skipped"):
        print("关键词索引已是最新，跳过。")
        return 0
    print(f"完成：{index['count']} 块，词表 {len(index['df'])} 个词")
    print(f"索引文件：{KEYWORD_FILE}")
    return 0


def cmd_chunk(args: argparse.Namespace) -> int:
    """只根据已解析的 JSON 重新切片，不重新解析 PDF。"""
    parsed_files = sorted(config.PARSED_DIR.glob("*.json"))
    if not parsed_files:
        print(f"没有解析结果。先运行 rag build，或把 PDF 放进 {config.PDF_DIR}")
        return 1

    print(f"从 {len(parsed_files)} 篇解析结果重新切片（不重新解析 PDF）……")
    total = 0
    for path in parsed_files:
        result = chunk_file(path)
        stats = result["stats"]
        total += result["chunk_count"]
        print(
            f"  {result['paper_id']}: {result['chunk_count']} 块 | "
            f"平均 {stats['avg_chars']} 字 | "
            f"范围 {stats['min_chars']}～{stats['max_chars']} 字"
        )

    docs = [
        d
        for d in load_chunks(config.CHUNKS_DIR)
        if d.get("type") != "keywords"
    ]
    kw = build_keyword_index(docs, force=True)
    print(
        f"\n完成：{len(parsed_files)} 篇，{total} 块；"
        f"关键词索引 {kw['count']} 块，词表 {len(kw['df'])} 个词。"
    )
    _sync_database()
    return 0


def _sync_database() -> None:
    """把当前 JSON/npz 同步到数据库；数据库没开时只提示，不影响检索。"""
    try:
        from .db.migrate import import_legacy

        result = import_legacy()
    except Exception as exc:  # noqa: BLE001
        print(f"提示：数据库未同步（{exc}）。JSON 仍然可以正常检索。")
        return
    print(
        f"数据库：{result['papers']} 篇 | "
        f"{result['chunks']} 个切片 | {result['embeddings']} 个向量"
    )


def _build_vectors(force: bool = False) -> int:
    docs = [
        d
        for d in load_chunks(config.CHUNKS_DIR)
        if d.get("type") != "keywords"
    ]
    if not docs:
        print("还没有切片，先运行：rag build")
        return 1
    print(f"\n向量模型：{config.EMBED_MODEL}（第一次运行会下载约 90MB，之后完全本地）")
    print(f"正在为 {len(docs)} 个切片生成向量……")
    started = time.time()
    try:
        meta = build_vector_index(docs, force=force)
    except Exception as exc:
        print(f"建立向量索引失败：{exc}")
        return 1
    if meta.get("skipped"):
        print("向量索引已是最新，跳过。")
        return 0
    print(
        f"完成：{meta.get('count')} 个切片，{meta.get('dim')} 维，"
        f"用时 {time.time() - started:.1f} 秒。"
    )
    print(f"索引文件：{config.INDEX_DIR / 'vectors.npz'}")
    return 0


def _print_results(payload: dict, full: bool = False) -> None:
    print(f"\n问题：{payload['query']}")
    print(f"分词：{' / '.join(payload.get('query_tokens', []))}")
    expanded = payload.get("expanded_tokens")
    if expanded:
        print(f"扩展：{' / '.join(expanded)}")
    mode = payload.get("mode", "bm25")
    backend = payload.get("vector_backend")
    backend_label = f" | 向量后端：{backend}" if backend else ""
    print(f"模式：{MODE_NAMES.get(mode, mode)}{backend_label}")
    if payload.get("vector_error") and payload.get("requested_mode") in ("vector", "hybrid"):
        print(f"提示：{payload['vector_error']}，本次已改用关键词检索。")
    elif payload.get("requested_mode") == "auto" and payload.get("vector_error"):
        print("提示：还没有向量索引，先运行 rag embed 可启用混合检索。")
    if payload.get("rerank_requested"):
        if payload.get("rerank_applied"):
            print(
                f"重排序：{payload.get('rerank_model')} | "
                f"候选 {payload.get('rerank_candidate_count')} 个"
            )
        elif payload.get("rerank_error"):
            print(f"重排序：未启用（{payload['rerank_error']}）")
    if payload.get("error"):
        print(f"\n{payload['error']}")
        return
    results = payload.get("results", [])
    if not results:
        print("\n没有找到相关内容。")
        return
    if results[0].get("confidence") == "较低":
        print("\n提示：相关度较低，这篇论文里可能没有直接答案，下面是相关度最高的段落。")
    for item in results:
        pages = (
            f"第 {item['page_start']} 页"
            if item.get("page_start") == item.get("page_end")
            else f"第 {item['page_start']}–{item['page_end']} 页"
        )
        section = " > ".join(item.get("section_path") or [])
        semantic = item.get("vector_score")
        extra = f" | 语义 {semantic:.2f}" if semantic is not None else ""
        rerank_score = item.get("rerank_score")
        if rerank_score is not None:
            extra += f" | 重排 {rerank_score:.2f}"
            before = item.get("rank_before_rerank")
            if before and before != item.get("rank"):
                extra += f"（原第 {before} 名）"
        anchor = " [保底]" if item.get("rerank_anchor") else ""
        print("\n" + "─" * 72)
        print(
            f"[{item['rank']}]{anchor} 得分 {item['score']:.3f}{extra} | "
            f"词命中 {item.get('coverage', 0):.0%} | 综合 {item.get('confidence', '')} | {pages}"
        )
        print(f"《{item['paper_title']}》 > {section}")
        print()
        print(item["text"] if full else item["snippet"])


def cmd_search(args: argparse.Namespace) -> int:
    payload = search_chunks(
        args.query, top_k=args.k, mode=args.mode, rerank=args.rerank
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    _print_results(payload, full=args.full)
    return 0


def _print_answer_payload(payload: dict) -> None:
    """打印一次问答的答案、出处和引用检查结果。"""
    if payload.get("structured_error"):
        print(
            "提示：结构化输出失败，已退回普通文本。\n"
            f"原因：{payload['structured_error']}\n"
        )

    print(payload["answer"])
    print("\n出处：")
    for source in payload.get("sources", []):
        print(
            f"  [{source['rank']}] 《{source['title']}》 "
            f"{source['pages']} | {source['section']}"
        )

    structured = payload.get("structured")
    if structured:
        citations = structured.get("citations") or []
        print(
            f"\n结构化输出：置信度 {structured.get('confidence', '')} | "
            f"引用 {len(citations)} 条"
        )
        for issue in payload.get("citation_issues") or []:
            print(f"  ⚠ {issue}")


def cmd_ask(args: argparse.Namespace) -> int:
    """检索原文，再让大模型基于原文回答并标出处。"""
    if args.prompt_only:
        try:
            prepared = prepare_answer(
                args.query,
                top_k=args.k,
                mode=args.mode,
                rerank=args.rerank,
                structured=not args.plain,
            )
        except Exception as exc:
            print(f"准备提示词失败：{exc}")
            return 1
        search = prepared["search"]
        if search.get("error"):
            print(search["error"])
            return 1
        if not prepared["messages"]:
            print("没有检索到相关段落，无法生成提示词。")
            return 1
        print("=== SYSTEM ===")
        print(prepared["messages"][0].content)
        print("\n=== USER ===")
        print(prepared["messages"][1].content)
        return 0

    try:
        payload = answer_question(
            args.query,
            top_k=args.k,
            mode=args.mode,
            rerank=args.rerank,
            structured=not args.plain,
        )
    except Exception as exc:
        print(f"生成失败：{exc}")
        print(
            "\n提示：可以先运行 "
            "`rag ask \"你的问题\" --prompt-only` 检查检索到的上下文；"
            "\n模型配置见 README 的“带出处的结构化问答”。"
        )
        return 1

    if payload.get("error"):
        print(payload["error"])
        return 1
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    _print_answer_payload(payload)
    return 0


def cmd_repl(args: argparse.Namespace) -> int:
    docs = load_chunks()
    if not docs:
        print("还没有入库内容，先运行：rag build")
        return 1
    print(f"已加载 {len(docs)} 个切片。输入问题回车检索，输入 q 退出。")
    while True:
        try:
            query = input("\n问题> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            continue
        if query.lower() in {"q", "quit", "exit"}:
            break
        payload = search_chunks(
            query, top_k=args.k, docs=docs, mode=args.mode, rerank=args.rerank
        )
        _print_results(payload, full=args.full)
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    files = sorted(config.CHUNKS_DIR.glob("*.json"))
    if not files:
        print("还没有入库内容，先运行：rag build")
        return 1
    total = 0
    print(f"入库文件：{len(files)} 篇\n")
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        meta = data.get("meta", {})
        stats = data.get("stats", {})
        total += data.get("chunk_count", 0)
        print(f"- {meta.get('title', path.stem)}")
        print(
            f"  {data.get('chunk_count', 0)} 块 | 平均 {stats.get('avg_chars')} 字 | "
            f"类型 {stats.get('by_type', {})}"
        )
    docs = load_chunks()
    kw = keyword_status(docs)
    if kw.get("exists"):
        state = "最新" if kw.get("up_to_date") else "已过期，请运行 rag index"
        print(
            f"\n关键词索引：jieba + BM25 | {kw.get('count')} 块 | "
            f"词表 {kw.get('vocab')} | 平均 {float(kw.get('avgdl') or 0):.1f} 词/块 | {state}"
        )
    else:
        print("\n关键词索引：未建立（运行 rag index）")

    status = index_status(docs)
    if status.get("exists"):
        state = "最新" if status.get("up_to_date") else "已过期，请运行 rag embed"
        print(
            f"向量索引：{status.get('model')} | {status.get('count')} 块 | "
            f"{status.get('dim')} 维 | {state}（可选）"
        )
    else:
        print("向量索引：未建立（可选，运行 rag embed 启用）")

    try:
        from .db.repo import stats as db_stats

        db = db_stats()
        print(
            f"数据库：{db['papers']} 篇 | {db['chunks']} 个切片 | "
            f"{db['embeddings']} 个向量 | {db['queries']} 条问答记录 | "
            f"{db.get('agent_runs', 0)} 次 Agent 运行"
        )
    except Exception:
        print("数据库：未连接（JSON 仍可正常检索）")
    print(f"合计：{total} 块")
    return 0


def cmd_db(args: argparse.Namespace) -> int:
    """数据库初始化、导入和统计。"""
    try:
        from .db.migrate import import_legacy, init_schema
        from .db.repo import stats as db_stats
    except Exception as exc:  # noqa: BLE001
        print(f"数据库依赖不可用：{exc}")
        return 1

    try:
        if args.db_command == "init":
            init_schema()
            print("数据库表已创建。")
            return 0

        if args.db_command == "import":
            init_schema()
            data_dir = Path(args.data_dir) if args.data_dir else None
            result = import_legacy(data_dir)
            print(
                f"导入完成：{result['papers']} 篇 | "
                f"{result['chunks']} 个切片 | "
                f"{result['embeddings']} 个向量"
            )
            return 0

        if args.db_command == "stats":
            result = db_stats()
            print(
                f"论文：{result['papers']} | 章节：{result['sections']} | "
                f"切片：{result['chunks']} | 向量：{result['embeddings']} | "
                f"问答记录：{result['queries']} | "
                f"Agent 运行：{result.get('agent_runs', 0)}"
            )
            return 0
    except Exception as exc:  # noqa: BLE001
        print(f"数据库操作失败：{exc}")
        return 1

    print("未知的数据库命令。")
    return 1


def _json_short(value: object, limit: int = 320) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False)
    except Exception:  # noqa: BLE001
        text = str(value)
    return text if len(text) <= limit else text[:limit] + "…"


def _print_agent_result(result: dict) -> None:
    """打印 Agent 的结果和每一轮判断摘要。"""
    status_names = {
        "completed": "完成",
        "failed": "失败",
        "running": "运行中",
    }
    status = status_names.get(result.get("status"), result.get("status"))
    resumed = "（从存档继续）" if result.get("resumed") else ""
    print(
        f"\nAgent 运行 #{result.get('run_id')} | {status}{resumed} | "
        f"检索 {result.get('search_rounds', 0)} 轮 | "
        f"{result.get('step_count', 0)} 步"
    )
    queries = result.get("queries") or []
    if queries:
        print("检索词：" + " → ".join(queries))

    for index, assessment in enumerate(
        result.get("assessments") or [], 1
    ):
        if assessment.get("sufficient"):
            verdict = "证据足够"
        else:
            verdict = "证据不足"
            missing = assessment.get("missing")
            if missing:
                verdict += f"（缺：{missing}）"
        print(f"  判断 {index}：{verdict}")
        next_query = assessment.get("next_query")
        if next_query:
            print(f"    下一轮检索：{next_query}")
        if assessment.get("error"):
            print(f"    ⚠ 判断过程有问题：{assessment['error']}")

    if result.get("error"):
        print(f"\n失败原因：{result['error']}")

    answer = result.get("answer")
    if answer:
        if answer.get("error"):
            print(f"\n{answer['error']}")
        else:
            print()
            _print_answer_payload(answer)

    print(f"\n查看步骤：.\\rag agent trace {result.get('run_id')}")


def cmd_agent(args: argparse.Namespace) -> int:
    """Agent 多步检索的命令行入口。"""
    try:
        from .agent import AgentRunner
        from .agent.checkpoint import PostgresCheckpointStore
    except Exception as exc:  # noqa: BLE001
        print(f"Agent 模块不可用：{exc}")
        return 1

    store = PostgresCheckpointStore()

    if args.agent_command == "list":
        try:
            runs = store.list_runs(limit=args.k)
        except Exception as exc:  # noqa: BLE001
            print(f"读取 Agent 运行记录失败：{exc}")
            print("提示：先运行 `rag db init` 创建 Agent 表。")
            return 1
        if not runs:
            print("还没有 Agent 运行记录。")
            return 0
        for run in runs:
            print(
                f"#{run['run_id']} | {run['status']} | "
                f"{run['step_count']} 步 | {run['updated_at']} | "
                f"{run['question']}"
            )
        return 0

    if args.agent_command == "trace":
        try:
            trace = store.trace(args.run_id)
        except Exception as exc:  # noqa: BLE001
            print(f"读取 Agent 追踪失败：{exc}")
            print("提示：先运行 `rag db init` 创建 Agent 表。")
            return 1
        if trace is None:
            print(f"找不到 Agent 运行 #{args.run_id}")
            return 1
        run = trace["run"]
        state = run.get("state") or {}
        print(
            f"任务 #{run['run_id']} | {run['status']} | "
            f"phase={run['phase']} | {run['step_count']} 步"
        )
        print(f"问题：{run['question']}")
        if run.get("error"):
            print(f"错误：{run['error']}")
        if run.get("created_at"):
            print(f"创建：{run['created_at']} | 更新：{run['updated_at']}")

        for step in trace["steps"]:
            print(
                f"\n[{step['step_no']}] {step['node']} | {step['status']} "
                f"| {step['started_at']} → {step['ended_at']}"
            )
            print(f"  输入：{_json_short(step['input_payload'])}")
            if step.get("output_payload"):
                print(f"  输出：{_json_short(step['output_payload'])}")
            if step.get("error"):
                print(f"  错误：{step['error']}")

        final_answer = state.get("answer")
        if final_answer and not final_answer.get("error"):
            print("\n最终答案：")
            _print_answer_payload(final_answer)
        return 0

    if args.agent_command == "ask":
        try:
            runner = AgentRunner(
                store=store,
                max_steps=args.max_steps,
                max_search_rounds=args.max_search_rounds,
                top_k=args.k,
                mode=args.mode,
                rerank=args.rerank,
            )
            result = runner.run(args.query)
        except Exception as exc:  # noqa: BLE001
            print(f"Agent 运行失败：{exc}")
            print(
                "提示：Agent 需要 PostgreSQL 和大模型配置；"
                "先运行 `rag db init`，并检查 config/llm.json。"
            )
            return 1
        if args.json:
            print(
                json.dumps(
                    result, ensure_ascii=False, indent=2, default=str
                )
            )
        else:
            _print_agent_result(result)
        return 0 if result.get("status") == "completed" else 1

    if args.agent_command == "resume":
        try:
            runner = AgentRunner(store=store)
            result = runner.run(
                resume_run_id=args.run_id,
                max_steps=args.max_steps,
                max_search_rounds=args.max_search_rounds,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Agent 恢复失败：{exc}")
            return 1
        if args.json:
            print(
                json.dumps(
                    result, ensure_ascii=False, indent=2, default=str
                )
            )
        else:
            _print_agent_result(result)
        return 0 if result.get("status") == "completed" else 1

    print("未知的 Agent 命令。")
    return 1


def cmd_embed(args: argparse.Namespace) -> int:
    code = _build_vectors(force=args.force)
    if code == 0:
        _sync_database()
    return code


def cmd_eval(args: argparse.Namespace) -> int:
    path = Path(args.questions or (config.ROOT / "tests" / "questions.json"))
    if not path.exists():
        print(f"找不到测试问题文件：{path}")
        return 1
    questions = json.loads(path.read_text(encoding="utf-8"))
    docs = load_chunks()
    if not docs:
        print("还没有入库内容，先运行：rag build")
        return 1

    rerank_label = " | 重排序：开" if args.rerank else ""
    print(
        f"检索模式：{MODE_NAMES.get(args.mode, args.mode)}"
        f"{rerank_label} | Hit@{args.k} + Recall@{args.k}\n"
    )
    passed = 0
    recall_sum = 0.0
    for i, item in enumerate(questions, 1):
        payload = search_chunks(
            item["question"],
            top_k=args.k,
            docs=docs,
            mode=args.mode,
            rerank=args.rerank,
        )
        results = payload.get("results", [])
        hit = False
        hit_result = None
        expected_papers = item.get("expect_paper_ids") or []
        if isinstance(expected_papers, str):
            expected_papers = [expected_papers]
        expected_chunks = item.get("expect_chunk_ids") or []
        if isinstance(expected_chunks, str):
            expected_chunks = [expected_chunks]

        relevant_results: list[dict] = []
        for r in results:
            haystack = r["text"] + " " + " ".join(r.get("section_path") or [])
            if expected_chunks:
                relevant = r.get("chunk_id") in expected_chunks
            else:
                relevant = bool(
                    expected_papers
                    and r.get("paper_id") in expected_papers
                    and (
                        not item.get("expect_section")
                        or item["expect_section"]
                        in " ".join(r.get("section_path") or [])
                    )
                    and any(
                        k.lower() in haystack.lower()
                        for k in item.get("expect_any", [])
                    )
                )
            if relevant:
                relevant_results.append(r)
                if hit_result is None:
                    hit_result = r
                    hit = True

        if expected_chunks:
            found = {
                r.get("chunk_id")
                for r in relevant_results
                if r.get("chunk_id") in expected_chunks
            }
            recall = len(found) / max(len(expected_chunks), 1)
        else:
            found_papers = {
                r.get("paper_id") for r in relevant_results
            }
            recall = (
                len(found_papers) / max(len(expected_papers), 1)
                if expected_papers
                else 0.0
            )

        passed += int(hit)
        recall_sum += recall
        print(
            f"[{'通过' if hit else '未通过'}] {i}. {item['question']}"
            f"  |  Recall@{args.k}: {recall:.0%}"
        )
        if hit_result is not None:
            print(
                f"      命中：第 {hit_result['rank']} 名 | "
                f"《{hit_result.get('paper_title', '')}》 | "
                f"{' > '.join(hit_result['section_path'])} | "
                f"第 {hit_result['page_start']} 页"
            )
        elif results:
            top = results[0]
            print(
                f"      未命中，第一名：《{top.get('paper_title', '')}》 | "
                f"{' > '.join(top['section_path'])} | 第 {top['page_start']} 页"
            )
        else:
            print("      没有召回结果")
    mean_recall = recall_sum / max(len(questions), 1)
    print(
        f"\nHit@{args.k}：{passed}/{len(questions)} | "
        f"Recall@{args.k}：{mean_recall:.1%}"
    )
    return 0 if passed == len(questions) else 1


def _print_downloads(
    results: list[dict],
    extra_skipped: int = 0,
) -> None:
    downloaded = skipped = failed = 0
    for item in results:
        status = item.get("status")
        if status == "downloaded":
            downloaded += 1
            path = item["path"]
            print(f"  ✓ {item['paper_id']}  {path}  ({path.stat().st_size} 字节)")
        elif status == "skipped":
            skipped += 1
            print(f"  · {item['paper_id']}  已存在，跳过")
        else:
            failed += 1
            print(f"  ✗ {item['paper_id']}  {item.get('error')}")
    total_skipped = skipped + extra_skipped
    print(
        f"\n本次：下载 {downloaded} 篇，跳过 {total_skipped} 篇，失败 {failed} 篇"
    )
    if downloaded:
        print("下一步：运行 rag build 把新论文入库（需要向量再加 --embed）。")


def cmd_fetch(args: argparse.Namespace) -> int:
    if args.paper:
        codes = [c.strip() for c in args.paper.split(",") if c.strip()]
        print(f"从 ChinaXiv 下载 {len(codes)} 篇……")
        _print_downloads(download_chinaxiv_papers(codes, force=args.force))
        return 0

    if args.catalog:
        domain = args.catalog.strip()
        if domain.lower() in {"cs", "计算机", "计算机科学", "computer-science"}:
            domain = "243"
        if not domain.isdigit():
            print("分类编号必须是数字。默认的计算机科学是 243，可以直接写 --catalog。")
            return 1

        result = fetch_chinaxiv_catalog(pages=args.pages, domain=domain)
        items = result["items"]
        ignored_note = ""
        if result.get("filtered_ignored"):
            ignored_note = f"忽略名单 {result['filtered_ignored']} 篇；"
        print(
            f"ChinaXiv 计算机科学：抓取 {result['fetched']} 篇；"
            f"过滤英文标题 {result['filtered_english']} 篇；"
            f"{ignored_note}保留中文 {len(items)} 篇。"
        )
        path, merged, added, updated = merge_catalog_file(items)
        print(
            f"本次新增 {added} 篇，更新 {updated} 篇；列表共 {len(merged)} 篇。"
        )
        print(f"列表文件：{path}")
        for item in items[:10]:
            detail = item.get("subject_detail") or ""
            extra = f" | {detail}" if detail else ""
            print(f"  {item['paper_id']}  {item['title']}{extra}")
        if len(items) > 10:
            print(f"  …… 还有 {len(items) - 10} 篇")

        if args.download:
            results, skipped_existing, missing_total = download_catalog(
                merged, limit=args.limit, force=args.force
            )
            print(
                f"\n列表共 {len(merged)} 篇；已有 PDF {skipped_existing} 篇；"
                f"待下载 {missing_total} 篇。"
            )
            if missing_total == 0:
                print("所有论文的 PDF 都已经在本地，无需重复下载。")
                return 0
            if args.limit > 0:
                print(f"本次最多下载 {args.limit} 篇。")
            _print_downloads(results, extra_skipped=skipped_existing)
        return 0

    if not args.url:
        print(
            "用法：\n"
            "  rag fetch --paper 202610.00065\n"
            "  rag fetch --catalog                 # 抓 1 页计算机科学论文列表\n"
            "  rag fetch --catalog --pages 3\n"
            "  rag fetch --catalog --pages 3 --download\n"
            "  rag fetch --catalog --pages 3 --download --limit 10\n"
            "  rag fetch --url https://.../xxx.pdf --name xxx"
        )
        return 1
    path = download_pdf(args.url, name=args.name, referer=args.referer)
    print(f"已保存：{path} ({path.stat().st_size} 字节)")
    return 0


def _resolve_pdf_arg(value: str) -> Path:
    path = Path(value)
    if path.exists():
        return path
    for candidate in (config.PDF_DIR / value, config.PDF_DIR / f"{value}.pdf"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"找不到 PDF：{value}\n"
        f"可以直接写 data/pdfs 下的文件名或论文编号，例如 202609.00318。"
    )


def cmd_inspect(args: argparse.Namespace) -> int:
    """只检查一页的解析细节，不写任何数据文件。"""
    try:
        pdf_path = _resolve_pdf_arg(args.pdf)
        info = inspect_page(pdf_path, page_no=args.page)
    except Exception as exc:
        print(f"检查失败：{exc}")
        return 1

    width, height = info["page_size"]
    print(f"文件：{info['pdf']}")
    print(
        f"第 {info['page']}/{info['page_count']} 页 | "
        f"页面 {width}×{height} | 正文标准字号 {info['body_font_size']}"
    )
    ref = info.get("reference_start")
    print(f"参考文献起点：{ref if ref else '未识别到'}")
    front = info.get("front_matter_end")
    print(f"前置内容截止：{front if front else '未识别到（按第一页处理）'}")

    removed = info["removed_lines"]
    counts = info.get("page_removed_counts", {})
    print(f"\n本页删除 {len(removed)} 行：" + " | ".join(
        f"{reason} {count}" for reason, count in counts.items()
    ))
    if not removed:
        print("  （无）")
    groups: dict[str, list[dict]] = {}
    for item in removed:
        groups.setdefault(item["reason"], []).append(item)
    for reason, items in groups.items():
        show = len(items) if args.all_removed else min(len(items), 8)
        print(f"\n  [{reason}] {len(items)} 行")
        for item in items[:show]:
            text = item["text"].replace("\n", " ")
            direction = tuple(item["direction"])
            print(
                f"    size={item['size']} dir={direction} | {text[:100]}"
            )
        if show < len(items):
            print(f"    …… 还有 {len(items) - show} 行（加 --all-removed 查看）")

    print(
        f"\n本页保留 {len(info['kept_lines'])} 行 | "
        f"正文字符 {len(info['text'])}"
    )
    body = info["text"]
    print(body[:1500] + ("…" if len(body) > 1500 else ""))

    if args.raw:
        print(f"\n原始行（{len(info['raw_lines'])}）：")
        for index, line in enumerate(info["raw_lines"], 1):
            x0, y0, x1, y1 = line["bbox"]
            mark = " [竖排]" if line["vertical"] else ""
            print(
                f"  {index:3d} y={y0:6.1f}-{y1:6.1f} "
                f"x={x0:6.1f}-{x1:6.1f} size={line['size']:4.1f}"
                f"{mark} | {line['text'][:100]}"
            )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag", description="本地论文 RAG：解析、切片、关键词 + 向量检索（暂不接大模型生成）"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="读取 MinerU JSON、切片并入库")
    p_build.add_argument("--pdf", help="只处理某一份 PDF")
    p_build.add_argument("--ids", nargs="*", help="只处理这些论文编号")
    p_build.add_argument(
        "--mineru-dir",
        default=str(config.MINERU_JSON_DIR),
        help="MinerU middle JSON 所在目录",
    )
    p_build.add_argument("--force", action="store_true", help="已入库也重做")
    p_build.add_argument("--embed", action="store_true", help="入库后顺便建立向量索引")
    p_build.set_defaults(func=cmd_build)

    p_index = sub.add_parser("index", help="用 jieba 分词建立关键词索引（混合检索的一路）")
    p_index.add_argument("--force", action="store_true", help="强制重建")
    p_index.set_defaults(func=cmd_index)

    p_chunk = sub.add_parser(
        "chunk", help="只重新切片并重建关键词索引（不重新解析 PDF）"
    )
    p_chunk.set_defaults(func=cmd_chunk)

    p_embed = sub.add_parser("embed", help="为已有切片建立/更新向量索引（可选）")
    p_embed.add_argument("--force", action="store_true", help="强制重建")
    p_embed.set_defaults(func=cmd_embed)

    mode_choices = ["auto", "hybrid", "bm25", "vector"]

    p_search = sub.add_parser("search", help="检索一个问题")
    p_search.add_argument("query")
    p_search.add_argument("-k", type=int, default=config.DEFAULT_TOP_K, help="返回几条，默认 5")
    p_search.add_argument(
        "--mode", choices=mode_choices, default="hybrid",
        help="默认 hybrid（BM25 + 本地向量）；没有向量索引时自动退回 bm25",
    )
    p_search.add_argument("--full", action="store_true", help="显示完整切片，不只显示片段")
    p_search.add_argument(
        "--rerank",
        action="store_true",
        help="先用本地模型对候选段落重排序（第一次运行会下载模型）",
    )
    p_search.add_argument("--json", action="store_true", help="输出 JSON")
    p_search.set_defaults(func=cmd_search)

    p_ask = sub.add_parser(
        "ask", help="检索原文并让大模型基于原文回答（需要 OpenAI 兼容接口）"
    )
    p_ask.add_argument("query")
    p_ask.add_argument("-k", type=int, default=config.DEFAULT_TOP_K)
    p_ask.add_argument("--mode", choices=mode_choices, default="hybrid")
    p_ask.add_argument(
        "--rerank",
        action="store_true",
        help="检索后先对候选段落重排序，再交给大模型",
    )
    p_ask.add_argument(
        "--plain",
        "--no-structured",
        dest="plain",
        action="store_true",
        help="不使用结构化输出，直接让模型返回普通文本",
    )
    p_ask.add_argument(
        "--prompt-only",
        action="store_true",
        help="只显示检索上下文和提示词，不调用大模型",
    )
    p_ask.add_argument("--json", action="store_true", help="输出 JSON")
    p_ask.set_defaults(func=cmd_ask)

    p_repl = sub.add_parser("repl", help="连续提问模式")
    p_repl.add_argument("-k", type=int, default=config.DEFAULT_TOP_K)
    p_repl.add_argument("--mode", choices=mode_choices, default="hybrid")
    p_repl.add_argument("--rerank", action="store_true", help="对候选段落重排序")
    p_repl.add_argument("--full", action="store_true")
    p_repl.set_defaults(func=cmd_repl)

    p_stats = sub.add_parser("stats", help="查看已入库情况")
    p_stats.set_defaults(func=cmd_stats)

    p_db = sub.add_parser("db", help="PostgreSQL 数据库管理")
    db_sub = p_db.add_subparsers(dest="db_command", required=True)
    p_db_init = db_sub.add_parser("init", help="创建数据库表")
    p_db_init.set_defaults(func=cmd_db)
    p_db_import = db_sub.add_parser(
        "import", help="把现有 JSON/npz 导入数据库"
    )
    p_db_import.add_argument(
        "--data-dir", help="数据目录，默认 data/（一般不用改）"
    )
    p_db_import.set_defaults(func=cmd_db)
    p_db_stats = db_sub.add_parser("stats", help="查看数据库统计")
    p_db_stats.set_defaults(func=cmd_db)

    p_agent = sub.add_parser(
        "agent", help="多步检索 Agent：证据不够时补查，并保存运行进度"
    )
    agent_sub = p_agent.add_subparsers(
        dest="agent_command", required=True
    )

    agent_ask = agent_sub.add_parser(
        "ask", help="运行一次 Agent 问答（需要 PostgreSQL 和大模型）"
    )
    agent_ask.add_argument("query")
    agent_ask.add_argument("-k", type=int, default=config.DEFAULT_TOP_K)
    agent_ask.add_argument("--mode", choices=mode_choices, default="hybrid")
    agent_ask.add_argument(
        "--rerank", action="store_true", help="每轮检索都启用重排序"
    )
    agent_ask.add_argument(
        "--max-steps", type=int, default=8, help="最多执行几个步骤，默认 8"
    )
    agent_ask.add_argument(
        "--max-search-rounds",
        type=int,
        default=3,
        help="最多检索几轮（含第一轮），默认 3",
    )
    agent_ask.add_argument("--json", action="store_true")
    agent_ask.set_defaults(func=cmd_agent)

    agent_resume = agent_sub.add_parser(
        "resume", help="从 checkpoint 继续一次未完成的运行"
    )
    agent_resume.add_argument("run_id", type=int)
    agent_resume.add_argument(
        "--max-steps", type=int, help="可选：覆盖原来的最大步数"
    )
    agent_resume.add_argument(
        "--max-search-rounds", type=int, help="可选：覆盖原来的最大检索轮数"
    )
    agent_resume.add_argument("--json", action="store_true")
    agent_resume.set_defaults(func=cmd_agent)

    agent_trace = agent_sub.add_parser(
        "trace", help="查看一次运行的每一步"
    )
    agent_trace.add_argument("run_id", type=int)
    agent_trace.set_defaults(func=cmd_agent)

    agent_list = agent_sub.add_parser(
        "list", help="列出最近的 Agent 运行"
    )
    agent_list.add_argument("-k", type=int, default=20)
    agent_list.set_defaults(func=cmd_agent)

    p_eval = sub.add_parser("eval", help="跑一组测试问题")
    p_eval.add_argument("--questions", help="测试问题 JSON 路径")
    p_eval.add_argument(
        "-k", "--k", type=int, default=5, help="按 Hit@K 计算，默认 5"
    )
    p_eval.add_argument("--mode", choices=mode_choices, default="hybrid")
    p_eval.add_argument(
        "--rerank",
        action="store_true",
        help="评测时启用本地重排序",
    )
    p_eval.set_defaults(func=cmd_eval)

    p_fetch = sub.add_parser("fetch", help="从 ChinaXiv 抓论文列表并下载 PDF")
    p_fetch.add_argument(
        "--paper",
        help="ChinaXiv 论文编号或详情页链接，多个用逗号隔开，如 202610.00065",
    )
    p_fetch.add_argument("--url", help="PDF 直链")
    p_fetch.add_argument("--name", help="保存文件名")
    p_fetch.add_argument("--referer", help="Referer")
    p_fetch.add_argument(
        "--catalog",
        nargs="?",
        const="243",
        help="抓 ChinaXiv 分类列表；省略值就是计算机科学（243）",
    )
    p_fetch.add_argument(
        "--pages", type=int, default=1, help="抓列表的页数，每页约 20 篇"
    )
    p_fetch.add_argument(
        "--download",
        action="store_true",
        help="按合并后的论文列表下载所有缺失的 PDF（已下载的自动跳过）",
    )
    p_fetch.add_argument(
        "--limit", type=int, default=0,
        help="配合 --download，本次最多下载几篇缺失论文，0 表示全部",
    )
    p_fetch.add_argument("--force", action="store_true", help="已存在的 PDF 也重新下载")
    p_fetch.set_defaults(func=cmd_fetch)

    p_inspect = sub.add_parser(
        "inspect", help="检查单页解析：原始行、删除原因、保留正文（只读）"
    )
    p_inspect.add_argument("pdf", help="PDF 路径，或 data/pdfs 下的论文编号")
    p_inspect.add_argument("--page", type=int, default=1, help="页码，从 1 开始")
    p_inspect.add_argument(
        "--raw", action="store_true", help="显示这一页所有原始行及位置"
    )
    p_inspect.add_argument(
        "--all-removed", action="store_true", help="显示所有被删除的行"
    )
    p_inspect.set_defaults(func=cmd_inspect)

    return parser


def main(argv: list[str] | None = None) -> int:
    _utf8()
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
