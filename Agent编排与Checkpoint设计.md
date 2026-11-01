# Agent 编排与 Checkpoint 设计

## 1. 这一版要解决什么

普通 `rag ask` 是固定流程：

```text
问题 → 检索一次 → 拼提示词 → 回答
```

它适合单篇论文、单次就能查到的问题，但不适合：

- 问题比较口语化，和论文用词对不上；
- 需要把“这些论文有哪些路线”拆成几个子问题；
- 需要同时查多篇论文再归纳。

旧版 Agent 走的是“检索 → 判断证据 → 不够再补查”，
每一轮都要额外调用一次大模型判断，实际比较慢。

这一版改成**开头分析一次，后面批量检索一次，最后回答一次**：

```text
分析问题 → 批量检索 → 回答
```

## 2. 整体流程

```text
analyze：分析问题
  ├─ intent：问题类型
  ├─ rewritten_query：改写后的检索词
  └─ sub_queries：0～3 个子查询
        ↓
retrieve：依次执行 rewritten_query 和 sub_queries
  └─ 按 chunk_id 去重，合并成一个证据池
        ↓
answer：基于证据池生成一次带引用的回答
        ↓
done
```

整个流程固定 3 个步骤：

1. `analyze`
2. `retrieve`
3. `answer`

不再在每轮检索后单独判断“证据够不够”。

## 3. 问题怎么分类

分类由 `src/rag/agent/analyze.py` 里的一次结构化大模型调用完成。

意图只保留 3 类：

| intent | 含义 | 例子 |
|---|---|---|
| `single` | 单篇问题：定义、事实、数字、追问等 | 双沙漏模型是什么？ |
| `comparison` | 两个对象对比 | Roofline 和 BigCodeBench 有什么区别？ |
| `multi_paper` | 多篇论文归纳、分类、路线 | 这些论文里幻觉治理有哪几类？ |

模型只输出分类，不回答问题。

如果模型分析失败，代码会用少量规则做兜底判断：

- 出现“这些论文、哪些路线、分几类、综述” → `multi_paper`；
- 出现“区别、对比、相比、优劣” → `comparison`；
- 其他情况统一归到 `single`。

## 4. 查询怎么改写

同一个大模型调用还会输出：

```text
rewritten_query
sub_queries
```

改写规则写进提示词：

1. 保留专有名词、模型名、系统名和英文缩写；
2. 补上论文里常见的同义说法；
3. 写成简短、具体的检索词，不写成空泛问句；
4. 不重复原问题，也不重复已生成的子查询；
5. 不编造论文里没有的术语。

子查询数量：

- `single`：不生成子查询，只检索改写后的一条查询；
- `comparison`：1～2 个，分别覆盖对比对象；
- `multi_paper`：2～3 个，分别覆盖不同路线或不同方面。

例如：

```text
原问题：这些论文里，减少大模型胡说的办法大致可以分成哪几类？

intent: multi_paper
rewritten_query: 大语言模型 幻觉治理 方法分类
sub_queries:
  - 元认知门控 动态检索 幻觉治理
  - 系统性证伪 固定策略 对照实验
  - 自适应路由 幻觉治理 能力边界
```

代码还会再做一次清理：

- 去掉空查询；
- 去掉和原问题、改写查询重复的子查询；
- 同一批查询内去重；
- `single` 不保留子查询；
- `comparison` 最多保留 2 个，`multi_paper` 最多保留 3 个；
- 总数受 `--max-search-rounds` 限制，默认最多 4 次检索。

如果分析调用失败，不会让整次任务失败，而是退回：

```text
intent = 规则推断
rewritten_query = 原问题
sub_queries = []
```

## 5. 为什么先拆开“检索”和“生成”

原来的 `answer_question()` 内部同时做检索和回答。
Agent 需要先查多个查询，最后只生成一次，所以拆成了：

```text
search_chunks       只负责找片段
answer_from_search  只负责基于片段生成答案
answer_question     普通 RAG 的固定组合
```

这样 Agent 不会每查一次就生成一次答案，也方便单独测试。

## 6. 状态里存什么

`AgentState` 是完整的运行进度：

- 原始问题；
- 当前阶段：`analyze` / `retrieve` / `answer` / `done`；
- 已执行步数、最大步数；
- 分析结果：`intent`、`rewritten_query`、`sub_queries`；
- 已执行查询、等待执行查询；
- 去重后的证据片段；
- 每次查询的摘要；
- 最终答案或错误信息；
- 检索参数：`top_k`、`mode`、`rerank`。

状态整体存进 PostgreSQL 的 JSONB 字段，不依赖进程内存。

旧版 `assess` 阶段只在读取旧 checkpoint 时兼容，新流程不会进入。

## 7. Checkpoint 表

### `agent_runs`

一次运行一行：

- `run_id`：运行编号；
- `question`：原始问题；
- `status`：`running` / `completed` / `failed`；
- `phase`：当前阶段；
- `step_count`：已执行步数；
- `state`：完整状态 JSON；
- `error`：失败原因；
- `created_at` / `updated_at`。

### `agent_steps`

每一步一行：

- `run_id` + `step_no`：属于哪次运行的第几步；
- `node`：`analyze` / `retrieve` / `answer`；
- `status`：`running` / `success` / `failed` / `interrupted`；
- `input_payload`：这一步的输入摘要；
- `output_payload`：这一步的输出摘要；
- `error`；
- `started_at` / `ended_at`。

完整正文放在 `agent_runs.state` 里，步骤表只存摘要，避免 trace 太长。

## 8. 断点续跑怎么保证

每个节点成功后：

1. 先更新步骤状态；
2. 再保存完整状态。

如果进程在两步之间中断：

- 最后完成的节点已经存好；
- 下一次 `resume` 从当前 `phase` 继续；
- 没完成的节点会重新执行一次。

这是“至少执行一次”，不是严格“只执行一次”。

`resume` 时会先把遗留的 `running` 步骤标成 `interrupted`，再用同一个步号重新执行。

旧版 Agent 的状态版本是 1，新流程是 2。
旧进度不兼容时会直接提示重新发起任务。

## 9. 命令行

```powershell
# 新建一次运行
.\rag agent ask "问题" --max-steps 8 --max-search-rounds 4

# 列出最近的运行
.\rag agent list

# 查看每一步：分析、检索、回答
.\rag agent trace 1

# 中断后继续；可以顺便提高最大步数
.\rag agent resume 1 --max-steps 10
```

普通单篇问题仍然使用：

```powershell
.\rag ask "问题"
```

Agent 不会接管所有问题，避免简单问题也变慢、变贵。

## 10. 测试

离线测试不连数据库、不连真实模型，覆盖：

- 分析后只查一次；
- 多个子查询会在一个检索步骤里全部执行；
- 重复子查询会被去掉；
- 没有检索结果时直接返回“信息不足”；
- 步数不够时失败，`resume` 后能完成；
- trace 能看到 `analyze → retrieve → answer`；
- 分析失败时退回原问题；
- 意图规则兜底和子查询数量限制。

## 11. 当前限制

- 意图分类和查询改写质量取决于模型；
- 多论文归纳仍然只靠一次检索 + 多子查询，没有真正逐篇阅读；
- 没有并发控制，同一 `run_id` 不要同时开两个进程 `resume`；
- 还没有用真实大模型系统评测新流程和普通 `rag ask` 的差异。

下一步更值得做的是：用复杂测试集比较普通 RAG、旧版 Agent 和新版 Agent 的
答案质量、检索覆盖和耗时，而不是继续加工具数量。
