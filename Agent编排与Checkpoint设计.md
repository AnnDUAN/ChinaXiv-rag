# Agent 编排与 Checkpoint 设计

## 1. 这一版要解决什么

原来的 `rag ask` 是固定流程：

```text
问题 → 检索一次 → 拼提示词 → 回答
```

它适合“一个问题、一次就能查到”的情况，但不适合：

- 第一次没查到关键证据；
- 需要换一个角度再查一次；
- 跨论文对比，需要多轮收集片段。

这一版只加一件事：**让流程能判断“证据够不够”，不够时最多补查几轮。**

没有做：

- 自由调用网页、代码、数据库等工具；
- 多智能体互相讨论；
- 自动拆解很复杂的长任务；
- 后台任务队列和分布式锁。

## 2. 整体流程

```text
开始
  ↓
retrieve：检索
  ↓
assess：判断证据够不够
  ├─ 够 → answer
  ├─ 不够，但给出了新的检索词，且轮数没用完 → retrieve
  └─ 不够，但没有新角度 / 没有新片段 / 轮数用完 → answer
  ↓
answer：基于已有片段生成一次带引用的回答
  ↓
done
```

### 停止条件

满足任意一条就停止补查：

1. 模型判断证据已经够用；
2. 模型没有给出下一轮检索词；
3. 下一轮检索词和之前重复；
4. 补查没有带回新的切片；
5. 达到最大检索轮数；
6. 达到最大步骤数。

达到最大步骤数时任务标记为失败，可以用 `resume` 提高步数后继续。

## 3. 为什么先拆开“检索”和“生成”

原来的 `answer_question()` 内部同时做检索和回答。
Agent 需要先多轮检索，最后只生成一次，所以拆成了：

```text
prepare_answer / search_chunks   只负责找片段
answer_from_search               只负责基于片段生成答案
answer_question                  普通 RAG 的固定组合
```

这样 Agent 不会每补查一次就重复生成一次答案，也方便单独测试。

## 4. 状态里存什么

`AgentState` 是完整的运行进度：

- 原始问题；
- 当前阶段：`retrieve` / `assess` / `answer` / `done`；
- 已执行步数、最大步数；
- 已检索轮数、最大检索轮数；
- 每一轮用过的检索词；
- 去重后的证据片段；
- 每一轮检索摘要；
- 每一次“证据够不够”的判断；
- 最终答案或错误信息；
- 检索参数：`top_k`、`mode`、`rerank`。

状态整体存进 PostgreSQL 的 JSONB 字段，不依赖进程内存。

## 5. Checkpoint 表

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
- `node`：`retrieve` / `assess` / `answer`；
- `status`：`running` / `success` / `failed` / `interrupted`；
- `input_payload`：这一步的输入摘要；
- `output_payload`：这一步的输出摘要；
- `error`；
- `started_at` / `ended_at`。

完整正文放在 `agent_runs.state` 里，步骤表只存摘要，避免 trace 太长。

## 6. 断点续跑怎么保证

每个节点成功后：

1. 先更新步骤状态；
2. 再保存完整状态。

如果进程在两步之间中断：

- 最后完成的节点已经存好；
- 下一次 `resume` 从当前 `phase` 继续；
- 没完成的节点会重新执行一次。

这是“至少执行一次”，不是严格“只执行一次”。
对于本地检索和问答，重跑一次最多多花一次模型调用，风险可以接受。

`resume` 时会先把遗留的 `running` 步骤标成 `interrupted`，再用同一个步号重新执行。

## 7. 命令行

```powershell
# 新建一次运行
.\rag agent ask "问题" --max-steps 8 --max-search-rounds 3

# 列出最近的运行
.\rag agent list

# 查看每一步
.\rag agent trace 1

# 中断后继续；可以顺便提高最大步数
.\rag agent resume 1 --max-steps 10
```

普通问题仍然使用：

```powershell
.\rag ask "问题"
```

Agent 默认不会接管所有问题，避免简单问题也变慢、变贵。

## 8. 测试

离线测试不连数据库、不连真实模型：

- 第一轮证据够：只查一次就回答；
- 第一轮不够：自动补查第二轮；
- 模型重复给出同一个检索词：不进入死循环；
- 没有检索结果：直接返回“信息不足”；
- 步数不够时失败，`resume` 后能完成；
- trace 能看到 `retrieve → assess → answer`。

另外做了一次本地假模型冒烟测试，完整走通：

```text
真实检索 → 本地假 OpenAI 接口 → 结构化答案 → PostgreSQL checkpoint
```

## 9. 当前限制

- 证据判断和下一轮检索词都由大模型给出，判断质量取决于模型；
- 没有做真正的问题分解，只是“补查”；
- 没有自动把长期记忆带进 Agent；
- 没有并发控制，同一 `run_id` 不要同时开两个进程 `resume`；
- 还没有用真实大模型系统评测复杂问题。

下一步更值得做的是：用一批跨论文、多跳问题比较普通 RAG 和 Agent 的效果，而不是继续加工具数量。
