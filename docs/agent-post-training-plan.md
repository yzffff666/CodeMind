# Pico Agent 后训练方向改进方案

## 1. 背景与目标

Pico 当前是一个面向本地代码仓库的轻量级 coding agent。它已经具备 agent 后训练项目非常重要的基础能力：

- 能在真实工作区中执行多步任务。
- 能调用受约束的工具，例如读文件、改文件、执行命令。
- 能为每次运行保存 `trace.jsonl`、`report.json`、`task_state.json`。
- 已经有 benchmark 任务、verifier、step budget 和 metrics 聚合逻辑。

这些能力本身不是后训练，但它们非常适合扩展成一个 agent post-training data pipeline。也就是说，项目可以从“一个能跑的 agent”升级为“一个能采集、筛选、构造、评估 agent 后训练数据的系统”。

目标不是夸大成“从零训练大模型”，而是更真实、更适合求职表达地定位为：

> 基于本地 coding agent 构建 agent 后训练数据闭环，将 tool-use trajectory 和 verifier 结果转换为 SFT 数据、DPO preference pairs，并用 reward signal 和 benchmark 评估驱动 agent 行为改进。

这个定位更贴近国内大模型后训练、Agent 算法、LLM 数据策略、AI 应用算法岗位的实际要求。

## 2. 与国内岗位 JD 的匹配度

国内相关岗位通常会强调以下能力：

- SFT、LoRA、DPO、RLHF、PPO、GRPO 等后训练方法。
- 指令数据、偏好数据、领域数据的构建、清洗和质量控制。
- Reward model、verifier、自动评测体系和效果迭代。
- Agent、RAG、tool-use、function calling、多步推理等应用场景。
- Qwen、LLaMA、InternLM、ChatGLM 等开源模型实践。
- PyTorch、Transformers、PEFT、TRL、DeepSpeed 等训练框架经验。
- 业务落地、数据飞轮、评估闭环和工程化能力。

Pico 当前和这些要求的匹配关系如下：

| JD 要求 | Pico 当前基础 | 需要补齐 |
| --- | --- | --- |
| Agent / tool-use | 已有本地 coding agent、工具调用、运行时约束 | 增加 trajectory 数据导出 |
| 数据构建 | 已有 trace/report/task_state | 增加 SFT/DPO 数据格式转换 |
| 自动评测 | 已有 benchmark、verifier、metrics | 增加 post-training report |
| Reward / preference | 有成功率、step、错误事件等信号 | 增加 reward scoring 和 pair 规则 |
| SFT / DPO 实践 | 暂无训练脚本 | 增加可选 Qwen LoRA SFT/DPO demo |
| 工程落地 | CLI、runtime、artifact 保存较完整 | 增加独立 post_training 模块和文档 |

因此，Pico 更适合作为“Agent 后训练数据与评测闭环”项目，而不是“大规模训练 infra”项目。如果目标岗位偏千卡训练、Megatron、NCCL、CUDA、FSDP，则还需要另起一条训练系统路线；如果目标岗位偏 Agent 后训练、SFT/DPO 数据、应用算法、评测体系，Pico 的改造方向是匹配的。

## 3. 总体改进思路

建议采用“旁路增强”的方式，不修改现有核心 agent 逻辑。

现有 `pico/` 包继续负责：

- agent runtime
- tool execution
- memory/session
- benchmark evaluation
- trace/report/task_state 写入

新增 `post_training/` 模块负责：

- 读取已有运行产物。
- 构造训练样本。
- 计算 reward signal。
- 生成 SFT JSONL。
- 生成 DPO preference pair JSONL。
- 输出数据质量和评估报告。

这样做有几个好处：

- 不破坏原项目稳定性。
- 项目边界更清楚，方便面试讲解。
- 可以先用已有 `.pico/runs` 数据做离线处理。
- 后续如果要接入真实模型训练，也能自然扩展。

## 4. 推荐新增目录结构

```text
pico/
  post_training/
    __init__.py
    schemas.py
    trace_loader.py
    reward.py
    sft_builder.py
    dpo_builder.py
    summary.py

  scripts/
    export_sft_dataset.py
    build_dpo_pairs.py
    summarize_post_training.py

  docs/
    agent-post-training-plan.md
    post-training-data-schema.md
    post-training-experiment-report.md

  artifacts/
    datasets/
      sft.sample.jsonl
      dpo.sample.jsonl

  experiments/
    qwen_lora_sft/
      README.md
      train_sft.py
      train_dpo.py
      requirements.txt
```

其中 `post_training/` 是核心扩展层，`experiments/` 是可选训练实验层。第一阶段可以先做 `post_training/`，不急着真的跑 GPU 训练。

## 5. 核心模块设计

### 5.1 Trace Loader

输入：

```text
.pico/runs/<run_id>/trace.jsonl
.pico/runs/<run_id>/report.json
.pico/runs/<run_id>/task_state.json
```

职责：

- 解析一次 agent 运行的完整事件序列。
- 提取 user request、assistant output、tool call、tool result、final answer。
- 合并 report 中的状态、stop reason、tool steps、attempts、prompt metadata。
- 过滤损坏、缺字段、格式不完整的 run。

输出统一结构：

```json
{
  "run_id": "...",
  "task_id": "...",
  "user_request": "...",
  "events": [],
  "final_answer": "...",
  "status": "completed",
  "stop_reason": "final_answer_returned",
  "tool_steps": 3,
  "attempts": 1,
  "verifier_passed": true
}
```

### 5.2 Reward Signal

设计一个可解释的规则 reward，而不是直接训练 reward model。

建议初版：

```text
base_success_reward:
  +1.0 if verifier passed or status completed

efficiency_reward:
  +0.2 if within step budget
  -0.05 for each extra tool step

tool_quality_penalty:
  -0.3 for invalid tool call
  -0.2 for repeated read rejection
  -0.5 for path escape or security violation

completion_penalty:
  -0.5 if step limit reached
  -0.5 if retry limit reached
```

输出：

```json
{
  "score": 0.8,
  "signals": {
    "success": true,
    "within_budget": true,
    "invalid_tool_calls": 0,
    "security_violations": 0,
    "tool_steps": 3
  }
}
```

这个模块的求职价值很高，因为它体现了对 agent 后训练中“什么是好行为”的建模能力。

### 5.3 SFT Dataset Builder

目标：把高质量成功轨迹转换成 supervised fine-tuning 数据。

筛选规则：

- run 状态为 completed。
- verifier 通过，或者 report 中没有明显失败信号。
- 没有安全违规。
- tool call 格式有效。
- final answer 存在。

输出格式：

```json
{
  "messages": [
    {"role": "user", "content": "Fix the failing test."},
    {"role": "assistant", "content": "<tool name=\"read_file\" ...></tool>"},
    {"role": "tool", "content": "..."},
    {"role": "assistant", "content": "<final>Done.</final>"}
  ],
  "metadata": {
    "run_id": "...",
    "task_id": "...",
    "reward": 1.2,
    "tool_steps": 3,
    "source": "pico_trace"
  }
}
```

求职表达：

> 将 agent tool-use trajectory 自动转换为 SFT messages 数据，用于训练模型学习任务分解、工具调用、观察反馈和最终回答格式。

### 5.4 DPO Pair Builder

目标：构造 chosen/rejected 偏好对。

pair 来源可以分三类：

1. 同一 task 的成功 run vs 失败 run。
2. 同一 prompt 下 reward 高的 run vs reward 低的 run。
3. 人工或规则生成的 rejected，例如非法工具调用、路径逃逸、超步数、无 final answer。

输出格式：

```json
{
  "prompt": "Fix the README placeholder.",
  "chosen": "<tool name=\"patch_file\" ...></tool>\n<final>Done.</final>",
  "rejected": "<tool>{\"name\":\"patch_file\",\"args\":{...}}</tool>\n<final>Failed.</final>",
  "metadata": {
    "chosen_run_id": "...",
    "rejected_run_id": "...",
    "reason": "verifier_pass_vs_fail",
    "chosen_reward": 1.2,
    "rejected_reward": -0.4
  }
}
```

求职表达：

> 基于 verifier、工具调用质量和安全约束自动构建 DPO preference pairs，用于优化 agent 的工具使用可靠性和任务完成率。

### 5.5 Summary Report

生成一个训练数据质量报告：

- SFT 样本数量。
- DPO pair 数量。
- 平均 reward。
- 成功率和失败率。
- 平均 tool steps。
- error/rejection reason 分布。
- task category 分布。
- 安全违规数量。
- 数据过滤率。

示例输出：

```text
Post-Training Dataset Summary

Runs scanned: 120
Valid SFT samples: 74
DPO pairs: 38
Average reward: 0.82
Verifier pass rate: 71.6%
Average tool steps: 3.4
Top rejection reasons:
- verifier_failed: 21
- step_limit_reached: 8
- invalid_tool_call: 6
- security_violation: 3
```

这个报告可以放进 `docs/post-training-experiment-report.md`，作为简历项目展示材料。

## 6. 可选训练实验

为了更贴国内 JD，可以增加一个独立实验目录，不要求默认运行。

建议选择：

- Qwen2.5 / Qwen3 小模型。
- LoRA / QLoRA。
- HuggingFace Transformers。
- PEFT。
- TRL 的 SFTTrainer / DPOTrainer。

实验目录：

```text
experiments/qwen_lora_sft/
  train_sft.py
  train_dpo.py
  README.md
  requirements.txt
```

重点不是追求效果，而是证明自己理解完整链路：

```text
agent trace -> SFT dataset -> LoRA SFT -> DPO pairs -> DPO training -> benchmark evaluation
```

如果没有 GPU，也可以保留脚本、样例数据和实验说明。面试中可以诚实说明：

> 当前项目重点是 agent 后训练数据管线和评估闭环，训练脚本提供 Qwen LoRA SFT/DPO 的可复现实验入口；实际大规模训练依赖 GPU 资源。

## 7. 分阶段落地计划

### 阶段一：数据管线 MVP

目标：不训练模型，只完成数据构建闭环。

交付物：

- `post_training/trace_loader.py`
- `post_training/reward.py`
- `post_training/sft_builder.py`
- `post_training/dpo_builder.py`
- `scripts/export_sft_dataset.py`
- `scripts/build_dpo_pairs.py`
- `scripts/summarize_post_training.py`
- `artifacts/datasets/*.sample.jsonl`

验收标准：

- 能从 `.pico/runs` 读取已有 run。
- 能导出 SFT JSONL。
- 能导出 DPO JSONL。
- 能生成 summary report。

### 阶段二：评估与文档

目标：把项目包装成可展示的后训练项目。

交付物：

- `docs/post-training-data-schema.md`
- `docs/post-training-experiment-report.md`
- README 增加 post-training 亮点章节。
- 一个小规模 benchmark 数据集导出示例。

验收标准：

- 面试官能从文档中看懂问题定义、数据格式、reward 设计、评估指标。
- 简历中的项目描述能和仓库内容一一对应。

### 阶段三：Qwen LoRA SFT/DPO Demo

目标：补齐“真的做过 SFT/DPO 实践”的证据。

交付物：

- `experiments/qwen_lora_sft/train_sft.py`
- `experiments/qwen_lora_sft/train_dpo.py`
- `experiments/qwen_lora_sft/README.md`
- 小样例训练配置。

验收标准：

- 脚本能读取阶段一导出的数据。
- 支持小模型 LoRA SFT。
- 支持 DPOTrainer 读取 preference pairs。
- 文档说明硬件需求、运行方式和预期结果。

### 阶段四：数据飞轮

目标：进一步贴近企业岗位里的“持续迭代”和“数据飞轮”。

流程：

```text
run benchmark
  -> collect traces
  -> score runs
  -> export SFT/DPO data
  -> train or simulate policy improvement
  -> re-run benchmark
  -> compare metrics
```

交付物：

- 自动化脚本。
- before/after report。
- badcase 分析。
- rejection reason 改进记录。

## 8. 简历与面试表达

推荐简历描述：

> 基于本地 Coding Agent 构建 Agent 后训练数据闭环：采集 tool-use trajectory、runtime trace 和 verifier outcome，设计 reward signal，将成功/失败轨迹自动转换为 SFT 数据与 DPO preference pairs，并输出数据质量报告和 benchmark 评估结果；支持后续接入 Qwen LoRA SFT/DPO 实验。

可以展开讲的技术点：

- 为什么 agent 后训练不能只用普通问答数据。
- 如何从工具调用轨迹构造 messages。
- chosen/rejected 的规则如何设计。
- verifier 和 reward 的关系。
- 如何过滤脏数据、失败数据和安全违规数据。
- 如何用 benchmark 指标验证数据管线是否有效。
- 如何扩展到 Qwen + LoRA + TRL。

需要避免的表述：

- 不要说“训练了一个大模型”，除非确实完成了训练实验。
- 不要说“实现了完整 RLHF”，除非有 reward model 和 RL 训练。
- 不要把规则 reward 包装成 reward model。

更准确的说法：

- “构建 agent 后训练数据管线。”
- “基于 verifier 自动构造偏好数据。”
- “支持 SFT/DPO 数据导出。”
- “设计了 agent tool-use 行为的 reward signal。”
- “建立了 benchmark 驱动的数据评估闭环。”

## 9. 20-30 天最小学习与项目推进计划

这一阶段的目标不是系统学习所有后训练知识，而是围绕 Pico 项目快速补齐能支撑实习面试的最小知识闭环：

```text
SFT / DPO 基础
  -> Agent trajectory 数据
  -> verifier-based reward
  -> SFT/DPO 数据导出
  -> Qwen LoRA / TRL 基础认知
  -> 项目文档与面试表达
```

不建议在这个阶段分散到 DeepSpeed、Megatron、PPO 细节、CUDA、复杂 RAG、多模态等方向。它们有价值，但不是当前最短路径。

### 9.1 最小知识范围

| 知识模块 | 要学到什么程度 | 对应项目改进 | 面试中能回答什么 |
| --- | --- | --- | --- |
| SFT | 理解 supervised fine-tuning、chat messages、instruction data、agent SFT 和普通问答 SFT 的区别 | `sft_builder.py`，将 trace 转为 SFT JSONL | “Agent trajectory 如何转成 SFT 数据？” |
| DPO | 理解 chosen/rejected、preference pair、DPO 不显式训练 reward model 的特点 | `dpo_builder.py`，构造成功/失败轨迹偏好对 | “你的 DPO pair 怎么构造？” |
| Agent trajectory | 理解 user request、tool call、observation、next action、final answer 的链路 | `trace_loader.py`，解析 `trace.jsonl`、`report.json` | “为什么 Agent 后训练需要轨迹数据？” |
| Verifier / reward | 理解 outcome reward、process signal、规则 reward、数据筛选 | `reward.py`，根据成功率、工具错误、安全事件打分 | “你如何定义好的 agent 行为？” |
| Evaluation | 理解 pass rate、tool success rate、failure reason、badcase analysis | `summary.py`，输出数据质量和评估报告 | “怎么验证改造有效？” |
| Qwen / LoRA / TRL | 理解 Qwen 开源模型、LoRA 参数高效微调、SFTTrainer、DPOTrainer 的用途 | `experiments/qwen_lora_sft/` 可选 demo | “如果接入真实训练，你会怎么做？” |

### 9.2 20 天核心计划

#### 第 1-3 天：SFT、DPO 和 Agent trajectory 基础

学习内容：

- SFT 的输入输出格式。
- Chat messages 格式。
- DPO 的 chosen/rejected 偏好对。
- SFT 和 DPO 的区别。
- Agent trajectory 的基本结构。
- ReAct / Tool Use 的基本思路。

建议只掌握到能解释以下问题：

- SFT 在训练模型模仿什么？
- DPO 为什么需要一好一坏两条回答？
- Agent SFT 为什么不能只用普通问答数据？
- Tool call 和 observation 在训练数据里如何表示？

项目交付：

- 新增 `docs/interview-notes.md`，整理 SFT、DPO、trajectory 的一页笔记。
- 在 `docs/post-training-data-schema.md` 中定义初版 SFT/DPO JSONL 格式。

面试表达：

> Pico 的运行 trace 记录了 agent 的多步工具调用过程，因此可以把成功轨迹转为 SFT 样本，把成功/失败轨迹转为 DPO preference pairs。

#### 第 4-8 天：实现 trace 到 SFT/DPO 的数据管线

学习内容：

- JSONL 数据格式。
- 如何解析事件序列。
- 如何从日志中提取 user、assistant、tool、final answer。
- 数据过滤的基本规则。

项目交付：

```text
post_training/
  __init__.py
  schemas.py
  trace_loader.py
  sft_builder.py
  dpo_builder.py

scripts/
  export_sft_dataset.py
  build_dpo_pairs.py

artifacts/datasets/
  sft.sample.jsonl
  dpo.sample.jsonl
```

最低验收标准：

- 能扫描 `.pico/runs`。
- 能读取 `trace.jsonl`、`report.json`、`task_state.json`。
- 能导出至少一份 SFT sample。
- 能导出至少一份 DPO pair sample。

面试表达：

> 我没有手写训练样本，而是从 agent 真实运行轨迹中自动构建训练数据，保留了工具调用、工具返回、最终回答和运行元数据。

#### 第 9-12 天：补 reward signal 和数据质量报告

学习内容：

- Outcome reward 和 process signal 的区别。
- Verifier-based reward 的适用场景。
- 常见 agent 失败原因：超步数、非法工具调用、路径逃逸、final answer 缺失、verifier fail。
- 数据质量统计。

项目交付：

```text
post_training/
  reward.py
  summary.py

scripts/
  summarize_post_training.py

docs/
  post-training-experiment-report.md
```

初版 reward 可以采用规则：

```text
+1.0 verifier pass or completed
+0.2 within step budget
-0.05 per extra tool step
-0.3 invalid tool call
-0.5 security/path escape violation
-0.5 step limit or retry limit reached
```

报告至少包含：

- 扫描 run 数量。
- SFT 样本数量。
- DPO pair 数量。
- 平均 reward。
- 平均 tool steps。
- failure reason 分布。
- verifier pass rate。

面试表达：

> 我先用 verifier-based reward 做离线数据筛选和偏好对构造，没有把它包装成 reward model。这样更适合小规模项目，也更容易解释每条数据为什么被选为 chosen 或 rejected。

#### 第 13-16 天：Qwen LoRA / TRL 最小认知

学习内容：

- Qwen2.5 / Qwen3 小模型的基本使用。
- LoRA 是什么，为什么适合资源有限的微调。
- PEFT 的作用。
- TRL 中 SFTTrainer 和 DPOTrainer 的用途。
- SFT 和 DPO 脚本读取什么数据。

项目交付：

```text
experiments/qwen_lora_sft/
  README.md
  train_sft.py
  train_dpo.py
  requirements.txt
```

如果本地没有 GPU，不强求真实训练结果。脚本可以作为可选实验入口，但需要写清楚：

- 输入数据路径。
- 模型名称。
- LoRA 配置。
- 运行命令。
- 硬件要求。

面试表达：

> 项目核心是数据管线和评估闭环；训练层面预留了 Qwen + LoRA + TRL 的实验入口，可以读取导出的 SFT/DPO 数据进行小规模验证。

#### 第 17-20 天：项目包装和面试准备

学习内容：

- 如何讲清项目背景、方案和边界。
- 如何承认局限但说明下一步优化。
- 国内 JD 中常见关键词：SFT、DPO、Agentic RL、Function Calling、Trajectory、Verifier、Evaluation。

项目交付：

- 修复 README 乱码。
- README 增加 Agent Post-Training Pipeline 亮点。
- 清理 `__MACOSX` 和不必要的大文件，或在文档中说明其来源。
- 完成 `docs/post-training-experiment-report.md`。
- 准备一段简历项目描述。

必须准备好的 8 个面试问题：

1. 你的项目解决什么问题？
2. 为什么它和后训练有关？
3. Agent trajectory 怎么转 SFT？
4. DPO pair 怎么构造？
5. chosen/rejected 的标准是什么？
6. reward signal 怎么设计？
7. 怎么评估 agent 变好了？
8. 这个方案有什么局限和下一步？

### 9.3 30 天增强计划

如果有 30 天，多出来的 10 天不要继续扩大学习面，而是提高项目完成度。

建议补充：

- 增加更多 benchmark task，制造更多成功/失败轨迹。
- 生成一份更完整的 `post-training-experiment-report.md`。
- 尝试用 Qwen 小模型跑通一次 LoRA SFT。
- 如果资源允许，再跑一个极小规模 DPO demo。
- 补一个流程图，展示 `trace -> reward -> SFT/DPO -> eval`。
- 将 README 首页改成求职友好的项目介绍。

增强后的简历表达：

> 构建面向 Coding Agent 的后训练数据闭环，支持从 tool-use trace 自动生成 SFT 数据与 DPO preference pairs，设计 verifier-based reward，并基于 Qwen LoRA/TRL 完成小规模 SFT/DPO 实验和 benchmark 评估。

### 9.4 学习资源关键词

优先搜索这些关键词：

```text
SFT instruction tuning chat messages
DPO chosen rejected preference data
Agent trajectory SFT
ReAct tool use function calling
verifier based reward
Qwen LoRA fine-tuning
PEFT LoRA tutorial
TRL SFTTrainer DPOTrainer
LLaMA Factory SFT DPO
OpenRLHF DPO
```

中文平台可以重点看：

- B 站：Qwen LoRA 微调、LLaMA Factory、TRL DPO。
- 知乎/博客：DPO 原理、RLHF 流程、PEFT LoRA。
- GitHub：Qwen examples、PEFT、TRL、LLaMA-Factory、OpenRLHF。

学习时要始终围绕 Pico 的主线：

```text
Pico agent 运行轨迹
  -> trace 解析
  -> SFT 数据构建
  -> DPO preference pair
  -> verifier reward
  -> summary report
  -> optional Qwen LoRA SFT/DPO
```

只要能把这条线讲清楚，20-30 天内就足够支撑大多数 Agent 后训练/LLM 数据方向实习面试。

## 10. 数据规模、训练收益与成本权衡

Agent 后训练数据的成本明显高于普通问答数据。普通 SFT 样本通常只需要 `prompt -> response`，而 coding agent trajectory 需要真实或模拟环境、工具调用、状态变化、verifier 检查、失败重试和质量筛选。因此，数据规模目标需要结合训练收益和个人项目成本来设定。

### 10.1 不同数据量能证明什么

| 数据规模 | 适合证明 | 不适合声称 |
| --- | --- | --- |
| 50-100 条 trajectory | 跑通 pipeline，验证 trace loader、reward、SFT/DPO schema | 证明训练后模型能力稳定提升 |
| 100-200 条 trajectory | 小规模数据闭环展示，能做初步 SFT/DPO 样例和数据质量报告 | 训练出泛化能力强的 coding agent |
| 300-500 条 trajectory | 比较适合实习项目：可做小规模 LoRA SFT/DPO 实验，观察 tool-use 稳定性变化 | 工业级后训练或复杂代码能力提升 |
| 500-1000 条 trajectory | 更完整的展示版：能覆盖更多任务类型、错误类型和 DPO pairs | 大规模通用 agent 能力提升 |
| 数千条以上 | 更接近真实训练数据积累，但成本和清洗压力明显增加 | 个人短周期项目中不一定划算 |

因此，本项目推荐的目标不是一开始追求上万条数据，而是先做高质量、可验证、可复现的小规模数据闭环。

### 10.2 推荐数据规模

对 20-30 天实习项目冲刺，建议分三档：

```text
最低可交付：100-200 条 trajectory
推荐目标：300-500 条 trajectory
冲刺目标：500-1000 条 trajectory
```

其中最推荐的是：

```text
50-80 个 benchmark task
每个 task 运行 5 次
总计 250-400 条 trajectory
```

如果时间和模型调用成本允许，可以扩展到：

```text
100 个 benchmark task
每个 task 运行 5 次
总计 500 条 trajectory
```

这个量级比较适合个人项目：成本仍然可控，同时足够生成 SFT candidates、DPO pairs、badcase 分析和 summary report。

### 10.3 小数据能带来什么提升

100-500 条 trajectory 不足以训练出通用能力很强的 coding agent，但可能带来窄域、可观测的优化：

- 工具调用格式更稳定。
- `<tool>` / `<final>` 等协议输出更规范。
- `invalid_arguments` 等工具参数错误减少。
- `path_escape` 等安全违规减少。
- 简单 text-edit / documentation benchmark 的 pass rate 改善。
- 平均 tool steps 下降。
- final answer rate 提升。

这些更像是 **agent tool-use protocol alignment**，而不是通用代码能力提升。

因此，实验目标应该写成：

> 验证小规模 agent trajectory 后训练数据对工具调用稳定性、安全偏好和简单 benchmark 任务完成率的影响。

不应该写成：

> 训练出一个强大的通用 coding agent。

### 10.4 数据配比建议

不要只收集成功轨迹。Agent 后训练需要成功样本，也需要失败样本和边界样本。

推荐配比：

```text
高质量成功轨迹：40%
错误恢复轨迹：25%
失败轨迹：25%
安全/边界问题轨迹：10%
```

以 500 条 trajectory 为例：

```text
高质量成功轨迹：约 200 条
错误恢复轨迹：约 125 条
失败轨迹：约 125 条
安全/边界问题轨迹：约 50 条
```

这样可以同时支持：

- SFT：使用高质量成功轨迹。
- Recovery SFT：使用可恢复错误轨迹。
- DPO：使用高分轨迹 vs 低分轨迹。
- Safety preference：使用安全轨迹 vs 越界轨迹。
- Badcase analysis：分析失败和边界问题。

### 10.5 成本控制策略

为了降低 agent trajectory 数据成本，可以采用分层采集策略：

```text
synthetic benchmark traces
  -> pipeline 开发、schema 测试、reward 规则验证

real model benchmark traces
  -> 固定任务、多次采样、自动 verifier、真实模型行为分析

real project task traces
  -> 少量高价值真实 coding task，用于项目展示和人工复盘
```

具体方法：

- 用 synthetic traces 先测试 parser、reward、SFT/DPO builder。
- 对同一个 benchmark task 多次采样，获得 same-prompt trajectories。
- 用 verifier 自动判断 pass/fail，减少人工标注。
- 失败轨迹不丢弃，转为 DPO rejected 或 badcase。
- 用 reward score 自动排序，降低人工筛选成本。
- 只对少量高价值样本做人工 review。
- 优先收集带测试、assert、lint 的任务，因为它们容易自动验收。

### 10.6 面试时的准确表达

如果被问“几百条数据够不够训练出效果”，可以这样回答：

> 对通用 coding 能力提升肯定不够，但对窄域 agent tool-use 协议对齐是可能有效的。我的目标不是用几百条数据训练出工业级 coding agent，而是验证一套可复现的数据闭环：采集带 verifier 的 trajectories，基于 reward 做质量评估，导出 SFT 和 DPO 数据，并观察 tool parse error、invalid_arguments、path_escape、final answer rate 和 benchmark pass rate 等指标是否改善。

这个表达更符合项目真实边界，也能体现对数据成本和后训练收益的理解。

## 11. 风险与注意事项

### 数据规模风险

如果只有少量 benchmark run，数据规模不足以证明训练效果。

解决方案：

- 增加更多 coding task。
- 对同一任务生成多个失败/成功轨迹。
- 引入真实小仓库任务。
- 保留 sample dataset，同时说明这是 pipeline demo。

### 训练资源风险

本地可能没有 GPU，无法跑完整 SFT/DPO。

解决方案：

- 第一阶段重点放在数据管线和评估。
- 训练脚本作为可选实验。
- 使用小模型和 LoRA。
- 文档中明确硬件依赖。

### 项目边界风险

如果把训练逻辑塞进 runtime，项目会变乱。

解决方案：

- 保持 `pico/` 核心 agent 不动。
- 用 `post_training/` 做离线处理。
- 用 `experiments/` 做训练 demo。

### 展示风险

当前仓库中有一些大 PDF、`__MACOSX` 和乱码文件名，可能影响面试官第一印象。

解决方案：

- 将大文件移出主仓库或改用 Git LFS。
- 清理 `__MACOSX`。
- 修复 README 编码问题。
- 保留清晰的英文/中文技术文档。

## 12. 结论

Pico 很适合向 Agent 后训练方向扩展。它已经有 agent runtime、tool-use trace、benchmark、verifier 和 metrics，这些正是构建后训练数据闭环的关键基础。

最合适的改进路线不是重写现有项目，而是在旁路新增 `post_training/` 模块：

```text
existing agent runtime
  -> collect traces
  -> score with verifier/reward
  -> export SFT dataset
  -> build DPO preference pairs
  -> summarize and evaluate
  -> optional Qwen LoRA SFT/DPO experiment
```

完成这些后，项目会从“一个本地 coding agent”升级为“一个面向 agent 后训练的数据构建与评估系统”。这和国内大模型后训练、Agent 算法、LLM 数据工程、应用算法岗位的 JD 是匹配的，也更容易在简历和面试中讲出深度。
