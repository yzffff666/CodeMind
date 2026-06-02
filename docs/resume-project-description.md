# 简历项目描述：CodeMind 本地代码智能体 Harness

## 推荐简历版本

**CodeMind 本地代码智能体 Harness**  
2026年03月 - 2026年05月  
**技术栈：** Python、Agent Harness、Tool Calling、Context Management、Checkpoint / Resume、Run Trace、Benchmark Eval、SFT / DPO Data Pipeline、Reward Modeling

- **运行链路：** 设计本地 Coding Agent 的核心执行流程，打通 CLI 启动、模型适配、工具调用、会话状态和运行工件落盘，使代码理解、缺陷定位、轻量修复和文本编辑任务可以被持续执行、恢复、评测和复盘；支持 2 类模型后端、7 类工具和 trace / report / task_state 等运行工件。
- **上下文压缩：** 针对多轮代码任务中 prompt 不断膨胀的问题，将上下文拆分为任务前缀、工作记忆、相关记忆、历史记录和当前请求等区块，并按优先级裁剪；在 12 组长上下文配置中，将平均 prompt 长度从 7082 降至 5664，平均压缩率 16.19%。
- **分层记忆：** 优化 Agent 的多轮任务记忆，将任务摘要、文件摘要、过程笔记和相关记忆召回分层管理，让 follow-up 阶段优先复用已确认事实，减少重复读取和重复确认；在 12 个记忆依赖任务中，将重复读文件次数从 60 次降至 0 次。
- **任务恢复：** 实现 checkpoint / resume 与 workspace 漂移校验，记录任务状态、工具调用结果、上下文摘要和工作区指纹，在会话中断、上下文超预算或文件变更后判断旧状态是否仍可复用；覆盖 10 个恢复场景，workspace 漂移识别率达到 100%。
- **工具安全治理：** 将文件读取、代码搜索、Shell 执行、文件写入和 Patch 修改等工具收口到统一执行网关，加入参数校验、路径逃逸防护、高风险操作审批、重复调用拦截和敏感信息脱敏，降低模型误调用工具带来的副作用风险。
- **评测闭环：** 建立 run trace 与 benchmark 评测体系，聚合每次任务的工具调用、prompt 长度、失败类型、恢复结果和任务完成情况，对比上下文压缩、记忆策略和恢复机制的实际收益，避免仅凭主观体验判断 Agent 效果。
- **后训练数据闭环：** 基于真实 Agent 运行轨迹构建 SFT / DPO 数据处理链路，从 `.core agent package/runs/*` 中解析 tool-use trajectory、verifier 结果、stop reason、tool error 和安全事件，设计 rule-based reward 自动筛选 SFT candidates，并通过 same-prompt repeated sampling 构造 DPO chosen / rejected pairs。
- **数据增强与偏好构造：** 针对 held-out eval 中暴露的 `completion_without_final`、`verifier_failed` 和精确指令执行失败等 badcase，设计相似但不泄漏评测集的 targeted benchmark；通过 repeated sampling 将“完成 verifier 条件并及时 final”的轨迹作为 chosen，将“完成修改但未及时 final”的轨迹作为 rejected，形成面向 Agent 行为偏好的训练数据。

## 更适合一页简历的压缩版本

**CodeMind 本地代码智能体 Harness**  
2026年03月 - 2026年05月  
**技术栈：** Python、Agent Harness、Tool Calling、Context Management、Checkpoint / Resume、Benchmark Eval、SFT / DPO Data Pipeline

- 设计本地 Coding Agent 执行链路，打通 CLI 启动、模型适配、工具调用、会话状态和运行工件落盘，支持代码理解、缺陷定位、轻量修复、文本编辑等任务的持续执行、恢复和复盘。
- 构建上下文压缩与分层记忆机制，将 prompt 拆分为任务前缀、工作记忆、相关记忆、历史记录和当前请求等区块，在 12 组长上下文配置中将平均 prompt 长度从 7082 降至 5664，平均压缩率 16.19%；在 12 个记忆依赖任务中将重复读文件次数从 60 次降至 0 次。
- 实现 checkpoint / resume 与 workspace 漂移校验，记录任务状态、工具调用结果、上下文摘要和工作区指纹，覆盖 10 个恢复场景，workspace 漂移识别率达到 100%。
- 建立工具安全治理与 benchmark eval 体系，对文件读取、代码搜索、Shell 执行、Patch 修改等工具进行统一网关管理，聚合 run trace、失败类型、恢复结果和 verifier 结果，用于分析 Agent 行为稳定性。
- 扩展 Agent 后训练数据闭环，从真实 tool-use trajectory 中构造 rule-based reward、SFT candidates 与 same-prompt DPO preference pairs，并通过 held-out eval、badcase taxonomy 和 targeted data augmentation 改进 `completion_without_final`、精确指令执行等失败模式。

## 偏后训练岗位的版本

**CodeMind：面向代码 Agent 的后训练数据与评估闭环**  
2026年03月 - 2026年05月  
**技术栈：** Python、Agent Harness、Tool Calling、Run Trace、Benchmark Eval、SFT、DPO、Reward Modeling、Data Curation

- 构建代码 Agent 后训练数据闭环，基于真实运行 trace 解析多步 tool-use trajectory，保留 prompt、工具调用、文件修改、verifier、stop reason、task_state 和 report 等信息，用于 SFT / DPO 数据构造与行为复盘。
- 设计 rule-based reward，将 verifier pass、final answer、tool failure、安全事件、路径逃逸、工具步数和 benchmark pass 等信号纳入打分，自动筛选高质量 SFT 样本，并对失败轨迹进行 badcase taxonomy 归因。
- 基于 same-prompt repeated sampling 构造 DPO preference pairs，将成功完成任务且及时 final 的轨迹作为 chosen，将 verifier 通过但未及时结束、工具调用失败或违反协议的轨迹作为 rejected，用于训练 Agent 的过程偏好。
- 构建 train / eval split 与 held-out benchmark，避免 prompt leakage；基于 held-out baseline 定位精确指令执行失败，并设计相似但不泄漏评测集的 targeted benchmark 进行数据增强。
- 打通从 benchmark 运行、trajectory 收集、reward 打分、SFT / DPO JSONL 导出、dry run 验证到 badcase 报告生成的闭环流程，形成可复现的 Agent 后训练数据工程方案。

## 更靠近预定目标的投递版本

这版适合投递 Agent 后训练、数据、SFT / DPO、模型评估相关实习岗位。写法更贴近目标 JD，强调“后训练闭环”和“可持续迭代”，但仍然保留可解释的工程落点。

**CodeMind：代码 Agent 后训练数据、偏好构造与评估闭环**  
2026年03月 - 2026年05月  
**技术栈：** Python、Agent Harness、Tool Calling、Run Trace、Benchmark Eval、Reward Modeling、SFT、DPO、Data Curation

- 面向代码 Agent 后训练场景，构建从任务运行、trajectory 采集、reward 打分、SFT / DPO 数据构造到 held-out eval 的闭环 pipeline，支持基于真实 tool-use 轨迹持续迭代 Agent 的任务完成行为。
- 设计 Agent Harness 执行链路，打通模型适配、工具调用、上下文管理、checkpoint / resume、workspace 指纹校验与 run artifact 落盘，使每次任务的 prompt、工具调用、文件修改、verifier 结果和 stop reason 均可追踪复盘。
- 实现面向后训练的数据清洗与 reward 规则，将 verifier pass、benchmark pass、final answer、tool failure、安全事件、路径逃逸和工具步数等信号纳入轨迹打分，自动筛选高质量 SFT 样本和低质量 rejected 样本。
- 基于 same-prompt repeated sampling 构造 DPO preference pairs，将“完成 verifier 条件并及时 final”的轨迹作为 chosen，将“verifier 通过但未及时结束 / 工具调用失败 / 协议不合规”的轨迹作为 rejected，用于优化 Agent 的过程偏好。
- 建立 held-out benchmark 与 badcase taxonomy，定位 `completion_without_final`、`verifier_failed`、`tool_failed`、精确指令执行失败等问题，并通过 targeted benchmark 生成相似但不泄漏评测集的训练样本。
- 支持将真实 trajectory 导出为 SFT / DPO JSONL，并通过 dry run 验证 teacher forcing 数据格式、assistant mask 和训练样本质量，为后续 LoRA SFT、DPO 训练和 base-vs-finetuned 行为对比评估预留接口。

## 面试展开讲法

### 1. 项目背景

普通 SFT / DPO 多是 prompt-answer 数据，但代码 Agent 的后训练更关注完整运行轨迹：模型是否会读文件、是否会定位 bug、是否会正确 patch、是否会跑 verifier、完成后是否及时 final。因此我把本地 Agent Harness 做成了可以采集、评估和复盘 trajectory 的系统。

### 2. Harness 主体

我先搭了 Agent 的执行链路，包括 CLI、模型后端、工具调用、会话状态、checkpoint / resume、workspace 指纹和 run artifact。这样每次任务不是一次黑盒调用，而是可以落盘为 trace、report 和 task_state，方便后续评测与数据处理。

### 3. Context 与 Memory

为了解决多轮任务 prompt 变长的问题，我把上下文拆成 prefix、memory、relevant memory、history 和 request，再按优先级裁剪。记忆上区分任务摘要、文件摘要、过程笔记和相关记忆召回，减少 follow-up 阶段重复读文件。

### 4. 安全与评测

所有工具经过统一网关，做路径校验、参数校验、敏感信息脱敏和高风险操作控制。评测上用 benchmark 固定任务，记录 pass rate、verifier pass、tool steps、failure category、resume status 等指标。

### 5. 后训练扩展

在 Harness 之上，我又做了后训练数据 pipeline：从 `.core agent package/runs/*` 中加载 trajectory，结合 verifier、stop reason、tool error 和安全事件做 rule-based reward。高质量成功轨迹导出为 SFT；同 prompt 下 reward 差距明显的轨迹构造成 DPO pair。

### 6. Badcase 到数据增强

我没有只堆数据，而是先做 held-out eval 和 badcase taxonomy。比如发现模型会完成文件修改但不及时 final，就用 repeated sampling 构造 chosen / rejected：chosen 是完成 verifier 后及时 final，rejected 是 verifier 通过但撞到 step limit。这样 DPO 训练目标就更贴近 Agent 行为偏好。

## 面试关键词

- Agent Harness
- Tool Calling
- Context Management
- Checkpoint / Resume
- Workspace Fingerprint
- Run Trace
- Benchmark Eval
- Verifier-based Evaluation
- Rule-based Reward
- SFT Data Construction
- DPO Preference Pair
- Same-prompt Repeated Sampling
- Badcase Taxonomy
- Held-out Eval
- Prompt Leakage
- Targeted Data Augmentation

## 面试时可以强调的亮点

- 不是只调 API，而是做了可复盘的 Agent Harness。
- 不是只看最终答案，而是评估完整 tool-use trajectory。
- 不是只做 SFT，而是区分 SFT 成功示范和 DPO 偏好对。
- 不是把评测题混进训练，而是做 train / eval split 和 held-out eval。
- 不是盲目加数据，而是从 badcase 反推 targeted data augmentation。

## 建议避免的说法

不建议说：

- 我训练出了一个更强的代码大模型。
- 我完成了大规模 SFT / DPO 训练。
- 数据量已经证明模型能力显著提升。

更稳的说法：

- 我完成了面向代码 Agent 的后训练数据采集、评估和偏好构造闭环。
- 我构建了从真实 trajectory 到 SFT / DPO 数据的 pipeline。
- 我通过 held-out eval 和 badcase taxonomy 指导数据增强，并用 dry run 验证训练数据格式。
- 当前重点是 Agent 后训练数据工程、reward 设计和评估闭环，而不是大规模算力训练。
