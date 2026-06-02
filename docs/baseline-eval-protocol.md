# Baseline 与评估协议

这份协议用于冻结下一阶段的实验口径：从训练前的数据诊断，切换到小规模、可复现的 SFT/DPO 实验。

## 当前项目状态

CodeMind 目前已经具备 agent 后训练数据闭环的核心组件：

- 能针对文本编辑和代码修复任务采集真实 DeepSeek trajectory。
- 能从 `.core agent package/runs/*` artifacts 中加载 trace、report 和 task_state。
- 能基于任务完成、verifier 结果、安全性和工具步数做规则 reward 打分。
- 能从干净的成功轨迹中导出 SFT candidate。
- 能从同 prompt 的不同轨迹中构造带 reward gap 的 DPO pair。
- 能对 protocol、tool、verifier 和 safety failure 做 badcase taxonomy。
- 能用跨数据集对比报告分析 before/after 行为变化。

最近一轮训练前诊断发现了两个有价值的失败模式：

- `completion_without_final`：任务已经完成，但 agent 在 step limit 前没有返回 final answer。
- `non_unique_old_text`：`patch_file.old_text` 匹配到多个位置，导致编辑位置不明确。

## 采用的 Baseline

第一轮训练实验采用的 baseline 应包含：

- DeepSeek provider 对空文本或无法解析的响应进行重试。
- Runtime 提示：验证通过后应及时返回 final answer。
- Runtime 提示：`patch_file.old_text` 必须唯一，必要时应包含足够上下文。
- 对 `invalid_arguments` 做更细 badcase subtype，包括 `non_unique_old_text`、`old_text_not_found`、`missing_new_text`、`bad_file_path` 和 `bad_directory_path`。
- 通过 `compare_post_training_datasets.py` 生成数据集对比报告。

不采用更丰富的 `patch_file` matching-context feedback 作为 baseline。专项实验中，它把 `non_unique_old_text` 从 3 次降到 1 次，但 benchmark pass rate 从 100% 降到 80%，平均 reward 从 0.96 降到 0.18。因此这里把它视为 error migration 的证据，而不是安全的全局改进。

## 固定评估指标

每一次 base-vs-finetuned 对比都应报告同一组指标：

- Benchmark pass rate。
- Verifier pass rate。
- 平均 reward。
- 平均 tool steps。
- Final-answer return rate。
- SFT candidate 数量和比例。
- DPO pair 数量。
- Quality label 分布。
- Failure subtype 分布，尤其关注 `non_unique_old_text`、`completion_without_final`、`bad_file_path` 和 `tool_failed`。

## 数据划分规则

训练数据和评估数据必须分开：

- 训练数据可以包含历史成功轨迹和同 prompt preference pairs。
- 评估数据必须从训练导出中 held out。
- 用于 DPO 的 repeated-sampling runs 不应同时作为同 prompt 的 held-out eval examples。
- Synthetic trajectories 可以用于验证 loader 和 reward 逻辑，但不能声称为真实训练证据。

## 下一阶段里程碑

下一阶段目标是跑通第一轮小规模微调闭环：

1. 固定一版包含文本编辑和代码修复任务的 eval benchmark。
2. 从干净成功轨迹导出 `train_sft.jsonl`。
3. 从同 prompt chosen/rejected 轨迹导出 `train_dpo.jsonl`。
4. 如果本地或租用 GPU 资源允许，跑一轮小规模 Qwen LoRA SFT。
5. 使用同一套 benchmark 和 comparison script 对比 base model 与 SFT model。
6. 只有当 SFT baseline 产出可测量或可诊断的结果后，再加入 DPO。

这个阶段的目标不是证明小数据集能带来通用 coding-agent 能力提升，而是展示一个可复现的 agent 后训练闭环：采集带 verifier 的 trajectories，挖掘失败模式，构造 SFT/DPO 数据，微调小模型，并评估目标 tool-use 行为是否发生变化。
