# 简历项目描述：CodeMind / Pico Agent 后训练数据闭环

## 简历项目名

CodeMind：面向代码 Agent 的后训练数据采集与评估闭环

## 一句话概括

基于 Pico 代码 Agent，构建真实 trajectory 采集、reward 打分、SFT/DPO 数据构造、badcase 分析和 held-out eval 的后训练数据闭环，用于分析并改进 agent 的工具使用和任务完成行为。

## 简历 Bullet 版本

- 基于 Pico agent runtime 接入 DeepSeek API，设计代码修复、文本编辑、精确指令执行等 benchmark，采集真实多步 tool-use trajectories，并保留 verifier、trace、report、task_state 等可复现运行信息。
- 实现 agent 后训练数据处理 pipeline：从 `.pico/runs/*` 加载轨迹，结合 verifier 结果、stop reason、tool error、安全事件和工具步数构造 rule-based reward，并筛选 SFT candidates 与 same-prompt DPO pairs。
- 设计 badcase taxonomy，对 `completion_without_final`、`non_unique_old_text`、`verifier_failed`、`tool_failed` 等失败模式进行分类统计，用于指导 prompt/runtime 修正和 targeted data augmentation。
- 构建 train/eval split 与 held-out benchmark，避免将评测 prompt 直接放入训练集；基于 held-out baseline 发现“近似改写但不满足精确 verifier”的失败模式，并设计相似但不泄漏的 exact-instruction 训练任务。
- 通过 repeated sampling 构造 DPO 偏好数据，将“改对文件且及时 final”的轨迹作为 chosen，将“改对文件但未及时 final”的轨迹作为 rejected，最终形成 SFT 18 条、DPO 9 对的小规模可验证训练数据集。

## 更短的简历版

- 构建代码 Agent 后训练数据闭环，支持真实 DeepSeek trajectory 采集、reward 打分、SFT/DPO 数据导出、badcase 分析与 held-out eval。
- 设计 train/eval split，避免 prompt leakage；基于 held-out baseline 定位精确指令执行失败，并通过 targeted benchmark 与 repeated sampling 构造 SFT/DPO 数据。
- 实现 rule-based reward 与 badcase taxonomy，覆盖 verifier pass、final answer、tool failure、安全事件、工具步数等信号，产出 18 条 SFT 和 9 对 DPO 样本用于小规模训练验证。

## 面试展开讲法

可以按这个顺序讲：

1. 先讲背景：普通 SFT/DPO 教程多是 prompt-answer，但 agent 后训练更关注完整轨迹，包括工具调用、文件修改、测试/verifier 和 final answer。
2. 再讲你做了什么：我把 Pico 改造成一个可以采集真实 agent trajectory 的数据闭环，接入 DeepSeek，保存 trace/report/task_state。
3. 讲 reward：我没有只看最终文本，而是综合 verifier 是否通过、是否及时 final、工具是否失败、是否有安全事件、工具步数是否过多。
4. 讲 SFT/DPO：成功且高 reward 的轨迹导出为 SFT；同 prompt 下成功/失败差距明显的轨迹构造成 DPO chosen/rejected pair。
5. 讲评估意识：我单独做了 held-out eval，避免把评测题混进训练；发现 badcase 后，设计相似但不同的训练任务做 targeted data augmentation。
6. 讲结果：当前是小规模实验，已经形成 SFT 18 条、DPO 9 对，并通过 dry run 验证数据格式和 teacher forcing objective。

## 面试关键词

- Agent trajectory
- Tool-use data
- SFT data construction
- DPO preference pair
- Rule-based reward
- Verifier-based evaluation
- Badcase taxonomy
- Train/eval split
- Held-out evaluation
- Data leakage / prompt leakage
- Repeated sampling
- Targeted data augmentation
- Teacher forcing

## 不要夸大的说法

不要说：

- “我训练出了一个更强的代码大模型。”
- “我完成了完整的大规模 SFT/DPO 训练。”
- “这个数据量已经证明模型能力显著提升。”

更准确的说法：

- “我完成了 agent 后训练数据闭环和小规模可验证数据集构建。”
- “我通过 dry run 验证了 SFT 数据格式和 teacher forcing 目标。”
- “我设计了 held-out eval，用于后续比较 base model 与 SFT/DPO model 的行为变化。”
- “当前项目重点是数据、reward、评估和偏好构造，而不是大规模训练算力。”

## 推荐简历最终写法

**CodeMind：代码 Agent 后训练数据采集与评估闭环**

基于 Pico agent runtime 构建面向代码修复与文本编辑任务的后训练数据 pipeline，接入 DeepSeek API 采集真实 tool-use trajectories，并保存 trace、report、verifier 和 task_state 等可复现运行信息。设计 rule-based reward，将 verifier pass、final answer、tool failure、安全事件和工具步数纳入打分，自动筛选 SFT candidates 与 same-prompt DPO preference pairs。构建 held-out eval 与 badcase taxonomy，定位 `completion_without_final`、`verifier_failed` 等失败模式，并通过 targeted benchmark 和 repeated sampling 进行数据增强，形成 18 条 SFT 与 9 对 DPO 小规模训练样本，完成 SFT dry run 验证。

