# Exact-instruction 数据增强记录

## 背景

held-out baseline v1 暴露了一个失败模式：模型能理解大方向，但没有严格满足 verifier 的完整目标短语。例如它把 `Status: draft` 改成 `Status: ready`，但评测要求是 `Status: ready for held-out evaluation`。

## 这次新增了什么

新增训练专用 benchmark：

- `benchmarks/exact_instruction_train_tasks.json`
- `tests/fixtures/bench_repo_release`
- `tests/fixtures/bench_repo_policy`

这组任务和 held-out eval 相似，但不是同一个 prompt、fixture 或目标短语，因此不会直接泄漏评测题。

## 真实采样结果

DeepSeek baseline 在该训练 benchmark 上：

- 任务数：2
- 通过：1
- 失败：1
- pass rate：0.5
- 主要失败类型：`completion_without_final`

其中 `policy_mode_strict_collection` 是成功轨迹，可以作为 SFT candidate；`release_status_candidate_review` 的文件修改通过了 verifier，但没有及时返回 final，适合用于 badcase 分析。

随后对 `release_status_candidate_review` 做了 4 次 repeated sampling：

- 成功：1
- 失败：3
- verifier passed：4
- 主要失败类型：`completion_without_final`
- DPO pairs：3

这批 pair 的 chosen 是及时返回 final 的成功轨迹，rejected 是虽然文件修改正确、但没有在 step limit 内 final 的轨迹。它更像是在训练 agent 偏好“完成 verifier 条件后及时停止并返回 final answer”。

## 学习点

这一步不是直接训练，而是在做针对性数据增强：

1. 从 held-out eval 找到失败模式。
2. 设计相似但不相同的训练任务。
3. 采集真实 trajectory。
4. 只把高质量成功轨迹导入 SFT，失败轨迹用于 badcase 或 DPO 候选。

这就是后训练里常见的闭环：评测发现问题，数据针对问题补齐，再用 held-out eval 重新验证。
