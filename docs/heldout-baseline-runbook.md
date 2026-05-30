# Held-out Baseline 运行手册

这份手册用于生成训练前 baseline。它会调用 DeepSeek API，在 `benchmarks/heldout_eval_tasks.json` 上跑 agent，并把真实 trajectory 保存在 `artifacts/heldout-deepseek-baseline-v1`。

## 前置条件

- `.env.local` 中已经配置 `DEEPSEEK_API_KEY`。
- 不要把 `.env.local` 或 `artifacts/` 提交到 Git。
- 当前 held-out eval 只用于评测，不要把这些轨迹加入 `train_sft.jsonl` 或 `train_dpo.jsonl`。

## 运行命令

在项目根目录执行：

```powershell
python scripts\run_heldout_deepseek_baseline.py
```

如果 `python` 指向 WindowsApps 的占位启动器，可能会直接失败且没有输出。可以改用真实 Python 路径，例如：

```powershell
C:\Users\13670\AppData\Local\Programs\Python\Python311\python.exe scripts\run_heldout_deepseek_baseline.py
```

成功后会生成：

- `artifacts/heldout-deepseek-baseline-v1.json`
- `artifacts/heldout-deepseek-baseline-v1/**/.pico/runs/*`

## 看结果

重点看 summary：

- `pass_rate`：任务整体通过率。
- `verifier_pass_rate`：自动测试/校验通过率。
- `failed`：失败任务数量。
- `failure_category_counts`：失败类型分布。

当前 DeepSeek baseline v1 的结果：

- 任务数：2
- 通过：1
- 失败：1
- pass rate：0.5
- 主要失败类型：`verifier_failed`

失败样例是 `profile_status_ready`：模型把 `Status: draft` 改成了 `Status: ready`，但 verifier 要求精确包含 `Status: ready for held-out evaluation`。这类 badcase 说明模型理解了大方向，但没有严格执行评测条件。

## 为什么这一步重要

SFT 前先跑 baseline，SFT 后再跑同一份 held-out benchmark，才能比较模型行为是否真的变好。这里评估的不是 loss，而是 agent 在未见过任务上的工具使用、代码修改、verifier 通过和 final answer 返回能力。
