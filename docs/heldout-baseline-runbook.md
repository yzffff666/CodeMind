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

成功后会生成：

- `artifacts/heldout-deepseek-baseline-v1.json`
- `artifacts/heldout-deepseek-baseline-v1/**/.pico/runs/*`

## 看结果

重点看 summary：

- `pass_rate`：任务整体通过率。
- `verifier_pass_rate`：自动测试/校验通过率。
- `failed`：失败任务数量。
- `failure_category_counts`：失败类型分布。

## 为什么这一步重要

SFT 前先跑 baseline，SFT 后再跑同一份 held-out benchmark，才能比较模型行为是否真的变好。这里评估的不是 loss，而是 agent 在未见过任务上的工具使用、代码修改、verifier 通过和 final answer 返回能力。

