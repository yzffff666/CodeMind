# Base-vs-SFT 评估协议

## 目标

训练后不能只看 loss，需要用 held-out benchmark 对比 base model 和 SFT model 的真实 agent 行为。

这个协议用于回答：

- SFT 后任务通过率有没有变化？
- verifier pass rate 有没有变化？
- 模型是否更容易在完成任务后及时返回 final？
- reward 和 badcase 分布有没有改善？
- 是否出现 error migration？

## 输入

- Base artifact：`artifacts/heldout-deepseek-baseline-v1.json`
- SFT artifact：训练后在同一份 `benchmarks/heldout_eval_tasks.json` 上跑出来的 benchmark artifact
- 配置：`configs/base_vs_sft_eval.json`

## 对比指标

- pass rate
- verifier pass rate
- final-answer return rate
- average reward
- average tool steps
- failure category distribution
- quality label distribution
- task-level passed / stop_reason / reward

## 运行命令

当 SFT artifact 准备好后运行：

```powershell
python scripts\compare_base_vs_sft_eval.py --config configs\base_vs_sft_eval.json
```

如果还没有 SFT artifact，可以先用 base artifact 对比自身做 smoke：

```powershell
python scripts\compare_base_vs_sft_eval.py --config configs\base_vs_sft_eval.json --sft-artifact artifacts\heldout-deepseek-baseline-v1.json --sft-name smoke_same_as_base --out artifacts\eval\base-vs-base-smoke.md
```

## 判断原则

- 如果 loss 下降但 held-out 行为指标没有改善，不能声称模型能力提升。
- 如果 verifier pass 下降，即使 final-answer rate 提升，也要检查是否出现 error migration。
- 如果 final-answer rate 提升且 verifier pass 不下降，说明 SFT 可能改善了任务完成后的停止行为。
- held-out eval 的轨迹不能加入训练数据。

## 面试表达

可以说：我把评估协议固定为 base-vs-SFT held-out 对比，不用训练 loss 单独下结论，而是比较 pass rate、verifier pass、final-answer rate、reward 和 badcase 分布，避免把数据构造项目停留在“只会导 JSONL”的阶段。

