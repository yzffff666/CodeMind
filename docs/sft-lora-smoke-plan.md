# SFT LoRA Smoke 训练计划

## 当前定位

当前数据量适合做最小 SFT 训练闭环验证，而不是证明模型能力显著提升。

- SFT：18 条
- DPO：9 对
- 目标：验证训练入口、assistant-only label mask、checkpoint 输出和后续评估协议
- 非目标：声称训练出了稳定更强的代码 Agent

## 新增入口

- 配置：`configs/sft_lora_smoke.json`
- 脚本：`scripts/train_sft_lora.py`

默认推荐先运行 dry run：

```powershell
python scripts\train_sft_lora.py --config configs\sft_lora_smoke.json --dry-run
```

dry run 会验证：

- `train_sft.jsonl` 能被加载
- messages 结构合法
- assistant-only 训练目标能被定位
- 配置中的模型、LoRA、训练参数和评估计划可解析
- 输出 dry-run report 到 `artifacts/models/sft-lora-smoke`

## 真正 LoRA 训练需要什么

真实 LoRA SFT 需要额外依赖：

- `torch`
- `transformers`
- `peft`

有可用模型和环境后，可以运行：

```powershell
python scripts\train_sft_lora.py --config configs\sft_lora_smoke.json
```

默认模型配置为 `Qwen/Qwen2.5-0.5B-Instruct`。如果本地已有更小或已下载模型，可以把 `model.name_or_path` 改成本地路径。

## 训练后怎么评估

训练后不直接看 loss 下结论，而是回到 held-out benchmark：

1. 用 base model 跑 `benchmarks/heldout_eval_tasks.json`
2. 用 SFT adapter 跑同一份 held-out benchmark
3. 对比：
   - pass rate
   - verifier pass rate
   - final-answer return rate
   - 平均 reward
   - badcase subtype 分布

## 面试表达

可以说：我把项目推进到最小 SFT LoRA 训练闭环，先通过 dry run 验证数据、mask 和配置，再预留真实 LoRA 训练入口；训练效果不只看 loss，而是计划用 held-out eval 做 base-vs-SFT 行为对比。

