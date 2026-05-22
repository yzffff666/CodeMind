# SFT Dry Run 说明

这一步的目标不是证明模型能力提升，而是验证第一版 SFT 数据能进入训练链路。

当前 `artifacts/datasets/v1/train_sft.jsonl` 只有 6 条样本，适合做 pipeline validation，不适合得出泛化提升结论。因此这里先使用一个不依赖 `torch`、`transformers` 或 GPU 的 dry run 脚本：

```powershell
python scripts\sft_dry_run.py --data artifacts\datasets\v1\train_sft.jsonl --out artifacts\datasets\v1\sft_dry_run_report.json
```

## 它验证什么

- JSONL 每行是否能正确解析。
- 每条样本是否包含合法的 `messages`。
- 第一条 message 是否是 user。
- 是否存在 assistant message。
- 能否把多轮 agent trajectory 序列化成训练文本。
- 能否只在 assistant 输出位置计算 next-token loss。

## 为什么只看 assistant 位置

普通语言模型预训练会对整段文本做 next-token prediction。Chat SFT 通常会把 user prompt、tool observation 等上下文作为输入，但 loss 主要计算在 assistant 需要学习输出的位置。

在 agent trajectory 里，这意味着模型主要学习：

- 什么时候发起 tool call。
- tool call 应该包含什么参数。
- 看到 tool observation 后下一步怎么决策。
- 什么时候返回 final answer。

因此 dry run 使用 masked assistant next-character loss：上下文会被放进序列，但统计 loss 时只看 assistant span。

## 它不代表什么

这个 dry run 不是 LoRA，也不是神经网络微调。它只是一个 toy bigram learner，用来确认数据格式、mask 逻辑和 teacher forcing 目标没有明显问题。

后续真正训练时，需要换成 Qwen + LoRA + SFTTrainer 或类似框架，并在 held-out eval benchmark 上做 base-vs-SFT 对比。
