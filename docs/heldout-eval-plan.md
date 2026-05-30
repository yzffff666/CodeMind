# Held-out Eval 计划

这份评测集用于后续做 base model 和 SFT/DPO model 的对比，不进入训练数据。

## 为什么要单独做 held-out

如果训练数据和评测任务来自同一批 prompt，模型可能只是记住了相似操作，不能说明它学会了更通用的 agent 行为。held-out eval 的作用是把“训练用轨迹”和“评测用任务”分开，让结果更像真实实验。

## 当前新增内容

- `benchmarks/heldout_eval_tasks.json`：新的评测入口。
- `tests/fixtures/bench_repo_profile`：文本修改任务，检查 agent 是否能做最小编辑并及时 final。
- `tests/fixtures/bench_repo_stats`：代码修复任务，检查 agent 是否能读测试、定位 bug、修改实现并跑 verifier。

## 后续使用方式

1. 用 DeepSeek 或本地模型在 `benchmarks/heldout_eval_tasks.json` 上跑一遍，得到训练前 baseline。
2. SFT 后用同一份 held-out benchmark 再跑一遍。
3. 对比 pass rate、verifier pass rate、final-answer return rate、平均 reward、failure subtype 分布。

## 面试表达

可以这样说：项目不只是在收集成功样本，也专门拆出了 held-out benchmark，用来避免训练集和评测集 prompt 泄漏。这样后续微调前后的变化可以被复现和诊断，而不是只展示 loss 下降。

