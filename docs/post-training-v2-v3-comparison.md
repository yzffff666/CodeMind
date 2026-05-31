# Post-training 数据集 v2-v3 对比

## 总览

- 对比对象：`v2` -> `v3`
- 配置变化：`configs\post_training_split_v2.json` -> `configs\post_training_split_v3.json`
- SFT 样本：16 -> 18，变化 +2
- DPO pairs：6 -> 9，变化 +3

## 新增来源

### SFT

- `exact_instruction_text_edit_v1`：导出 1 条
- `exact_instruction_repeated_preferences_v1`：导出 1 条

### DPO

- `exact_instruction_text_edit_v1`：导出 0 条
- `exact_instruction_repeated_preferences_v1`：导出 3 条

## 结论

v3 相比 v2 的核心变化不是盲目扩大数据量，而是围绕 held-out baseline 暴露的问题做 targeted data augmentation。
新增 exact-instruction 数据让训练集覆盖“精确执行 verifier 目标短语”和“完成后及时 final”的行为偏好。
其中 repeated sampling 产生了同 prompt 的成功/失败轨迹，因此 DPO pairs 从 6 增至 9；SFT 样本从 16 增至 18。

## 面试表达

可以说：我先用 held-out eval 暴露失败模式，再设计相似但不泄漏评测集的训练任务，通过 repeated sampling 构造 chosen/rejected 偏好对，最后重新生成 v3 数据集并验证 SFT dry run。这个过程体现的是后训练数据闭环，而不是单次 prompt 调参。
