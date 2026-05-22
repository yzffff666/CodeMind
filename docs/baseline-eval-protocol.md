# Baseline and Evaluation Protocol

This protocol freezes the next project phase: move from pre-training data
diagnostics into a small, reproducible SFT/DPO experiment.

## Current Project State

Pico now has the core pieces for an agent post-training data loop:

- Real DeepSeek trajectory collection for text-edit and code-repair tasks.
- Trace loading from `.pico/runs/*` artifacts.
- Rule-based reward scoring for completed, verified, safe, and efficient runs.
- SFT candidate export from clean successful trajectories.
- DPO pair construction from same-prompt trajectories with reward gaps.
- Badcase taxonomy for protocol, tool, verifier, and safety failures.
- Cross-dataset comparison reports for before/after behavior analysis.

The latest pre-training diagnostics found two useful failure modes:

- `completion_without_final`: the task was solved, but the agent did not return
  a final answer before the step limit.
- `non_unique_old_text`: `patch_file.old_text` matched multiple locations, so
  the edit was ambiguous.

## Adopted Baseline

The baseline for the first training experiment should include:

- DeepSeek provider retry for empty or unparsable text responses.
- Runtime guidance to return a final answer after successful verification.
- Runtime guidance that `patch_file.old_text` must be unique and should include
  enough surrounding context.
- Badcase subtyping for `invalid_arguments`, including `non_unique_old_text`,
  `old_text_not_found`, `missing_new_text`, `bad_file_path`, and
  `bad_directory_path`.
- Dataset comparison reporting via `compare_post_training_datasets.py`.

The richer `patch_file` matching-context feedback is not adopted as the
baseline. In the targeted experiment it reduced `non_unique_old_text` from 3 to
1, but benchmark pass rate dropped from 100% to 80% and average reward dropped
from 0.96 to 0.18. This is treated as evidence of error migration rather than a
safe global improvement.

## Fixed Evaluation Metrics

Every base-vs-finetuned comparison should report the same metrics:

- Benchmark pass rate.
- Verifier pass rate.
- Average reward.
- Average tool steps.
- Final-answer return rate.
- SFT candidate count and rate.
- DPO pair count.
- Quality label distribution.
- Failure subtype distribution, especially `non_unique_old_text`,
  `completion_without_final`, `bad_file_path`, and `tool_failed`.

## Data Split Rules

Use separate data for training and evaluation:

- Training data can include historical successful trajectories and same-prompt
  preference pairs.
- Evaluation data must be held out from training exports.
- Repeated-sampling runs for DPO should not also be used as held-out evaluation
  examples for the same prompt.
- Synthetic trajectories can validate loaders and reward logic, but should not
  be claimed as real training evidence.

## Next Milestone

The next milestone is the first small fine-tuning loop:

1. Freeze an eval benchmark with text-edit and code-repair tasks.
2. Export `train_sft.jsonl` from clean successful trajectories.
3. Export `train_dpo.jsonl` from same-prompt chosen/rejected trajectories.
4. Run a small Qwen LoRA SFT experiment if local or rented GPU resources are
   available.
5. Evaluate base and SFT models with the same benchmark and comparison script.
6. Add DPO only after the SFT baseline produces a measurable or diagnosable
   result.

The goal is not to prove broad coding-agent improvement from a small dataset.
The goal is to show a reproducible agent post-training loop: collect
verifier-backed trajectories, mine failures, build SFT/DPO data, fine-tune a
small model, and evaluate whether targeted tool-use behaviors change.
