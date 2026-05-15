"""Export DPO JSONL preference pairs from scored Pico trajectories."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from post_training.dpo_builder import ScoredTrajectory, build_dpo_record, build_pairs_for_prompt
from post_training.reward import score_trajectory
from post_training.trace_loader import discover_benchmark_artifacts, find_run_dirs, load_benchmark_index, summarize_run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Directory that contains .pico/runs artifacts.")
    parser.add_argument("--out", type=Path, required=True, help="Output JSONL path.")
    parser.add_argument("--min-reward-gap", type=float, default=0.5, help="Minimum chosen/rejected reward gap.")
    parser.add_argument(
        "--demo-global-pair",
        action="store_true",
        help="If no same-prompt pairs exist, emit one educational, not-for-training pair using global best/worst trajectories.",
    )
    parser.add_argument(
        "--benchmark-artifact",
        action="append",
        default=[],
        help="Optional benchmark JSON artifact used to enrich runs with verifier results.",
    )
    args = parser.parse_args()

    artifact_paths = args.benchmark_artifact or discover_benchmark_artifacts(args.root)
    benchmark_index = load_benchmark_index(artifact_paths) if artifact_paths else {}
    by_prompt: dict[str, list[ScoredTrajectory]] = defaultdict(list)
    all_scored: list[ScoredTrajectory] = []
    for run_dir in find_run_dirs(args.root):
        summary = summarize_run(run_dir, benchmark_index=benchmark_index)
        scored = ScoredTrajectory(summary=summary, reward=score_trajectory(summary))
        all_scored.append(scored)
        if summary.user_request:
            by_prompt[summary.user_request].append(scored)

    records = []
    for trajectories in by_prompt.values():
        records.extend(build_pairs_for_prompt(trajectories, min_reward_gap=args.min_reward_gap))

    if not records and args.demo_global_pair and len(all_scored) >= 2:
        ordered = sorted(all_scored, key=lambda item: item.reward.score, reverse=True)
        best = ordered[0]
        worst = ordered[-1]
        if best.reward.score - worst.reward.score >= args.min_reward_gap:
            demo = build_dpo_record(best, worst, reason="demo_global_high_vs_low_reward")
            demo.metadata["same_prompt"] = best.summary.user_request == worst.summary.user_request
            demo.metadata["valid_for_training"] = False
            demo.metadata["note"] = "Educational shape demo only; DPO training pairs should share the same prompt."
            records.append(demo)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True))
            handle.write("\n")

    print(f"scanned={len(all_scored)} exported={len(records)} out={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
