"""Export SFT JSONL records from Pico run artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from post_training.reward import score_trajectory
from post_training.sft_builder import build_sft_record, is_sft_candidate
from post_training.trace_loader import discover_benchmark_artifacts, find_run_dirs, load_benchmark_index, summarize_run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Directory that contains .pico/runs artifacts.")
    parser.add_argument("--out", type=Path, required=True, help="Output JSONL path.")
    parser.add_argument("--min-reward", type=float, default=1.0, help="Minimum reward score for SFT candidates.")
    parser.add_argument(
        "--benchmark-artifact",
        action="append",
        default=[],
        help="Optional benchmark JSON artifact used to enrich runs with verifier results.",
    )
    args = parser.parse_args()

    artifact_paths = args.benchmark_artifact or discover_benchmark_artifacts(args.root)
    benchmark_index = load_benchmark_index(artifact_paths) if artifact_paths else {}
    records = []
    scanned = 0
    for run_dir in find_run_dirs(args.root):
        scanned += 1
        summary = summarize_run(run_dir, benchmark_index=benchmark_index)
        reward = score_trajectory(summary)
        if is_sft_candidate(summary, reward, min_reward=args.min_reward):
            records.append(build_sft_record(summary, reward).to_dict())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")

    print(f"scanned={scanned} exported={len(records)} out={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
