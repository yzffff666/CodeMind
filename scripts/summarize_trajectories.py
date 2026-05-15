"""Print compact trajectory summaries for Pico run artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from post_training.trace_loader import discover_benchmark_artifacts, find_run_dirs, load_benchmark_index, summarize_run
from post_training.reward import score_trajectory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Directory that contains .pico/runs artifacts.")
    parser.add_argument("--pretty", action="store_true", help="Print indented JSON instead of JSONL.")
    parser.add_argument(
        "--benchmark-artifact",
        action="append",
        default=[],
        help="Optional benchmark JSON artifact used to enrich runs with verifier results.",
    )
    args = parser.parse_args()

    artifact_paths = args.benchmark_artifact or discover_benchmark_artifacts(args.root)
    benchmark_index = load_benchmark_index(artifact_paths) if artifact_paths else {}
    summaries = []
    for run_dir in find_run_dirs(args.root):
        summary = summarize_run(run_dir, benchmark_index=benchmark_index)
        payload = summary.to_dict()
        payload["reward"] = score_trajectory(summary).to_dict()
        summaries.append(payload)
    if args.pretty:
        print(json.dumps(summaries, ensure_ascii=False, indent=2))
    else:
        for summary in summaries:
            print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
