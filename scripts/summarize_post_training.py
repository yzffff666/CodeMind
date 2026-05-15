"""Summarize Pico post-training data quality signals."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from post_training.summary import render_markdown, summarize_post_training


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Directory that contains .pico/runs artifacts.")
    parser.add_argument("--out", type=Path, help="Optional output path.")
    parser.add_argument("--format", choices=("json", "markdown"), default="markdown")
    parser.add_argument("--min-sft-reward", type=float, default=1.0)
    parser.add_argument("--min-dpo-gap", type=float, default=0.5)
    parser.add_argument(
        "--benchmark-artifact",
        action="append",
        default=[],
        help="Optional benchmark JSON artifact used to enrich runs with verifier results.",
    )
    args = parser.parse_args()

    summary = summarize_post_training(
        args.root,
        min_sft_reward=args.min_sft_reward,
        min_dpo_gap=args.min_dpo_gap,
        benchmark_artifacts=args.benchmark_artifact or None,
    )
    if args.format == "json":
        output = json.dumps(summary.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    else:
        output = render_markdown(summary)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
