"""Compare multiple Pico post-training experiment directories."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from post_training.comparison import DatasetSpec, compare_datasets, render_markdown


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        nargs=2,
        metavar=("NAME", "ROOT"),
        required=True,
        help="Dataset name followed by the directory that contains .pico/runs artifacts.",
    )
    parser.add_argument(
        "--benchmark-artifact",
        action="append",
        nargs=2,
        metavar=("NAME", "PATH"),
        default=[],
        help="Optional benchmark JSON artifact associated with one dataset name.",
    )
    parser.add_argument("--out", type=Path, help="Optional output path.")
    parser.add_argument("--format", choices=("json", "markdown"), default="markdown")
    parser.add_argument("--min-sft-reward", type=float, default=1.0)
    parser.add_argument("--min-dpo-gap", type=float, default=0.5)
    args = parser.parse_args()

    artifacts_by_name: dict[str, list[Path]] = {}
    for name, path in args.benchmark_artifact:
        artifacts_by_name.setdefault(name, []).append(Path(path))

    datasets = [
        DatasetSpec(
            name=name,
            root=Path(root),
            benchmark_artifacts=tuple(artifacts_by_name.get(name, [])),
        )
        for name, root in args.dataset
    ]
    report = compare_datasets(
        datasets,
        min_sft_reward=args.min_sft_reward,
        min_dpo_gap=args.min_dpo_gap,
    )
    if args.format == "json":
        text = json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    else:
        text = render_markdown(report) + "\n"

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
