"""Render representative post-training badcases from Pico run artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from post_training.badcases import build_badcase_report, render_markdown


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Directory that contains .pico/runs artifacts.")
    parser.add_argument("--out", type=Path, help="Optional output path.")
    parser.add_argument("--format", choices=("json", "markdown"), default="markdown")
    parser.add_argument(
        "--benchmark-artifact",
        action="append",
        default=[],
        help="Optional benchmark JSON artifact used to enrich runs with verifier results.",
    )
    args = parser.parse_args()

    report = build_badcase_report(
        args.root,
        benchmark_artifacts=args.benchmark_artifact or None,
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
