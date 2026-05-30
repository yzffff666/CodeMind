#!/usr/bin/env python3
"""Run the held-out DeepSeek baseline used for future SFT/DPO comparisons."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_deepseek_trajectory_batch import main as run_batch  # noqa: E402


def main() -> int:
    return run_batch(
        [
            "--benchmark-path",
            str(ROOT / "benchmarks" / "heldout_eval_tasks.json"),
            "--workspace-root",
            str(ROOT / "artifacts" / "heldout-deepseek-baseline-v1"),
            "--artifact-path",
            str(ROOT / "artifacts" / "heldout-deepseek-baseline-v1.json"),
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())

