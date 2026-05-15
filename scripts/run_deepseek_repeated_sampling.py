#!/usr/bin/env python3
"""Repeat one real DeepSeek task to collect same-prompt trajectories for DPO."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pico.evaluator import run_fixed_benchmark  # noqa: E402
from pico.models import DeepSeekChatModelClient  # noqa: E402
from scripts.run_deepseek_trajectory_batch import _load_local_env, _write_benchmark_subset  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", default="readme_intro_locked")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--benchmark-path", type=Path, default=ROOT / "benchmarks" / "coding_tasks.json")
    parser.add_argument("--output-root", type=Path, default=ROOT / "artifacts" / "real-deepseek-dpo-samples")
    parser.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"))
    parser.add_argument("--base-url", default=os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com"))
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args(argv)

    _load_local_env(ROOT / ".env.local")
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("DEEPSEEK_API_KEY is not set. Put it in .env.local or the current environment.", file=sys.stderr)
        return 1

    results = []
    for index in range(1, args.repeats + 1):
        round_root = args.output_root / f"round_{index:02d}"
        subset_path = round_root / "benchmark_subset.json"
        artifact_path = round_root / "benchmark.json"
        _write_benchmark_subset(args.benchmark_path, subset_path, (args.task_id,))

        def factory(task, workspace):
            del task, workspace
            return DeepSeekChatModelClient(
                model=args.model,
                base_url=args.base_url,
                api_key=api_key,
                temperature=args.temperature,
                timeout=300,
            )

        payload = run_fixed_benchmark(
            benchmark_path=subset_path,
            artifact_path=artifact_path,
            workspace_root=round_root / "workspace",
            model_name="deepseek",
            model_version=args.model,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            model_client_factory=factory,
        )
        row = payload["rows"][0]
        results.append(
            {
                "round": index,
                "status": row["status"],
                "failure_category": row["failure_category"],
                "tool_steps": row["tool_steps"],
                "attempts": row["attempts"],
                "run_id": row["run_id"],
            }
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_root / "sampling_summary.json"
    summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
