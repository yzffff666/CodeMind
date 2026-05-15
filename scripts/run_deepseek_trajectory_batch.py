#!/usr/bin/env python3
"""Run a small real DeepSeek trajectory batch for post-training data collection."""

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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-path", type=Path, default=ROOT / "benchmarks" / "coding_tasks.json")
    parser.add_argument("--workspace-root", type=Path, default=ROOT / "artifacts" / "real-deepseek-trajectories")
    parser.add_argument("--artifact-path", type=Path, default=ROOT / "artifacts" / "real-deepseek-benchmark.json")
    parser.add_argument("--task-id", action="append", dest="task_ids", help="Task id to include. Can be repeated.")
    parser.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"))
    parser.add_argument("--base-url", default=os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com"))
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args(argv)

    _load_local_env(ROOT / ".env.local")
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("DEEPSEEK_API_KEY is not set. Put it in .env.local or the current environment.", file=sys.stderr)
        return 1

    task_ids = tuple(args.task_ids or _all_task_ids(args.benchmark_path))
    subset_path = args.workspace_root / "benchmark_subset.json"
    _write_benchmark_subset(args.benchmark_path, subset_path, task_ids)

    def factory(task, workspace):
        del task, workspace
        return DeepSeekChatModelClient(
            model=args.model,
            base_url=args.base_url,
            api_key=api_key,
            temperature=0.0,
            timeout=300,
        )

    payload = run_fixed_benchmark(
        benchmark_path=subset_path,
        artifact_path=args.artifact_path,
        workspace_root=args.workspace_root,
        model_name="deepseek",
        model_version=args.model,
        temperature=0.0,
        max_new_tokens=args.max_new_tokens,
        model_client_factory=factory,
    )
    summary = payload.get("summary", {})
    print(
        json.dumps(
            {
                "artifact_path": str(args.artifact_path),
                "workspace_root": str(args.workspace_root),
                "tasks": len(payload.get("rows", [])),
                "passed": summary.get("passed"),
                "failed": summary.get("failed"),
                "pass_rate": summary.get("pass_rate"),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _load_local_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().lstrip("\ufeff")
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _write_benchmark_subset(source_path: Path, output_path: Path, task_ids: tuple[str, ...]) -> None:
    data = json.loads(source_path.read_text(encoding="utf-8"))
    wanted = set(task_ids)
    tasks = [task for task in data.get("tasks", []) if task.get("id") in wanted]
    found = {task.get("id") for task in tasks}
    missing = sorted(wanted - found)
    if missing:
        raise ValueError(f"Unknown task ids: {', '.join(missing)}")
    subset = {
        "schema_version": data.get("schema_version", 1),
        "description": "Small real DeepSeek trajectory batch for post-training data collection.",
        "tasks": [_with_absolute_fixture_path(task) for task in tasks],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(subset, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _all_task_ids(source_path: Path) -> tuple[str, ...]:
    data = json.loads(source_path.read_text(encoding="utf-8"))
    return tuple(str(task.get("id", "")).strip() for task in data.get("tasks", []) if str(task.get("id", "")).strip())


def _with_absolute_fixture_path(task: dict) -> dict:
    updated = dict(task)
    updated["fixture_repo"] = str((ROOT / str(task["fixture_repo"])).resolve())
    if os.name == "nt":
        python_cmd = f'"{sys.executable}"'
        verifier = str(updated["verifier"])
        if verifier.startswith("python3 "):
            verifier = python_cmd + verifier[len("python3") :]
        elif verifier.startswith("python "):
            verifier = python_cmd + verifier[len("python") :]
        updated["verifier"] = verifier
    if str(updated.get("id", "")).startswith("sample_"):
        updated["step_budget"] = max(int(updated["step_budget"]), 6)
    return updated


if __name__ == "__main__":
    raise SystemExit(main())
