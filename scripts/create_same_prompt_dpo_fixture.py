"""Create a same-prompt DPO fixture from existing Pico run artifacts.

This script is for pipeline testing and teaching. It copies two existing run
directories and normalizes their user_request to the same prompt so the DPO
builder can exercise its same-prompt pairing path. The resulting fixture is not
intended to be used as real training data.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


DEFAULT_PROMPT = "Demo task: complete the requested file edit safely and efficiently."


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chosen-run", type=Path, required=True, help="High-quality run directory to copy.")
    parser.add_argument("--rejected-run", type=Path, required=True, help="Low-quality run directory to copy.")
    parser.add_argument("--out", type=Path, required=True, help="Output fixture root.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Shared prompt assigned to both copied runs.")
    args = parser.parse_args()

    if args.out.exists():
        shutil.rmtree(args.out)
    chosen_out = args.out / "chosen" / ".pico" / "runs" / args.chosen_run.name
    rejected_out = args.out / "rejected" / ".pico" / "runs" / args.rejected_run.name
    shutil.copytree(args.chosen_run, chosen_out)
    shutil.copytree(args.rejected_run, rejected_out)

    for run_dir, label in ((chosen_out, "chosen"), (rejected_out, "rejected")):
        _normalize_prompt(run_dir, args.prompt, label)

    print(f"wrote {args.out}")
    print("note: this is a same-prompt DPO pairing fixture for tests and teaching, not real training data")
    return 0


def _normalize_prompt(run_dir: Path, prompt: str, label: str) -> None:
    task_state_path = run_dir / "task_state.json"
    report_path = run_dir / "report.json"

    task_state = _read_json(task_state_path)
    task_state["user_request"] = prompt
    task_state["dpo_fixture_label"] = label
    _write_json(task_state_path, task_state)

    report = _read_json(report_path)
    prompt_metadata = report.setdefault("prompt_metadata", {})
    current_request = prompt_metadata.setdefault("current_request", {})
    current_request["text"] = prompt
    current_request["dpo_fixture_label"] = label
    report["dpo_fixture_note"] = "same-prompt fixture for DPO builder testing; not real training data"
    _write_json(report_path, report)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
