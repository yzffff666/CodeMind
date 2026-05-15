"""Representative badcase reporting for Pico post-training datasets."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .dpo_builder import ScoredTrajectory
from .reward import score_trajectory
from .trace_loader import discover_benchmark_artifacts, find_run_dirs, load_benchmark_index, summarize_run


@dataclass(frozen=True)
class BadcaseReport:
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload)


def build_badcase_report(
    root: str | Path,
    benchmark_artifacts: list[str | Path] | tuple[str | Path, ...] | None = None,
) -> BadcaseReport:
    artifact_paths = list(benchmark_artifacts or discover_benchmark_artifacts(root))
    benchmark_index = load_benchmark_index(artifact_paths) if artifact_paths else {}
    scored: list[ScoredTrajectory] = []
    by_label: dict[str, list[ScoredTrajectory]] = defaultdict(list)
    subtype_counts: Counter[str] = Counter()

    for run_dir in find_run_dirs(root):
        summary = summarize_run(run_dir, benchmark_index=benchmark_index)
        item = ScoredTrajectory(summary=summary, reward=score_trajectory(summary))
        scored.append(item)
        label = str(item.reward.signals["quality_label"])
        by_label[label].append(item)
        subtype_counts.update(_subtypes(item))

    return BadcaseReport(
        payload={
            "runs_scanned": len(scored),
            "benchmark_artifacts": [str(path) for path in artifact_paths],
            "quality_label_counts": {label: len(items) for label, items in sorted(by_label.items())},
            "subtype_counts": dict(sorted(subtype_counts.items())),
            "examples": {
                label: [_example(item) for item in _representatives(items)]
                for label, items in sorted(by_label.items())
            },
        }
    )


def render_markdown(report: BadcaseReport) -> str:
    data = report.to_dict()
    lines = [
        "# Pico Badcase Report",
        "",
        f"- Runs scanned: {data['runs_scanned']}",
        "",
        "## Quality Labels",
        "",
    ]
    lines.extend(_format_counts(data["quality_label_counts"]))
    lines.extend(["", "## Failure Subtypes", ""])
    lines.extend(_format_counts(data["subtype_counts"]))
    lines.extend(["", "## Representative Examples", ""])
    for label, examples in data["examples"].items():
        lines.append(f"### {label}")
        lines.append("")
        for example in examples:
            lines.append(
                f"- run={example['run_id']} score={example['score']} "
                f"verifier={example['verifier_passed']} stop={example['stop_reason']} "
                f"errors={example['errors']} security={example['security_events']}"
            )
            lines.append(f"  prompt={example['prompt']}")
            lines.append(f"  tool_path={example['tool_path']}")
        lines.append("")
    return "\n".join(lines)


def _subtypes(item: ScoredTrajectory) -> list[str]:
    signals = item.reward.signals
    subtypes = []
    if signals.get("completion_without_final"):
        subtypes.append("completion_without_final")
    if signals.get("task_failure"):
        subtypes.append("verifier_failed")
    if signals.get("invalid_arguments"):
        subtypes.append("invalid_arguments")
    if signals.get("failed_tool_calls"):
        subtypes.append("tool_failed")
    if "repeated_identical_call" in signals.get("errors", []):
        subtypes.append("repeated_identical_call")
    if signals.get("path_escape"):
        subtypes.append("path_escape")
    return subtypes


def _representatives(items: list[ScoredTrajectory]) -> list[ScoredTrajectory]:
    return sorted(items, key=lambda item: (item.reward.score, item.summary.run_id))[:2]


def _example(item: ScoredTrajectory) -> dict[str, Any]:
    summary = item.summary
    return {
        "run_id": summary.run_id,
        "score": round(item.reward.score, 4),
        "prompt": summary.user_request,
        "stop_reason": summary.stop_reason,
        "verifier_passed": summary.verifier_passed,
        "errors": list(summary.errors),
        "security_events": list(summary.security_events),
        "tool_path": " -> ".join(call.name for call in summary.tool_calls) or "(no tools)",
    }


def _format_counts(counts: dict[str, int]) -> list[str]:
    if not counts:
        return ["- none"]
    return [f"- {key}: {value}" for key, value in sorted(counts.items())]
