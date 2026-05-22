"""Cross-dataset comparison reports for Pico post-training experiments."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .badcases import build_badcase_report
from .summary import summarize_post_training
from .trace_loader import (
    discover_benchmark_artifacts,
    find_run_dirs,
    load_benchmark_index,
    summarize_run,
)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    root: Path
    benchmark_artifacts: tuple[Path, ...] = ()


@dataclass(frozen=True)
class ComparisonReport:
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload)


def compare_datasets(
    datasets: list[DatasetSpec],
    min_sft_reward: float = 1.0,
    min_dpo_gap: float = 0.5,
) -> ComparisonReport:
    rows = [
        _summarize_dataset(
            dataset,
            min_sft_reward=min_sft_reward,
            min_dpo_gap=min_dpo_gap,
        )
        for dataset in datasets
    ]
    return ComparisonReport(
        payload={
            "datasets": rows,
            "settings": {
                "min_sft_reward": min_sft_reward,
                "min_dpo_gap": min_dpo_gap,
            },
        }
    )


def render_markdown(report: ComparisonReport) -> str:
    rows = report.to_dict()["datasets"]
    lines = [
        "# Pico Post-Training Dataset Comparison",
        "",
        "## Overview",
        "",
        "| Dataset | Runs | Benchmark Pass | Verifier Pass | Avg Steps | Avg Reward | SFT Candidates | DPO Pairs |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {name} | {runs} | {benchmark_pass_rate} | {verifier_pass_rate} | "
            "{avg_tool_steps} | {avg_reward} | {sft_candidates} | {dpo_pairs} |".format(**row)
        )

    lines.extend(["", "## Quality Labels", ""])
    for row in rows:
        lines.append(f"### {row['name']}")
        lines.append("")
        lines.extend(_format_counts(row["quality_label_counts"]))
        lines.append("")

    lines.extend(["## Failure Subtypes", ""])
    for row in rows:
        lines.append(f"### {row['name']}")
        lines.append("")
        lines.extend(_format_counts(row["failure_subtype_counts"]))
        lines.append("")

    lines.extend(["## Reading Guide", ""])
    lines.extend(
        [
            "- Benchmark pass rate answers whether the task outcome is correct.",
            "- Verifier pass rate answers whether an external checker confirms that outcome.",
            "- SFT candidates show how many traces are clean enough to imitate directly.",
            "- DPO pairs show whether repeated sampling already exposes useful preference signals.",
        ]
    )
    return "\n".join(lines)


def _summarize_dataset(
    dataset: DatasetSpec,
    min_sft_reward: float,
    min_dpo_gap: float,
) -> dict[str, Any]:
    summary = summarize_post_training(
        dataset.root,
        min_sft_reward=min_sft_reward,
        min_dpo_gap=min_dpo_gap,
        benchmark_artifacts=dataset.benchmark_artifacts or None,
    ).to_dict()
    badcases = build_badcase_report(
        dataset.root,
        benchmark_artifacts=dataset.benchmark_artifacts or None,
    ).to_dict()
    benchmark_pass_rate, avg_tool_steps = _runtime_metrics(dataset)
    verifier_counts = summary["verifier_counts"]
    verifier_pass_rate = _rate(verifier_counts.get("passed", 0), summary["runs_scanned"])
    return {
        "name": dataset.name,
        "runs": summary["runs_scanned"],
        "benchmark_pass_rate": benchmark_pass_rate,
        "verifier_pass_rate": verifier_pass_rate,
        "avg_tool_steps": avg_tool_steps,
        "avg_reward": summary["reward"]["avg"],
        "sft_candidates": summary["sft_candidates"],
        "dpo_pairs": summary["dpo_pairs"],
        "quality_label_counts": summary["quality_label_counts"],
        "failure_subtype_counts": badcases["subtype_counts"],
    }


def _runtime_metrics(dataset: DatasetSpec) -> tuple[str, float]:
    artifact_paths = (
        list(dataset.benchmark_artifacts)
        if dataset.benchmark_artifacts
        else discover_benchmark_artifacts(dataset.root)
    )
    benchmark_index = (
        load_benchmark_index(artifact_paths)
        if artifact_paths
        else {}
    )
    summaries = [
        summarize_run(run_dir, benchmark_index=benchmark_index)
        for run_dir in find_run_dirs(dataset.root)
    ]
    benchmark_counts = Counter(summary.benchmark_passed for summary in summaries)
    benchmark_pass_rate = _rate(benchmark_counts.get(True, 0), len(summaries))
    avg_tool_steps = round(
        sum(summary.tool_steps for summary in summaries) / len(summaries),
        4,
    ) if summaries else 0.0
    return benchmark_pass_rate, avg_tool_steps


def _rate(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "0.0%"
    return f"{(numerator / denominator) * 100:.1f}%"


def _format_counts(counts: dict[str, int]) -> list[str]:
    if not counts:
        return ["- none"]
    return [f"- {key}: {value}" for key, value in sorted(counts.items())]
