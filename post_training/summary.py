"""Dataset quality summaries for Pico post-training artifacts."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .dpo_builder import ScoredTrajectory, build_pairs_for_prompt
from .reward import score_trajectory
from .sft_builder import is_sft_candidate
from .trace_loader import discover_benchmark_artifacts, find_run_dirs, load_benchmark_index, summarize_run


@dataclass(frozen=True)
class PostTrainingSummary:
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload)


def summarize_post_training(
    root: str | Path,
    min_sft_reward: float = 1.0,
    min_dpo_gap: float = 0.5,
    benchmark_artifacts: list[str | Path] | tuple[str | Path, ...] | None = None,
) -> PostTrainingSummary:
    scored: list[ScoredTrajectory] = []
    error_counts: Counter[str] = Counter()
    security_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    stop_reason_counts: Counter[str] = Counter()
    tool_name_counts: Counter[str] = Counter()
    quality_label_counts: Counter[str] = Counter()
    verifier_counts: Counter[str] = Counter()
    prompt_counts: Counter[str] = Counter()
    by_prompt: dict[str, list[ScoredTrajectory]] = defaultdict(list)
    artifact_paths = list(benchmark_artifacts or discover_benchmark_artifacts(root))
    benchmark_index = load_benchmark_index(artifact_paths) if artifact_paths else {}

    for run_dir in find_run_dirs(root):
        summary = summarize_run(run_dir, benchmark_index=benchmark_index)
        reward = score_trajectory(summary)
        item = ScoredTrajectory(summary=summary, reward=reward)
        scored.append(item)
        if summary.user_request:
            prompt_counts[summary.user_request] += 1
            by_prompt[summary.user_request].append(item)
        status_counts[summary.status] += 1
        stop_reason_counts[summary.stop_reason] += 1
        error_counts.update(summary.errors)
        security_counts.update(summary.security_events)
        tool_name_counts.update(call.name for call in summary.tool_calls if call.name)
        quality_label_counts[str(reward.signals["quality_label"])] += 1
        verifier_counts[_verifier_key(summary.verifier_passed)] += 1

    rewards = [item.reward.score for item in scored]
    sft_candidates = [
        item
        for item in scored
        if is_sft_candidate(item.summary, item.reward, min_reward=min_sft_reward)
    ]
    dpo_pairs = []
    for trajectories in by_prompt.values():
        dpo_pairs.extend(build_pairs_for_prompt(trajectories, min_reward_gap=min_dpo_gap))

    payload = {
        "runs_scanned": len(scored),
        "sft_candidates": len(sft_candidates),
        "dpo_pairs": len(dpo_pairs),
        "same_prompt_groups": sum(1 for items in by_prompt.values() if len(items) >= 2),
        "unique_prompts": len(prompt_counts),
        "reward": {
            "min": round(min(rewards), 4) if rewards else 0.0,
            "max": round(max(rewards), 4) if rewards else 0.0,
            "avg": round(sum(rewards) / len(rewards), 4) if rewards else 0.0,
        },
        "status_counts": dict(status_counts),
        "stop_reason_counts": dict(stop_reason_counts),
        "error_counts": dict(error_counts),
        "security_event_counts": dict(security_counts),
        "tool_name_counts": dict(tool_name_counts),
        "quality_label_counts": dict(quality_label_counts),
        "verifier_counts": dict(verifier_counts),
        "benchmark_artifacts": [str(path) for path in artifact_paths],
        "top_runs": [
            _ranked_run(item)
            for item in sorted(scored, key=lambda value: value.reward.score, reverse=True)[:5]
        ],
        "bottom_runs": [
            _ranked_run(item)
            for item in sorted(scored, key=lambda value: value.reward.score)[:5]
        ],
        "settings": {
            "min_sft_reward": min_sft_reward,
            "min_dpo_gap": min_dpo_gap,
        },
    }
    return PostTrainingSummary(payload=payload)


def render_markdown(summary: PostTrainingSummary) -> str:
    data = summary.to_dict()
    reward = data["reward"]
    lines = [
        "# Pico Post-Training Data Summary",
        "",
        "## Overview",
        "",
        f"- Runs scanned: {data['runs_scanned']}",
        f"- SFT candidates: {data['sft_candidates']}",
        f"- DPO pairs: {data['dpo_pairs']}",
        f"- Unique prompts: {data['unique_prompts']}",
        f"- Same-prompt groups: {data['same_prompt_groups']}",
        f"- Reward range: {reward['min']} to {reward['max']} (avg {reward['avg']})",
        "",
        "## Counts",
        "",
        _format_counts("Status", data["status_counts"]),
        _format_counts("Stop reasons", data["stop_reason_counts"]),
        _format_counts("Tool names", data["tool_name_counts"]),
        _format_counts("Verifier", data["verifier_counts"]),
        _format_counts("Quality labels", data["quality_label_counts"]),
        _format_counts("Errors", data["error_counts"]),
        _format_counts("Security events", data["security_event_counts"]),
        "",
        "## Highest Reward Runs",
        "",
        _format_runs(data["top_runs"]),
        "",
        "## Lowest Reward Runs",
        "",
        _format_runs(data["bottom_runs"]),
        "",
    ]
    return "\n".join(lines)


def _ranked_run(item: ScoredTrajectory) -> dict[str, Any]:
    summary = item.summary
    return {
        "run_id": summary.run_id,
        "score": round(item.reward.score, 4),
        "tool_steps": summary.tool_steps,
        "errors": list(summary.errors),
        "security_events": list(summary.security_events),
        "prompt": summary.user_request,
    }


def _format_counts(title: str, counts: dict[str, int]) -> str:
    if not counts:
        return f"### {title}\n\n- none\n"
    lines = [f"### {title}", ""]
    lines.extend(f"- {key}: {value}" for key, value in sorted(counts.items()))
    return "\n".join(lines) + "\n"


def _format_runs(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "- none"
    lines = []
    for row in rows:
        prompt = str(row["prompt"]).replace("\n", " ")
        if len(prompt) > 100:
            prompt = prompt[:97] + "..."
        lines.append(
            f"- score={row['score']} steps={row['tool_steps']} run={row['run_id']} "
            f"errors={row['errors']} security={row['security_events']} prompt={prompt}"
        )
    return "\n".join(lines)


def _verifier_key(value: bool | None) -> str:
    if value is True:
        return "passed"
    if value is False:
        return "failed"
    return "unknown"
