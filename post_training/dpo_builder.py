"""Build DPO preference records from scored Pico trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .reward import RewardResult, score_trajectory
from .sft_builder import serialize_tool_call
from .trace_loader import TrajectorySummary


DEFAULT_MIN_REWARD_GAP = 0.5


@dataclass(frozen=True)
class ScoredTrajectory:
    summary: TrajectorySummary
    reward: RewardResult


@dataclass(frozen=True)
class DPORecord:
    prompt: str
    chosen: str
    rejected: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "metadata": dict(self.metadata),
        }


def score(summary: TrajectorySummary) -> ScoredTrajectory:
    return ScoredTrajectory(summary=summary, reward=score_trajectory(summary))


def build_dpo_record(
    chosen: ScoredTrajectory,
    rejected: ScoredTrajectory,
    reason: str = "higher_reward",
) -> DPORecord:
    reward_gap = chosen.reward.score - rejected.reward.score
    return DPORecord(
        prompt=chosen.summary.user_request,
        chosen=serialize_trajectory_response(chosen.summary),
        rejected=serialize_trajectory_response(rejected.summary),
        metadata={
            "reason": reason,
            "reward_gap": round(reward_gap, 4),
            "chosen_run_id": chosen.summary.run_id,
            "rejected_run_id": rejected.summary.run_id,
            "chosen_reward": chosen.reward.to_dict(),
            "rejected_reward": rejected.reward.to_dict(),
        },
    )


def build_pairs_for_prompt(
    trajectories: list[ScoredTrajectory],
    min_reward_gap: float = DEFAULT_MIN_REWARD_GAP,
) -> list[DPORecord]:
    if len(trajectories) < 2:
        return []
    ordered = sorted(trajectories, key=lambda item: item.reward.score, reverse=True)
    best = ordered[0]
    pairs = []
    for candidate in ordered[1:]:
        if best.reward.score - candidate.reward.score >= min_reward_gap:
            pairs.append(build_dpo_record(best, candidate))
    return pairs


def serialize_trajectory_response(summary: TrajectorySummary) -> str:
    parts: list[str] = []
    for tool_call in summary.tool_calls:
        parts.append(serialize_tool_call(tool_call))
        if tool_call.result:
            parts.append(f"<observation>{tool_call.result}</observation>")
    if summary.final_answer:
        parts.append(f"<final>{summary.final_answer}</final>")
    return "\n".join(parts)
