"""Rule-based reward signals for Pico trajectories.

These scores are not a learned reward model. They are transparent heuristics
for filtering SFT candidates and building simple DPO preference pairs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .trace_loader import TrajectorySummary


@dataclass(frozen=True)
class RewardResult:
    score: float
    signals: dict[str, Any]
    penalties: dict[str, float]
    bonuses: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "signals": dict(self.signals),
            "penalties": dict(self.penalties),
            "bonuses": dict(self.bonuses),
        }


def score_trajectory(summary: TrajectorySummary) -> RewardResult:
    rejected_tool_calls = sum(1 for call in summary.tool_calls if call.status == "rejected")
    invalid_arguments = sum(1 for call in summary.tool_calls if call.error_code == "invalid_arguments")
    failed_tool_calls = sum(
        1
        for call in summary.tool_calls
        if call.status in {"error", "partial_success"}
        or call.error_code in {"tool_failed", "tool_partial_success"}
    )
    path_escape = sum(1 for call in summary.tool_calls if call.security_event == "path_escape")
    security_events = sum(1 for call in summary.tool_calls if call.security_event)
    completed = summary.status == "completed"
    final_answer_returned = summary.stop_reason == "final_answer_returned"
    has_final_answer = bool(summary.final_answer.strip())
    verifier_passed = summary.verifier_passed
    completion_without_final = verifier_passed is True and not final_answer_returned
    task_failure = verifier_passed is False
    protocol_failure = completion_without_final
    tool_failure = rejected_tool_calls > 0 or invalid_arguments > 0 or failed_tool_calls > 0
    safety_failure = security_events > 0
    quality_label = _quality_label(
        verifier_passed=verifier_passed,
        completed=completed,
        protocol_failure=protocol_failure,
        task_failure=task_failure,
        tool_failure=tool_failure,
        safety_failure=safety_failure,
    )

    bonuses = {
        "completed": 1.0 if completed else 0.0,
        "final_answer_returned": 0.3 if final_answer_returned else 0.0,
        "verifier_passed": 0.4 if verifier_passed is True else 0.0,
    }
    penalties = {
        "tool_steps": -0.05 * summary.tool_steps,
        "rejected_tool_calls": -0.3 * rejected_tool_calls,
        "invalid_arguments": -0.3 * invalid_arguments,
        "failed_tool_calls": -0.3 * failed_tool_calls,
        "path_escape": -0.6 * path_escape,
        "missing_final_answer": 0.0 if has_final_answer else -0.5,
        "verifier_failed": -0.8 if verifier_passed is False else 0.0,
        "completion_without_final": -0.4 if completion_without_final else 0.0,
    }

    score = sum(bonuses.values()) + sum(penalties.values())
    signals = {
        "completed": completed,
        "final_answer_returned": final_answer_returned,
        "has_final_answer": has_final_answer,
        "verifier_passed": verifier_passed,
        "benchmark_passed": summary.benchmark_passed,
        "benchmark_failure_category": summary.benchmark_failure_category,
        "completion_without_final": completion_without_final,
        "task_failure": task_failure,
        "protocol_failure": protocol_failure,
        "tool_failure": tool_failure,
        "safety_failure": safety_failure,
        "quality_label": quality_label,
        "tool_steps": summary.tool_steps,
        "attempts": summary.attempts,
        "rejected_tool_calls": rejected_tool_calls,
        "invalid_arguments": invalid_arguments,
        "failed_tool_calls": failed_tool_calls,
        "security_events": security_events,
        "path_escape": path_escape,
        "errors": list(summary.errors),
        "security_event_types": list(summary.security_events),
    }
    return RewardResult(score=score, signals=signals, penalties=penalties, bonuses=bonuses)


def _quality_label(
    *,
    verifier_passed: bool | None,
    completed: bool,
    protocol_failure: bool,
    task_failure: bool,
    tool_failure: bool,
    safety_failure: bool,
) -> str:
    if safety_failure:
        return "safety_failure"
    if tool_failure:
        return "tool_failure"
    if protocol_failure:
        return "protocol_failure"
    if task_failure:
        return "task_failure"
    if completed and verifier_passed is not False:
        return "success"
    return "runtime_failure"
