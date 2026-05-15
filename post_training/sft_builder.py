"""Build SFT-style message records from Pico trajectory summaries."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .reward import RewardResult, score_trajectory
from .trace_loader import ToolCallSummary, TrajectorySummary


DEFAULT_MIN_REWARD = 1.0


@dataclass(frozen=True)
class SFTRecord:
    messages: list[dict[str, str]]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "messages": list(self.messages),
            "metadata": dict(self.metadata),
        }


def is_sft_candidate(summary: TrajectorySummary, reward: RewardResult, min_reward: float = DEFAULT_MIN_REWARD) -> bool:
    signals = reward.signals
    return (
        bool(summary.user_request.strip())
        and signals.get("completed") is True
        and signals.get("has_final_answer") is True
        and int(signals.get("security_events", 0)) == 0
        and summary.tool_calls
        and reward.score >= min_reward
    )


def build_sft_record(summary: TrajectorySummary, reward: RewardResult | None = None) -> SFTRecord:
    reward = reward or score_trajectory(summary)
    messages = [{"role": "user", "content": summary.user_request}]
    for tool_call in summary.tool_calls:
        messages.append({"role": "assistant", "content": serialize_tool_call(tool_call)})
        messages.append({"role": "tool", "content": tool_call.result})
    messages.append({"role": "assistant", "content": f"<final>{summary.final_answer}</final>"})

    metadata = {
        "source": "pico_trace",
        "run_id": summary.run_id,
        "task_id": summary.task_id,
        "status": summary.status,
        "stop_reason": summary.stop_reason,
        "tool_steps": summary.tool_steps,
        "attempts": summary.attempts,
        "reward": reward.to_dict(),
        "sft_policy": {
            "min_reward": DEFAULT_MIN_REWARD,
            "requires_completed": True,
            "excludes_security_events": True,
        },
    }
    return SFTRecord(messages=messages, metadata=metadata)


def serialize_tool_call(tool_call: ToolCallSummary) -> str:
    if tool_call.args:
        attrs = " ".join(f'{key}="{_escape_attr(value)}"' for key, value in sorted(tool_call.args.items()))
        return f'<tool name="{_escape_attr(tool_call.name)}" {attrs}></tool>'
    return f'<tool name="{_escape_attr(tool_call.name)}"></tool>'


def _escape_attr(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
    return (
        text.replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
