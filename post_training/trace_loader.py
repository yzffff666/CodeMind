"""Load Pico run artifacts into a compact trajectory summary.

This module is intentionally read-only: it does not modify Pico runtime state.
It turns `.pico/runs/<run_id>/` artifacts into a structure that later SFT,
DPO, reward, and summary builders can consume.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ToolCallSummary:
    name: str
    status: str
    error_code: str
    security_event: str
    args: dict[str, Any]
    result: str

    @classmethod
    def from_event(cls, event: dict[str, Any]) -> "ToolCallSummary":
        return cls(
            name=str(event.get("name", "")),
            status=str(event.get("tool_status", "")),
            error_code=str(event.get("tool_error_code", "")),
            security_event=str(event.get("security_event_type", "")),
            args=dict(event.get("args") or {}),
            result=str(event.get("result", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "error_code": self.error_code,
            "security_event": self.security_event,
            "args": self.args,
            "result": self.result,
        }


@dataclass(frozen=True)
class TrajectorySummary:
    run_id: str
    task_id: str
    user_request: str
    status: str
    stop_reason: str
    final_answer: str
    tool_steps: int
    attempts: int
    tool_calls: list[ToolCallSummary] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    security_events: list[str] = field(default_factory=list)
    event_counts: dict[str, int] = field(default_factory=dict)
    verifier_passed: bool | None = None
    benchmark_passed: bool | None = None
    benchmark_failure_category: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "user_request": self.user_request,
            "status": self.status,
            "stop_reason": self.stop_reason,
            "final_answer": self.final_answer,
            "tool_steps": self.tool_steps,
            "attempts": self.attempts,
            "tool_calls": [tool_call.to_dict() for tool_call in self.tool_calls],
            "errors": list(self.errors),
            "security_events": list(self.security_events),
            "event_counts": dict(self.event_counts),
            "verifier_passed": self.verifier_passed,
            "benchmark_passed": self.benchmark_passed,
            "benchmark_failure_category": self.benchmark_failure_category,
        }


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_trace_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events


def summarize_run(run_dir: str | Path, benchmark_index: dict[str, dict[str, Any]] | None = None) -> TrajectorySummary:
    run_path = Path(run_dir)
    report = load_json(run_path / "report.json")
    task_state = load_json(run_path / "task_state.json")
    events = load_trace_events(run_path / "trace.jsonl")

    tool_calls = [
        ToolCallSummary.from_event(event)
        for event in events
        if event.get("event") == "tool_executed"
    ]
    errors = sorted({tool.error_code for tool in tool_calls if tool.error_code})
    security_events = sorted({tool.security_event for tool in tool_calls if tool.security_event})

    event_counts: dict[str, int] = {}
    for event in events:
        event_name = str(event.get("event", ""))
        if event_name:
            event_counts[event_name] = event_counts.get(event_name, 0) + 1

    benchmark_row = (benchmark_index or {}).get(str(report.get("run_id", run_path.name)), {})

    return TrajectorySummary(
        run_id=str(report.get("run_id", run_path.name)),
        task_id=str(report.get("task_id", "")),
        user_request=_user_request(report, task_state),
        status=str(report.get("status", "")),
        stop_reason=str(report.get("stop_reason", "")),
        final_answer=str(report.get("final_answer", "")),
        tool_steps=int(report.get("tool_steps", 0) or 0),
        attempts=int(report.get("attempts", 0) or 0),
        tool_calls=tool_calls,
        errors=errors,
        security_events=security_events,
        event_counts=event_counts,
        verifier_passed=_optional_bool(benchmark_row.get("verifier_passed")),
        benchmark_passed=_optional_bool(benchmark_row.get("passed")),
        benchmark_failure_category=str(benchmark_row.get("failure_category") or ""),
    )


def _user_request(report: dict[str, Any], task_state: dict[str, Any]) -> str:
    task_request = str(task_state.get("user_request", "")).strip()
    if task_request:
        return task_request
    prompt_metadata = report.get("prompt_metadata") or {}
    current_request = prompt_metadata.get("current_request") or {}
    return str(current_request.get("text", "")).strip()


def find_run_dirs(root: str | Path) -> list[Path]:
    root_path = Path(root)
    return sorted(
        path
        for path in root_path.rglob("*")
        if path.is_dir()
        and (path / "trace.jsonl").is_file()
        and (path / "report.json").is_file()
        and (path / "task_state.json").is_file()
    )


def discover_benchmark_artifacts(root: str | Path) -> list[Path]:
    artifacts: list[Path] = []
    for path in Path(root).rglob("*.json"):
        try:
            payload = load_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
            artifacts.append(path)
    return sorted(artifacts)


def load_benchmark_index(paths: list[str | Path] | tuple[str | Path, ...]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for path in paths:
        payload = load_json(Path(path))
        for row in payload.get("rows", []):
            run_id = str(row.get("run_id", "")).strip()
            if run_id:
                index[run_id] = dict(row)
    return index


def _optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None
