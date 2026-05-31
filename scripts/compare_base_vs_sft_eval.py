"""Compare base and SFT held-out benchmark artifacts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/base_vs_sft_eval.json"))
    parser.add_argument("--base-artifact", type=Path)
    parser.add_argument("--sft-artifact", type=Path)
    parser.add_argument("--base-name")
    parser.add_argument("--sft-name")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--format", choices=("json", "markdown"), default="markdown")
    args = parser.parse_args()

    config = _load_config(args.config)
    base_path = args.base_artifact or Path(config["base"]["artifact"])
    sft_path = args.sft_artifact or Path(config["sft"]["artifact"])
    base_name = args.base_name or str(config["base"].get("name", "base"))
    sft_name = args.sft_name or str(config["sft"].get("name", "sft"))

    report = _compare(
        _summarize_artifact(base_name, base_path),
        _summarize_artifact(sft_name, sft_path),
        config=config,
    )

    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n" if args.format == "json" else _render_markdown(report)
    out_path = args.out or _default_output_path(config, args.format)
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        print(f"wrote {out_path}")
    else:
        print(text, end="")
    return 0


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if int(config.get("schema_version", 0)) != 1:
        raise ValueError("unsupported config schema_version")
    for key in ("base", "sft"):
        if key not in config:
            raise ValueError(f"config missing required section: {key}")
    return config


def _summarize_artifact(name: str, path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = list(payload.get("rows", []))
    total = len(rows)
    passed = sum(1 for row in rows if row.get("passed") is True)
    verifier_passed = sum(1 for row in rows if row.get("verifier_passed") is True)
    final_answer = sum(1 for row in rows if row.get("stop_reason") == "final_answer_returned")
    rewards = [_reward_for_row(row) for row in rows]
    tool_steps = [int(row.get("tool_steps", 0) or 0) for row in rows]
    return {
        "name": name,
        "artifact": str(path),
        "total_tasks": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": _rate(passed, total),
        "verifier_pass_rate": _rate(verifier_passed, total),
        "final_answer_rate": _rate(final_answer, total),
        "avg_reward": round(sum(rewards) / total, 4) if total else 0.0,
        "avg_tool_steps": round(sum(tool_steps) / total, 4) if total else 0.0,
        "failure_category_counts": dict(Counter(str(row.get("failure_category") or "pass") for row in rows)),
        "quality_label_counts": dict(Counter(_quality_label(row) for row in rows)),
        "task_rows": [
            {
                "id": row.get("id", ""),
                "passed": row.get("passed") is True,
                "verifier_passed": row.get("verifier_passed"),
                "stop_reason": row.get("stop_reason", ""),
                "failure_category": row.get("failure_category") or "",
                "reward": _reward_for_row(row),
                "tool_steps": int(row.get("tool_steps", 0) or 0),
            }
            for row in rows
        ],
    }


def _compare(base: dict[str, Any], sft: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    metric_keys = ("pass_rate", "verifier_pass_rate", "final_answer_rate", "avg_reward", "avg_tool_steps")
    delta = {key: round(float(sft[key]) - float(base[key]), 4) for key in metric_keys}
    return {
        "schema_version": 1,
        "config_description": config.get("description", ""),
        "base": base,
        "sft": sft,
        "delta": delta,
        "notes": list(config.get("notes", [])),
    }


def _reward_for_row(row: dict[str, Any]) -> float:
    report = row.get("report") or {}
    reward = report.get("reward") or row.get("reward") or {}
    if isinstance(reward, dict) and isinstance(reward.get("score"), (int, float)):
        return round(float(reward["score"]), 4)
    # Fallback mirrors the transparent reward intuition for benchmark rows.
    score = 0.0
    if row.get("passed") is True:
        score += 1.0
    if row.get("stop_reason") == "final_answer_returned":
        score += 0.3
    if row.get("verifier_passed") is True:
        score += 0.4
    if row.get("verifier_passed") is False:
        score -= 0.8
    score -= 0.05 * int(row.get("tool_steps", 0) or 0)
    if row.get("verifier_passed") is True and row.get("stop_reason") != "final_answer_returned":
        score -= 0.4
    return round(score, 4)


def _quality_label(row: dict[str, Any]) -> str:
    if row.get("passed") is True:
        return "success"
    if row.get("verifier_passed") is False:
        return "task_failure"
    if row.get("verifier_passed") is True and row.get("stop_reason") != "final_answer_returned":
        return "protocol_failure"
    return "runtime_failure"


def _rate(count: int, total: int) -> float:
    return round(count / total, 4) if total else 0.0


def _default_output_path(config: dict[str, Any], fmt: str) -> Path | None:
    output = config.get("output") or {}
    key = "json" if fmt == "json" else "markdown"
    path = output.get(key)
    return Path(path) if path else None


def _render_markdown(report: dict[str, Any]) -> str:
    base = report["base"]
    sft = report["sft"]
    delta = report["delta"]
    lines = [
        "# Base-vs-SFT Held-out Eval 对比",
        "",
        "## 总览",
        "",
        f"- Base：`{base['name']}` (`{base['artifact']}`)",
        f"- SFT：`{sft['name']}` (`{sft['artifact']}`)",
        f"- 任务数：{base['total_tasks']} -> {sft['total_tasks']}",
        "",
        "## 指标",
        "",
        "| 指标 | Base | SFT | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key, label in [
        ("pass_rate", "Pass rate"),
        ("verifier_pass_rate", "Verifier pass rate"),
        ("final_answer_rate", "Final-answer rate"),
        ("avg_reward", "Avg reward"),
        ("avg_tool_steps", "Avg tool steps"),
    ]:
        lines.append(f"| {label} | {base[key]} | {sft[key]} | {delta[key]:+g} |")
    lines.extend(
        [
            "",
            "## 失败类型",
            "",
            f"- Base failure categories：`{base['failure_category_counts']}`",
            f"- SFT failure categories：`{sft['failure_category_counts']}`",
            f"- Base quality labels：`{base['quality_label_counts']}`",
            f"- SFT quality labels：`{sft['quality_label_counts']}`",
            "",
            "## 任务级对比",
            "",
            "| Task | Base passed | SFT passed | Base stop | SFT stop | Base reward | SFT reward |",
            "| --- | ---: | ---: | --- | --- | ---: | ---: |",
        ]
    )
    sft_by_id = {str(row["id"]): row for row in sft["task_rows"]}
    for base_row in base["task_rows"]:
        task_id = str(base_row["id"])
        sft_row = sft_by_id.get(task_id, {})
        lines.append(
            "| {task} | {base_passed} | {sft_passed} | {base_stop} | {sft_stop} | {base_reward} | {sft_reward} |".format(
                task=task_id,
                base_passed=base_row.get("passed"),
                sft_passed=sft_row.get("passed", ""),
                base_stop=base_row.get("stop_reason", ""),
                sft_stop=sft_row.get("stop_reason", ""),
                base_reward=base_row.get("reward", ""),
                sft_reward=sft_row.get("reward", ""),
            )
        )
    lines.extend(
        [
            "",
            "## 解释原则",
            "",
            "- 如果 loss 下降但 held-out pass rate、final-answer rate 或 reward 没有改善，不能声称行为提升。",
            "- 如果 SFT 改善了 final-answer rate，但 verifier pass 下降，需要继续看 badcase 是否发生 error migration。",
            "- held-out trajectories 只用于评估，不进入训练集。",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())

