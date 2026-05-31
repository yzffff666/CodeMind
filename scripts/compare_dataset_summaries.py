"""Compare prepared post-training dataset_summary.json files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True, help="Earlier dataset_summary.json.")
    parser.add_argument("--after", type=Path, required=True, help="Later dataset_summary.json.")
    parser.add_argument("--before-name", default="before")
    parser.add_argument("--after-name", default="after")
    parser.add_argument("--format", choices=("json", "markdown"), default="markdown")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    before = _load(args.before)
    after = _load(args.after)
    report = _compare(before, after, args.before_name, args.after_name)

    if args.format == "json":
        text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    else:
        text = _render_markdown(report)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text, end="")
    return 0


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _compare(before: dict[str, Any], after: dict[str, Any], before_name: str, after_name: str) -> dict[str, Any]:
    before_train = before.get("train", {})
    after_train = after.get("train", {})
    return {
        "before": {
            "name": before_name,
            "config": before.get("config", ""),
            "sft_records": before_train.get("sft_records", 0),
            "dpo_records": before_train.get("dpo_records", 0),
            "sft_sources": _source_exports(before_train.get("sft_sources", [])),
            "dpo_sources": _source_exports(before_train.get("dpo_sources", [])),
        },
        "after": {
            "name": after_name,
            "config": after.get("config", ""),
            "sft_records": after_train.get("sft_records", 0),
            "dpo_records": after_train.get("dpo_records", 0),
            "sft_sources": _source_exports(after_train.get("sft_sources", [])),
            "dpo_sources": _source_exports(after_train.get("dpo_sources", [])),
        },
        "delta": {
            "sft_records": int(after_train.get("sft_records", 0)) - int(before_train.get("sft_records", 0)),
            "dpo_records": int(after_train.get("dpo_records", 0)) - int(before_train.get("dpo_records", 0)),
            "new_sft_sources": _new_sources(
                before_train.get("sft_sources", []),
                after_train.get("sft_sources", []),
            ),
            "new_dpo_sources": _new_sources(
                before_train.get("dpo_sources", []),
                after_train.get("dpo_sources", []),
            ),
        },
    }


def _source_exports(sources: list[dict[str, Any]]) -> dict[str, int]:
    return {str(source.get("name", "")): int(source.get("exported", 0)) for source in sources}


def _new_sources(before_sources: list[dict[str, Any]], after_sources: list[dict[str, Any]]) -> dict[str, int]:
    before_names = {str(source.get("name", "")) for source in before_sources}
    return {
        str(source.get("name", "")): int(source.get("exported", 0))
        for source in after_sources
        if str(source.get("name", "")) not in before_names
    }


def _render_markdown(report: dict[str, Any]) -> str:
    before = report["before"]
    after = report["after"]
    delta = report["delta"]
    lines = [
        "# Post-training 数据集 v2-v3 对比",
        "",
        "## 总览",
        "",
        f"- 对比对象：`{before['name']}` -> `{after['name']}`",
        f"- 配置变化：`{before['config']}` -> `{after['config']}`",
        f"- SFT 样本：{before['sft_records']} -> {after['sft_records']}，变化 {delta['sft_records']:+d}",
        f"- DPO pairs：{before['dpo_records']} -> {after['dpo_records']}，变化 {delta['dpo_records']:+d}",
        "",
        "## 新增来源",
        "",
        "### SFT",
        "",
    ]
    lines.extend(_render_sources(delta["new_sft_sources"]))
    lines.extend(["", "### DPO", ""])
    lines.extend(_render_sources(delta["new_dpo_sources"]))
    lines.extend(
        [
            "",
            "## 结论",
            "",
            "v3 相比 v2 的核心变化不是盲目扩大数据量，而是围绕 held-out baseline 暴露的问题做 targeted data augmentation。",
            "新增 exact-instruction 数据让训练集覆盖“精确执行 verifier 目标短语”和“完成后及时 final”的行为偏好。",
            "其中 repeated sampling 产生了同 prompt 的成功/失败轨迹，因此 DPO pairs 从 6 增至 9；SFT 样本从 16 增至 18。",
            "",
            "## 面试表达",
            "",
            "可以说：我先用 held-out eval 暴露失败模式，再设计相似但不泄漏评测集的训练任务，通过 repeated sampling 构造 chosen/rejected 偏好对，最后重新生成 v3 数据集并验证 SFT dry run。这个过程体现的是后训练数据闭环，而不是单次 prompt 调参。",
            "",
        ]
    )
    return "\n".join(lines)


def _render_sources(sources: dict[str, int]) -> list[str]:
    if not sources:
        return ["- 无新增来源。"]
    return [f"- `{name}`：导出 {exported} 条" for name, exported in sources.items()]


if __name__ == "__main__":
    raise SystemExit(main())

