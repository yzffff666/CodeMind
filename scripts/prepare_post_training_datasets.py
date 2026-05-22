"""Prepare first-version SFT/DPO train files and an eval manifest from a split config."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from post_training.dpo_builder import ScoredTrajectory, build_pairs_for_prompt
from post_training.reward import score_trajectory
from post_training.sft_builder import build_sft_record, is_sft_candidate
from post_training.summary import summarize_post_training
from post_training.trace_loader import (
    discover_benchmark_artifacts,
    find_run_dirs,
    load_benchmark_index,
    summarize_run,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "post_training_split.json",
        help="Train/eval split config.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "datasets" / "v1",
        help="Output directory for train_sft.jsonl, train_dpo.jsonl, and eval_manifest.json.",
    )
    args = parser.parse_args()

    config = _load_config(args.config)
    settings = config.get("settings", {})
    min_sft_reward = float(settings.get("min_sft_reward", 1.0))
    min_dpo_gap = float(settings.get("min_dpo_gap", 0.5))

    sft_records, sft_manifest = _collect_sft_records(
        config.get("train", {}).get("sft_sources", []),
        min_reward=min_sft_reward,
    )
    dpo_records, dpo_manifest = _collect_dpo_records(
        config.get("train", {}).get("dpo_sources", []),
        min_reward_gap=min_dpo_gap,
    )
    eval_manifest = _build_eval_manifest(config.get("eval", {}).get("benchmarks", []))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sft_path = args.out_dir / "train_sft.jsonl"
    dpo_path = args.out_dir / "train_dpo.jsonl"
    manifest_path = args.out_dir / "eval_manifest.json"
    summary_path = args.out_dir / "dataset_summary.json"

    _write_jsonl(sft_path, sft_records)
    _write_jsonl(dpo_path, [record.to_dict() for record in dpo_records])
    manifest_path.write_text(json.dumps(eval_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary = {
        "schema_version": 1,
        "config": str(args.config),
        "outputs": {
            "train_sft": str(sft_path),
            "train_dpo": str(dpo_path),
            "eval_manifest": str(manifest_path),
        },
        "settings": {
            "min_sft_reward": min_sft_reward,
            "min_dpo_gap": min_dpo_gap,
        },
        "train": {
            "sft_records": len(sft_records),
            "dpo_records": len(dpo_records),
            "sft_sources": sft_manifest,
            "dpo_sources": dpo_manifest,
        },
        "eval": eval_manifest,
        "excluded": config.get("excluded", []),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _collect_sft_records(sources: list[dict[str, Any]], min_reward: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    manifest = []
    seen_run_ids: set[str] = set()
    for source in sources:
        benchmark_index = _benchmark_index_for_source(source)
        scanned = 0
        exported = 0
        for run_dir in find_run_dirs(_root(source)):
            scanned += 1
            summary = summarize_run(run_dir, benchmark_index=benchmark_index)
            if summary.run_id in seen_run_ids:
                continue
            reward = score_trajectory(summary)
            if is_sft_candidate(summary, reward, min_reward=min_reward):
                record = build_sft_record(summary, reward).to_dict()
                record["metadata"]["split"] = "train"
                record["metadata"]["source_name"] = source["name"]
                records.append(record)
                seen_run_ids.add(summary.run_id)
                exported += 1
        manifest.append({"name": source["name"], "root": source["root"], "scanned": scanned, "exported": exported})
    return records, manifest


def _collect_dpo_records(sources: list[dict[str, Any]], min_reward_gap: float) -> tuple[list[Any], list[dict[str, Any]]]:
    records = []
    manifest = []
    seen_pairs: set[tuple[str, str]] = set()
    for source in sources:
        benchmark_index = _benchmark_index_for_source(source)
        by_prompt: dict[str, list[ScoredTrajectory]] = defaultdict(list)
        scanned = 0
        for run_dir in find_run_dirs(_root(source)):
            scanned += 1
            summary = summarize_run(run_dir, benchmark_index=benchmark_index)
            if summary.user_request:
                by_prompt[summary.user_request].append(
                    ScoredTrajectory(summary=summary, reward=score_trajectory(summary))
                )

        exported = 0
        for trajectories in by_prompt.values():
            for record in build_pairs_for_prompt(trajectories, min_reward_gap=min_reward_gap):
                pair_key = (record.metadata["chosen_run_id"], record.metadata["rejected_run_id"])
                if pair_key in seen_pairs:
                    continue
                record.metadata["split"] = "train"
                record.metadata["source_name"] = source["name"]
                records.append(record)
                seen_pairs.add(pair_key)
                exported += 1
        manifest.append(
            {
                "name": source["name"],
                "root": source["root"],
                "scanned": scanned,
                "same_prompt_groups": sum(1 for items in by_prompt.values() if len(items) >= 2),
                "exported": exported,
            }
        )
    return records, manifest


def _build_eval_manifest(benchmarks: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for benchmark in benchmarks:
        summary = summarize_post_training(
            _root(benchmark),
            benchmark_artifacts=_artifact_paths(benchmark) or None,
        ).to_dict()
        rows.append(
            {
                "name": benchmark["name"],
                "root": benchmark["root"],
                "purpose": benchmark.get("purpose", ""),
                "benchmark_artifacts": [str(path) for path in _artifact_paths(benchmark)],
                "runs_scanned": summary["runs_scanned"],
                "sft_candidates": summary["sft_candidates"],
                "dpo_pairs": summary["dpo_pairs"],
                "reward": summary["reward"],
                "quality_label_counts": summary["quality_label_counts"],
                "verifier_counts": summary["verifier_counts"],
            }
        )
    return {"benchmarks": rows}


def _load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _root(source: dict[str, Any]) -> Path:
    path = Path(source["root"])
    return path if path.is_absolute() else REPO_ROOT / path


def _artifact_paths(source: dict[str, Any]) -> list[Path]:
    explicit = [Path(path) for path in source.get("benchmark_artifacts", [])]
    paths = [path if path.is_absolute() else REPO_ROOT / path for path in explicit]
    if paths:
        return paths
    return discover_benchmark_artifacts(_root(source))


def _benchmark_index_for_source(source: dict[str, Any]) -> dict[str, dict[str, Any]]:
    paths = _artifact_paths(source)
    return load_benchmark_index(paths) if paths else {}


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
