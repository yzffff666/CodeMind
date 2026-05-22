"""Run a dependency-free SFT dry run over exported Pico SFT JSONL data.

This is not a real model fine-tune. It validates the data shape and simulates
the teacher-forcing next-token objective with a tiny character bigram learner.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROLE_ORDER = {"user", "assistant", "tool"}
ASSISTANT_ROLE = "assistant"
START_TOKEN = "\0"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("artifacts/datasets/v1/train_sft.jsonl"),
        help="SFT JSONL exported by prepare_post_training_datasets.py.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/datasets/v1/sft_dry_run_report.json"),
        help="Output JSON report.",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.1, help="Additive smoothing.")
    args = parser.parse_args()

    records = _load_jsonl(args.data)
    examples = [_to_example(record, index) for index, record in enumerate(records)]
    assistant_targets = [char for example in examples for char in example["assistant_chars"]]
    vocab = sorted(set(char for example in examples for char in example["text"]) | set(assistant_targets))

    counts: dict[str, Counter[str]] = defaultdict(Counter)
    history = []
    initial_loss = _masked_bigram_loss(examples, counts, vocab, args.alpha)
    history.append({"epoch": 0, "assistant_char_loss": round(initial_loss, 6)})
    for epoch in range(1, args.epochs + 1):
        _fit_one_epoch(examples, counts)
        loss = _masked_bigram_loss(examples, counts, vocab, args.alpha)
        history.append({"epoch": epoch, "assistant_char_loss": round(loss, 6)})

    report = {
        "data": str(args.data),
        "records": len(records),
        "examples": len(examples),
        "assistant_target_chars": len(assistant_targets),
        "vocab_size": len(vocab),
        "epochs": args.epochs,
        "objective": "masked assistant next-character negative log likelihood",
        "dry_run_only": True,
        "note": "This validates SFT data formatting and teacher forcing; it is not a neural LoRA fine-tune.",
        "loss_history": history,
        "records_by_source": dict(Counter(example["source_name"] for example in examples)),
        "records_by_quality": dict(Counter(example["quality_label"] for example in examples)),
        "sample": {
            "run_id": examples[0]["run_id"] if examples else "",
            "source_name": examples[0]["source_name"] if examples else "",
            "text_preview": examples[0]["text"][:500] if examples else "",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        _validate_record(record, line_no)
        records.append(record)
    return records


def _validate_record(record: dict[str, Any], line_no: int) -> None:
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"line {line_no}: messages must be a non-empty list")
    if messages[0].get("role") != "user":
        raise ValueError(f"line {line_no}: first message must be user")
    if not any(message.get("role") == ASSISTANT_ROLE for message in messages):
        raise ValueError(f"line {line_no}: record has no assistant messages")
    for index, message in enumerate(messages):
        role = message.get("role")
        content = message.get("content")
        if role not in ROLE_ORDER:
            raise ValueError(f"line {line_no}, message {index}: unsupported role {role!r}")
        if not isinstance(content, str) or not content:
            raise ValueError(f"line {line_no}, message {index}: content must be a non-empty string")


def _to_example(record: dict[str, Any], index: int) -> dict[str, Any]:
    pieces = []
    assistant_spans: list[tuple[int, int]] = []
    for message in record["messages"]:
        role = message["role"]
        header = f"<|{role}|>\n"
        pieces.append(header)
        start = sum(len(piece) for piece in pieces)
        pieces.append(message["content"])
        end = sum(len(piece) for piece in pieces)
        pieces.append("\n")
        if role == ASSISTANT_ROLE:
            assistant_spans.append((start, end))

    text = "".join(pieces)
    assistant_chars = []
    for start, end in assistant_spans:
        assistant_chars.extend(text[start:end])
    metadata = record.get("metadata", {})
    reward = metadata.get("reward", {})
    signals = reward.get("signals", {})
    return {
        "index": index,
        "text": text,
        "assistant_spans": assistant_spans,
        "assistant_chars": assistant_chars,
        "run_id": metadata.get("run_id", ""),
        "source_name": metadata.get("source_name", ""),
        "quality_label": signals.get("quality_label", ""),
    }


def _fit_one_epoch(examples: list[dict[str, Any]], counts: dict[str, Counter[str]]) -> None:
    for example in examples:
        text = example["text"]
        for start, end in example["assistant_spans"]:
            previous = text[start - 1] if start > 0 else START_TOKEN
            for position in range(start, end):
                current = text[position]
                counts[previous][current] += 1
                previous = current


def _masked_bigram_loss(
    examples: list[dict[str, Any]],
    counts: dict[str, Counter[str]],
    vocab: list[str],
    alpha: float,
) -> float:
    if not examples:
        return 0.0
    vocab_size = max(len(vocab), 1)
    total_loss = 0.0
    total_targets = 0
    for example in examples:
        text = example["text"]
        for start, end in example["assistant_spans"]:
            previous = text[start - 1] if start > 0 else START_TOKEN
            for position in range(start, end):
                current = text[position]
                row = counts[previous]
                denominator = sum(row.values()) + alpha * vocab_size
                probability = (row[current] + alpha) / denominator
                total_loss += -math.log(probability)
                total_targets += 1
                previous = current
    return total_loss / max(total_targets, 1)


if __name__ == "__main__":
    raise SystemExit(main())
