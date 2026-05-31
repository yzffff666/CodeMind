"""Minimal LoRA SFT training entrypoint for Pico trajectory JSONL data.

The default dry-run validates config, data formatting, and assistant-label
mask construction without importing heavyweight training dependencies. A real
LoRA run requires torch, transformers, and peft.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ASSISTANT_ROLE = "assistant"
SUPPORTED_ROLES = {"user", "assistant", "tool"}


@dataclass
class SftExample:
    text: str
    assistant_spans: list[tuple[int, int]]
    metadata: dict[str, Any]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/sft_lora_smoke.json"))
    parser.add_argument("--dry-run", action="store_true", help="Validate data and masks without loading a model.")
    args = parser.parse_args()

    config = _load_config(args.config)
    examples = _load_examples(Path(config["data"]["train_path"]), max_records=int(config["data"].get("max_records", 0) or 0))
    report = _build_data_report(args.config, config, examples)

    if args.dry_run:
        _write_report(config, report, dry_run=True)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    train_report = _run_lora_training(config, examples)
    report.update(train_report)
    _write_report(config, report, dry_run=False)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if int(config.get("schema_version", 0)) != 1:
        raise ValueError("unsupported config schema_version")
    for key in ("data", "model", "training", "lora"):
        if key not in config:
            raise ValueError(f"config missing required section: {key}")
    return config


def _load_examples(path: Path, max_records: int = 0) -> list[SftExample]:
    examples: list[SftExample] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        examples.append(_record_to_example(record, line_no))
        if max_records and len(examples) >= max_records:
            break
    if not examples:
        raise ValueError(f"no SFT examples loaded from {path}")
    return examples


def _record_to_example(record: dict[str, Any], line_no: int) -> SftExample:
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"line {line_no}: messages must be a non-empty list")
    if messages[0].get("role") != "user":
        raise ValueError(f"line {line_no}: first message must be user")

    pieces: list[str] = []
    assistant_spans: list[tuple[int, int]] = []
    for index, message in enumerate(messages):
        role = message.get("role")
        content = message.get("content")
        if role not in SUPPORTED_ROLES:
            raise ValueError(f"line {line_no}, message {index}: unsupported role {role!r}")
        if not isinstance(content, str) or not content:
            raise ValueError(f"line {line_no}, message {index}: content must be a non-empty string")
        pieces.append(f"<|{role}|>\n")
        start = sum(len(piece) for piece in pieces)
        pieces.append(content)
        end = sum(len(piece) for piece in pieces)
        pieces.append("\n")
        if role == ASSISTANT_ROLE:
            assistant_spans.append((start, end))

    if not assistant_spans:
        raise ValueError(f"line {line_no}: record has no assistant messages")
    return SftExample(text="".join(pieces), assistant_spans=assistant_spans, metadata=record.get("metadata", {}))


def _build_data_report(config_path: Path, config: dict[str, Any], examples: list[SftExample]) -> dict[str, Any]:
    assistant_chars = sum(end - start for example in examples for start, end in example.assistant_spans)
    total_chars = sum(len(example.text) for example in examples)
    source_counts: dict[str, int] = {}
    for example in examples:
        source = str(example.metadata.get("source_name", ""))
        source_counts[source] = source_counts.get(source, 0) + 1
    return {
        "config": str(config_path),
        "dry_run_only": None,
        "data": {
            "train_path": config["data"]["train_path"],
            "examples": len(examples),
            "total_chars": total_chars,
            "assistant_target_chars": assistant_chars,
            "assistant_target_ratio": round(assistant_chars / max(total_chars, 1), 6),
            "records_by_source": source_counts,
        },
        "model": {
            "name_or_path": config["model"]["name_or_path"],
            "trust_remote_code": bool(config["model"].get("trust_remote_code", False)),
        },
        "training": config["training"],
        "lora": config["lora"],
        "evaluation": config.get("evaluation", {}),
        "status": "validated",
    }


def _write_report(config: dict[str, Any], report: dict[str, Any], dry_run: bool) -> None:
    output_dir = Path(config["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    report["dry_run_only"] = dry_run
    report_path = output_dir / ("sft_lora_dry_run_report.json" if dry_run else "sft_lora_train_report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_lora_training(config: dict[str, Any], examples: list[SftExample]) -> dict[str, Any]:
    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Real LoRA training requires optional dependencies: torch, transformers, peft. "
            "Run with --dry-run to validate the pipeline without them."
        ) from exc

    seed = int(config["training"].get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["name_or_path"],
        trust_remote_code=bool(config["model"].get("trust_remote_code", False)),
        use_fast=True,
    )
    if not tokenizer.is_fast:
        raise RuntimeError("A fast tokenizer is required to build assistant-only labels from char offsets.")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = _resolve_device(torch, str(config["training"].get("device", "auto")))
    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["name_or_path"],
        trust_remote_code=bool(config["model"].get("trust_remote_code", False)),
    )
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(config["lora"].get("r", 8)),
        lora_alpha=int(config["lora"].get("alpha", 16)),
        lora_dropout=float(config["lora"].get("dropout", 0.05)),
        target_modules=list(config["lora"].get("target_modules", [])),
    )
    model = get_peft_model(model, lora_config)
    model.to(device)
    model.train()

    encoded = [_tokenize_example(tokenizer, example, int(config["data"].get("max_seq_length", 2048))) for example in examples]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"].get("learning_rate", 2e-4)),
        weight_decay=float(config["training"].get("weight_decay", 0.0)),
    )

    batch_size = int(config["training"].get("batch_size", 1))
    grad_accum = int(config["training"].get("gradient_accumulation_steps", 1))
    max_steps = int(config["training"].get("max_steps", 20))
    epochs = int(config["training"].get("epochs", 1))
    output_dir = Path(config["training"]["output_dir"])
    losses: list[float] = []
    step = 0
    optimizer.zero_grad(set_to_none=True)

    for _epoch in range(epochs):
        random.shuffle(encoded)
        for batch_start in range(0, len(encoded), batch_size):
            batch = _collate(encoded[batch_start : batch_start + batch_size], tokenizer.pad_token_id, torch, device)
            outputs = model(**batch)
            loss = outputs.loss / grad_accum
            loss.backward()
            if (step + 1) % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(outputs.loss.detach().cpu()))
            step += 1
            if step >= max_steps:
                break
        if step >= max_steps:
            break

    model.save_pretrained(output_dir / "adapter")
    tokenizer.save_pretrained(output_dir / "tokenizer")
    return {
        "status": "trained",
        "train_steps": step,
        "loss": {
            "first": round(losses[0], 6) if losses else None,
            "last": round(losses[-1], 6) if losses else None,
            "history": [round(value, 6) for value in losses],
        },
        "outputs": {
            "adapter": str(output_dir / "adapter"),
            "tokenizer": str(output_dir / "tokenizer"),
        },
    }


def _resolve_device(torch_module: Any, requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch_module.cuda.is_available() else "cpu"
    return requested


def _tokenize_example(tokenizer: Any, example: SftExample, max_length: int) -> dict[str, list[int]]:
    encoded = tokenizer(
        example.text,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    labels = list(encoded["input_ids"])
    for index, (start, end) in enumerate(encoded["offset_mapping"]):
        if start == end or not _overlaps_any_span(start, end, example.assistant_spans):
            labels[index] = -100
    if all(label == -100 for label in labels):
        raise ValueError("tokenized example has no assistant labels after truncation")
    return {
        "input_ids": list(encoded["input_ids"]),
        "attention_mask": list(encoded["attention_mask"]),
        "labels": labels,
    }


def _overlaps_any_span(start: int, end: int, spans: list[tuple[int, int]]) -> bool:
    return any(start < span_end and end > span_start for span_start, span_end in spans)


def _collate(batch: list[dict[str, list[int]]], pad_token_id: int, torch_module: Any, device: str) -> dict[str, Any]:
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids = []
    attention_mask = []
    labels = []
    for item in batch:
        pad = max_len - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_token_id] * pad)
        attention_mask.append(item["attention_mask"] + [0] * pad)
        labels.append(item["labels"] + [-100] * pad)
    return {
        "input_ids": torch_module.tensor(input_ids, dtype=torch_module.long, device=device),
        "attention_mask": torch_module.tensor(attention_mask, dtype=torch_module.long, device=device),
        "labels": torch_module.tensor(labels, dtype=torch_module.long, device=device),
    }


if __name__ == "__main__":
    raise SystemExit(main())

