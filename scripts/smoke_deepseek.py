"""Smoke test the DeepSeek chat-completions provider."""

from __future__ import annotations

import os
import sys
import traceback
from argparse import ArgumentParser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pico.models import DeepSeekChatModelClient


def main() -> int:
    parser = ArgumentParser(description="Smoke test the DeepSeek chat-completions provider.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only verify local configuration without sending an API request.",
    )
    args = parser.parse_args()

    _load_local_env(REPO_ROOT / ".env.local")
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("DEEPSEEK_API_KEY is not set", file=sys.stderr, flush=True)
        return 1
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")
    base_url = os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com")
    print(
        f"DeepSeek config: model={model}, base_url={base_url}, api_key_len={len(api_key)}",
        flush=True,
    )
    if args.dry_run:
        print("Dry run passed: local config is readable.", flush=True)
        return 0

    client = DeepSeekChatModelClient(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=0.0,
        timeout=60,
    )
    print("Sending smoke request...", flush=True)
    text = client.complete("Reply with exactly the lowercase word: ok", max_new_tokens=128)
    print(text, flush=True)
    print(client.last_completion_metadata, flush=True)
    return 0


def _load_local_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().lstrip("\ufeff")
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc(file=sys.stdout)
        raise SystemExit(1)
