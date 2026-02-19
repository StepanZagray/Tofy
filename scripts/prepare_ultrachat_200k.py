#!/usr/bin/env python3
"""
Prepare UltraChat 200k into context<TAB>next_turn pairs for world-model training.

Output format:
    context<TAB>next_turn

Each pair is built from assistant turns:
- context = up to N previous turns with role prefixes
- next_turn = current assistant turn text
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable


def _turn_role(turn: dict) -> str:
    role = (
        str(turn.get("role") or turn.get("from") or turn.get("speaker") or "")
        .strip()
        .lower()
    )
    if role in {"assistant", "gpt", "bot", "model"}:
        return "assistant"
    if role in {"user", "human", "prompt", "instruction"}:
        return "user"
    return "other"


def _turn_text(turn: dict) -> str:
    text = str(
        turn.get("content")
        or turn.get("value")
        or turn.get("text")
        or turn.get("message")
        or ""
    ).strip()
    return text.replace("\t", " ")


def _extract_turns(row: dict) -> list[dict]:
    for key in ("messages", "conversation", "conversations", "dialog", "turns"):
        value = row.get(key)
        if isinstance(value, list) and value:
            return [x for x in value if isinstance(x, dict)]
    return []


def _format_context(turns: Iterable[dict]) -> str:
    lines: list[str] = []
    for t in turns:
        role = _turn_role(t)
        text = _turn_text(t)
        if not text:
            continue
        prefix = "User" if role == "user" else "Assistant" if role == "assistant" else "Other"
        lines.append(f"{prefix}: {text}")
    return "\n".join(lines).strip()


def prepare(
    output_path: Path,
    *,
    split: str = "train_sft",
    max_rows: int | None = None,
    min_tokens: int = 2,
    context_window: int = 6,
) -> tuple[int, int]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install datasets first: pip install datasets") from exc

    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=split)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    with output_path.open("w", encoding="utf-8") as out:
        for row in ds:
            turns = _extract_turns(row)
            if len(turns) < 2:
                skipped += 1
                continue

            for i, turn in enumerate(turns):
                if _turn_role(turn) != "assistant":
                    continue
                target = _turn_text(turn)
                if len(target.split()) < min_tokens:
                    continue

                start = max(0, i - context_window)
                context = _format_context(turns[start:i])
                if not context or len(context.split()) < min_tokens:
                    continue

                out.write(f"{context}\t{target}\n")
                written += 1
                if max_rows is not None and written >= max_rows:
                    return written, skipped

    return written, skipped


def main() -> None:
    p = argparse.ArgumentParser(
        description="Prepare UltraChat 200k context<TAB>next_turn pairs"
    )
    p.add_argument("--output", default="data/ultrachat_pairs.txt")
    p.add_argument("--split", default="train_sft")
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--min-tokens", type=int, default=2)
    p.add_argument("--context-window", type=int, default=6)
    args = p.parse_args()

    written, skipped = prepare(
        Path(args.output),
        split=args.split,
        max_rows=args.max_rows,
        min_tokens=args.min_tokens,
        context_window=args.context_window,
    )
    print(f"wrote {written} pairs, skipped {skipped}, output={args.output}")


if __name__ == "__main__":
    main()
