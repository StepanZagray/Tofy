#!/usr/bin/env python3
from __future__ import annotations

"""
Build a code-first decoder mix with extra weight on instruction-shaped Rust tasks.
"""

import argparse
import random
from pathlib import Path


def read_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f if line.strip()]


def run(
    output: Path,
    base_pairs: Path,
    instruction_pairs: Path,
    extra_pairs: list[Path],
    instruction_repeat: int,
    extra_repeat: int,
    max_rows: int | None,
    seed: int,
) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    base = read_lines(base_pairs)
    instruction = read_lines(instruction_pairs)
    rows: list[str] = []
    for _ in range(max(instruction_repeat, 1)):
        rows.extend(instruction)
    for extra_path in extra_pairs:
        extra = read_lines(extra_path)
        for _ in range(max(extra_repeat, 1)):
            rows.extend(extra)
    rows.extend(base)
    random.Random(seed).shuffle(rows)
    if max_rows is not None and max_rows > 0:
        rows = rows[:max_rows]
    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(row)
            f.write("\n")
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a code-first decoder training mix.")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--base-pairs", type=Path, required=True)
    ap.add_argument("--instruction-pairs", type=Path, required=True)
    ap.add_argument("--extra-pairs", type=Path, action="append", default=[])
    ap.add_argument("--instruction-repeat", type=int, default=3)
    ap.add_argument("--extra-repeat", type=int, default=1)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()
    count = run(
        output=args.output,
        base_pairs=args.base_pairs,
        instruction_pairs=args.instruction_pairs,
        extra_pairs=args.extra_pairs,
        instruction_repeat=args.instruction_repeat,
        extra_repeat=args.extra_repeat,
        max_rows=args.max_rows if args.max_rows > 0 else None,
        seed=args.seed,
    )
    print(f"Wrote {count} mixed code POC rows to {args.output}")


if __name__ == "__main__":
    main()
