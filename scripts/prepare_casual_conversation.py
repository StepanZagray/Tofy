#!/usr/bin/env python3
"""
Download SohamGhadge/casual-conversation from Hugging Face and convert to
tab-separated format for this repo:

    context<TAB>response

Requires: pip install datasets
"""

from __future__ import annotations

import argparse
from pathlib import Path


def prepare_casual_conversation(
    output_path: Path,
    *,
    min_tokens: int = 2,
    lowercase: bool = False,
    max_pairs: int | None = None,
    split: str = "train",
) -> tuple[int, int]:
    """
    Load SohamGhadge/casual-conversation and write question<TAB>answer per line.
    Returns: (written_pairs, skipped_pairs)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "Install Hugging Face datasets: pip install datasets"
        ) from None

    ds = load_dataset("SohamGhadge/casual-conversation", split=split)
    # Columns: Unnamed: 0, question, answer
    if "question" not in ds.column_names or "answer" not in ds.column_names:
        raise RuntimeError(
            f"Dataset has columns: {ds.column_names}. Expected 'question' and 'answer'."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0

    with output_path.open("w", encoding="utf-8") as out:
        for row in ds:
            ctx = (row.get("question") or "").strip().replace("\t", " ")
            rsp = (row.get("answer") or "").strip().replace("\t", " ")
            if not ctx or not rsp:
                skipped += 1
                continue
            if lowercase:
                ctx = ctx.lower()
                rsp = rsp.lower()
            if len(ctx.split()) < min_tokens or len(rsp.split()) < min_tokens:
                skipped += 1
                continue
            out.write(f"{ctx}\t{rsp}\n")
            written += 1
            if max_pairs is not None and written >= max_pairs:
                break

    return written, skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download casual-conversation from Hugging Face and write context<TAB>response pairs"
    )
    parser.add_argument(
        "--output",
        default="data/casual_pairs.txt",
        help="Output .txt path (default: data/casual_pairs.txt)",
    )
    parser.add_argument("--min-tokens", type=int, default=2)
    parser.add_argument("--lower", action="store_true")
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Optional cap (e.g. for quick tests)",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split (default: train)",
    )
    args = parser.parse_args()

    output = Path(args.output).expanduser()
    written, skipped = prepare_casual_conversation(
        output,
        min_tokens=args.min_tokens,
        lowercase=args.lower,
        max_pairs=args.max_pairs,
        split=args.split,
    )
    print(f"wrote {written} pairs, skipped {skipped}, output={output}")


if __name__ == "__main__":
    main()
