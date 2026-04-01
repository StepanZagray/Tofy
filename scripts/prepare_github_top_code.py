#!/usr/bin/env python3
from __future__ import annotations

"""
Build context<TAB>completion pairs from ronantakizawa/github-top-code for world/decoder training.

Usage:
  pip install datasets
  python scripts/prepare_github_top_code.py --output data/github_top_code_pairs.txt
  python scripts/prepare_github_top_code.py --output data/rust_code_pairs.txt --languages Rust --max-files 50000
  python scripts/prepare_github_top_code.py --output data/multilang_pairs.txt --default-languages --max-files 200000

Output format: one line per pair, prefix<TAB>completion (TAB in content is replaced by spaces).
"""

# Default language set for --default-languages (file_language values in ronantakizawa/github-top-code)
DEFAULT_LANGUAGES = [
    "Rust",
    "TypeScript",
    "Go",
    "JavaScript",
    "C/C++ Header",
    "C",
    "C++",
    "TSX",
    "CSS",
    "HTML",
]

import argparse
import re
from pathlib import Path


def split_file_into_pairs(
    content: str,
    *,
    min_lines_prefix: int = 2,
    min_lines_completion: int = 2,
    split_ratio: float = 0.5,
) -> list[tuple[str, str]]:
    """Split a single file's content into (prefix, completion) pairs.
    Uses line count; split_ratio is fraction of lines in prefix (rest = completion).
    """
    lines = content.splitlines()
    if len(lines) < min_lines_prefix + min_lines_completion:
        return []
    n = len(lines)
    cut = max(min_lines_prefix, min(round(n * split_ratio), n - min_lines_completion))
    if cut < min_lines_prefix or (n - cut) < min_lines_completion:
        return []
    prefix = "\n".join(lines[:cut]).strip()
    completion = "\n".join(lines[cut:]).strip()
    if not prefix or not completion:
        return []
    # Avoid TAB in content so we don't break the format
    prefix = prefix.replace("\t", " ")
    completion = completion.replace("\t", " ")
    return [(prefix, completion)]


def run(
    output_path: Path,
    *,
    split: str = "train",
    languages: list[str] | None = None,
    max_files: int | None = None,
    min_lines: int = 5,
    min_lines_prefix: int = 2,
    min_lines_completion: int = 2,
    split_ratio: float = 0.5,
) -> int:
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError("Install: pip install datasets") from e

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("ronantakizawa/github-top-code", split=split)
    if languages:
        lang_set = {l.strip() for l in languages}
        ds = ds.filter(lambda x: (x.get("file_language") or "").strip() in lang_set)

    written = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(ds):
            if max_files is not None and i >= max_files:
                break
            content = (row.get("content") or "").strip()
            if not content:
                continue
            lines = content.splitlines()
            if len(lines) < min_lines:
                continue
            for prefix, completion in split_file_into_pairs(
                content,
                min_lines_prefix=min_lines_prefix,
                min_lines_completion=min_lines_completion,
                split_ratio=split_ratio,
            ):
                line = f"{prefix}\t{completion}\n"
                f.write(line)
                written += 1

    return written


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build code pairs from ronantakizawa/github-top-code (context TAB completion)."
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("data/github_top_code_pairs.txt"),
        help="Output .txt path (default: data/github_top_code_pairs.txt)",
    )
    ap.add_argument(
        "--split",
        default="train",
        choices=("train", "test", "validation"),
        help="Dataset split (default: train)",
    )
    ap.add_argument(
        "--languages",
        nargs="*",
        default=None,
        help="Filter by file_language (e.g. Rust TypeScript Go). Default: all.",
    )
    ap.add_argument(
        "--default-languages",
        action="store_true",
        help="Use preset: Rust, TypeScript, Go, JavaScript, C/C++ Header, C, C++, TSX, CSS, HTML.",
    )
    ap.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Max number of source files to process (default: all)",
    )
    ap.add_argument(
        "--min-lines",
        type=int,
        default=5,
        help="Skip files with fewer lines (default: 5)",
    )
    ap.add_argument(
        "--min-lines-prefix",
        type=int,
        default=2,
        help="Min lines in prefix (default: 2)",
    )
    ap.add_argument(
        "--min-lines-completion",
        type=int,
        default=2,
        help="Min lines in completion (default: 2)",
    )
    ap.add_argument(
        "--split-ratio",
        type=float,
        default=0.5,
        help="Fraction of lines in prefix, rest completion (default: 0.5)",
    )
    args = ap.parse_args()
    languages = args.languages
    if args.default_languages:
        languages = DEFAULT_LANGUAGES
    n = run(
        args.output,
        split=args.split,
        languages=languages,
        max_files=args.max_files,
        min_lines=args.min_lines,
        min_lines_prefix=args.min_lines_prefix,
        min_lines_completion=args.min_lines_completion,
        split_ratio=args.split_ratio,
    )
    print(f"Wrote {n} pairs to {args.output}")


if __name__ == "__main__":
    main()
