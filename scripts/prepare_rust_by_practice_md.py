#!/usr/bin/env python3
"""
Prepare Rust-by-Practice markdown docs (data/sunface_rust-by-practice_en) for Tofy training.

Two modes:

1. JEPA / encoder (--latent): one text chunk per line.
   Use as: --mode jepa --output data/rust_docs_jepa.txt
   Then: cargo run --release -- --latent data/rust_docs_jepa.txt ...

2. Pairs for world/decoder: context<TAB>next (e.g. previous section TAB next section).
   Use as: --mode pairs --output data/rust_docs_pairs.txt
   Then: use alongside or mixed with code pairs for --train-world / --train-decoder.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def iter_md_files(root: Path) -> list[Path]:
    root = Path(root)
    out: list[Path] = []
    for p in root.rglob("*.md"):
        if p.name.startswith(".") or "/.git/" in str(p):
            continue
        out.append(p)
    return sorted(out)


def split_by_heading(content: str) -> list[str]:
    """Split markdown by ## headings (or # if no ##). Keeps heading in the chunk."""
    parts = re.split(r"\n(?=##?\s)", content)
    return [p.strip() for p in parts if p.strip()]


def run_jepa(root: Path, output_path: Path, *, by_heading: bool = True) -> int:
    """One chunk per line. If by_heading=True, split each file by ##; else one file per line."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    files = iter_md_files(root)
    written = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for path in files:
            text = path.read_text(encoding="utf-8").strip()
            if not text:
                continue
            if by_heading:
                chunks = split_by_heading(text)
                for chunk in chunks:
                    if len(chunk.split()) < 5:  # skip tiny fragments
                        continue
                    line = chunk.replace("\n", " ").replace("\t", " ").strip()
                    if line:
                        f.write(line + "\n")
                        written += 1
            else:
                line = text.replace("\t", " ").replace("\n", " ")
                if len(line.split()) >= 5:
                    f.write(line + "\n")
                    written += 1
    return written


def run_pairs(root: Path, output_path: Path) -> int:
    """context<TAB>next: consecutive sections (or consecutive chunks) per file."""
    import re
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    files = iter_md_files(root)
    written = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for path in files:
            text = path.read_text(encoding="utf-8").strip()
            if not text:
                continue
            chunks = split_by_heading(text)
            for i in range(len(chunks) - 1):
                prev = chunks[i].replace("\t", " ").strip()
                next_ = chunks[i + 1].replace("\t", " ").strip()
                if len(prev.split()) < 3 or len(next_.split()) < 3:
                    continue
                f.write(f"{prev}\t{next_}\n")
                written += 1
    return written


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Prepare sunface Rust-by-Practice .md for JEPA or world/decoder training."
    )
    ap.add_argument(
        "--input",
        type=Path,
        default=Path("data/sunface_rust-by-practice_en"),
        help="Root folder of .md files (default: data/sunface_rust-by-practice_en)",
    )
    ap.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path (e.g. data/rust_docs_jepa.txt or data/rust_docs_pairs.txt)",
    )
    ap.add_argument(
        "--mode",
        choices=("jepa", "pairs"),
        required=True,
        help="jepa = one chunk per line for --latent; pairs = context TAB next for world/decoder",
    )
    ap.add_argument(
        "--no-split-headings",
        action="store_true",
        help="(JEPA only) One file per line instead of splitting by ##",
    )
    args = ap.parse_args()
    if not args.input.is_dir():
        raise SystemExit(f"Not a directory: {args.input}")
    if args.mode == "jepa":
        n = run_jepa(args.input, args.output, by_heading=not args.no_split_headings)
    else:
        n = run_pairs(args.input, args.output)
    print(f"Wrote {n} lines to {args.output}")


if __name__ == "__main__":
    main()
