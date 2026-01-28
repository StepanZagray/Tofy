#!/usr/bin/env python3
"""
Convert Cornell Movie Dialog Corpus to the tab-separated format this repo uses:

    context<TAB>response

This works with the standard Cornell extraction that contains:
  - movie_lines.txt
  - movie_conversations.txt
"""

from __future__ import annotations

import argparse
import ast
import os
from pathlib import Path


def _read_movie_lines(movie_lines_path: Path) -> dict[str, str]:
    # movie_lines.txt is latin-1 / iso-8859-1 in many distributions.
    lines: dict[str, str] = {}
    with movie_lines_path.open("r", encoding="iso-8859-1", errors="replace") as f:
        for raw in f:
            parts = raw.split(" +++$+++ ")
            if len(parts) < 5:
                continue
            line_id = parts[0].strip()
            text = parts[4].strip()
            if line_id and text:
                # Ensure no tabs sneak into our TSV format.
                lines[line_id] = text.replace("\t", " ").strip()
    return lines


def prepare_cornell(
    corpus_dir: Path,
    output_path: Path,
    *,
    min_tokens: int = 2,
    lowercase: bool = False,
    max_pairs: int | None = None,
) -> tuple[int, int]:
    """
    Returns: (written_pairs, skipped_pairs)
    """
    movie_lines = corpus_dir / "movie_lines.txt"
    movie_conversations = corpus_dir / "movie_conversations.txt"

    if not movie_lines.exists() or not movie_conversations.exists():
        cwd = os.getcwd()
        raise FileNotFoundError(
            "Cornell corpus files not found.\n"
            f"Current working directory: {cwd}\n"
            f"Provided/used corpus_dir: {corpus_dir}\n"
            f"Expected:\n  - {movie_lines}\n  - {movie_conversations}\n"
        )

    lines = _read_movie_lines(movie_lines)

    written = 0
    skipped = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        with movie_conversations.open("r", encoding="iso-8859-1", errors="replace") as f:
            for raw in f:
                parts = raw.split(" +++$+++ ")
                if len(parts) < 4:
                    continue
                try:
                    # Example: "['L194', 'L195', ...]"
                    utterance_ids = ast.literal_eval(parts[3].strip())
                except Exception:
                    continue
                if not isinstance(utterance_ids, list) or len(utterance_ids) < 2:
                    continue

                for i in range(len(utterance_ids) - 1):
                    ctx = lines.get(str(utterance_ids[i]), "").strip()
                    rsp = lines.get(str(utterance_ids[i + 1]), "").strip()
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
                        return written, skipped

    return written, skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Cornell Movie Dialog Corpus to context<TAB>response pairs"
    )
    parser.add_argument(
        "--corpus-dir",
        required=True,
        help="Path to extracted Cornell corpus directory (contains movie_lines.txt)",
    )
    parser.add_argument("--output", required=True, help="Output .txt path (TSV pairs)")
    parser.add_argument("--min-tokens", type=int, default=2)
    parser.add_argument("--lower", action="store_true")
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Optional cap for quick tests (e.g. 100000)",
    )
    args = parser.parse_args()

    # Make --corpus-dir work from any current working directory:
    # - first try as provided
    # - if missing, try relative to the repo root (parent of scripts/)
    corpus_dir = Path(args.corpus_dir).expanduser()
    repo_root = Path(__file__).resolve().parents[1]

    def _looks_like_cornell_dir(p: Path) -> bool:
        return (p / "movie_lines.txt").exists() and (p / "movie_conversations.txt").exists()

    if not _looks_like_cornell_dir(corpus_dir):
        # Try relative to repo root.
        alt = (repo_root / corpus_dir).resolve()
        if _looks_like_cornell_dir(alt):
            corpus_dir = alt
        else:
            # Try a few common names (your repo currently uses `cornell movie-dialogs/`).
            candidates = [
                repo_root / "cornell movie-dialogs",
                repo_root / "cornell movie-dialogs corpus",
                repo_root / "cornell_movie_dialogs_corpus",
            ]
            # Also scan repo root for any directory starting with "cornell".
            candidates.extend([p for p in repo_root.glob("cornell*") if p.is_dir()])
            for cand in candidates:
                if _looks_like_cornell_dir(cand):
                    corpus_dir = cand
                    break

    written, skipped = prepare_cornell(
        corpus_dir,
        Path(args.output).expanduser(),
        min_tokens=args.min_tokens,
        lowercase=args.lower,
        max_pairs=args.max_pairs,
    )
    print(f"wrote {written} pairs, skipped {skipped}, output={args.output}")


if __name__ == "__main__":
    main()

