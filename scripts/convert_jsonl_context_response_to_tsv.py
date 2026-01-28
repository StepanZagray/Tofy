#!/usr/bin/env python3
"""
Convert JSONL shards (one JSON object per line) that contain:
  - "context"
  - "response"
into a single tab-separated training file:

    context<TAB>response

This is a generic converter (not tied to PolyAI). PolyAI/OpenSubtitles JSON
shards are one common source of this format.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def convert_dir(
    input_dir: Path,
    output_path: Path,
    *,
    min_tokens: int = 1,
    max_tokens: int = 200,
) -> tuple[int, int]:
    if not input_dir.exists():
        raise FileNotFoundError(
            f"input_dir not found: {input_dir}\n\n"
            "Create the directory and put your JSON shard files there (files ending in .json).\n"
            "Each line in each .json file must be a JSON object with keys: context, response.\n"
            'Example line: {"context":"hello","response":"hi"}\n'
        )

    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(
            f"no .json files found in: {input_dir}\n\n"
            "Put JSONL shard files in this directory (e.g. train-00000-of-00010.json).\n"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    with output_path.open("w", encoding="utf-8") as out:
        for jf in json_files:
            with jf.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        skipped += 1
                        continue

                    ctx = str(obj.get("context", "")).strip().replace("\t", " ")
                    rsp = str(obj.get("response", "")).strip().replace("\t", " ")
                    if not ctx or not rsp:
                        skipped += 1
                        continue

                    ct = ctx.split()
                    rt = rsp.split()
                    if len(ct) < min_tokens or len(rt) < min_tokens:
                        skipped += 1
                        continue
                    if len(ct) > max_tokens or len(rt) > max_tokens:
                        skipped += 1
                        continue

                    out.write(f"{ctx}\t{rsp}\n")
                    written += 1

    return written, skipped


def main() -> None:
    p = argparse.ArgumentParser(description="Convert JSONL(context/response) shards to TSV pairs")
    p.add_argument("--input_dir", required=True, help="Directory with *.json JSONL shard files")
    p.add_argument("--output", required=True, help="Output TSV file path")
    p.add_argument("--min_tokens", type=int, default=1)
    p.add_argument("--max_tokens", type=int, default=200)
    args = p.parse_args()

    written, skipped = convert_dir(
        Path(args.input_dir),
        Path(args.output),
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
    )
    print(f"wrote {written} pairs, skipped {skipped}, output={args.output}")


if __name__ == "__main__":
    main()

