#!/usr/bin/env python3
import argparse
from pathlib import Path


def unescape_pair_field(text: str) -> str:
    return (
        text.replace("\\n", "\n")
        .replace("\\t", "\t")
        .replace("\\r", "\r")
        .replace("\\\\", "\\")
    )


def flatten_for_encoder(text: str) -> str:
    return " ".join(unescape_pair_field(text).split())


def iter_lines(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if "\t" in line:
                left, right = line.split("\t", 1)
                if left.strip():
                    yield flatten_for_encoder(left.strip())
                if right.strip():
                    yield flatten_for_encoder(right.strip())
                continue
            if "|||" in line:
                left, right = line.split("|||", 1)
                if left.strip():
                    yield flatten_for_encoder(left.strip())
                if right.strip():
                    yield flatten_for_encoder(right.strip())
                continue
            yield flatten_for_encoder(line)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a plain-text encoder corpus from mixed text/code datasets."
    )
    parser.add_argument("--output", required=True, help="Output text file")
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input files. Pair files contribute both left and right sides.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with output_path.open("w", encoding="utf-8") as out:
        for input_path in args.inputs:
            path = Path(input_path)
            if not path.exists():
                raise SystemExit(f"input file not found: {path}")
            for line in iter_lines(path):
                out.write(line)
                out.write("\n")
                written += 1

    print(f"Wrote {written} encoder lines to {output_path}")


if __name__ == "__main__":
    main()
