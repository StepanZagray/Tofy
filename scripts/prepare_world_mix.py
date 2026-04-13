#!/usr/bin/env python3
import argparse
import random
from pathlib import Path


def parse_pair(row: str) -> tuple[str, str] | None:
    if "\t" in row:
        left, right, *_ = row.split("\t")
        return left, right
    if "|||" in row:
        left, right, *_ = row.split("|||")
        return left, right
    return None


def read_pairs(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            pair = parse_pair(line)
            if pair is not None:
                rows.append(pair)
    return rows


def terminal_state(left: str, right: str, action: str) -> str:
    if action == "text_reply":
        return f"{left}\\nAssistant: {right}"
    return f"{left}\\n{right}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a labeled world-model/orchestrator mix with controlled text/code/done ratios."
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--text-pairs", required=True)
    parser.add_argument("--code-pairs", required=True, action="append")
    parser.add_argument("--code-ratio", type=float, default=0.35)
    parser.add_argument(
        "--done-ratio",
        type=float,
        default=0.18,
        help="Approximate fraction of terminal done rows in the final mix.",
    )
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    text_rows = read_pairs(Path(args.text_pairs))
    code_rows: list[tuple[str, str]] = []
    for path in args.code_pairs:
        code_rows.extend(read_pairs(Path(path)))
    if not text_rows:
        raise SystemExit(f"no usable text rows found in {args.text_pairs}")
    if not code_rows:
        raise SystemExit(f"no usable code rows found in {args.code_pairs}")

    rng.shuffle(text_rows)
    rng.shuffle(code_rows)

    max_rows = args.max_rows if args.max_rows > 0 else len(text_rows) + len(code_rows)
    code_ratio = min(max(args.code_ratio, 0.0), 0.8)
    done_ratio = min(max(args.done_ratio, 0.0), 0.4)
    output_rows: list[str] = []
    terminal_candidates: list[tuple[str, str, str]] = []
    chosen_code = 0
    chosen_text = 0
    chosen_done = 0

    while len(output_rows) < max_rows and (text_rows or code_rows):
        want_code = rng.random() < code_ratio
        if want_code and code_rows:
            left, right = code_rows.pop()
            output_rows.append(f"{left}\t{right}\tcode")
            terminal_candidates.append((left, right, "code"))
            chosen_code += 1
        elif text_rows:
            left, right = text_rows.pop()
            output_rows.append(f"{left}\t{right}\ttext_reply")
            terminal_candidates.append((left, right, "text_reply"))
            chosen_text += 1
        elif code_rows:
            left, right = code_rows.pop()
            output_rows.append(f"{left}\t{right}\tcode")
            terminal_candidates.append((left, right, "code"))
            chosen_code += 1

    target_done = 0
    if done_ratio > 0.0 and output_rows:
        target_done = int(round((len(output_rows) * done_ratio) / max(1e-6, 1.0 - done_ratio)))
    rng.shuffle(terminal_candidates)
    for left, right, action in terminal_candidates[:target_done]:
        output_rows.append(f"{terminal_state(left, right, action)}\t<done>\tdone")
        chosen_done += 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(row)
            handle.write("\n")

    actual_code_rate = 0.0 if not output_rows else chosen_code / len(output_rows)
    actual_done_rate = 0.0 if not output_rows else chosen_done / len(output_rows)
    print(
        f"Wrote {len(output_rows)} rows to {output_path} "
        f"(text_rows={chosen_text}, code_rows={chosen_code}, done_rows={chosen_done}, requested code_ratio={code_ratio:.2f}, requested done_ratio={done_ratio:.2f})"
    )
    print(f"Actual code ratio: {actual_code_rate:.2f}")
    print(f"Actual done ratio: {actual_done_rate:.2f}")


if __name__ == "__main__":
    main()
