#!/usr/bin/env python3
from __future__ import annotations

"""
Build Rust instruction->function pairs from escaped code pair files.

Input format:
- context<TAB>completion with escaped newlines like the output of
  scripts/prepare_github_top_code.py, or
- direct Rust source files from ronantakizawa/github-top-code via --github-top-code.

Output format: one escaped pair per line:
  prompt<TAB>function_code

The prompt is shaped like the hard Rust eval suite:
  Return only Rust code. Implement exactly this function:
  <signature>
"""

import argparse
import hashlib
import re
from pathlib import Path


RUST_TAG_RE = re.compile(r"<lang:rust>\s*<(?:ctx|reply)>\s*", re.IGNORECASE)
FUNC_START_RE = re.compile(
    r"(?m)^(?P<indent>\s*)(?P<sig>(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+[A-Za-z_][A-Za-z0-9_]*[\s\S]*?)\{"
)


def unescape_pair_field(text: str) -> str:
    out = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "\\" and i + 1 < len(text):
            nxt = text[i + 1]
            if nxt == "n":
                out.append("\n")
                i += 2
                continue
            if nxt == "t":
                out.append("\t")
                i += 2
                continue
            if nxt == "\\":
                out.append("\\")
                i += 2
                continue
        out.append(ch)
        i += 1
    return "".join(out)


def escape_pair_field(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace("\r", "")
        .replace("\t", "    ")
        .replace("\n", "\\n")
        .strip()
    )


def strip_tags(text: str) -> str:
    text = unescape_pair_field(text)
    return RUST_TAG_RE.sub("", text, count=1).strip()


def find_matching_brace(src: str, open_idx: int) -> int | None:
    depth = 0
    in_string = False
    in_char = False
    in_line_comment = False
    in_block_comment = 0
    escape = False
    i = open_idx
    while i < len(src):
        ch = src[i]
        nxt = src[i + 1] if i + 1 < len(src) else ""
        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
        elif in_block_comment:
            if ch == "/" and nxt == "*":
                in_block_comment += 1
                i += 1
            elif ch == "*" and nxt == "/":
                in_block_comment -= 1
                i += 1
        elif in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        elif in_char:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == "'":
                in_char = False
        else:
            if ch == "/" and nxt == "/":
                in_line_comment = True
                i += 1
            elif ch == "/" and nxt == "*":
                in_block_comment = 1
                i += 1
            elif ch == '"':
                in_string = True
            elif ch == "'":
                in_char = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return i
        i += 1
    return None


def normalize_signature(signature: str) -> str:
    lines = [line.rstrip() for line in signature.strip().splitlines()]
    return "\n".join(lines).strip()


def extract_rust_functions(src: str) -> list[tuple[str, str]]:
    results: list[tuple[str, str]] = []
    for match in FUNC_START_RE.finditer(src):
        sig = normalize_signature(match.group("sig"))
        open_idx = match.end() - 1
        close_idx = find_matching_brace(src, open_idx)
        if close_idx is None:
            continue
        body = src[match.start("sig") : close_idx + 1].strip()
        line_count = body.count("\n") + 1
        if line_count < 2 or line_count > 120:
            continue
        if len(sig) > 240 or len(body) > 8000:
            continue
        if "fn " not in sig:
            continue
        results.append((sig, body))
    return results


def build_prompt_variants(signature: str) -> list[str]:
    rules = (
        "Rules:\n"
        "- Keep the exact function name and signature.\n"
        "- Return only compilable Rust code for that function.\n"
        "- Do not add explanation.\n"
    )
    return [
        (
            "Return only Rust code. Implement exactly this function:\n"
            f"{signature}\n\n"
            f"{rules}"
        ),
        (
            "Write the Rust function below and return code only:\n"
            f"{signature}\n\n"
            f"{rules}"
        ),
        (
            "Complete this Rust function implementation. Output only the function body in its full function form:\n"
            f"{signature}\n\n"
            f"{rules}"
        ),
    ]


def write_pairs(
    samples: list[tuple[str, str]],
    output_path: Path,
    max_rows: int | None,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    written = 0
    with output_path.open("w", encoding="utf-8") as out:
        for signature, body in samples:
            if max_rows is not None and written >= max_rows:
                break
            for prompt in build_prompt_variants(signature):
                if max_rows is not None and written >= max_rows:
                    break
                digest = hashlib.sha1(f"{prompt}\t{body}".encode("utf-8")).hexdigest()
                if digest in seen:
                    continue
                seen.add(digest)
                out.write(f"{escape_pair_field(prompt)}\t{escape_pair_field(body)}\n")
                written += 1
    return written


def collect_from_pairs(input_path: Path) -> list[tuple[str, str]]:
    samples: list[tuple[str, str]] = []
    with input_path.open("r", encoding="utf-8") as src:
        for line in src:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) != 2:
                continue
            left, right = parts
            combined = "\n".join(
                part for part in (strip_tags(left), strip_tags(right)) if part
            ).strip()
            samples.extend(extract_rust_functions(combined))
    return samples


def collect_from_github_top_code(split: str, max_files: int | None) -> list[tuple[str, str]]:
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError("Install: pip install datasets") from e

    ds = load_dataset("ronantakizawa/github-top-code", split=split)
    ds = ds.filter(lambda row: (row.get("file_language") or "").strip() == "Rust")
    samples: list[tuple[str, str]] = []
    seen_files = 0
    for row in ds:
        if max_files is not None and seen_files >= max_files:
            break
        content = (row.get("content") or "").replace("\r\n", "\n").replace("\r", "\n")
        content = content.strip()
        if not content:
            continue
        samples.extend(extract_rust_functions(content))
        seen_files += 1
    return samples


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build Rust instruction/function pairs from escaped code pair files."
    )
    ap.add_argument("--input", type=Path, help="Input pair file")
    ap.add_argument("--output", type=Path, required=True, help="Output pair file")
    ap.add_argument(
        "--github-top-code",
        action="store_true",
        help="Read Rust source files directly from ronantakizawa/github-top-code",
    )
    ap.add_argument("--split", default="train", choices=("train", "test", "validation"))
    ap.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional file cap when using --github-top-code",
    )
    ap.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row cap for the output dataset",
    )
    args = ap.parse_args()
    if not args.github_top_code and args.input is None:
        ap.error("either --input or --github-top-code is required")
    if args.github_top_code:
        samples = collect_from_github_top_code(args.split, args.max_files)
    else:
        samples = collect_from_pairs(args.input)
    written = write_pairs(samples, args.output, args.max_rows)
    print(f"Wrote {written} Rust instruction pairs to {args.output}")


if __name__ == "__main__":
    main()
