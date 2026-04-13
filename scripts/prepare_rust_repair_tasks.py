#!/usr/bin/env python3
import argparse
import hashlib
import random
import re
import subprocess
import tempfile
from pathlib import Path


FUNC_NAME_RE = re.compile(r"\bfn\s+([A-Za-z_][A-Za-z0-9_]*)")


def escape_pair_field(text: str) -> str:
    return text.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n")


def unescape_pair_field(text: str) -> str:
    return text.replace("\\n", "\n").replace("\\t", "\t").replace("\\\\", "\\")


def load_pairs(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if not line or "\t" not in line:
                continue
            left, right = line.split("\t", 1)
            rows.append((unescape_pair_field(left), unescape_pair_field(right)))
    return rows


def compile_feedback(rustc_bin: str, code: str, timeout_sec: float) -> str:
    with tempfile.TemporaryDirectory(prefix="tofy-rust-repair-") as td:
        src = Path(td) / "bad.rs"
        src.write_text(code + "\n", encoding="utf-8")
        proc = subprocess.run(
            [rustc_bin, "--crate-type", "lib", str(src)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_sec,
        )
        if proc.returncode == 0:
            return ""
        stderr = proc.stderr.strip()
        lines = [line for line in stderr.splitlines() if line.strip()]
        return "\n".join(lines[:12]).strip()


def corruption_variants(code: str) -> list[tuple[str, str]]:
    variants: list[tuple[str, str]] = []

    variants.append(("stray_role_prefix", "assistant\n" + code))

    if "{" in code:
        variants.append(("missing_open_brace", code.replace("{", "", 1)))
    if "}" in code:
        idx = code.rfind("}")
        variants.append(("missing_close_brace", code[:idx] + code[idx + 1 :]))
    if "->" in code:
        variants.append(("broken_arrow", code.replace("->", "=>", 1)))
    if ")" in code:
        variants.append(("broken_paren", code.replace(")", "]", 1)))

    m = FUNC_NAME_RE.search(code)
    if m:
        name = m.group(1)
        variants.append(("wrong_fn_name", code.replace(f"fn {name}", f"fn broken_{name}", 1)))

    return variants


def build_prompt(task_prompt: str, broken_code: str, feedback: str) -> str:
    return (
        "Return only corrected Rust code.\n"
        "Fix the previous attempt using the compiler feedback.\n\n"
        "Original request:\n"
        f"{task_prompt}\n\n"
        "Previous attempt:\n"
        "```rust\n"
        f"{broken_code}\n"
        "```\n\n"
        "Compiler feedback:\n"
        f"{feedback}\n\n"
        "Rules:\n"
        "- Keep the exact requested function name and signature.\n"
        "- Return only compilable Rust code.\n"
        "- Do not add explanation.\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build Rust code-repair pairs from instruction/function pairs."
    )
    ap.add_argument("--input", required=True, help="Prompt<TAB>correct_code pairs")
    ap.add_argument("--output", required=True)
    ap.add_argument("--rustc", default="rustc")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout-sec", type=float, default=4.0)
    ap.add_argument("--variants-per-sample", type=int, default=2)
    ap.add_argument("--max-rows", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    pairs = load_pairs(Path(args.input))
    if not pairs:
        raise SystemExit(f"no usable pairs found in {args.input}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    seen: set[str] = set()
    with output_path.open("w", encoding="utf-8") as out:
        for task_prompt, correct_code in pairs:
            variants = corruption_variants(correct_code)
            rng.shuffle(variants)
            kept = 0
            for corruption_name, broken_code in variants:
                if args.max_rows > 0 and written >= args.max_rows:
                    break
                feedback = compile_feedback(args.rustc, broken_code, args.timeout_sec)
                if not feedback:
                    if corruption_name == "wrong_fn_name":
                        feedback = "error: function name does not match the requested signature"
                    else:
                        continue
                prompt = build_prompt(task_prompt, broken_code, feedback)
                digest = hashlib.sha1(f"{prompt}\t{correct_code}".encode("utf-8")).hexdigest()
                if digest in seen:
                    continue
                seen.add(digest)
                out.write(
                    f"{escape_pair_field(prompt)}\t{escape_pair_field(correct_code)}\n"
                )
                written += 1
                kept += 1
                if kept >= max(1, args.variants_per_sample):
                    break
            if args.max_rows > 0 and written >= args.max_rows:
                break

    print(f"Wrote {written} repair rows to {output_path}")


if __name__ == "__main__":
    main()
