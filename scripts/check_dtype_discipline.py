#!/usr/bin/env python3
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

FORBIDDEN = [
    re.compile(r"\.to_scalar::<f32>\(\)"),
    re.compile(r"\.to_vec1::<f32>\(\)"),
    re.compile(r"\.to_vec2::<f32>\(\)"),
]

ALLOW_FILES = {"src/util.rs"}

violations = []
for path in SRC.rglob("*.rs"):
    rel = path.relative_to(ROOT).as_posix()
    if rel in ALLOW_FILES:
        continue
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("//") or stripped.startswith("///"):
            continue
        for pattern in FORBIDDEN:
            if pattern.search(line):
                violations.append(f"{rel}:{lineno}:{stripped}")

if violations:
    print("dtype discipline violations found:")
    for violation in violations:
        print(violation)
    print("\nUse util::scalar_f32 / util::vec1_f32 / util::vec2_f32 instead.")
    sys.exit(1)

print("dtype discipline check passed.")
