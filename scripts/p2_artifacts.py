#!/usr/bin/env python3
"""Root-authoritative checksum verification for relocatable P2 run artifacts."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def resolve_artifact_path(path: Path, root: Path) -> Path | None:
    """Resolve a root-relative or legacy absolute path strictly beneath ``root``."""
    resolved_root = root.resolve()
    if path.is_absolute():
        matching = [index for index, part in enumerate(path.parts) if part == root.name]
        if not matching:
            return None
        path = Path(*path.parts[matching[-1] + 1 :])
    candidate = (resolved_root / path).resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError:
        return None
    return candidate


def verify_sha256_file(path: Path, root: Path) -> list[str]:
    errors: list[str] = []
    if not path.is_file():
        return [f"missing {path}"]
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2:
            errors.append(f"invalid {path} line {line_number}")
            continue
        expected, artifact_text = parts
        recorded_path = Path(artifact_text.lstrip("*"))
        artifact = resolve_artifact_path(recorded_path, root)
        if artifact is None:
            errors.append(f"hashed artifact is outside experiment root: {recorded_path}")
            continue
        if not artifact.is_file():
            errors.append(f"missing hashed artifact {artifact}")
            continue
        actual = hashlib.sha256(artifact.read_bytes()).hexdigest()
        if actual != expected:
            errors.append(f"sha256 mismatch for {artifact}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--sha-file", required=True, type=Path)
    args = parser.parse_args()
    errors = verify_sha256_file(args.sha_file, args.root)
    if errors:
        for error in errors:
            print(error)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
