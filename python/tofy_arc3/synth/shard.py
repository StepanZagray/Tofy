"""Shard exporter/loader for the ADR 0005 §2.6 contract.

Tensors (exact names/dtypes): ``frames u8[N,64,64]``, ``next_frames u8[N,64,64]``,
``actions u8[N,3]``, ``available_actions u8[N]``, ``episode u32[N]``,
``level u16[N]``, ``transition_index u32[N]``, ``rule_id_lo u32[N]``,
``rule_id_hi u32[N]``, ``level_completed u8[N]``, ``game_over u8[N]``.
Sidecar ``manifest.json`` with ``source: "tofy_synth_arcengine"``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from safetensors.numpy import load_file, save_file

from .rollout import Transition
from .rules import split_rule_id

SOURCE = "tofy_synth_arcengine"
MAX_SHARD_ROWS = 50_000

TENSOR_DTYPES: dict[str, tuple[np.dtype, tuple[int, ...]]] = {
    "frames": (np.dtype(np.uint8), (64, 64)),
    "next_frames": (np.dtype(np.uint8), (64, 64)),
    "actions": (np.dtype(np.uint8), (3,)),
    "available_actions": (np.dtype(np.uint8), ()),
    "episode": (np.dtype(np.uint32), ()),
    "level": (np.dtype(np.uint16), ()),
    "transition_index": (np.dtype(np.uint32), ()),
    "rule_id_lo": (np.dtype(np.uint32), ()),
    "rule_id_hi": (np.dtype(np.uint32), ()),
    "level_completed": (np.dtype(np.uint8), ()),
    "game_over": (np.dtype(np.uint8), ()),
}


def rows_to_tensors(rows: Sequence[Transition]) -> dict[str, np.ndarray]:
    n = len(rows)
    if n == 0:
        raise ValueError("cannot write an empty shard")
    if n > MAX_SHARD_ROWS:
        raise ValueError(f"shard has {n} rows > {MAX_SHARD_ROWS}")
    lo_hi = [split_rule_id(r.rule_id) for r in rows]
    t = {
        "frames": np.stack([r.frame for r in rows]).astype(np.uint8),
        "next_frames": np.stack([r.next_frame for r in rows]).astype(np.uint8),
        "actions": np.array([r.action for r in rows], dtype=np.uint8).reshape(n, 3),
        "available_actions": np.array([r.available_actions for r in rows], dtype=np.uint8),
        "episode": np.array([r.episode for r in rows], dtype=np.uint32),
        "level": np.array([r.level for r in rows], dtype=np.uint16),
        "transition_index": np.array([r.transition_index for r in rows], dtype=np.uint32),
        "rule_id_lo": np.array([lo for lo, _ in lo_hi], dtype=np.uint32),
        "rule_id_hi": np.array([hi for _, hi in lo_hi], dtype=np.uint32),
        "level_completed": np.array([int(r.level_completed) for r in rows], dtype=np.uint8),
        "game_over": np.array([int(r.game_over) for r in rows], dtype=np.uint8),
    }
    validate_tensors(t)
    return t


def validate_tensors(t: dict[str, np.ndarray]) -> int:
    """Check names, dtypes and shapes; return N.  Raises on any deviation."""
    if set(t) != set(TENSOR_DTYPES):
        raise ValueError(f"tensor names {sorted(t)} != {sorted(TENSOR_DTYPES)}")
    n = t["frames"].shape[0]
    for name, (dtype, tail) in TENSOR_DTYPES.items():
        arr = t[name]
        if arr.dtype != dtype:
            raise ValueError(f"{name}: dtype {arr.dtype} != {dtype}")
        if arr.shape != (n,) + tail:
            raise ValueError(f"{name}: shape {arr.shape} != {(n,) + tail}")
    if t["frames"].max(initial=0) > 15 or t["next_frames"].max(initial=0) > 15:
        raise ValueError("frame colour index > 15")
    return n


def write_shard(path: Path, rows: Sequence[Transition]) -> dict:
    """Write one safetensors shard; return its manifest entry."""
    tensors = rows_to_tensors(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "file": path.name,
        "rows": int(len(rows)),
        "sha256": digest,
        "episodes": sorted({int(r.episode) for r in rows}),
    }


def read_shard(path: Path) -> dict[str, np.ndarray]:
    t = load_file(str(path))
    validate_tensors(t)
    return t


def write_manifest(path: Path, manifest: dict) -> None:
    if manifest.get("source") != SOURCE:
        raise ValueError("manifest.source must be " + SOURCE)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_manifest(path: Path) -> dict:
    m = json.loads(path.read_text(encoding="utf-8"))
    if m.get("source") != SOURCE:
        raise ValueError(f"manifest source {m.get('source')!r} != {SOURCE!r}")
    return m
