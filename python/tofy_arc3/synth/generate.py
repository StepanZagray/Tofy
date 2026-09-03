"""CLI: generate ARCEngine synthetic-game shards.

    python -m tofy_arc3.synth.generate --seed S --games N --out DIR

``--games N`` is the number of base games; each base game is emitted with its
twin, so the run produces ``2N`` games / episodes.  Episodes ``2i`` (base) and
``2i+1`` (twin) share the level-0 initial frame and the random policy prefix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from .levelgen import SHARED_PREFIX_LEN, generate_pair
from .public_games import GENERATED_GAME_ID_PREFIX, PUBLIC_GAME_IDS, assert_not_public
from .rollout import EPS_END, EPS_START, Transition, rollout_episode
from .rules import GameSpec
from .shard import MAX_SHARD_ROWS, SOURCE, write_manifest, write_shard

GAMES_FILE = "games.json"
MANIFEST_FILE = "manifest.json"


def git_revision() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent, capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except Exception:  # noqa: BLE001 - best effort provenance
        return "unknown"


def policy_seed_for(seed: int, index: int) -> int:
    return int.from_bytes(hashlib.blake2b(f"policy:{seed}:{index}".encode(), digest_size=4).digest(), "little")


def rule_census(specs: list[GameSpec]) -> dict:
    return {
        "distinct_rule_ids": len({s.rule.rule_id() for s in specs}),
        "games": len(specs),
        "by_goal": dict(sorted(Counter(s.rule.goal for s in specs).items())),
        "by_action5": dict(sorted(Counter(s.rule.action5 for s in specs).items())),
        "by_action6": dict(sorted(Counter(s.rule.action6 for s in specs).items())),
        "by_move_perm": dict(sorted(Counter("".join(d[0] for d in s.rule.move_perm) for s in specs).items())),
        "by_background": dict(sorted(Counter(str(s.rule.background) for s in specs).items())),
        "by_available_mask": dict(sorted(Counter(str(s.mask) for s in specs).items())),
        "levels_per_game": dict(sorted(Counter(str(len(s.levels)) for s in specs).items())),
    }


def generate(seed: int, games: int, out: Path, episode_steps: int = 250, shard_rows: int = MAX_SHARD_ROWS, n_levels: int | None = None, verbose: bool = True) -> dict:
    if shard_rows > MAX_SHARD_ROWS:
        raise ValueError(f"--shard-rows must be <= {MAX_SHARD_ROWS}")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    specs: list[GameSpec] = []
    episodes: list[dict] = []
    twin_pairs: list[list[int]] = []
    buffer: list[Transition] = []
    shards: list[dict] = []
    total = 0

    def flush() -> None:
        nonlocal buffer
        if not buffer:
            return
        path = out / f"shard-{len(shards):05d}.safetensors"
        shards.append(write_shard(path, buffer))
        buffer = []

    for i in range(games):
        pseed = policy_seed_for(seed, i)
        base, twin = generate_pair(seed, i, pseed, n_levels)
        for k, spec in enumerate((base, twin)):
            ep = 2 * i + k
            rows = rollout_episode(spec, ep, pseed, episode_steps)
            specs.append(spec)
            episodes.append(
                {
                    "episode": ep,
                    "game_id": spec.game_id,
                    "rule_id": spec.rule.rule_id(),
                    "twin_of": spec.twin_of,
                    "levels": len(spec.levels),
                    "available_actions": spec.mask,
                    "rows": len(rows),
                    "levels_completed_events": sum(r.level_completed for r in rows),
                    "game_over_events": sum(r.game_over for r in rows),
                }
            )
            for r in rows:
                buffer.append(r)
                if len(buffer) >= shard_rows:
                    flush()
            total += len(rows)
        twin_pairs.append([2 * i, 2 * i + 1])
        if verbose:
            el = time.perf_counter() - t0
            print(f"[generate] pair {i + 1}/{games} rows={total} ({total / el:.0f} rows/s)", file=sys.stderr)
    flush()
    elapsed = time.perf_counter() - t0

    game_ids = [s.game_id for s in specs]
    assert_not_public(game_ids)
    (out / GAMES_FILE).write_text(json.dumps([s.to_dict() for s in specs], indent=1, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "source": SOURCE,
        "schema": "adr-0005-2.6",
        "generator": {"package": "tofy_arc3.synth", "git_revision": git_revision(), "engine": "arcengine"},
        "seed": seed,
        "args": {"games": games, "episode_steps": episode_steps, "shard_rows": shard_rows, "levels": n_levels},
        "policy": {"epsilon_start": EPS_START, "epsilon_end": EPS_END, "shared_random_prefix_len": SHARED_PREFIX_LEN, "solver": "bfs"},
        "game_id_prefix": GENERATED_GAME_ID_PREFIX,
        "game_ids": game_ids,
        "public_game_ids_excluded": list(PUBLIC_GAME_IDS),
        "public_game_ids_intersection": [],
        "total_rows": total,
        "shards": shards,
        "episodes": episodes,
        "twin_pairs": twin_pairs,
        "twin_pair_count": len(twin_pairs),
        "rule_census": rule_census(specs),
        "games_file": GAMES_FILE,
        "frame_semantics": {
            "frame_source": "last frame of ARCEngine perform_action frame list",
            "level_completed_next_frame": "initial frame of the next level (engine advances within the same action)",
            "reset_rows": "action id 0 rows are emitted when the policy issues RESET (after GAME_OVER/WIN or unsolvable state)",
            "available_actions_mask": "bit0=RESET, bit i=ACTIONi",
        },
    }
    write_manifest(out / MANIFEST_FILE, manifest)
    if verbose:
        print(f"[generate] wrote {total} rows in {len(shards)} shard(s) to {out} in {elapsed:.1f}s ({total / max(elapsed, 1e-9):.0f} rows/s)", file=sys.stderr)
    return {"manifest": manifest, "elapsed_s": elapsed, "rows_per_s": total / max(elapsed, 1e-9)}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--games", type=int, required=True, help="number of base games (each emitted with a twin)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--episode-steps", type=int, default=250, help="transitions per episode")
    p.add_argument("--shard-rows", type=int, default=MAX_SHARD_ROWS)
    p.add_argument("--levels", type=int, default=None, help="levels per game (default: random 4..6)")
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args(argv)
    generate(a.seed, a.games, a.out, a.episode_steps, a.shard_rows, a.levels, verbose=not a.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
