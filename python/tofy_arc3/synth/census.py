"""CLI: acceptance census over a generated shard directory.

    python -m tofy_arc3.synth.census DIR [--random-steps 20000] [--json OUT]

Reports: distinct rule ids, twin pairs, ``single_frame_rule_identifiable``
(must be 0), first-divergence check on twin pairs, and the bounded random-play
level-completion rate per level index (uniform policy over the game's
available actions on the exact simulator).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from .levelgen import _grid_click
from .public_games import PUBLIC_GAME_IDS
from .rules import GameSpec, apply, initial_state
from .shard import read_manifest, read_shard

RANDOM_PLAY_MAX_RATE = 1.0 / 10_000


def _h(arr: np.ndarray) -> str:
    return hashlib.blake2b(np.ascontiguousarray(arr).tobytes(), digest_size=16).hexdigest()


def load_rows(out: Path) -> tuple[dict, dict[str, np.ndarray]]:
    manifest = read_manifest(out / "manifest.json")
    parts = [read_shard(out / s["file"]) for s in manifest["shards"]]
    if not parts:
        raise ValueError("no shards")
    merged = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    return manifest, merged


def twin_census(manifest: dict, t: dict[str, np.ndarray]) -> dict:
    ep = t["episode"]
    ti = t["transition_index"]
    rid = t["rule_id_lo"].astype(np.uint64) | (t["rule_id_hi"].astype(np.uint64) << np.uint64(32))
    by_ep: dict[int, np.ndarray] = {}
    for e in np.unique(ep):
        idx = np.where(ep == e)[0]
        by_ep[int(e)] = idx[np.argsort(ti[idx], kind="stable")]

    frame0_to_rules: dict[str, set[int]] = defaultdict(set)
    for e, idx in by_ep.items():
        frame0_to_rules[_h(t["frames"][idx[0]])].add(int(rid[idx[0]]))

    pairs = manifest["twin_pairs"]
    single_frame_identifiable = 0
    frame0_mismatch = 0
    first_div = {"pass": 0, "fail_frames_or_actions_differ": 0, "no_divergence": 0}
    joint_conflicts = 0
    for a, b in pairs:
        ia, ib = by_ep[a], by_ep[b]
        ha, hb = _h(t["frames"][ia[0]]), _h(t["frames"][ib[0]])
        if ha != hb:
            frame0_mismatch += 1
        if len(frame0_to_rules[ha]) < 2:
            single_frame_identifiable += 1
        n = min(len(ia), len(ib))
        fa, fb = t["frames"][ia[:n]], t["frames"][ib[:n]]
        na, nb = t["next_frames"][ia[:n]], t["next_frames"][ib[:n]]
        aa, ab = t["actions"][ia[:n]], t["actions"][ib[:n]]
        diff = np.where((na != nb).reshape(n, -1).any(axis=1))[0]
        if len(diff) == 0:
            first_div["no_divergence"] += 1
        else:
            d = diff[0]
            if np.array_equal(fa[d], fb[d]) and np.array_equal(aa[d], ab[d]):
                first_div["pass"] += 1
            else:
                first_div["fail_frames_or_actions_differ"] += 1
        # Joint evidence: any (frame, action) seen in both with different next frames.
        seen: dict[tuple[str, bytes], set[str]] = defaultdict(set)
        for idx in (ia, ib):
            for i in idx:
                seen[(_h(t["frames"][i]), t["actions"][i].tobytes())].add(_h(t["next_frames"][i]))
        joint_conflicts += sum(1 for v in seen.values() if len(v) > 1)
    return {
        "twin_pairs": len(pairs),
        "twin_pairs_frame0_mismatch": frame0_mismatch,
        "single_frame_rule_identifiable": single_frame_identifiable,
        "first_divergence": first_div,
        "joint_frame_action_conflicts": joint_conflicts,
        "distinct_rule_ids": int(len(np.unique(rid))),
        "episodes": len(by_ep),
        "rows": int(len(ep)),
        "level_completed_rows": int(t["level_completed"].sum()),
        "game_over_rows": int(t["game_over"].sum()),
        "reset_rows": int((t["actions"][:, 0] == 0).sum()),
        "rows_per_level": {str(int(k)): int(v) for k, v in zip(*np.unique(t["level"], return_counts=True))},
    }


def random_play(spec: GameSpec, level_index: int, steps: int, seed: int) -> dict:
    """Uniform random policy on the exact simulator, starting at ``level_index``.

    An episode ends on death, energy exhaustion or completion (then the level
    resets).  Returns episode/completion counts.
    """
    rng = random.Random(f"census-random:{seed}:{spec.game_id}:{level_index}")
    level = spec.levels[level_index]
    avail = spec.available_actions
    state = initial_state(level)
    energy = level.budget
    episodes = 0
    completions = 0
    for _ in range(steps):
        a = rng.choice(avail)
        click = _grid_click(level, rng.randrange(64), rng.randrange(64)) if a == 6 else None
        state, outcome = apply(level, spec.rule, state, a, click)
        energy -= 1
        if outcome == "complete":
            completions += 1
        if outcome != "ok" or energy <= 0:
            episodes += 1
            state = initial_state(level)
            energy = level.budget
    return {"steps": steps, "episodes": episodes, "completions": completions}


def random_play_census(specs: list[GameSpec], steps: int, seed: int = 0) -> dict:
    per_level: dict[int, dict] = defaultdict(lambda: {"steps": 0, "episodes": 0, "completions": 0, "games": 0})
    worst: list[dict] = []
    for spec in specs:
        for li in range(len(spec.levels)):
            r = random_play(spec, li, steps, seed)
            agg = per_level[li]
            for k in ("steps", "episodes", "completions"):
                agg[k] += r[k]
            agg["games"] += 1
            if li > 0 and r["completions"]:
                worst.append({"game_id": spec.game_id, "level": li, **r, "goal": spec.rule.goal, "budget": spec.levels[li].budget})
    out = {}
    for li in sorted(per_level):
        agg = per_level[li]
        out[str(li)] = {
            **agg,
            "completions_per_episode": agg["completions"] / max(agg["episodes"], 1),
            "completions_per_step": agg["completions"] / max(agg["steps"], 1),
            "tutorial": li == 0,
        }
    non_tut = [v for k, v in out.items() if k != "0"]
    ep_total = sum(v["episodes"] for v in non_tut)
    comp_total = sum(v["completions"] for v in non_tut)
    step_total = sum(v["steps"] for v in non_tut)
    return {
        "steps_per_game_level": steps,
        "per_level": out,
        "non_tutorial": {
            "episodes": ep_total,
            "completions": comp_total,
            "steps": step_total,
            "completions_per_episode": comp_total / max(ep_total, 1),
            "completions_per_step": comp_total / max(step_total, 1),
            "meets_1_in_10000_per_episode": comp_total / max(ep_total, 1) <= RANDOM_PLAY_MAX_RATE,
        },
        "non_tutorial_levels_with_completions": sorted(worst, key=lambda d: -d["completions"])[:20],
    }


def run_census(out: Path, random_steps: int = 20_000) -> dict:
    manifest, t = load_rows(out)
    specs = [GameSpec.from_dict(d) for d in json.loads((out / manifest["games_file"]).read_text())]
    clash = sorted(set(manifest["game_ids"]) & set(PUBLIC_GAME_IDS))
    result = {
        "dir": str(out),
        "source": manifest["source"],
        "generator_revision": manifest["generator"]["git_revision"],
        "seed": manifest["seed"],
        "public_game_ids_intersection": clash,
        "rule_census": manifest["rule_census"],
        "twins": twin_census(manifest, t),
        "random_play": random_play_census(specs, random_steps, manifest["seed"]) if random_steps > 0 else None,
    }
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dir", type=Path)
    p.add_argument("--random-steps", type=int, default=20_000)
    p.add_argument("--json", type=Path, default=None)
    a = p.parse_args(argv)
    result = run_census(a.dir, a.random_steps)
    text = json.dumps(result, indent=2)
    print(text)
    if a.json:
        a.json.write_text(text + "\n")
    tw = result["twins"]
    ok = tw["single_frame_rule_identifiable"] == 0 and tw["twin_pairs_frame0_mismatch"] == 0 and not result["public_game_ids_intersection"]
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
