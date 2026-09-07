# SPDX-License-Identifier: MIT-0
"""Deterministic non-ARC fixtures and strict scorers for local-agent readout."""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any

from .frame_memory import PALETTE

SEED = 2060906
MODEL_SEEDS = (0, 1)


def _separated(a: dict[str, int], b: dict[str, int]) -> bool:
    return (
        a["x"] + a["size"] + 1 <= b["x"]
        or b["x"] + b["size"] + 1 <= a["x"]
        or a["y"] + a["size"] + 1 <= b["y"]
        or b["y"] + b["size"] + 1 <= a["y"]
    )


def _center(rectangle: dict[str, int]) -> tuple[int, int]:
    half = rectangle["size"] // 2
    return rectangle["x"] + half, rectangle["y"] + half


def _layouts(layout_seed: int = SEED) -> list[dict[str, Any]]:
    if type(layout_seed) is not int:
        raise ValueError("layout_seed must be an integer")
    rng = random.Random(layout_seed)
    layouts = []
    while len(layouts) < 12:
        rectangles = []
        for _ in range(2):
            size = rng.choice((3, 5))
            rectangles.append(
                {
                    "x": rng.randrange(65 - size),
                    "y": rng.randrange(65 - size),
                    "size": size,
                }
            )
        centers = [_center(rectangle) for rectangle in rectangles]
        if (
            not _separated(*rectangles)
            or centers[0][0] == centers[1][0]
            or (31, 31) in centers
        ):
            continue
        order = sorted(range(2), key=lambda index: centers[index][0])
        palettes = rng.sample(range(1, 16), 4)
        layouts.append(
            {
                "rectangles": [rectangles[index] for index in order],
                "grounding_palettes": palettes[:2],
                "grounding_target": rng.randrange(2),
                "initial_palette": palettes[2],
                "final_palette": palettes[3],
            }
        )
    return layouts


def _frame(rectangles: list[dict[str, int]], palettes: list[int]) -> list[list[int]]:
    frame = [[0] * 64 for _ in range(64)]
    for rectangle, palette in zip(rectangles, palettes, strict=True):
        for y in range(rectangle["y"], rectangle["y"] + rectangle["size"]):
            start = rectangle["x"]
            frame[y][start : start + rectangle["size"]] = [palette] * rectangle["size"]
    return frame


def _observation(frame: list[list[int]]) -> dict[str, Any]:
    return {
        "game_id": "synthetic-readout",
        "guid": "fresh-session",
        "state": "NOT_FINISHED",
        "levels_completed": 0,
        "win_levels": 1,
        "frame": [[row[:] for row in frame]],
        "available_actions": [6],
        "full_reset": False,
    }


def _action(rectangle: dict[str, int]) -> dict[str, int]:
    x, y = _center(rectangle)
    return {"action_id": 6, "x": x, "y": y}


def generate_readout_fixtures(
    *, layout_seed: int = SEED
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    """Return model-visible inputs and index-aligned private oracle metadata."""
    visible: dict[str, list[dict[str, Any]]] = {"grounding": [], "effects": []}
    oracle: dict[str, list[dict[str, Any]]] = {"grounding": [], "effects": []}
    for layout_index, layout in enumerate(_layouts(layout_seed)):
        rectangles = layout["rectangles"]
        target_index = layout["grounding_target"]
        target = rectangles[target_index]
        visible["grounding"].append(
            {
                "observation": _observation(
                    _frame(rectangles, layout["grounding_palettes"])
                ),
                "instruction": (
                    "Click the exact integer centroid of the component whose palette "
                    f"symbol is {PALETTE[layout['grounding_palettes'][target_index]]}."
                ),
            }
        )
        oracle["grounding"].append(
            {
                "layout_index": layout_index,
                "target_action": _action(target),
                "target_bbox": [
                    target["x"],
                    target["y"],
                    target["x"] + target["size"] - 1,
                    target["y"] + target["size"] - 1,
                ],
            }
        )
        final_frame = _frame(rectangles, [layout["final_palette"]] * 2)
        for changed_index in range(2):
            start_palettes = [layout["final_palette"]] * 2
            start_palettes[changed_index] = layout["initial_palette"]
            frames = [_frame(rectangles, start_palettes)]
            changes = []
            for action_index in range(2):
                next_palettes = start_palettes[:]
                next_palettes[action_index] = layout["final_palette"]
                next_frame = _frame(rectangles, next_palettes)
                changes.append(
                    sum(
                        a != b
                        for rows in zip(frames[-1], next_frame, strict=True)
                        for a, b in zip(*rows, strict=True)
                    )
                )
                frames.append(next_frame)
                start_palettes = next_palettes
            actions = [_action(rectangle) for rectangle in rectangles]
            raw = [
                {
                    "observation": _observation(frames[index]),
                    "action": actions[index],
                    "next_observation": _observation(frames[index + 1]),
                }
                for index in range(2)
            ]
            visible["effects"].append(
                {
                    "current_observation": _observation(final_frame),
                    "raw_history": raw,
                    "summary_history": [
                        {"action": action, "changed_pixel_count": count}
                        for action, count in zip(actions, changes, strict=True)
                    ],
                    "instruction": "Repeat the prior coordinate action that changed the visible board.",
                }
            )
            oracle["effects"].append(
                {
                    "layout_index": layout_index,
                    "changed_side": "left" if changed_index == 0 else "right",
                    "target_action": actions[changed_index].copy(),
                    "changed_pixel_counts": changes,
                }
            )
    return visible, oracle


def fixture_digest(*, layout_seed: int = SEED) -> str:
    visible, oracle = generate_readout_fixtures(layout_seed=layout_seed)
    encoded = json.dumps(
        [visible, oracle], sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _prediction_map(
    rows: list[dict[str, Any]], task: str, arms: tuple[str, ...], item_count: int
) -> dict[tuple[int, str, int], dict[str, int]]:
    expected = {
        (seed, arm, item)
        for seed in MODEL_SEEDS
        for arm in arms
        for item in range(item_count)
    }
    predictions = {}
    for row in rows:
        arm = row.get("arm", "grounding")
        seed, item = row.get("seed"), row.get("item")
        if type(seed) is not int or type(item) is not int:
            raise ValueError("prediction seed and item must be integers")
        key = (seed, arm, item)
        action = row.get("action")
        if row.get("task") != task or key not in expected or key in predictions:
            raise ValueError("prediction rows contain an unknown or duplicate item")
        if (
            not isinstance(action, dict)
            or set(action) != {"action_id", "x", "y"}
            or type(action["action_id"]) is not int
            or action["action_id"] != 6
            or any(
                isinstance(action[name], bool)
                or not isinstance(action[name], int)
                or not 0 <= action[name] < 64
                for name in ("x", "y")
            )
        ):
            raise ValueError("prediction action must be a strict ACTION6 tuple")
        predictions[key] = action
    if set(predictions) != expected:
        raise ValueError("prediction rows are incomplete")
    return predictions


def score_grounding(
    rows: list[dict[str, Any]], *, layout_seed: int = SEED, mode: str | None = None
) -> dict[str, Any]:
    _, oracle = generate_readout_fixtures(layout_seed=layout_seed)
    if mode is not None:
        rows = [row for row in rows if row.get("mode") == mode]
    predictions = _prediction_map(rows, "A", ("grounding",), 12)
    by_seed = {}
    for seed in MODEL_SEEDS:
        exact = membership = 0
        for item, target in enumerate(oracle["grounding"]):
            action = predictions[(seed, "grounding", item)]
            exact += action == target["target_action"]
            left, top, right, bottom = target["target_bbox"]
            membership += left <= action["x"] <= right and top <= action["y"] <= bottom
        by_seed[str(seed)] = {"exact_centroid": exact, "target_membership": membership}
    return {
        "by_seed": by_seed,
        "pass": all(value["exact_centroid"] >= 11 for value in by_seed.values()),
    }


def score_effects(
    rows: list[dict[str, Any]], *, layout_seed: int = SEED, mode: str | None = None
) -> dict[str, Any]:
    _, oracle = generate_readout_fixtures(layout_seed=layout_seed)
    if mode is not None:
        rows = [row for row in rows if row.get("mode") == mode]
    predictions = _prediction_map(rows, "B", ("raw", "summary"), 24)
    by_seed = {}
    for seed in MODEL_SEEDS:
        outcomes = {
            "both_correct": 0,
            "raw_only": 0,
            "summary_only": 0,
            "neither": 0,
        }
        raw = summary = 0
        for item, target in enumerate(oracle["effects"]):
            raw_correct = predictions[(seed, "raw", item)] == target["target_action"]
            summary_correct = (
                predictions[(seed, "summary", item)] == target["target_action"]
            )
            raw += raw_correct
            summary += summary_correct
            outcome = (
                "both_correct"
                if raw_correct and summary_correct
                else "raw_only"
                if raw_correct
                else "summary_only"
                if summary_correct
                else "neither"
            )
            outcomes[outcome] += 1
        counts = {"raw": raw, "summary": summary}
        counts["summary_gain"] = counts["summary"] - counts["raw"]
        counts["paired_outcomes"] = outcomes
        by_seed[str(seed)] = counts
    return {
        "by_seed": by_seed,
        "pass": all(
            value["summary"] >= 21 and value["summary_gain"] >= 3
            for value in by_seed.values()
        ),
    }


def score_reasoning_effects(
    rows: list[dict[str, Any]], *, layout_seed: int = SEED
) -> dict[str, Any]:
    """Score paired raw-history reasoning modes for the controlled readout."""
    _, oracle = generate_readout_fixtures(layout_seed=layout_seed)
    modes = ("off", "on")
    if any(row.get("mode") not in modes for row in rows):
        raise ValueError("prediction rows contain an unknown reasoning mode")
    predictions = {
        mode: _prediction_map(
            [row for row in rows if row.get("mode") == mode], "B", ("raw",), 24
        )
        for mode in modes
    }
    by_seed = {}
    for seed in MODEL_SEEDS:
        paired = {"both_correct": 0, "off_only": 0, "on_only": 0, "neither": 0}
        correct = {mode: {} for mode in modes}
        for item, target in enumerate(oracle["effects"]):
            off = predictions["off"][(seed, "raw", item)] == target["target_action"]
            on = predictions["on"][(seed, "raw", item)] == target["target_action"]
            correct["off"][item] = off
            correct["on"][item] = on
            outcome = (
                "both_correct"
                if off and on
                else "off_only"
                if off
                else "on_only"
                if on
                else "neither"
            )
            paired[outcome] += 1
        off = paired["both_correct"] + paired["off_only"]
        on = paired["both_correct"] + paired["on_only"]
        gain = on - off
        counterfactual_pairs = {}
        for mode in modes:
            counts = {
                "both_correct": 0,
                "first_only": 0,
                "second_only": 0,
                "neither_correct": 0,
                "prediction_changed": 0,
                "prediction_same": 0,
            }
            for pair in range(12):
                first, second = 2 * pair, 2 * pair + 1
                first_correct = correct[mode][first]
                second_correct = correct[mode][second]
                outcome = (
                    "both_correct"
                    if first_correct and second_correct
                    else "first_only"
                    if first_correct
                    else "second_only"
                    if second_correct
                    else "neither_correct"
                )
                counts[outcome] += 1
                changed = (
                    predictions[mode][(seed, "raw", first)]
                    != predictions[mode][(seed, "raw", second)]
                )
                counts["prediction_changed" if changed else "prediction_same"] += 1
            counterfactual_pairs[mode] = counts
        by_seed[str(seed)] = {
            "off": off,
            "on": on,
            "gain": gain,
            "paired_mode_outcomes": paired,
            "mode_prediction_changed": sum(
                predictions["off"][(seed, "raw", item)]
                != predictions["on"][(seed, "raw", item)]
                for item in range(24)
            ),
            "counterfactual_pairs": counterfactual_pairs,
            "gate_margins": {
                "on_accuracy_margin_from_21": on - 21,
                "gain_margin_from_3": gain - 3,
            },
        }
    return {
        "by_seed": by_seed,
        "pass": all(
            value["on"] >= 21 and value["gain"] >= 3 for value in by_seed.values()
        ),
        "population_controls": {
            "counterfactual_pairs": 12,
            "target_side_counts": {
                side: sum(
                    target["changed_side"] == side for target in oracle["effects"]
                )
                for side in ("left", "right")
            },
            "unique_target_actions": len(
                {
                    tuple(
                        target["target_action"][key] for key in ("action_id", "x", "y")
                    )
                    for target in oracle["effects"]
                }
            ),
        },
    }
