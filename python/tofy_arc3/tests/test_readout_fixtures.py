"""Tests for deterministic controlled readout fixtures and strict scorers."""

from __future__ import annotations

import copy
import unittest

from tofy_arc3.readout_fixtures import (
    fixture_digest,
    generate_readout_fixtures,
    score_effects,
    score_grounding,
    score_reasoning_effects,
)


def oracle_predictions(
    task: str, oracle: list[dict], arms: tuple[str, ...]
) -> list[dict]:
    return [
        {
            "task": task,
            "seed": seed,
            "arm": arm,
            "item": item,
            "action": copy.deepcopy(target["target_action"]),
        }
        for seed in (0, 1)
        for arm in arms
        for item, target in enumerate(oracle)
    ]


class ReadoutFixtureTests(unittest.TestCase):
    def test_paired_current_frames_alias_with_opposite_answers(self) -> None:
        visible, oracle = generate_readout_fixtures()
        for layout in range(12):
            left, right = 2 * layout, 2 * layout + 1
            self.assertEqual(
                visible["effects"][left]["current_observation"],
                visible["effects"][right]["current_observation"],
            )
            self.assertEqual(
                visible["effects"][left]["instruction"],
                visible["effects"][right]["instruction"],
            )
            self.assertNotEqual(
                oracle["effects"][left]["target_action"],
                oracle["effects"][right]["target_action"],
            )
        visible["effects"][0]["raw_history"][0]["action"]["x"] = (
            oracle["effects"][0]["target_action"]["x"] + 1
        ) % 64
        self.assertNotEqual(
            visible["effects"][0]["raw_history"][0]["action"],
            oracle["effects"][0]["target_action"],
        )

    def test_historical_changes_are_rectangle_area_then_zero(self) -> None:
        visible, oracle = generate_readout_fixtures()
        for item, target in zip(visible["effects"], oracle["effects"], strict=True):
            counts = target["changed_pixel_counts"]
            self.assertIn(max(counts), (9, 25))
            self.assertEqual(min(counts), 0)
            self.assertEqual(
                counts,
                [summary["changed_pixel_count"] for summary in item["summary_history"]],
            )
            for transition, count in zip(item["raw_history"], counts, strict=True):
                before = transition["observation"]["frame"][0]
                after = transition["next_observation"]["frame"][0]
                self.assertEqual(
                    sum(
                        a != b
                        for rows in zip(before, after, strict=True)
                        for a, b in zip(*rows, strict=True)
                    ),
                    count,
                )

    def test_grounding_labels_are_valid_and_fixed_center_fails(self) -> None:
        visible, oracle = generate_readout_fixtures()
        for item, target in zip(visible["grounding"], oracle["grounding"], strict=True):
            action = target["target_action"]
            self.assertNotEqual((action["x"], action["y"]), (31, 31))
            left, top, right, bottom = target["target_bbox"]
            self.assertTrue(
                left <= action["x"] <= right and top <= action["y"] <= bottom
            )
            self.assertNotIn("fixture", item["instruction"].lower())
            self.assertNotIn("answer", item["instruction"].lower())
        fixed = [
            {**row, "action": {"action_id": 6, "x": 31, "y": 31}}
            for row in oracle_predictions("A", oracle["grounding"], ("grounding",))
        ]
        self.assertEqual(score_grounding(fixed)["by_seed"]["0"]["exact_centroid"], 0)

    def test_oracle_scores_perfect_and_digest_repeats(self) -> None:
        _, oracle = generate_readout_fixtures()
        grounding = oracle_predictions("A", oracle["grounding"], ("grounding",))
        effects = oracle_predictions("B", oracle["effects"], ("raw", "summary"))
        for row in effects:
            if row["arm"] == "raw":
                row["action"] = {"action_id": 6, "x": 31, "y": 31}
        self.assertEqual(
            score_grounding(grounding)["by_seed"]["1"]["exact_centroid"], 12
        )
        effect_score = score_effects(effects)
        self.assertEqual(effect_score["by_seed"]["0"]["summary"], 24)
        self.assertEqual(effect_score["by_seed"]["0"]["raw"], 0)
        self.assertEqual(
            effect_score["by_seed"]["0"]["paired_outcomes"],
            {"both_correct": 0, "raw_only": 0, "summary_only": 24, "neither": 0},
        )
        self.assertTrue(effect_score["pass"])
        self.assertEqual(fixture_digest(), fixture_digest())
        self.assertEqual(
            fixture_digest(),
            "313ac9d7f554d771bbe86fe4040246f59c22cb2fe7cd1c52dbc805bde4a3e24e",
        )

    def test_effect_score_requires_complete_unique_rows_and_seeds(self) -> None:
        _, oracle = generate_readout_fixtures()
        rows = oracle_predictions("B", oracle["effects"], ("raw", "summary"))
        with self.assertRaisesRegex(ValueError, "incomplete"):
            score_effects(rows[:-1])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            score_effects(rows + [copy.deepcopy(rows[0])])

    def test_alternate_layout_seed_is_fixed_and_mode_filter_is_strict(self) -> None:
        layout_seed = 2060907
        _, oracle = generate_readout_fixtures(layout_seed=layout_seed)
        self.assertEqual(
            fixture_digest(layout_seed=layout_seed),
            "30715f4a1cfe932bd480f149db2bde8394662a0953d613b9cb6c8207b29b7841",
        )
        rows = oracle_predictions("A", oracle["grounding"], ("grounding",))
        on = [{**row, "mode": "on"} for row in rows]
        off = [
            {**row, "mode": "off", "action": {"action_id": 6, "x": 31, "y": 31}}
            for row in rows
        ]
        self.assertTrue(
            score_grounding(on + off, layout_seed=layout_seed, mode="on")["pass"]
        )
        self.assertFalse(
            score_grounding(on + off, layout_seed=layout_seed, mode="off")["pass"]
        )
        with self.assertRaisesRegex(ValueError, "layout_seed must be an integer"):
            generate_readout_fixtures(layout_seed=True)

    def test_reasoning_effect_score_reports_paired_discordance(self) -> None:
        _, oracle = generate_readout_fixtures(layout_seed=2060907)
        rows = []
        for seed in (0, 1):
            for mode in ("off", "on"):
                for item, target in enumerate(oracle["effects"]):
                    action = copy.deepcopy(target["target_action"])
                    if mode == "off" and item < 3:
                        action["x"] = (action["x"] + 1) % 64
                    rows.append(
                        {
                            "task": "B",
                            "seed": seed,
                            "mode": mode,
                            "arm": "raw",
                            "item": item,
                            "action": action,
                        }
                    )
        result = score_reasoning_effects(rows, layout_seed=2060907)
        self.assertTrue(result["pass"])
        self.assertEqual(result["by_seed"]["0"]["gain"], 3)
        self.assertEqual(
            result["by_seed"]["0"]["paired_mode_outcomes"],
            {"both_correct": 21, "off_only": 0, "on_only": 3, "neither": 0},
        )
        self.assertEqual(result["by_seed"]["0"]["mode_prediction_changed"], 3)
        self.assertEqual(
            result["by_seed"]["0"]["counterfactual_pairs"]["on"],
            {
                "both_correct": 12,
                "first_only": 0,
                "second_only": 0,
                "neither_correct": 0,
                "prediction_changed": 12,
                "prediction_same": 0,
            },
        )
        self.assertEqual(
            result["by_seed"]["0"]["gate_margins"],
            {
                "on_accuracy_margin_from_21": 3,
                "gain_margin_from_3": 0,
            },
        )
        self.assertEqual(
            result["population_controls"],
            {
                "counterfactual_pairs": 12,
                "target_side_counts": {"left": 12, "right": 12},
                "unique_target_actions": 24,
            },
        )
        with self.assertRaisesRegex(ValueError, "unknown reasoning mode"):
            score_reasoning_effects([{**rows[0], "mode": "summary"}] + rows[1:])
        with self.assertRaisesRegex(ValueError, "incomplete"):
            score_reasoning_effects(rows[:-1])

    def test_score_rejects_non_integer_keys_and_has_no_ceiling_exception(self) -> None:
        _, oracle = generate_readout_fixtures()
        rows = oracle_predictions("B", oracle["effects"], ("raw", "summary"))
        self.assertFalse(score_effects(rows)["pass"])
        self.assertEqual(
            score_effects(rows)["by_seed"]["1"]["paired_outcomes"]["both_correct"],
            24,
        )
        for field in ("seed", "item"):
            for value in (True, 0.0):
                malformed = copy.deepcopy(rows)
                malformed[0][field] = value
                with (
                    self.subTest(field=field, value=value),
                    self.assertRaisesRegex(ValueError, "must be integers"),
                ):
                    score_effects(malformed)
        for value in (True, 6.0):
            malformed = copy.deepcopy(rows)
            malformed[0]["action"]["action_id"] = value
            with (
                self.subTest(action_id=value),
                self.assertRaisesRegex(ValueError, "strict ACTION6"),
            ):
                score_effects(malformed)


if __name__ == "__main__":
    unittest.main()
