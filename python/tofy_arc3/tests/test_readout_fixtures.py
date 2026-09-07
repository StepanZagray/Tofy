"""Tests for deterministic controlled readout fixtures and strict scorers."""

from __future__ import annotations

import copy
import unittest

from tofy_arc3.readout_fixtures import (
    fixture_digest,
    generate_readout_fixtures,
    score_effects,
    score_grounding,
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
