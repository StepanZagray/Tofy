#!/usr/bin/env python3
"""Black-box regression tests for P2 evaluation artifact validation."""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = REPO_ROOT / "scripts" / "p2_validate_eval.sh"
CAMPAIGN = REPO_ROOT / "scripts" / "p2_sigreg_cell_dose_response_v1.sh"
EXPECTED_SEED = 424248
COUNTS = {
    "synthetic_dynamics": {4: 128, 8: 64, 16: 0},
    "synthetic_planner": {4: 122, 8: 94, 16: 33},
}


def horizon_metrics(counts: dict[int, int]) -> dict[str, object]:
    result: dict[str, object] = {}
    for horizon, count in counts.items():
        result[f"n{horizon}"] = count
        result[f"h{horizon}"] = {"n": count, "finite_n": count}
    return result


def report() -> dict[str, object]:
    return {
        "schema": "p2.eval_report.v14",
        "mode": "full",
        "seed": EXPECTED_SEED,
        "board_probe": {
            "population_fingerprint": "fnv1a64:test",
            "metrics": {"trusted": False},
        },
        "synthetic_dynamics": {
            "n_samples": 1408,
            "representation": {
                "effective_rank_fraction": 0.04,
                "noncollapse_pass": False,
            },
            "changed_transitions": {
                "improvement_fraction": -1.0,
                "improvement_ci95_low": -2.0,
                "ten_percent_improvement_pass": False,
            },
            "action_diagnostics": {
                "aggregate": {
                    "shuffle": {
                        "ratio": 1.0,
                        "ratio_ci95_low": 0.99,
                        "action_conditioning_pass": False,
                    }
                }
            },
            "rollout": {
                **horizon_metrics(COUNTS["synthetic_dynamics"]),
                "h4": {
                    "n": 128,
                    "finite_n": 128,
                    "normalized_mean": 1.0,
                    "normalized_cvar95": 2.0,
                },
                "h8": {
                    "n": 64,
                    "finite_n": 64,
                    "normalized_mean": 1.0,
                    "normalized_cvar95": 2.0,
                    "fraction_beating_copy": 0.0,
                },
            },
            "q": {
                "n": 1408,
                "brier": 0.1,
                "positive_label_rate": 0.0,
                "saturated": True,
                "balanced_accuracy": None,
            },
        },
        "synthetic_planner": {
            "rollout": horizon_metrics(COUNTS["synthetic_planner"]),
        },
        "outcome_counterfactuals": outcome_counterfactuals(),
    }


def counterfactual_interval(groups: int, pairs: int) -> dict[str, object]:
    return {
        "estimate": 0.05,
        "lower_95": 0.01,
        "upper_95": 0.09,
        "lower_98_75": 0.0,
        "upper_98_75": 0.1,
        "groups": groups,
        "pairs": pairs,
        "resamples": 10_000,
        "unit": "whole_branch_group",
    }


def outcome_counterfactuals() -> dict[str, object]:
    digest = "sha256:" + "1" * 64
    ledger = []
    for group_index in range(256):
        population = "movement" if group_index % 2 == 0 else "coordinate"
        group = {
            "group_index": group_index,
            "family": "factual_branch",
            "population": population,
            "content_fingerprint": digest,
            "current_sha256": digest,
            "next_sha256": [digest] * 4,
            "actions": [{"id": action} for action in range(1, 5)],
        }
        for left in range(4):
            for right in range(left + 1, 4):
                changed_unchanged = left == 0 and right == 1
                ledger.append(
                    {
                        "group": group,
                        "left_branch_index": left,
                        "right_branch_index": right,
                        "left_action": {"id": left + 1},
                        "right_action": {"id": right + 1},
                        "left_outcome_class": left,
                        "right_outcome_class": right,
                        "left_changed": not changed_unchanged,
                        "right_changed": True,
                        "left_changed_cells": [] if changed_unchanged else [0],
                        "right_changed_cells": [1],
                        "target_pair_mse": 1.0,
                        "concordant_loss": 0.95,
                        "crossed_loss": 1.05,
                        "margin": 0.05,
                        "eligible": True,
                        "reason": "distinct_canonical_board_outcomes",
                    }
                )
    anchors = {str(action): {"changed": 64, "unchanged": 64} for action in range(1, 5)}
    return {
        "population_fingerprint": digest,
        "groups": 256,
        "movement_groups": 128,
        "coordinate_groups": 128,
        "unordered_pairs": 1536,
        "eligible_pairs": 1536,
        "outcome_equivalent_pairs": 0,
        "changed_changed_pairs": 1280,
        "changed_unchanged_pairs": 256,
        "epsilon": 1e-30,
        "material_threshold": 0.1,
        "overall": counterfactual_interval(256, 1536),
        "movement": counterfactual_interval(128, 768),
        "coordinate": counterfactual_interval(128, 768),
        "changed_changed": counterfactual_interval(256, 1280),
        "changed_unchanged": counterfactual_interval(256, 256),
        "action_separation_pass": False,
        "controls": {
            "pixel_oracle_estimate": 1.0,
            "pixel_oracle_exactly_one": True,
            "latent_oracle_estimate": 1.0,
            "latent_oracle_at_least_0_99": True,
            "target_collapse_failure": False,
            "target_collapsed_pairs": 0,
            "swapped_oracle_estimate": -1.0,
            "swapped_oracle_at_most_negative_0_99": True,
            "action_masked_max_abs_margin": 0.0,
            "action_masked_max_abs_at_most_1e_6": True,
            "identity_max_abs_margin": 0.0,
            "identity_max_abs_at_most_1e_6": True,
            "outcome_equivalent_pairs": 0,
            "outcome_equivalent_max_abs_margin": 0.0,
            "outcome_equivalent_max_abs_at_most_1e_6": True,
            "state_scrambled_same_action_template": {
                "available": True,
                "estimate": 0.0,
                "groups": 128,
                "pairs": 768,
                "reason": None,
            },
            "required_controls_pass": True,
        },
        "population_gates": {
            "eligible_simulator_groups": 128,
            "eligible_simulator_groups_at_least_100": True,
            "movement_action_anchors": anchors,
            "each_movement_action_at_least_16_changed_and_16_unchanged": True,
            "simulator_changed_changed_pairs": 640,
            "simulator_changed_changed_pairs_at_least_100": True,
            "target_collapse_failure": False,
            "population_pass": True,
        },
        "pair_ledger": ledger,
        "ledger_reconciled": True,
    }


def rows() -> list[dict[str, object]]:
    result = []
    episode_id = 0
    for source, counts in COUNTS.items():
        for horizon, count in counts.items():
            for _ in range(count):
                result.append(
                    {
                        "schema": "p2.episode_rollout.v2",
                        "source": source,
                        "seed": EXPECTED_SEED,
                        "episode_id": episode_id,
                        "families_through_horizon": ["test"],
                        "horizon": horizon,
                        "open_mse": 0.2,
                        "closed_mse": 0.2,
                        "copy_forward_mse": 0.1,
                        "normalized_open_mse": 2.0,
                    }
                )
                episode_id += 1
    return result


class EvalValidationTests(unittest.TestCase):
    def test_campaign_uses_validator_interface_without_fixed_total(self) -> None:
        campaign = CAMPAIGN.read_text()

        self.assertIn('eval_validator="$script_dir/p2_validate_eval.sh"', campaign)
        self.assertNotIn('wc -l <"$episodes"', campaign)
        self.assertEqual(campaign.count('bash "$eval_validator"'), 2)

    def write_fixture(
        self, directory: Path, episode_rows: list[dict[str, object]]
    ) -> tuple[Path, Path]:
        report_path = directory / "eval_report.json"
        episodes_path = directory / "episodes.jsonl"
        report_path.write_text(json.dumps(report()) + "\n")
        episodes_path.write_text(
            "".join(json.dumps(row) + "\n" for row in episode_rows)
        )
        return report_path, episodes_path

    def validate(self, report_path: Path, episodes_path: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(VALIDATOR), str(report_path), str(episodes_path), str(EXPECTED_SEED)],
            text=True,
            capture_output=True,
            check=False,
        )

    def test_accepts_report_reconciled_441_row_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            report_path, episodes_path = self.write_fixture(Path(temp), rows())

            completed = self.validate(report_path, episodes_path)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(len(episodes_path.read_text().splitlines()), 441)

    def test_rejects_truncated_episode_stream(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            report_path, episodes_path = self.write_fixture(Path(temp), rows()[:-1])

            completed = self.validate(report_path, episodes_path)

            self.assertNotEqual(completed.returncode, 0)

    def test_rejects_malformed_episode_stream(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            report_path, episodes_path = self.write_fixture(Path(temp), rows())
            with episodes_path.open("a") as stream:
                stream.write("{malformed\n")

            completed = self.validate(report_path, episodes_path)

            self.assertNotEqual(completed.returncode, 0)

    def test_rejects_multiple_json_values_on_one_jsonl_line(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            report_path, episodes_path = self.write_fixture(Path(temp), rows())
            episodes_path.write_text("".join(json.dumps(row) for row in rows()) + "\n")

            completed = self.validate(report_path, episodes_path)

            self.assertNotEqual(completed.returncode, 0)

    def test_rejects_structurally_corrupt_episode_row(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            corrupt_rows = rows()
            corrupt_rows[0]["source"] = "unregistered_source"
            report_path, episodes_path = self.write_fixture(Path(temp), corrupt_rows)

            completed = self.validate(report_path, episodes_path)

            self.assertNotEqual(completed.returncode, 0)

    def test_rejects_wrong_episode_seed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            corrupt_rows = rows()
            corrupt_rows[0]["seed"] = EXPECTED_SEED + 1
            report_path, episodes_path = self.write_fixture(Path(temp), corrupt_rows)

            completed = self.validate(report_path, episodes_path)

            self.assertNotEqual(completed.returncode, 0)

    def test_rejects_duplicate_row_substituted_for_a_missing_episode(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            corrupt_rows = rows()
            corrupt_rows[-1] = corrupt_rows[-2].copy()
            report_path, episodes_path = self.write_fixture(Path(temp), corrupt_rows)

            completed = self.validate(report_path, episodes_path)

            self.assertNotEqual(completed.returncode, 0)


if __name__ == "__main__":
    unittest.main()
