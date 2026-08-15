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
        "schema": "p2.eval_report.v13",
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
